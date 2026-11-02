"""
NIH ChestX-ray14 Kaggle Ingestion, Remote Archive Shard Inspector, Disk Preflight, and Shard Extractor.

Official Dataset Source:
- Kaggle Dataset: `nih-chest-xrays/data` (https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Primary Citation: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri,
  Ronald M. Summers. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
  Weakly-Supervised Classification and Localization of Common Thorax Diseases." CVPR 2017.
  https://arxiv.org/abs/1705.02315
- Originating Institution: NIH Clinical Center (112,120 frontal-view radiographs, 30,805 patients).

Governance & Safety Guardrails:
1. ZERO dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
2. Enforces disk-space preflight checks before downloading or extracting archives
   (prevents exhausting constrained partitions such as A:\\ with ~30.8 GB free).
3. Verifies Kaggle credentials (`kaggle.json` or env vars) are never tracked or unignored inside Git.
4. Uses HTTP Range ZIP64 Central Directory inspection + 32 MiB buffered streaming from Kaggle's
   signed Google Cloud Storage archive so metadata or individual shards (`images_001`..`images_012`)
   download cleanly with built-in ZIP CRC-32 + SHA-256 verification without fetching all 42 GiB.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET_SLUG = "nih-chest-xrays/data"
METADATA_FILENAMES = [
    "Data_Entry_2017.csv",
    "BBox_List_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
]
DOC_FILENAMES = [
    "ARXIV_V5_CHESTXRAY.pdf",
    "FAQ_CHESTXRAY.pdf",
    "LOG_CHESTXRAY.pdf",
    "README_CHESTXRAY.pdf",
]
ALL_SHARD_NUMBERS = list(range(1, 13))

# Conservative free-space requirements (GiB) to prevent disk exhaustion
MIN_FREE_GB_METADATA_ONLY = 1.0
MIN_FREE_GB_FULL_CLEANUP = 60.0
MIN_FREE_GB_FULL_KEEP_ZIPS = 95.0
PER_SHARD_ZIP_PLUS_EXTRACT_GB = 8.0
PER_SHARD_CLEANUP_GB = 4.5


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when target drive lacks sufficient free disk space for requested NIH ingestion mode."""


class SecurityConfigurationError(RuntimeError):
    """Raised when Kaggle API credentials violate repository security policies."""


class KaggleCredentialsMissingError(RuntimeError):
    """Raised when Kaggle authentication credentials cannot be located and public access fails."""


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hex digest of a file in 1 MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_existing_parent(path: Path) -> Path:
    curr = path.resolve()
    while not curr.exists() and curr.parent != curr:
        curr = curr.parent
    return curr


def get_free_disk_gb(target_path: Path) -> float:
    """Returns free disk space in GiB on the volume backing `target_path`."""
    probe = _resolve_existing_parent(target_path)
    usage = shutil.disk_usage(probe)
    return round(usage.free / (1024 ** 3), 2)


def get_all_local_drives_audit() -> Dict[str, Dict[str, Any]]:
    """Audits total/used/free GiB across available Windows drive roots (e.g. C:\\, A:\\)."""
    audits: Dict[str, Dict[str, Any]] = {}
    candidates = [f"{ch}:\\" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{ch}:\\").exists()] if os.name == "nt" else ["/"]
    for drv in candidates:
        try:
            u = shutil.disk_usage(drv)
            audits[drv] = {
                "total_gb": round(u.total / (1024 ** 3), 2),
                "used_gb": round(u.used / (1024 ** 3), 2),
                "free_gb": round(u.free / (1024 ** 3), 2),
            }
        except Exception:
            continue
    return audits


def required_free_gb_for_plan(
    mode: str,
    shards: Sequence[int],
    cleanup_zips: bool,
) -> float:
    """Computes required free space in GiB for the requested download/extraction mode."""
    if mode == "metadata_only":
        return MIN_FREE_GB_METADATA_ONLY
    if mode == "full":
        return MIN_FREE_GB_FULL_CLEANUP if cleanup_zips else MIN_FREE_GB_FULL_KEEP_ZIPS
    if mode == "shards":
        n_shards = max(1, len(shards))
        if cleanup_zips:
            return round(max(6.0, 2.0 + n_shards * PER_SHARD_CLEANUP_GB), 2)
        return round(max(8.0, 2.0 + n_shards * PER_SHARD_ZIP_PLUS_EXTRACT_GB), 2)
    raise ValueError(f"Unsupported mode: {mode}")


def check_disk_space(target_dir: Path, required_gb: float) -> Dict[str, Any]:
    """
    Verifies that `target_dir` has at least `required_gb` GiB free.
    Raises InsufficientDiskSpaceError if free space is below threshold.
    """
    probe = _resolve_existing_parent(target_dir)
    usage = shutil.disk_usage(probe)
    total_gb = round(usage.total / (1024 ** 3), 2)
    used_gb = round(usage.used / (1024 ** 3), 2)
    free_gb = round(usage.free / (1024 ** 3), 2)
    drive_label = probe.drive or str(probe)

    if free_gb < required_gb:
        raise InsufficientDiskSpaceError(
            f"Insufficient disk space on drive '{drive_label}' (target: {target_dir.resolve()}): "
            f"available={free_gb:.2f} GB, required={required_gb:.2f} GB. "
            "Use a larger partition (e.g. C:\\ with >100 GB free) or run incremental batch mode "
            "(`--mode shards --shards 1 --cleanup-zips`)."
        )

    return {
        "drive": drive_label,
        "target_dir": str(target_dir.resolve()),
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "required_gb": required_gb,
        "preflight_passed": True,
    }


def verify_git_token_safety(repo_root: Path = PROJECT_ROOT) -> None:
    """Ensures no `kaggle.json` file inside `repo_root` is tracked or unignored by Git."""
    candidates = list(repo_root.rglob("kaggle.json"))
    for cand in candidates:
        try:
            res = subprocess.run(
                ["git", "check-ignore", "-q", str(cand)],
                cwd=str(repo_root),
                capture_output=True,
                timeout=5,
                check=False,
            )
            if res.returncode != 0:
                raise SecurityConfigurationError(
                    f"SECURITY BLOCK: Found unignored credential file inside repository: {cand}. "
                    "Move `kaggle.json` to `~/.kaggle/kaggle.json` (outside the repo) and ensure `kaggle.json` is in `.gitignore`."
                )
        except FileNotFoundError:
            raise SecurityConfigurationError(
                f"SECURITY BLOCK: Credential file `{cand}` is stored inside repository directory `{repo_root}`. "
                "Store `kaggle.json` in `%USERPROFILE%\\.kaggle\\kaggle.json` instead."
            )


def _find_kaggle_cli() -> List[str]:
    venv_kaggle = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        return [which_kaggle]
    raise RuntimeError(
        "Kaggle CLI executable not found. Install it via `.\\venv\\Scripts\\python.exe -m pip install kaggle`."
    )


def check_kaggle_credentials(
    custom_config_dir: Optional[Path] = None,
    allow_public_cc0: bool = True,
) -> Dict[str, Any]:
    """
    Verifies Kaggle access either via explicit credentials (`kaggle.json` / env vars)
    or verified anonymous `CC0-1.0` access supported by Kaggle SDK v0.1.38+ / CLI v2.2+.
    Never logs secret token values.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"auth_method": "env:KAGGLE_API_TOKEN", "configured": True}
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {
            "auth_method": "env:KAGGLE_USERNAME+KAGGLE_KEY",
            "username": os.environ.get("KAGGLE_USERNAME"),
            "configured": True,
        }

    cfg_dir = (
        Path(custom_config_dir)
        if custom_config_dir
        else Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    )
    token_file = cfg_dir / "kaggle.json"
    if token_file.exists() and token_file.is_file():
        return {
            "auth_method": "file:kaggle.json",
            "token_location": str(token_file),
            "configured": True,
        }

    if allow_public_cc0:
        try:
            url, total_bytes = resolve_kaggle_signed_archive_url()
            if url and total_bytes > 0:
                return {
                    "auth_method": "kagglesdk_public_cc0_signed_gcs_stream",
                    "token_location": None,
                    "configured": True,
                    "remote_archive_bytes": total_bytes,
                }
        except Exception:
            pass

    raise KaggleCredentialsMissingError(
        "Kaggle credentials not found and anonymous dataset probe failed. Place `kaggle.json` at "
        f"`{Path.home() / '.kaggle' / 'kaggle.json'}` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` "
        "(or `KAGGLE_API_TOKEN`) environment variables."
    )


def resolve_kaggle_signed_archive_url(dataset_slug: str = KAGGLE_DATASET_SLUG) -> Tuple[str, int]:
    """
    Resolves Kaggle's temporary signed Google Cloud Storage archive URL and exact byte length
    for `owner/dataset` without downloading its payload.
    """
    from kagglesdk import KaggleClient
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    owner_slug, ds_name = dataset_slug.split("/", 1)
    with KaggleClient() as client:
        req = ApiDownloadDatasetRequest()
        req.owner_slug = owner_slug
        req.dataset_slug = ds_name
        resp = client.datasets.dataset_api_client.download_dataset(req)
        try:
            signed_url = str(resp.url)
            content_length = int(resp.headers.get("Content-Length", "0"))
            accept_ranges = resp.headers.get("Accept-Ranges", "")
        finally:
            resp.close()

    if not signed_url or content_length <= 0:
        raise RuntimeError(f"Failed to resolve signed archive URL for {dataset_slug}")
    if "bytes" not in accept_ranges.lower():
        raise RuntimeError(f"Remote archive server does not advertise byte-range support: {accept_ranges}")
    return signed_url, content_length


class SeekableHttpRangeReader(io.RawIOBase):
    """
    Seekable read-only HTTP byte-range file object for standard-library `zipfile.ZipFile`.
    - Caches the final `tail_cache_bytes` (24 MiB) in memory so ZIP64 Central Directory parsing
      requires exactly 1 HTTP request.
    - Uses a sequential `read_ahead_bytes` (32 MiB) sliding window so streaming thousands of
      contiguous shard PNGs runs at full CDN throughput with O(32 MiB) RAM.
    """

    def __init__(
        self,
        url_resolver: Callable[[], Tuple[str, int]],
        read_ahead_bytes: int = 32 * 1024 * 1024,
        tail_cache_bytes: int = 24 * 1024 * 1024,
    ):
        super().__init__()
        self._url_resolver = url_resolver
        self.url, self.size = self._url_resolver()
        self.pos = 0
        self.read_ahead_bytes = read_ahead_bytes
        self.tail_cache_bytes = min(tail_cache_bytes, self.size)
        self._session = requests.Session()

        self._tail_start = self.size - self.tail_cache_bytes
        self._tail_buf: Optional[bytes] = None

        self._win_start = -1
        self._win_buf = b""
        self.http_range_requests = 0
        self.bytes_downloaded = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence={whence}")
        if new_pos < 0:
            raise ValueError(f"Negative seek position: {new_pos}")
        self.pos = min(new_pos, self.size)
        return self.pos

    def _fetch_range(self, start: int, end_inclusive: int) -> bytes:
        end_inclusive = min(end_inclusive, self.size - 1)
        if start > end_inclusive:
            return b""
        headers = {"Range": f"bytes={start}-{end_inclusive}"}
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                resp = self._session.get(self.url, headers=headers, timeout=90)
                if resp.status_code in (401, 403):
                    self.url, _ = self._url_resolver()
                    resp = self._session.get(self.url, headers=headers, timeout=90)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", str(5 * (attempt + 1))))
                    logger.warning("Rate limited (HTTP 429); backing off %.1fs before retry...", retry_after)
                    time.sleep(min(retry_after, 60.0))
                    continue
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP Range {start}-{end_inclusive} returned status {resp.status_code}")
                data = resp.content
                expected_len = end_inclusive - start + 1
                if resp.status_code == 206 and len(data) != expected_len:
                    raise RuntimeError(
                        f"Incomplete HTTP 206 Range {start}-{end_inclusive}: got {len(data)} bytes, expected {expected_len}"
                    )
                self.http_range_requests += 1
                self.bytes_downloaded += len(data)
                return data
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed HTTP Range {start}-{end_inclusive} after {max_attempts} attempts: {exc}"
                    ) from exc
                backoff_s = min(30.0, 2.0 ** attempt)
                logger.warning(
                    "Transient network error on Range %d-%d (attempt %d/%d): %s; retrying in %.1fs...",
                    start,
                    end_inclusive,
                    attempt + 1,
                    max_attempts,
                    exc,
                    backoff_s,
                )
                time.sleep(backoff_s)
        return b""

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        n_want = min(len(b), self.size - self.pos)
        if n_want <= 0:
            return 0

        # 1. Serve from cached EOF tail if request starts inside tail region
        if self.pos >= self._tail_start:
            if self._tail_buf is None:
                self._tail_buf = self._fetch_range(self._tail_start, self.size - 1)
            rel = self.pos - self._tail_start
            chunk = self._tail_buf[rel : rel + n_want]
            n_got = len(chunk)
            b[:n_got] = chunk
            self.pos += n_got
            return n_got

        # 2. Serve from current sequential read-ahead window if available
        if self._win_start >= 0 and self._win_start <= self.pos < self._win_start + len(self._win_buf):
            rel = self.pos - self._win_start
            chunk = self._win_buf[rel : rel + n_want]
            n_got = len(chunk)
            b[:n_got] = chunk
            self.pos += n_got
            return n_got

        # 3. Populate new sequential read-ahead window
        fetch_len = max(n_want, self.read_ahead_bytes)
        self._win_start = self.pos
        self._win_buf = self._fetch_range(self.pos, self.pos + fetch_len - 1)
        chunk = self._win_buf[:n_want]
        n_got = len(chunk)
        b[:n_got] = chunk
        self.pos += n_got
        return n_got

    def close(self) -> None:
        try:
            self._session.close()
        finally:
            super().close()


def inspect_remote_kaggle_archive(dataset_slug: str = KAGGLE_DATASET_SLUG) -> Dict[str, Any]:
    """
    Reads ONLY the ZIP64 Central Directory of Kaggle's remote `nih-chest-xrays/data` archive
    to return exact file counts, compressed sizes, and uncompressed sizes per shard (`images_001`..`images_012`)
    and per top-level metadata/documentation file without downloading image payloads.
    """
    raw_reader = SeekableHttpRangeReader(lambda: resolve_kaggle_signed_archive_url(dataset_slug))
    shard_re = re.compile(r"^(images_\d{3})/", re.IGNORECASE)

    shards_summary: Dict[str, Dict[str, Any]] = {
        f"images_{s:03d}": {
            "png_count": 0,
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
            "first_image": None,
            "last_image": None,
        }
        for s in ALL_SHARD_NUMBERS
    }
    top_level_files: Dict[str, Dict[str, Any]] = {}
    total_entries = 0
    total_png_entries = 0

    try:
        with io.BufferedReader(raw_reader, buffer_size=1024 * 1024) as buf_stream:
            with zipfile.ZipFile(buf_stream, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    total_entries += 1
                    fname = info.filename
                    m = shard_re.match(fname)
                    if m and fname.lower().endswith(".png"):
                        s_id = m.group(1).lower()
                        total_png_entries += 1
                        entry = shards_summary.setdefault(
                            s_id,
                            {
                                "png_count": 0,
                                "compressed_bytes": 0,
                                "uncompressed_bytes": 0,
                                "first_image": None,
                                "last_image": None,
                            },
                        )
                        entry["png_count"] += 1
                        entry["compressed_bytes"] += int(info.compress_size)
                        entry["uncompressed_bytes"] += int(info.file_size)
                        base_png = Path(fname).name
                        if entry["first_image"] is None or base_png < entry["first_image"]:
                            entry["first_image"] = base_png
                        if entry["last_image"] is None or base_png > entry["last_image"]:
                            entry["last_image"] = base_png
                    elif "/" not in fname:
                        top_level_files[fname] = {
                            "compressed_bytes": int(info.compress_size),
                            "uncompressed_bytes": int(info.file_size),
                            "uncompressed_mb": round(info.file_size / (1024 ** 2), 3),
                            "crc32_hex": f"{info.CRC:08x}",
                        }
    finally:
        raw_reader.close()

    for s_id, s_data in shards_summary.items():
        s_data["compressed_gb"] = round(s_data["compressed_bytes"] / (1024 ** 3), 3)
        s_data["uncompressed_gb"] = round(s_data["uncompressed_bytes"] / (1024 ** 3), 3)

    return {
        "dataset_slug": dataset_slug,
        "remote_archive_total_bytes": raw_reader.size,
        "remote_archive_total_gb": round(raw_reader.size / (1024 ** 3), 2),
        "central_directory_probe_bytes_transferred": raw_reader.bytes_downloaded,
        "total_archive_files": total_entries,
        "total_png_files_across_shards": total_png_entries,
        "top_level_metadata_and_docs": top_level_files,
        "shards": shards_summary,
    }


def stream_remote_kaggle_selection(
    nih_root: Path,
    include_metadata_and_docs: bool = True,
    shards: Sequence[int] = (),
    dataset_slug: str = KAGGLE_DATASET_SLUG,
) -> Dict[str, Any]:
    """
    Streams selected metadata/documentation files and/or selected shard PNGs (`images_001`..`images_012`)
    from Kaggle's remote ZIP64 archive in monotonic byte-offset order with built-in ZIP CRC-32 + SHA-256 verification.
    """
    nih_root = Path(nih_root)
    raw_dir = nih_root / "raw"
    metadata_dir = nih_root / "metadata"
    docs_dir = metadata_dir / "docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    wanted_top_files = set(METADATA_FILENAMES + DOC_FILENAMES) if include_metadata_and_docs else set()
    wanted_shard_prefixes = tuple(f"images_{s:03d}/" for s in shards)

    t0 = time.time()
    raw_reader = SeekableHttpRangeReader(lambda: resolve_kaggle_signed_archive_url(dataset_slug))
    staged_metadata: Dict[str, Any] = {}
    staged_docs: Dict[str, Any] = {}
    shard_stats: Dict[str, Dict[str, Any]] = {
        f"images_{s:03d}": {
            "shard_id": f"images_{s:03d}",
            "expected_in_archive": 0,
            "newly_extracted": 0,
            "already_present_verified": 0,
            "total_png_on_disk": 0,
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
            "Aggregate_sha256_manifest_digest": None,
        }
        for s in shards
    }

    try:
        with io.BufferedReader(raw_reader, buffer_size=1024 * 1024) as buf_stream:
            with zipfile.ZipFile(buf_stream, "r") as zf:
                selected_infos: List[Tuple[zipfile.ZipInfo, Path, str]] = []

                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    fname = info.filename
                    if "/" not in fname and fname in wanted_top_files:
                        dst = (metadata_dir / fname) if fname in METADATA_FILENAMES else (docs_dir / fname)
                        if not (dst.exists() and dst.stat().st_size == info.file_size):
                            selected_infos.append((info, dst, "top_level"))
                    elif wanted_shard_prefixes and fname.startswith(wanted_shard_prefixes) and fname.lower().endswith(".png"):
                        s_id = fname.split("/", 1)[0].lower()
                        shard_stats[s_id]["expected_in_archive"] += 1
                        shard_stats[s_id]["compressed_bytes"] += int(info.compress_size)
                        shard_stats[s_id]["uncompressed_bytes"] += int(info.file_size)
                        rel_inside = Path(fname)
                        # Prevent path traversal
                        dst = (raw_dir / rel_inside).resolve()
                        if not str(dst).startswith(str(raw_dir.resolve())):
                            raise SecurityConfigurationError(f"Blocked unsafe archive path: {fname}")
                        if dst.exists() and dst.stat().st_size == info.file_size:
                            shard_stats[s_id]["already_present_verified"] += 1
                        else:
                            selected_infos.append((info, dst, s_id))

                # Sort strictly by header_offset so HTTP Range streaming is 100% sequential!
                selected_infos.sort(key=lambda item: item[0].header_offset)
                logger.info(
                    "Streaming %d archive entries from Kaggle GCS (shards=%s)...",
                    len(selected_infos),
                    list(shards),
                )

                for idx, (info, dst_path, category) in enumerate(selected_infos, start=1):
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_dst = dst_path.with_suffix(dst_path.suffix + ".part")
                    try:
                        with zf.open(info, "r") as zsrc, open(tmp_dst, "wb") as fdst:
                            shutil.copyfileobj(zsrc, fdst, length=1024 * 1024)
                        if tmp_dst.stat().st_size != info.file_size:
                            raise RuntimeError(
                                f"Size mismatch for {info.filename}: got {tmp_dst.stat().st_size}, expected {info.file_size}"
                            )
                        tmp_dst.replace(dst_path)
                    except Exception:
                        tmp_dst.unlink(missing_ok=True)
                        raise

                    if category != "top_level":
                        shard_stats[category]["newly_extracted"] += 1
                        if idx % 500 == 0:
                            logger.info(
                                "  Progress: %d / %d files extracted (%.2f GiB transferred)...",
                                idx,
                                len(selected_infos),
                                raw_reader.bytes_downloaded / (1024 ** 3),
                            )

        # Record SHA-256 of all metadata & doc files
        for meta_name in METADATA_FILENAMES:
            mp = metadata_dir / meta_name
            staged_metadata[meta_name] = (
                {
                    "path": str(mp),
                    "size_bytes": mp.stat().st_size,
                    "sha256": compute_sha256(mp),
                }
                if mp.exists()
                else None
            )
        for doc_name in DOC_FILENAMES:
            dp = docs_dir / doc_name
            staged_docs[doc_name] = (
                {
                    "path": str(dp),
                    "size_bytes": dp.stat().st_size,
                    "sha256": compute_sha256(dp),
                }
                if dp.exists()
                else None
            )

        # Compute deterministic tree SHA-256 manifest digest per downloaded shard
        extracted_shards_list: List[Dict[str, Any]] = []
        for s in shards:
            s_id = f"images_{s:03d}"
            s_dir = raw_dir / s_id
            pngs = sorted(s_dir.rglob("*.png")) if s_dir.exists() else []
            tree_hasher = hashlib.sha256()
            for p in pngs:
                file_sha = compute_sha256(p)
                tree_hasher.update(f"{p.name}:{p.stat().st_size}:{file_sha}\n".encode("utf-8"))
            st = shard_stats[s_id]
            st["total_png_on_disk"] = len(pngs)
            st["extracted_dir"] = str(s_dir)
            st["png_count"] = len(pngs)
            st["shard_tree_sha256"] = tree_hasher.hexdigest() if pngs else None
            extracted_shards_list.append(st)

        elapsed_s = round(time.time() - t0, 2)
        return {
            "staged_metadata": staged_metadata,
            "staged_documentation": staged_docs,
            "extracted_shards": extracted_shards_list,
            "network_transfer_telemetry": {
                "http_range_requests": raw_reader.http_range_requests,
                "bytes_downloaded": raw_reader.bytes_downloaded,
                "gigabytes_downloaded": round(raw_reader.bytes_downloaded / (1024 ** 3), 3),
                "elapsed_seconds": elapsed_s,
            },
        }
    finally:
        raw_reader.close()


def safe_extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """
    Extracts `zip_path` into `extract_to` while blocking path-traversal ('ZipSlip') entries.
    Returns list of extracted file paths.
    """
    extract_root = extract_to.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted_files: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            dest_path = (extract_root / member.filename).resolve()
            if not str(dest_path).startswith(str(extract_root)):
                raise SecurityConfigurationError(
                    f"Unsafe ZIP path traversal entry blocked in {zip_path}: {member.filename}"
                )
            if member.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted_files.append(dest_path)

    return extracted_files


def _stage_single_metadata_file(source_dir: Path, metadata_dir: Path, filename: str) -> Optional[Path]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    target_path = metadata_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    candidate_names = [filename]
    if filename == "BBox_List_2017.csv":
        candidate_names.append("BBox_List_2017_Official.csv")

    for cand_name in candidate_names:
        direct_cand = source_dir / cand_name
        if direct_cand.exists() and direct_cand.is_file():
            if direct_cand.resolve() != target_path.resolve():
                shutil.copy2(direct_cand, target_path)
            return target_path

        zip_cand = source_dir / f"{cand_name}.zip"
        if zip_cand.exists() and zipfile.is_zipfile(zip_cand):
            extracted = safe_extract_zip(zip_cand, metadata_dir)
            for ext_file in extracted:
                if ext_file.name.lower() == cand_name.lower() and ext_file.resolve() != target_path.resolve():
                    ext_file.replace(target_path)
            if target_path.exists():
                return target_path

    for cand_name in candidate_names:
        matches = list(source_dir.rglob(cand_name))
        if matches:
            shutil.copy2(matches[0], target_path)
            return target_path

    return None


def extract_shard_archive(
    zip_path: Path,
    raw_dir: Path,
    shard_num: int,
    cleanup_zip: bool = False,
) -> Dict[str, Any]:
    """
    Extracts local `images_XXX.zip` into `raw_dir / images_XXX/` preserving PNGs cleanly,
    audits extracted PNG count & archive SHA-256, and optionally deletes the ZIP archive.
    """
    shard_name = f"images_{shard_num:03d}"
    shard_target_dir = raw_dir / shard_name
    check_disk_space(raw_dir, required_gb=3.5)

    archive_sha256 = compute_sha256(zip_path)
    archive_size_bytes = zip_path.stat().st_size

    safe_extract_zip(zip_path, shard_target_dir)
    png_files = sorted(shard_target_dir.rglob("*.png"))
    if len(png_files) == 0:
        raise RuntimeError(
            f"Shard extraction produced 0 PNG files for {zip_path} -> {shard_target_dir}!"
        )

    if cleanup_zip:
        zip_path.unlink()

    return {
        "shard_id": shard_name,
        "archive_filename": zip_path.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "extracted_dir": str(shard_target_dir),
        "png_count": len(png_files),
        "zip_cleaned_up": bool(cleanup_zip),
    }


def stage_from_local_directory(
    source_dir: Path,
    nih_root: Path,
    shards: Optional[Sequence[int]] = None,
    cleanup_zips: bool = False,
) -> Dict[str, Any]:
    """Stages metadata and extracts nested shard archives from an existing local directory into `nih_root`."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Local staging source directory does not exist: {source_dir}")

    raw_dir = nih_root / "raw"
    metadata_dir = nih_root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    selected_shards = list(shards) if shards is not None else list(ALL_SHARD_NUMBERS)

    for bundle_name in ("archive.zip", "data.zip", "nih-chest-xrays.zip"):
        bundle_zip = source_dir / bundle_name
        if bundle_zip.exists() and zipfile.is_zipfile(bundle_zip):
            check_disk_space(nih_root, required_gb=MIN_FREE_GB_FULL_CLEANUP)
            logger.info("Extracting top-level Kaggle bundle `%s`...", bundle_zip)
            safe_extract_zip(bundle_zip, nih_root / "_staging_unpack")
            source_dir = nih_root / "_staging_unpack"
            break

    staged_metadata: Dict[str, Any] = {}
    for meta_name in METADATA_FILENAMES:
        staged = _stage_single_metadata_file(source_dir, metadata_dir, meta_name)
        if staged and staged.exists():
            staged_metadata[meta_name] = {
                "path": str(staged),
                "size_bytes": staged.stat().st_size,
                "sha256": compute_sha256(staged),
            }
        else:
            staged_metadata[meta_name] = None

    shard_audits: List[Dict[str, Any]] = []
    for shard_num in selected_shards:
        shard_name = f"images_{shard_num:03d}"
        shard_zip_candidates = list(source_dir.rglob(f"{shard_name}.zip"))
        existing_shard_dir = source_dir / shard_name

        if shard_zip_candidates:
            audit = extract_shard_archive(
                shard_zip_candidates[0],
                raw_dir=raw_dir,
                shard_num=shard_num,
                cleanup_zip=cleanup_zips,
            )
            shard_audits.append(audit)
        elif existing_shard_dir.exists() and existing_shard_dir.is_dir():
            dest_shard_dir = raw_dir / shard_name
            if existing_shard_dir.resolve() != dest_shard_dir.resolve():
                shutil.copytree(existing_shard_dir, dest_shard_dir, dirs_exist_ok=True)
            png_count = len(list(dest_shard_dir.rglob("*.png")))
            shard_audits.append({
                "shard_id": shard_name,
                "archive_filename": None,
                "archive_sha256": None,
                "archive_size_bytes": None,
                "extracted_dir": str(dest_shard_dir),
                "png_count": png_count,
                "zip_cleaned_up": False,
            })

    return {
        "staged_metadata": staged_metadata,
        "staged_documentation": {},
        "extracted_shards": shard_audits,
        "network_transfer_telemetry": None,
    }


def run_nih_ingestion(
    nih_root: Path = Path("data/kaggle_nih"),
    mode: str = "shards",
    shards: Optional[Sequence[int]] = None,
    cleanup_zips: bool = True,
    stage_local_dir: Optional[Path] = None,
    audit_report_path: Path = Path("reports/nih_download_audit.json"),
    include_remote_inventory: bool = True,
) -> Dict[str, Any]:
    """
    Executes preflight checks, inspects remote Kaggle shard sizes, downloads/stages requested
    NIH ChestX-ray14 artifacts, and writes `reports/nih_download_audit.json`.
    """
    verify_git_token_safety(PROJECT_ROOT)

    nih_root = Path(nih_root)
    raw_dir = nih_root / "raw"
    metadata_dir = nih_root / "metadata"
    manifests_dir = nih_root / "manifests"
    processed_dir = nih_root / "processed"
    for d in (raw_dir, metadata_dir, manifests_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_shards = list(shards) if shards else ([1] if mode == "shards" else list(ALL_SHARD_NUMBERS))
    for s in target_shards:
        if s not in ALL_SHARD_NUMBERS:
            raise ValueError(f"Invalid shard index {s}. Valid NIH shards are 1..12.")

    req_gb = required_free_gb_for_plan(mode=mode, shards=target_shards, cleanup_zips=cleanup_zips)
    disk_audit = check_disk_space(nih_root, required_gb=req_gb)
    all_drives = get_all_local_drives_audit()

    remote_inventory: Optional[Dict[str, Any]] = None
    if stage_local_dir is not None:
        cred_info = {"auth_method": "local_staging_directory", "configured": True}
        transfer_result = stage_from_local_directory(
            source_dir=stage_local_dir,
            nih_root=nih_root,
            shards=([] if mode == "metadata_only" else target_shards),
            cleanup_zips=cleanup_zips,
        )
    else:
        cred_info = check_kaggle_credentials()
        if include_remote_inventory:
            logger.info("Inspecting remote Kaggle ZIP64 Central Directory for `%s`...", KAGGLE_DATASET_SLUG)
            remote_inventory = inspect_remote_kaggle_archive(KAGGLE_DATASET_SLUG)

        active_shards = [] if mode == "metadata_only" else (ALL_SHARD_NUMBERS if mode == "full" else target_shards)
        transfer_result = stream_remote_kaggle_selection(
            nih_root=nih_root,
            include_metadata_and_docs=True,
            shards=active_shards,
            dataset_slug=KAGGLE_DATASET_SLUG,
        )

    shard_inventory: Dict[str, int] = {}
    total_png_on_disk = 0
    for s in ALL_SHARD_NUMBERS:
        s_name = f"images_{s:03d}"
        s_dir = raw_dir / s_name
        count = len(list(s_dir.rglob("*.png"))) if s_dir.exists() else 0
        shard_inventory[s_name] = count
        total_png_on_disk += count

    audit_payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "NIH ChestX-ray14 (Kaggle nih-chest-xrays/data)",
        "kaggle_dataset_slug": KAGGLE_DATASET_SLUG,
        "kaggle_url": "https://www.kaggle.com/datasets/nih-chest-xrays/data",
        "license": "CC0-1.0",
        "primary_citation": (
            "Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray "
            "Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax "
            "Diseases. IEEE CVPR 2017 (arXiv:1705.02315). Originating institution: NIH Clinical Center."
        ),
        "physionet_or_mimic_used": False,
        "ingestion_mode": mode,
        "requested_shards": [] if mode == "metadata_only" else [f"images_{s:03d}" for s in target_shards],
        "host_drives_audit": all_drives,
        "disk_preflight": disk_audit,
        "credentials_check": cred_info,
        "remote_kaggle_archive_inventory": remote_inventory,
        "staged_metadata": transfer_result["staged_metadata"],
        "staged_documentation": transfer_result.get("staged_documentation", {}),
        "extracted_shards_this_run": transfer_result["extracted_shards"],
        "network_transfer_telemetry": transfer_result.get("network_transfer_telemetry"),
        "on_disk_shard_png_inventory": shard_inventory,
        "total_png_images_on_disk": total_png_on_disk,
    }

    audit_report_path = Path(audit_report_path)
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)

    logger.info(
        "NIH download audit saved to %s (total PNGs on disk: %d).",
        audit_report_path,
        total_png_on_disk,
    )
    return audit_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and download NIH ChestX-ray14 shards/metadata from Kaggle with disk preflight."
    )
    parser.add_argument(
        "--nih-root",
        type=Path,
        default=Path("data/kaggle_nih"),
        help="Target NIH dataset root directory (default: data/kaggle_nih).",
    )
    parser.add_argument(
        "--mode",
        choices=["metadata_only", "shards", "full"],
        default="shards",
        help="Ingestion mode: metadata_only, shards (incremental batch), or full (all 12 shards).",
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=[1],
        help="Shard numbers (1..12) to download when --mode shards is selected (default: 1).",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Retain local shard .zip archives if staging from local ZIPs.",
    )
    parser.add_argument(
        "--stage-local-dir",
        type=Path,
        default=None,
        help="Optional path to pre-downloaded local ZIPs/CSVs to stage offline.",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("reports/nih_download_audit.json"),
        help="Output path for download audit JSON (default: reports/nih_download_audit.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_nih_ingestion(
        nih_root=args.nih_root,
        mode=args.mode,
        shards=args.shards,
        cleanup_zips=not args.keep_zips,
        stage_local_dir=args.stage_local_dir,
        audit_report_path=args.audit_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
