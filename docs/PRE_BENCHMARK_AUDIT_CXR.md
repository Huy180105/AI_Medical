# Pre-Benchmark Methodological Audit & Execution Protocol (`NIH ChestX-ray14`)

> **Official Dataset Scope (`v2.0-nih14`)**: MedFusion-CXR's official Chest X-Ray (CXR) pipeline uses **exclusively `NIH ChestX-ray14` from Kaggle** ([`nih-chest-xrays/data`](https://www.kaggle.com/datasets/nih-chest-xrays/data)). **PhysioNet, `MIMIC-CXR-JPG`, and `MIMIC-IV-ED` are NOT used** as data sources or imports anywhere in this CXR pipeline.

---

## 1. Dataset Attribution & Scientific Governance

- **Dataset Name**: NIH ChestX-ray14 (`112,120` frontal-view chest radiographs from `30,805` unique patients).
- **Originating Institution**: NIH Clinical Center (National Institutes of Health, Bethesda, MD).
- **Primary Scientific Citation**:
  > Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. *"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases."* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 3462–3471. [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)
- **Kaggle Distribution**: [`https://www.kaggle.com/datasets/nih-chest-xrays/data`](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### Mandatory Label & Clinical Limitations Disclosure
1. **14 Binary NLP-Mined Disease Labels (`NIH_LABELS`)**:
   `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`.
2. **`No Finding` Rule**: `"No Finding"` in `Data_Entry_2017.csv` means none of the 14 target pathologies were extracted (`[0, 0, ..., 0]` across all 14 columns, `no_finding = 1`). It is **never** modeled or evaluated as a 15th disease class.
3. **Strict Binary `{0, 1}` Domain (No CheXpert `-1` Carryover)**: NIH ChestX-ray14 labels are strictly binary `{0, 1}`. `BenchmarkFusionEvaluator(label_domain_policy=LabelDomainPolicy.BINARY_STRICT)` explicitly rejects `-1` uncertainty values with `InvalidLabelError`. NIH labels are **not** equivalent to Stanford CheXpert labels.
4. **NLP Label Noise**: NIH ChestX-ray14 labels were text-mined from clinical radiology reports (~90% NLP extraction accuracy reported by Wang et al., 2017; visual re-read discrepancies documented by Oakden-Rayner, 2020). Metrics quantify agreement with NIH NLP report tags, **not** board-certified radiologist biopsy/consensus ground truth.

---

## 2. Canonical Directory Structure & Disk Capacity Audit

```text
data/kaggle_nih/
  raw/                  # Shards images_001/ .. images_012/ (extracted incrementally or full)
  metadata/             # Data_Entry_2017.csv, BBox_List_2017.csv, train_val_list.txt, test_list.txt
  manifests/            # nih_manifest.csv, patient_splits.csv, patient_split_audit.json, data_audit.json
  processed/            # Optional cached artifacts
reports/
  nih_ingestion_audit.json
  nih_dataset_audit.json
  nih_predictions.csv
  nih_benchmark_report.json
```

### Verified Local Partition Audit (Windows)
| Drive | Free Space | Suitability for NIH ChestX-ray14 (~42–45 GB raw PNGs) |
| :---: | :---: | :--- |
| **`C:`** | **~125.2 GB free** | **PASS**: Supports both incremental shard batches and `--mode full --cleanup-zips` (requires `>= 60 GB` free). |
| **`A:`** | **~30.8 GB free** | **BLOCKED for `--mode full`** by `check_disk_space()` (`InsufficientDiskSpaceError`); supports only 1–4 shard batches. |

---

## 3. Patient-Level Split & Statistical Evaluation Protocol

1. **Official `test_list.txt` Preserved Untouched**: All radiographs in NIH's official `test_list.txt` remain fixed in `split = "test"`.
2. **Deterministic Patient-Grouped `train` / `validate` Split (`scripts/split_nih_by_patient.py`)**:
   - `train_val_list.txt` is partitioned into `train` (85%) and `validate` (15%) **strictly by `patient_id` (`Patient ID`)** using fixed `--seed 42` and rare-pathology-first multi-label stratification (preserving positive patients for rare classes such as `Hernia` in both splits).
   - Hard mathematical guard (`verify_zero_patient_leakage`):
     $$(\text{train} \cap \text{validate}) = (\text{train} \cap \text{test}) = (\text{validate} \cap \text{test}) = \emptyset$$
3. **Integrity Accounting Without Silent Deletion (`scripts/build_nih_cxr_manifest.py`)**:
   - Retains all `112,120` rows of `Data_Entry_2017.csv`, tagging each row's physical PNG status as `VALID`, `MISSING` (shard not yet downloaded), or `CORRUPT` (0-byte / truncated PNG), while auditing counts across all 14 pathologies, splits, `patient_gender`, `age_group` (including NIH's known `>120` age typos flagged via `age_anomaly_flag`), and `view_position` (`PA`/`AP`).
4. **Streaming Mini-Batch Inference & Patient-Clustered Bootstrap (`scripts/run_nih_cxr_inference.py`)**:
   - Processes `VALID` images in bounded mini-batches (`--batch-size 32`), streaming predictions directly to `reports/nih_predictions.csv` (never loading the dataset into RAM).
   - Selects `validation_optimal_threshold` per pathology strictly on `validate` (never `test`, never `source_op_thresh`).
   - Computes `AUROC`, `AUPRC` (exact step-wise Average Precision), `F1`, `Sensitivity`, `Specificity`, `Brier score`, `ECE`, and **`patient_id`-clustered 95% bootstrap confidence intervals**.

---

## 4. Checkpoint Provenance Audit (`PretrainedNihDenseNet121`)

| Profile | Checkpoint File | SHA-256 | Active NIH-14 Classes | Training Overlap Disclosure |
| :--- | :--- | :--- | :---: | :--- |
| **`nih14`** *(Default)* | `models/nih-densenet121-nih14.pt` | `31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf` | **14 / 14** (slots `0..13` active; `14..17` masked) | Trained by TorchXRayVision (`densenet121-res224-nih`) on its internal NIH patient split (`seed=0`), which may partially overlap official `test_list.txt`. |
| **`chex_external`** *(Optional)* | `models/chex-densenet121-chexpert.pt` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` | **7 / 14** (`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Pneumonia`, `Pneumothorax`; 7 NIH-only classes masked as `None`) | **0% NIH training overlap** (trained solely on Stanford CheXpert). |

---

## 5. Step-by-Step Windows PowerShell Execution Commands

### Step 0: Configure Kaggle API Token Safely (Outside Git)
Place `kaggle.json` in `$env:USERPROFILE\.kaggle\kaggle.json` (never inside the repository):
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
# Copy your downloaded kaggle.json to $env:USERPROFILE\.kaggle\kaggle.json
```

### Step 1A (Recommended First): Download Metadata + Batch Shards (`images_001` & `images_002`)
```powershell
.\venv\Scripts\python.exe scripts/download_kaggle_nih.py --mode shards --shards 1 2
```
*(Or to download & extract all 12 shards on `C:` when ready, automatically deleting intermediate shard `.zip` files after PNG verification)*:
```powershell
.\venv\Scripts\python.exe scripts/download_kaggle_nih.py --mode full
```

### Step 2: Generate Deterministic Patient-Level Splits (`train` / `validate` / official `test`)
```powershell
.\venv\Scripts\python.exe scripts/split_nih_by_patient.py --seed 42 --val-fraction 0.15
```

### Step 3: Build Canonical 14-Label Manifest & Dataset Audit Report
```powershell
.\venv\Scripts\python.exe scripts/build_nih_cxr_manifest.py --verify-images
```
*(Add `--strict-full-count` to enforce exact `112,120` metadata rows.)*

### Step 4A: Run Batch/Shard Pilot Streaming Inference & Evaluation
```powershell
.\venv\Scripts\python.exe scripts/run_nih_cxr_inference.py --mode real_pilot --shards images_001 images_002 --batch-size 32 --bootstrap-iterations 500
```

### Step 4B: Run Full Official NIH ChestX-ray14 Benchmark (All 12 Shards Extracted)
```powershell
.\venv\Scripts\python.exe scripts/run_nih_cxr_inference.py --mode benchmark --checkpoint-profile nih14 --batch-size 32 --bootstrap-iterations 1000
```
*(Optional zero-overlap external baseline on the 7 shared classes)*:
```powershell
.\venv\Scripts\python.exe scripts/run_nih_cxr_inference.py --mode benchmark --checkpoint-profile chex_external --report-out reports/nih_benchmark_chex_external.json --predictions-out reports/nih_predictions_chex_external.csv
```
