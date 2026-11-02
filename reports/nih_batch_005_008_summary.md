# NIH ChestX-ray14 — Batch Acquisition Validation (`images_005` to `images_008`)

- **Timestamp (UTC)**: `2026-10-30T22:56:27.133385+00:00`
- **Overall Batch Status**: **`PASS`**
- **Dataset Source**: Kaggle `nih-chest-xrays/data` (`CC0-1.0`; PhysioNet/MIMIC used: `False`)
- **Accepted Shards `images_001`–`images_004` Unchanged**: `True` (100% SHA-256 tree match)
- **Unrequested Shards `images_009`–`images_012` Downloaded**: `0` files (`True`)
- **Network Transfer (`images_005`–`images_008`)**: `15,997,075,456` bytes (`14.898 GiB`) across `477` HTTP Range requests in `3959.25s`

## 1. Per-Shard Download Sizes, 1024x1024 Integrity & Full SHA-256 (`images_001`–`images_008`)

| Shard | Status | PNGs (Valid / Expected) | 1024x1024 | Corrupt / Missing / Orphan / Dup | Color Modes | Uncompressed Bytes (GiB) | Full Tree SHA-256 |
| :--- | :-: | --: | :-: | :-: | :--- | --: | :--- |
| `images_001` | **PASS** | 4,999 / 4,999 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 4928, `1024x1024:RGBA`: 71 | 2,007,961,024 (1.870) | `2c776f12f88bfaa7ca6d6ed29ac52b0207b7125de0552533a94a1c8399569d2f` |
| `images_002` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9918, `1024x1024:RGBA`: 82 | 3,951,069,542 (3.680) | `afc3742da9a23dabbe5f88a3add366ebe87d9c9779bb759e6197450e65f49f5d` |
| `images_003` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9817, `1024x1024:RGBA`: 183 | 3,928,657,321 (3.659) | `4c6f2d4ce61b9e9bb69676384ed7bcd06a1f1b3989f1c7e462ad363b86e79138` |
| `images_004` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9931, `1024x1024:RGBA`: 69 | 3,837,267,398 (3.574) | `18ba3e408bf1f90009242338c0ce461860fff50b380c94ceecfc8e89d95c41d6` |
| `images_005` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9984, `1024x1024:RGBA`: 16 | 3,933,273,109 (3.663) | `0b6eb6689dd32d673ef9492b52ced4c6257857279f68aba81181a081c6febaec` |
| `images_006` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9988, `1024x1024:RGBA`: 12 | 3,983,895,279 (3.710) | `706c6ab080c46ea971637a5383a045ccf24dfa0652de5f82a621dbb63efaf960` |
| `images_007` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9966, `1024x1024:RGBA`: 34 | 4,014,137,889 (3.738) | `a1c1ff11c52252a712c0af4c4f6130ac7004c45740c4c16d00d5affc7e300455` |
| `images_008` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9989, `1024x1024:RGBA`: 11 | 4,015,949,603 (3.740) | `1e2d75ad811206bca00797f0a46cb4293ca54a918df72655c0a1f7927639d7de` |

## 2. Cross-Shard & Patient Split Reconciliation (`74,999 / 112,120` Images — `66.89%`)

- **Total Verified PNGs (`images_001`–`images_008`)**: `74,999 / 74,999`
- **Cross-Shard Duplicate `Image Index`**: `0`
- **Manifest Global Accounting**: `VALID = 74,999`, `MISSING = 37,121` (`images_009`–`images_012`), `CORRUPT = 0`
- **Patient ID Metadata & Filename Prefix Match**: `100%` (`74,999 / 74,999` across `images_001`–`images_008`)
- **Downloaded Split Distribution (`74,999` images)**: `train = 50,173` (`14,938` pts), `validate = 9,157` (`2,641` pts), `test = 15,669` (`808` pts)
- **Official `test_list.txt` Untouched**: `True` (`25,596` test images)
- **Zero Patient Leakage**: `True` (`train ∩ validate = 0`, `train ∩ test = 0`, `validate ∩ test = 0`)

## 3. Stratified Model Validation (`images_005`–`images_008`)

- **Checkpoint**: `models\nih-densenet121-nih14.pt` (`SHA-256 = 31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf`)
- **Device & Mini-Batch Size**: `cpu`, `mini_batch_size = 16`
- **Sampled Real Radiographs**: `128` (`32` per new shard: `16 validate` + `16 test` from distinct patients)
- **Execution Time & Throughput**: `19.28s` (`6.64 images/second` — sample-level CPU measurement only; do not extrapolate linearly to full dataset)
- **Probability Domain**: `1,792` scalars in `[0.0, 0.890836]`, `NaN/Inf = False`
- **Threshold Governance**: `threshold_invariant_to_test_label_inversion = True` (`used_test_labels_for_threshold = False`)

| NIH-14 Pathology | Min Prob | Mean Prob | Std Prob | Max Prob |
| :--- | --: | --: | --: | --: |
| `Atelectasis` | 0.000243 | 0.100072 | 0.102070 | 0.375746 |
| `Cardiomegaly` | 0.000000 | 0.052745 | 0.132101 | 0.798037 |
| `Consolidation` | 0.000055 | 0.025602 | 0.033385 | 0.136756 |
| `Edema` | 0.000000 | 0.015397 | 0.039501 | 0.278660 |
| `Effusion` | 0.000035 | 0.144712 | 0.222669 | 0.890836 |
| `Emphysema` | 0.000000 | 0.020999 | 0.049658 | 0.368602 |
| `Fibrosis` | 0.000050 | 0.028438 | 0.035898 | 0.194426 |
| `Hernia` | 0.000000 | 0.006062 | 0.016490 | 0.108759 |
| `Infiltration` | 0.009642 | 0.183282 | 0.105204 | 0.538544 |
| `Mass` | 0.000102 | 0.058672 | 0.097353 | 0.749764 |
| `Nodule` | 0.004120 | 0.074735 | 0.084622 | 0.700124 |
| `Pleural_Thickening` | 0.000632 | 0.039022 | 0.059679 | 0.299635 |
| `Pneumonia` | 0.000005 | 0.014821 | 0.022806 | 0.130569 |
| `Pneumothorax` | 0.000002 | 0.027328 | 0.044120 | 0.259837 |

## 4. Regression Tests & Disk Capacity Forecast

- **NIH + Benchmark Regression Suite**: `passed = 32`, `failed = 0`, `skipped = 0` (`exit_code = 0`, `45.36s`)
- **Free Space Before Batch on `C:\`**: `111.77 GiB`
- **Remaining Free Space on `C:\` Post-Batch**: `95.58 GiB` (`A:\`: `30.73 GiB`)
- **Remaining Final 4 Shards (`images_009`–`images_012`, `37,121` PNGs)**: `14.329 GiB` uncompressed (`14.332 GiB` compressed)
- **Projected Free Space on `C:\` After All 12 Shards**: `~81.25 GiB`
- **Recommendation**: **`PROCEED_TO_FINAL_SHARDS_009_012_ON_USER_APPROVAL`**
