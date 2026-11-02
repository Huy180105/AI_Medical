# NIH ChestX-ray14 — Batch Acquisition Validation (`images_002` to `images_004`)

- **Timestamp (UTC)**: `2026-10-30T21:09:57.985616+00:00`
- **Overall Batch Status**: **`PASS`**
- **Dataset Source**: Kaggle `nih-chest-xrays/data` (`CC0-1.0`; PhysioNet/MIMIC used: `False`)
- **Validated `images_001` Untouched**: `True` (`shard_tree_sha256 = 2c776f12f88bfaa7ca6d6ed29ac52b0207b7125de0552533a94a1c8399569d2f`)
- **Unrequested Shards `images_005`–`images_012` Downloaded**: `0` files (`True`)
- **Network Transfer (`images_002`–`images_004`)**: `11,769,217,024` bytes (`10.961 GiB`) across `351` HTTP Range requests in `2,962.16s`

## 1. Per-Shard Download Sizes, PNG Integrity & Full SHA-256 (`images_001`–`images_004`)

| Shard | Status | PNGs (Valid / Expected) | Corrupt / Missing / Orphan / Dup | Color Modes (`1024x1024`) | Compressed GiB | Uncompressed Bytes (GiB) | Full Tree SHA-256 |
| :--- | :-: | --: | :-: | :--- | --: | --: | :--- |
| `images_001` | **PASS** | 4,999 / 4,999 | `0 / 0 / 0 / 0` | `1024x1024:L`: 4,928, `1024x1024:RGBA`: 71 | 1.870 | 2,007,961,024 (1.870) | `2c776f12f88bfaa7ca6d6ed29ac52b0207b7125de0552533a94a1c8399569d2f` |
| `images_002` | **PASS** | 10,000 / 10,000 | `0 / 0 / 0 / 0` | `1024x1024:L`: 9,918, `1024x1024:RGBA`: 82 | 3.680 | 3,951,069,542 (3.680) | `afc3742da9a23dabbe5f88a3add366ebe87d9c9779bb759e6197450e65f49f5d` |
| `images_003` | **PASS** | 10,000 / 10,000 | `0 / 0 / 0 / 0` | `1024x1024:L`: 9,817, `1024x1024:RGBA`: 183 | 3.658 | 3,928,657,321 (3.659) | `4c6f2d4ce61b9e9bb69676384ed7bcd06a1f1b3989f1c7e462ad363b86e79138` |
| `images_004` | **PASS** | 10,000 / 10,000 | `0 / 0 / 0 / 0` | `1024x1024:L`: 9,931, `1024x1024:RGBA`: 69 | 3.574 | 3,837,267,398 (3.574) | `18ba3e408bf1f90009242338c0ce461860fff50b380c94ceecfc8e89d95c41d6` |

## 2. Cross-Shard & Patient Split Reconciliation (`34,999 / 112,120` Images)

- **Total Verified PNGs (`images_001`–`images_004`)**: `34,999 / 34,999`
- **Cross-Shard Duplicate `Image Index`**: `0`
- **Manifest Global Accounting**: `VALID = 34,999`, `MISSING = 77,121` (`images_005`–`images_012`), `CORRUPT = 0`
- **Patient ID Metadata & Filename Prefix Match**: `100%` (`34,999 / 34,999` across `images_001`–`images_004`)
- **Downloaded Split Distribution (`34,999` images)**: `train = 24,821` (`7,532` pts), `validate = 4,573` (`1,369` pts), `test = 5,605` (`331` pts)
- **Official `test_list.txt` Untouched**: `True` (`25,596` test images)
- **Zero Patient Leakage**: `True` (`train ∩ validate = 0`, `train ∩ test = 0`, `validate ∩ test = 0`)

## 3. Stratified Model Validation (`images_002`–`images_004`)

- **Checkpoint**: `models\nih-densenet121-nih14.pt` (`SHA-256 = 31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf`)
- **Device & Mini-Batch Size**: `cpu`, `mini_batch_size = 16`
- **Sampled Real Radiographs**: `96` (`32` per new shard: `16 validate` + `16 test` from distinct patients)
- **Execution Time & Throughput**: `53.50s` (`1.79 images/second`; sample-level CPU measurement only — not for full-dataset extrapolation)
- **Probability Domain**: `1,344` scalars in `[0.0, 0.744738]`, `NaN/Inf = False`
- **Threshold Governance**: `threshold_invariant_to_test_label_inversion = True` (`used_test_labels_for_threshold = False`)

| NIH-14 Pathology | Min Prob | Mean Prob | Std Prob | Max Prob |
| :--- | --: | --: | --: | --: |
| `Atelectasis` | 0.000013 | 0.099576 | 0.101345 | 0.360349 |
| `Cardiomegaly` | 0.000000 | 0.029421 | 0.061054 | 0.322096 |
| `Consolidation` | 0.000052 | 0.020486 | 0.029864 | 0.162424 |
| `Edema` | 0.000000 | 0.008559 | 0.023787 | 0.166004 |
| `Effusion` | 0.000159 | 0.089062 | 0.148378 | 0.744738 |
| `Emphysema` | 0.000000 | 0.020758 | 0.034086 | 0.204158 |
| `Fibrosis` | 0.000073 | 0.025501 | 0.035826 | 0.196801 |
| `Hernia` | 0.000000 | 0.006528 | 0.020134 | 0.122067 |
| `Infiltration` | 0.007759 | 0.135881 | 0.094150 | 0.433859 |
| `Mass` | 0.000146 | 0.062077 | 0.082555 | 0.469806 |
| `Nodule` | 0.003027 | 0.065263 | 0.053332 | 0.309250 |
| `Pleural_Thickening` | 0.000075 | 0.031623 | 0.046832 | 0.311390 |
| `Pneumonia` | 0.000006 | 0.009923 | 0.014762 | 0.071756 |
| `Pneumothorax` | 0.000045 | 0.023619 | 0.045640 | 0.266764 |

## 4. Regression Tests & Disk Capacity Forecast

- **NIH + Benchmark Regression Suite**: `passed = 32`, `failed = 0`, `skipped = 0` (`exit_code = 0`, `72.37s`)
- **Free Space Before Batch on `C:\`**: `122.26 GiB`
- **Remaining Free Space on `C:\` Post-Batch**: `111.78 GiB` (`A:\`: `30.74 GiB`)
- **Remaining 8 Shards (`images_005`–`images_012`, `77,121` PNGs)**: `29.181 GiB` uncompressed (`29.187 GiB` compressed)
- **Projected Free Space on `C:\` After All 12 Shards**: `~82.60 GiB`
- **Recommendation**: **`PROCEED_TO_NEXT_BATCH_ON_USER_APPROVAL`**
