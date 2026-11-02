# NIH ChestX-ray14 — Final Batch (`images_009`–`images_012`) & Global 12-Shard Acceptance Report

- **Timestamp (UTC)**: `2026-10-31T00:35:34.620411+00:00`
- **Overall Final Status**: **`PASS`** (`12/12` shards `PASS`, `112,120 / 112,120` PNGs verified)
- **Dataset Source**: Kaggle `nih-chest-xrays/data` (`CC0-1.0`; PhysioNet/MIMIC used: `False`)
- **Accepted Shards `images_001`–`images_008` Unchanged**: `True` (100% SHA-256 tree match)
- **Governance Hashes Unchanged (`test_list.txt`, `patient_splits.csv`, Checkpoint)**: `True`
- **Network Transfer (`images_009`–`images_012`)**: `15,535,702,016` bytes (`14.469 GiB`) across `464` HTTP Range requests in `3480.22s`
- **Frozen 12-Shard Master Dataset SHA-256**: `d6583f3c6731898cfcda1fe633703c0d228f038f23640196fe7e071ecfdbc805`

## 1. Per-Shard Integrity, 1024x1024 Dimensions & Full SHA-256 (`images_001`–`images_012`)

| Shard | Status | PNGs (Valid / Expected) | 1024x1024 | Corrupt / Missing / Orphan / Dup | Color Modes (`L` / `RGBA`) | Uncompressed Bytes (GiB) | Full Tree SHA-256 |
| :--- | :-: | --: | :-: | :-: | :--- | --: | :--- |
| `images_001` | **PASS** | 4,999 / 4,999 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 4928, `1024x1024:RGBA`: 71 | 2,007,961,024 (1.870) | `2c776f12f88bfaa7ca6d6ed29ac52b0207b7125de0552533a94a1c8399569d2f` |
| `images_002` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9918, `1024x1024:RGBA`: 82 | 3,951,069,542 (3.680) | `afc3742da9a23dabbe5f88a3add366ebe87d9c9779bb759e6197450e65f49f5d` |
| `images_003` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9817, `1024x1024:RGBA`: 183 | 3,928,657,321 (3.659) | `4c6f2d4ce61b9e9bb69676384ed7bcd06a1f1b3989f1c7e462ad363b86e79138` |
| `images_004` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9931, `1024x1024:RGBA`: 69 | 3,837,267,398 (3.574) | `18ba3e408bf1f90009242338c0ce461860fff50b380c94ceecfc8e89d95c41d6` |
| `images_005` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9984, `1024x1024:RGBA`: 16 | 3,933,273,109 (3.663) | `0b6eb6689dd32d673ef9492b52ced4c6257857279f68aba81181a081c6febaec` |
| `images_006` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9988, `1024x1024:RGBA`: 12 | 3,983,895,279 (3.710) | `706c6ab080c46ea971637a5383a045ccf24dfa0652de5f82a621dbb63efaf960` |
| `images_007` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9966, `1024x1024:RGBA`: 34 | 4,014,137,889 (3.738) | `a1c1ff11c52252a712c0af4c4f6130ac7004c45740c4c16d00d5affc7e300455` |
| `images_008` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9989, `1024x1024:RGBA`: 11 | 4,015,949,603 (3.740) | `1e2d75ad811206bca00797f0a46cb4293ca54a918df72655c0a1f7927639d7de` |
| `images_009` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9987, `1024x1024:RGBA`: 13 | 4,108,933,524 (3.827) | `47e2887fe68af1be1cac7b4c0ba40315f00ee973d715bf0601288ac359cd7de6` |
| `images_010` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9991, `1024x1024:RGBA`: 9 | 4,179,144,121 (3.892) | `64a3c0f4b3a33637135e1b0fff2ae790050e6c0f81bcd5fe430738c67ae40ebf` |
| `images_011` | **PASS** | 10,000 / 10,000 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 9995, `1024x1024:RGBA`: 5 | 4,184,608,844 (3.897) | `2a5be7527c29f8d0bf9c06408e511cf66f1c9a92c6b2cf22a995c17e60d93903` |
| `images_012` | **PASS** | 7,121 / 7,121 | `100%` | `0 / 0 / 0 / 0` | `1024x1024:L`: 7107, `1024x1024:RGBA`: 14 | 2,912,543,044 (2.713) | `895ca212bc77f601b107eb3ab6d052148568795896963af03a59c3e7f1d87d14` |

## 2. Global 12-Shard Reconciliation & Patient Split Isolation (`112,120 / 112,120` Images — `100.00%`)

- **Final Batch New Valid PNGs (`images_009`–`images_012`)**: `37,121 / 37,121`
- **Global Verified PNGs (`images_001`–`images_012`)**: `112,120 / 112,120` (`30,805` unique patients)
- **Global Total Uncompressed Size**: `45,057,440,698` bytes (`41.963 GiB`)
- **Manifest Global Accounting**: `VALID = 112,120`, `MISSING = 0`, `CORRUPT = 0`
- **Global Anomalies**: `Duplicate Image Index = 0`, `Orphan = 0`, `Non-1024x1024 = 0`, `Leftover .part = 0`
- **Patient ID Metadata & Filename Prefix Match**: `100%` (`112,120 / 112,120`)
- **Global Color Modes (`112,120` PNGs)**: `1024x1024:L = 111,601`, `1024x1024:RGBA = 519`
- **Locked Split Distribution (`112,120` images / `30,805` patients)**: `train = 73,089` (`23,807` pts), `validate = 13,435` (`4,201` pts), `test = 25,596` (`2,797` pts)
- **Official `test_list.txt` & Split Membership Untouched**: `True` (`25,596` test images)
- **Zero Patient Leakage**: `True` (`train ∩ validate = 0`, `train ∩ test = 0`, `validate ∩ test = 0`)

## 3. Stratified Model Validation (`images_009`–`images_012`)

- **Checkpoint**: `models\nih-densenet121-nih14.pt` (`SHA-256 = 31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf`)
- **Device & Mini-Batch Size**: `cpu`, `mini_batch_size = 16`
- **Sampled Real Radiographs**: `128` (`32` per new shard: `16 validate` + `16 test` from distinct patients)
- **Execution Time & Throughput**: `18.68s` (`6.85 images/second` — sample-level CPU measurement only; do not extrapolate linearly to full dataset)
- **Probability Domain**: `1,792` scalars in `[0.0, 0.913044]`, `NaN/Inf = False`
- **Threshold Governance**: `threshold_invariant_to_test_label_inversion = True` (`used_test_labels_for_threshold = False`)

| NIH-14 Pathology | Global Positive Count (`112,120`) | Sample Min Prob | Sample Mean Prob | Sample Std Prob | Sample Max Prob |
| :--- | --: | --: | --: | --: | --: |
| `Atelectasis` | 11,559 | 0.000029 | 0.074124 | 0.089396 | 0.571192 |
| `Cardiomegaly` | 2,776 | 0.000000 | 0.053091 | 0.123523 | 0.762133 |
| `Consolidation` | 4,667 | 0.000182 | 0.029188 | 0.048573 | 0.280073 |
| `Edema` | 2,303 | 0.000000 | 0.012022 | 0.028673 | 0.199890 |
| `Effusion` | 13,317 | 0.000115 | 0.138320 | 0.212811 | 0.901970 |
| `Emphysema` | 2,516 | 0.000000 | 0.006730 | 0.013058 | 0.093074 |
| `Fibrosis` | 1,686 | 0.000042 | 0.024646 | 0.032218 | 0.156866 |
| `Hernia` | 227 | 0.000000 | 0.002981 | 0.006518 | 0.037925 |
| `Infiltration` | 19,894 | 0.019130 | 0.200999 | 0.089780 | 0.558602 |
| `Mass` | 5,782 | 0.000183 | 0.062302 | 0.109573 | 0.913044 |
| `Nodule` | 6,331 | 0.001497 | 0.081553 | 0.079880 | 0.426920 |
| `Pleural_Thickening` | 3,385 | 0.000259 | 0.044548 | 0.066567 | 0.391602 |
| `Pneumonia` | 1,431 | 0.000010 | 0.014400 | 0.023656 | 0.136367 |
| `Pneumothorax` | 5,302 | 0.000004 | 0.028733 | 0.056491 | 0.396827 |

## 4. Regression Tests, Disk Usage & Final Completion Status

- **NIH + Benchmark Regression Suite**: `passed = 32`, `failed = 0`, `skipped = 0`, `warnings = 17` (`exit_code = 0`, `50.81s`)
- **Free Space Before Final Batch on `C:\`**: `95.43 GiB`
- **Free Space After Final Batch on `C:\`**: `73.75 GiB` (`A:\`: `30.73 GiB`)
- **Dataset Completion Status**: **`COMPLETE_112120_VERIFIED`** (`112,120 / 112,120` PNGs)
- **Recommendation**: **`DATASET_ACQUISITION_100PCT_COMPLETE_READY_FOR_FINALIZATION_AND_BENCHMARKING`**
