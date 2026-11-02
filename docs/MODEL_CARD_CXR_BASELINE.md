# Model Card: Pretrained Chest X-Ray DenseNet-121 Baselines (`NIH-14` & `CheXpert-External`)

## 1. Provenance & Cryptographic Identity

MedFusion-CXR's official NIH ChestX-ray14 pipeline (`src/vision/pretrained_cxr.py` -> `PretrainedNihDenseNet121`) audits and supports two single-dataset DenseNet-121 checkpoints from `torchxrayvision` `v1.5.5` (Cohen et al., MIDL 2022, [arXiv:2002.02497](https://arxiv.org/abs/2002.02497), Apache-2.0 License):

| Field | Primary NIH-14 Profile (`nih14`) | External Zero-Overlap Profile (`chex_external`) |
| :--- | :--- | :--- |
| **Upstream Identifier** | `densenet121-res224-nih` | `densenet121-res224-chex` |
| **Local Checkpoint Path** | `models/nih-densenet121-nih14.pt` | `models/chex-densenet121-chexpert.pt` |
| **File Size** | `28,382,096` bytes | `28,381,998` bytes |
| **SHA-256 Checksum** | `31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| **Training Dataset** | NIH ChestX-ray14 (`nih` single-dataset weights) | Stanford CheXpert (`chex` single-dataset weights) |
| **Active NIH-14 Labels** | **14 / 14** (`0..13` trained; `14..17` masked) | **7 / 14** (`0, 1, 3, 4, 7, 8, 10` trained; 7 NIH-only masked as `None`) |
| **Training Overlap Note** | Trained upstream on TorchXRayVision's internal NIH patient split (`seed=0`), which may partially overlap NIH's official `test_list.txt`. | **0% NIH ChestX-ray14 training overlap** (100% out-of-domain external baseline). |

---

## 2. Direct 18-Slot Classifier Head Verification (`models/nih-densenet121-nih14.pt`)

Direct inspection of `model.classifier.weight` ($\|w_i\|_2$) and `model.classifier.bias` ($b_i$) inside `models/nih-densenet121-nih14.pt` confirms all 14 NIH ChestX-ray14 slots (`0..13`) have trained weights while the 4 non-NIH slots (`14..17`) have exact zero weights and are quarantined:

| Slot | Raw `model.targets[i]` | Canonical `NIH_LABELS` Mapping | Trained in `nih14`? | $\|w_i\|_2$ (`nih14`) | Bias $b_i$ (`nih14`) | Published `NIH_OP_THRESHS` | Trained in `chex_external`? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `Atelectasis` | `Atelectasis` | **Yes** | `1.9416` | `-0.067342` | `0.039118` | Yes (`1.7562`) |
| **1** | `Consolidation` | `Consolidation` | **Yes** | `2.2840` | `-0.052101` | `0.003453` | Yes (`2.2071`) |
| **2** | `Infiltration` | `Infiltration` | **Yes** | `1.9286` | `-0.077933` | `0.113963` | No (`0.0000` -> `None`) |
| **3** | `Pneumothorax` | `Pneumothorax` | **Yes** | `2.7955` | `-0.067032` | `0.005730` | Yes (`2.6213`) |
| **4** | `Edema` | `Edema` | **Yes** | `4.3826` | `-0.003064` | `0.000457` | Yes (`2.4843`) |
| **5** | `Emphysema` | `Emphysema` | **Yes** | `2.9448` | `-0.069527` | `0.001888` | No (`0.0000` -> `None`) |
| **6** | `Fibrosis` | `Fibrosis` | **Yes** | `2.2399` | `-0.029679` | `0.012038` | No (`0.0000` -> `None`) |
| **7** | `Effusion` | `Effusion` | **Yes** | `2.1978` | `-0.055370` | `0.038744` | Yes (`2.5101`) |
| **8** | `Pneumonia` | `Pneumonia` | **Yes** | `2.7563` | `-0.017621` | `0.003721` | Yes (`1.9007`) |
| **9** | `Pleural_Thickening` | `Pleural_Thickening` | **Yes** | `2.0202` | `-0.034995` | `0.014731` | No (`0.0000` -> `None`) |
| **10** | `Cardiomegaly` | `Cardiomegaly` | **Yes** | `2.8692` | `-0.049546` | `0.016150` | Yes (`2.2760`) |
| **11** | `Nodule` | `Nodule` | **Yes** | `1.7309` | `-0.067007` | `0.054241` | No (`0.0000` -> `None`) |
| **12** | `Mass` | `Mass` | **Yes** | `2.1821` | `-0.033516` | `0.037199` | No (`0.0000` -> `None`) |
| **13** | `Hernia` | `Hernia` | **Yes** | `3.4107` | `-0.100636` | `0.000440` | No (`0.0000` -> `None`) |
| **14** | `Lung Lesion` | *Not in NIH-14* | **No (Masked)** | `0.0000` | `+0.000000` | `NaN` | Yes (CheXpert-only) |
| **15** | `Fracture` | *Not in NIH-14* | **No (Masked)** | `0.0000` | `+0.000000` | `NaN` | Yes (CheXpert-only) |
| **16** | `Lung Opacity` | *Not in NIH-14* | **No (Masked)** | `0.0000` | `+0.000000` | `NaN` | Yes (CheXpert-only) |
| **17** | `Enlarged Cardiomediastinum` | *Not in NIH-14* | **No (Masked)** | `0.0000` | `+0.000000` | `NaN` | Yes (CheXpert-only) |

---

## 3. Input Preprocessing & Streaming Mini-Batch Pipeline

`PretrainedNihDenseNet121.preprocess_batch()` reproduces exact TorchXRayVision training normalization with bounded $O(B)$ memory:

1. **Grayscale Conversion**: 8-bit 2D grayscale (`L` mode, pixel domain $[0, 255]$).
2. **Dynamic Range Scaling**:
   $$x_{\text{norm}} = \left(2.0 \times \frac{x}{255.0} - 1.0\right) \times 1024.0 \in [-1024.0, +1024.0]$$
3. **Square Center Crop (`XRayCenterCrop`)**: $\min(H, W) \times \min(H, W)$ centered crop preserving aspect ratio.
4. **Spatial Resize**: Bilinear anti-aliased interpolation to $(B, 1, 224, 224)$.

---

## 4. Target Ontology, Threshold Provenance, & Bootstrap CIs

- **Evaluation Dataset**: NIH ChestX-ray14 ([Kaggle `nih-chest-xrays/data`](https://www.kaggle.com/datasets/nih-chest-xrays/data); Wang et al., CVPR 2017, NIH Clinical Center). PhysioNet/MIMIC datasets are not used.
- **Strict Binary Labels (`LabelDomainPolicy.BINARY_STRICT`)**: Ground-truth targets are strictly `{0, 1}` across the 14 `NIH_LABELS`. `"No Finding"` encodes all 14 targets as `0`. CheXpert `-1` uncertainty values are rejected.
- **Threshold Provenance**:
  - `source_op_thresh` (`NIH_OP_THRESHS` / `CHEX_OP_THRESHS`): Recorded strictly as reference metadata (`source_op_thresh_used_for_test_metrics: false`).
  - `validation_optimal_threshold`: Selected exclusively on the patient-disjoint `validate` split via a 91-point grid ($t \in [0.05, 0.95]$), frozen, and applied to official `test_list.txt` (`used_test_labels_for_threshold: false`).
- **Patient-Clustered Bootstrap (`patient_id`)**: 95% percentile CIs computed by resampling unique `patient_id` clusters with replacement, skipping single-class draws safely and computing CIs solely from valid dual-class iterations.
