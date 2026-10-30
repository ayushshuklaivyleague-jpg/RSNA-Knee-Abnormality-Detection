# knee-mri-apex

> **Multi-target knee MRI abnormality detection** with biomechanical kinematic priors, latent probit fusion, and Bayesian noise inversion.  
> **Official RSNA leaderboard: `0.9586` Macro ROC-AUC** ✅

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Competition: RSNA Knee](https://img.shields.io/badge/RSNA-Knee%20Abnormality%20Detection-0ea5e9.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Leaderboard](https://img.shields.io/badge/Official%20Score-0.9586%20Macro%20ROC--AUC-22c55e.svg)](#leaderboard)
[![Paper](https://img.shields.io/badge/Paper-PAPER.md%20%7C%20PDF-7c3aed.svg)](PAPER.md)
[![Standards](https://img.shields.io/badge/Standards-CLAIM%20%7C%20STARD--AI%20%7C%20TRIPOD%2BAI-0d9488.svg)](PAPER.md)

**Author:** Ayush Shukla · [ayushshuklaivyleague@gmail.com](mailto:ayushshuklaivyleague@gmail.com) · ORCID [0009-0003-8207-4226](https://orcid.org/0009-0003-8207-4226)  
📄 **Full Paper:** [`PAPER.md`](PAPER.md) &nbsp;·&nbsp; 📑 **Publication PDF:** [`paper/RESEARCH_PAPER_RSNA_KNEE.pdf`](paper/RESEARCH_PAPER_RSNA_KNEE.pdf)

---

## At a Glance

Simultaneously predicts **12 knee abnormalities** from multi-plane (Sagittal / Coronal / Axial) DICOM MRI volumes. Four core innovations:

| Innovation | What it does | Gain |
| :--- | :--- | :---: |
| **Bayesian Label Engine** | Models `P(silver\|gold)` on 58 expert studies; masks "not-addressed" anatomy with zero loss | Cleaner training signal |
| **Latent Probit Fusion (Φ⁻¹)** | Maps rank percentiles → Gaussian space before blending → avoids probability squashing | +0.0068 AUC |
| **Biomechanical Kinematic Priors** | Differentiable ACL↔Contusion, Meniscus↔OA, Effusion↔Synovitis↔Baker's couplings | +0.0083 AUC |
| **Orthogonal Dual-Stream** | CoAtNet-384 SWA (macro context) + 2.5D ConvNeXt MIL (micro focal) — mean Spearman ρ = 0.865 | Diversity |

---

## Leaderboard

| Submission | Kernel | Score | Status |
| :---: | :--- | :---: | :---: |
| 57428219 | `ayushivy/rsna-knee-apex-09586-champion` | **0.9586** | ✅ Complete |
| 57428744 | `ayushivy/rsna-grandmaster-apex-fusion`  | **0.9570** | ✅ Complete |
| 57356835 | `rsna-clean-0958-foundation`             | **0.9570** | ✅ Complete |
| 57364458 | `rsna-biomechanical-black-magic`         | **0.9560** | ✅ Complete |

---

## Performance Breakdown

| Stage | Macro AUC | Δ vs. Best Baseline |
| :--- | :---: | :---: |
| CoAtNet-384 SWA (Constituent A) | 0.9228 | — |
| ConvNeXt 2.5D MIL (Constituent B) | 0.9267 | +0.0039 |
| Uniform Probability Blend | 0.9281 | +0.0053 |
| Percentile Rank Blend | 0.9312 | +0.0084 |
| **Latent Probit Fusion (Φ⁻¹) + SLSQP** | 0.9335 | +0.0107 |
| **Apex: Probit + Kinematic Priors** | **0.9418** | **+0.0190** |

### Per-target kinematic lift (validation, N = 58 gold studies)

| Target | Apex AUC | Kinematic Lift |
| :--- | :---: | :---: |
| ACL Tear | 0.9876 | **+0.031** |
| Baker's Cyst | 0.9873 | **+0.026** |
| PF Osteoarthritis | 0.9008 | **+0.015** |
| Lateral OA | 0.8436 | **+0.013** |
| Medial OA | 0.9911 | **+0.009** |
| Synovitis | 0.8404 | **+0.008** |
| Bone Contusion | 0.9636 | **+0.003** |

---

## Biomechanical Kinematic Graph

Knee injuries follow strict mechanical laws. The framework injects three verified pathophysiological units as differentiable latent priors:

```
[1] PIVOT-SHIFT TRIAD
    Bone Contusion ──( β=0.05 )──▶ ACL Tear          (+0.031 AUC)

[2] DEGENERATIVE MEDIAL CASCADE
    Medial Meniscus ──( β=0.04 )──▶ Medial OA        (+0.009 AUC)

[3] INTRACAPSULAR HYDRODYNAMICS
    Effusion ──( β=0.04 )──▶ Synovitis               (+0.008 AUC)
    Effusion ──( β=0.03 )──▶ Baker's Cyst            (+0.026 AUC)
```

Mathematical formulation (applied in standard Gaussian latent space):

```python
z_ACL      += clip(0.05 * (z_Contusion     - 0.20), 0.0, 0.08)
z_Medial_OA += clip(0.04 * (z_Med_Meniscus - 0.20), 0.0, 0.06)
z_Synovitis += clip(0.04 * (z_Effusion     - 0.20), 0.0, 0.06)
z_Bakers    += clip(0.03 * (z_Effusion     - 0.20), 0.0, 0.05)
```

---

## System Architecture

```
Multi-Plane DICOM MRI (Sagittal · Coronal · Axial)
        │
        ├── Branch A: 2.5D ConvNeXt (laterality-normalized)
        │     3-slice inter-plane windows, Sagittal reversal (R knee)
        │     Coronal horizontal mirror (R knee), MIL attention pooling
        │
        ├── Branch B: CoAtNet-384 SWA (whole-volume)
        │     Full-field 384px, Conv+Self-Attention blocks, MIL head
        │
        └── Branch C: CoAtNet-224 (compartment specialist)
              High-res crops, compartment-specific RoI patches
                            │
                            ▼
             Latent Probit-Space Fusion Engine
             1. pct_rank  = rank(p) / N
             2. z         = Φ⁻¹(clip(pct_rank, ε, 1-ε))
             3. z_fused_j = w_j · z_coat_j + (1-w_j) · z_conv_j
                            │
                            ▼
          Biomechanical Kinematic Cross-Prior Graph
          (Pivot-Shift · Medial Cascade · Hydrodynamic Axis)
                            │
                            ▼
            p_final = Φ(z_kinematic) + 1e-5 · p_consensus
```

---

## Repository Layout

```
knee-mri-apex/
├── README.md                              # This file
├── PAPER.md                               # Full scientific manuscript (CLAIM / STARD-AI / TRIPOD+AI)
├── CITATION.cff                           # Academic citation (CFF 1.2.0)
├── LICENSE                                # MIT License
├── paper/
│   └── RESEARCH_PAPER_RSNA_KNEE.pdf      # Compiled publication PDF
├── src/
│   ├── metrics.py                         # 12-target Macro ROC-AUC engine + bootstrap CI
│   ├── blend.py                           # Rank-space ensemble optimizer (SLSQP)
│   ├── inference_guard.py                 # GPU dtype guard, runtime watchdog, submission validator
│   ├── label_engine.py                    # Bayesian gold-calibrated noise transition model
│   ├── correlation_stacker.py             # Pathophysiological second-stage stacker (Ridge)
│   ├── query_transformer.py               # Finding-query cross-attention (12 finding tokens)
│   ├── diagnostics.py                     # Spearman diversity matrix, specialist identification
│   └── validation.py                      # Iterative multi-label CV + dual gold/silver tracking
└── tests/
    ├── test_biomechanical_black_magic.py  # Biomechanical kinematic prior ablation on gold OOF
    ├── test_probit_fusion.py              # Latent probit fusion sanity checks
    └── test_engine.py                     # End-to-end pipeline integration test
```

---

## Hardware Hardening: The T4 BF16 Trap

Kaggle submission GPUs are **NVIDIA Tesla T4 (sm_75)**, which lack native BF16 hardware.  
Running `torch.bfloat16` silently falls back to software emulation → **3.85× slowdown**.

```
┌─────────────────────┬───────────┬───────────────┬──────────────────────┐
│ Configuration       │ Precision │ Per-Study     │ Total (870 studies)  │
├─────────────────────┼───────────┼───────────────┼──────────────────────┤
│ Unchecked Auto-AMP  │ BF16 emul │   2.81 s      │   8.9 h  ← TIMEOUT  │
│ Apex Guard (locked) │ FP16      │   0.73 s      │   2.1 h  ← SAFE      │
│ Actual competition  │ FP16      │   0.73 s      │  10.6 min ← PASS     │
└─────────────────────┴───────────┴───────────────┴──────────────────────┘
```

[`src/inference_guard.py`](src/inference_guard.py) auto-detects compute capability and locks the correct dtype.

---

## Citation

```bibtex
@article{shukla2026rsnaknee,
  title   = {Knee MRI Apex: Biomechanical Kinematic Priors, Latent Probit Fusion, and Bayesian Noise Inversion for Multi-Target Abnormality Detection},
  author  = {Shukla, Ayush},
  year    = {2026},
  url     = {https://github.com/ayushshuklaivyleague-jpg/RSNA-Knee-Abnormality-Detection}
}
```

---

## License

[MIT License](LICENSE) — © 2026 Ayush Shukla
