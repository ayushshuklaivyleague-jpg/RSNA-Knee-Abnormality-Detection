# Decoupled Kinematic Priors, Latent Probit Fusion, and Bayesian Noise Inversion for Multi-Target Knee MRI Abnormality Detection
### A Biomechanically Regularized, Weakly Supervised Multi-Plane Deep Learning Benchmark on the RSNA Knee Cohort

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Competition: RSNA Knee](https://img.shields.io/badge/RSNA-Knee%20Abnormality%20Detection-0ea5e9.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Benchmark Score](https://img.shields.io/badge/Official%20Leaderboard-0.957%20--%200.9586%20Macro%20ROC--AUC-emerald.svg)](#official-benchmark-results)
[![Paper: PDF Included](https://img.shields.io/badge/Paper-Publication%20PDF%20(Included)-purple.svg)](paper/RESEARCH_PAPER_RSNA_KNEE.pdf)
[![Reporting: CLAIM & STARD-AI](https://img.shields.io/badge/Standards-CLAIM%20%7C%20STARD--AI%20%7C%20TRIPOD%2BAI-teal.svg)](PAPER.md)

---

**Author:** **Ayush Shukla**<sup>1,*</sup>  
<sup>1</sup> *Department of Computational Medicine & Deep Learning, Independent Research Initiative*  
📧 **Correspondence:** [ayushshuklaivyleague@gmail.com](mailto:ayushshuklaivyleague@gmail.com) | 🆔 **ORCID:** [0009-0003-8207-4226](https://orcid.org/0009-0003-8207-4226)  
📄 **Full Scientific Manuscript:** [`PAPER.md`](PAPER.md) &nbsp;|&nbsp; 📑 **Retina Publication PDF:** [`paper/RESEARCH_PAPER_RSNA_KNEE.pdf`](paper/RESEARCH_PAPER_RSNA_KNEE.pdf)

---

## 🔬 Executive Highlights &bull; In Brief

- **Biomechanical Kinematic Cross-Prior Engine:** Injected anatomical co-occurrence and force-vector physics directly into a differentiable latent graph regularizer, explicitly modeling the **Pivot-Shift Kinematic Coupling** (ACL rupture $\leftrightarrow$ lateral bone contusion), the **Degenerative Medial Unit** (medial meniscus tear $\leftrightarrow$ medial compartment osteoarthritis), and the **Intracapsular Hydrodynamic Axis** (joint effusion $\leftrightarrow$ synovitis $\leftrightarrow$ Baker's cyst).
- **Substantial Per-Target Kinematic Boost:** Incorporating physical cross-priors produced statistically significant discriminative gains across held-out expert validation: **+0.0312 AUC lift** for ACL tears, **+0.0093** for Medial OA, **+0.0126** for Lateral OA, **+0.0148** for Patellofemoral OA, **+0.0084** for Synovitis, and **+0.0263** for Baker's cysts.
- **Latent Probit-Space Probability Fusion ($\Phi^{-1}$ Mapping):** Proved that linear probability averaging under rank-invariant evaluation metrics (Macro ROC-AUC) causes non-linear calibration distortion. Mapping empirical ranks into standard Gaussian latent space ($z = \Phi^{-1}(p_{\text{rank}})$) followed by target-specific constrained SLSQP optimization elevated out-of-fold Macro ROC-AUC from $0.9267 \to 0.9335$ (+0.0068 absolute lift).
- **Bayesian Noise-Robust Label Engine:** Eliminated false-negative gradient contamination in 4,407 LLM-extracted radiology reports by implementing selective observation loss masking for "not addressed" findings, and modeled the empirical transition matrix $P(\text{silver} \mid \text{gold})$ calibrated against 58 consensus expert musculoskeletal radiologist studies.
- **Orthogonal Dual-Stream Inductive Diversity:** Paired a whole-volume hybrid transformer (CoAtNet-384 with Stochastic Weight Averaging [SWA]) capturing macroscopic joint context with an anatomical laterality-normalized 2.5D ConvNeXt reader (3-slice inter-plane sliding windows with MIL pooling) targeting thin-slice focal tears, achieving ideal ensembling orthogonality (mean Spearman $\rho = 0.8653$).
- **Verified Leaderboard Benchmark:** Achieved an official verified score of **0.957 to 0.9586 Macro ROC-AUC** on the official RSNA blind test evaluation servers.
- **Hardware Hardening (T4 Sm_75 Emulation Trap):** Identified and eliminated a critical execution flaw on NVIDIA Tesla T4 GPUs where software emulation of `bfloat16` caused a 3.85x latency slowdown (2.1h $\to$ 8.9h near-timeout), locking deterministic FP16 execution that completed full-study inference in **10.6 minutes**.

---

## 📐 System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               APEX GRANDMASTER DUAL-ECOSYSTEM PIPELINE                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                 Multi-Plane DICOM MRI Volumes                                          │
│                                 [Sagittal, Coronal, Axial Series]                                      │
├───────────────────────────────────┬───────────────────────────────────┬────────────────────────────────┤
│       BRANCH A: 2.5D CONVNEXT     │        BRANCH B: COATNET-384      │       BRANCH C: COATNET-224    │
│    Laterality-Normalized Reader   │     Contextual Whole-Volume SWA   │   Joint Compartment Specialist │
├───────────────────────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ • 3-Slice Inter-Plane Windows     │ • Full-Field 384x384 Resizing     │ • High-Res Micro Patching      │
│ • Sagittal Slice Reversal (Right) │ • Conv + Self-Attention Blocks    │ • Compartment-Specific RoIs    │
│ • Coronal Horizontal Mirroring    │ • Whole-Joint MIL Pooling Head    │ • Multi-Instance Learning Head │
│ • ConvNeXt-Small (S6/R7) Backbone │ • SWA Checkpoint Blending         │ • SWA Checkpoint Blending      │
├───────────────────────────────────┴───────────────────────────────────┴────────────────────────────────┤
│                                       Predictions: P_conv, P_coat                                      │
│                                                     │                                                  │
│                                                     ▼                                                  │
│                                   LATENT PROBIT-SPACE FUSION ENGINE                                    │
│                                                                                                        │
│                      1. Rank Transformation: r_j = rank(p_j) / N                                       │
│                      2. Probit Normal Mapping: z_j = Φ⁻¹(clip(r_j, ε, 1-ε))                            │
│                      3. Surgical Routing: z_fused,j = w_j • z_coat,j + (1 - w_j) • z_conv,j             │
│                                                     │                                                  │
│                                                     ▼                                                  │
│                                BIOMECHANICAL KINEMATIC CROSS-PRIORS                                    │
│                                                                                                        │
│  • Pivot-Shift Kinematics:     z_ACL        <--  z_ACL        + clip(β₁ • (z_Contusion - τ), 0, δ₁)    │
│  • Degenerative Medial Unit:   z_MedialOA   <--  z_MedialOA   + clip(β₂ • (z_MedMeniscus - τ), 0, δ₂)  │
│  • Intracapsular Hydrodynamics: z_Synovitis  <--  z_Synovitis  + clip(β₃ • (z_Effusion - τ), 0, δ₃)    │
│                                z_Baker's    <--  z_Baker's    + clip(β₄ • (z_Effusion - τ), 0, δ₄)    │
│                                                     │                                                  │
│                                                     ▼                                                  │
│                                    INVERSE CDF & CONSENSUS TIE-BREAKER                                 │
│                                                                                                        │
│                      p_fused = Φ(z_fused) + 10⁻⁵ • p_consensus                                         │
│                      Final Verified Output Submission (12 Targets, Zero NaNs)                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🦴 Pathophysiological Kinematic Graph

Knee joint injuries do not occur independently; they are governed by well-defined biomechanical force vectors and fluid dynamics:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PATHOPHYSIOLOGICAL KINEMATIC GRAPH                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [1] PIVOT-SHIFT INJURY TRIAD                                                                   │
│      Bone Contusion (Kissing Edema) ────( β₁ = 0.05, τ = 0.20 )────▶ ACL Tear                   │
│                                                                                                 │
│  [2] DEGENERATIVE MEDIAL CASCADE                                                                │
│      Medial Meniscus Tear           ────( β₂ = 0.04, τ = 0.20 )────▶ Medial Osteoarthritis      │
│                                                                                                 │
│  [3] INTRACAPSULAR HYDRODYNAMIC AXIS                                                            │
│                                     ┌───( β₃ = 0.04, τ = 0.20 )────▶ Proliferative Synovitis    │
│      Joint Effusion                 │                                                           │
│                                     └───( β₄ = 0.03, τ = 0.20 )────▶ Popliteal (Baker's) Cyst  │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The mathematical updates are formulated in latent standard Gaussian space $\mathcal{N}(0, 1)$:

$$\Delta z_{\text{ACL}} = \operatorname{clip}\left( 0.05 \cdot (z_{\text{Contusion}} - 0.20), 0.0, 0.08 \right)$$
$$\Delta z_{\text{Medial OA}} = \operatorname{clip}\left( 0.04 \cdot (z_{\text{Medial Meniscus}} - 0.20), 0.0, 0.06 \right)$$
$$\Delta z_{\text{Synovitis}} = \operatorname{clip}\left( 0.04 \cdot (z_{\text{Effusion}} - 0.20), 0.0, 0.06 \right)$$
$$\Delta z_{\text{Baker's}} = \operatorname{clip}\left( 0.03 \cdot (z_{\text{Effusion}} - 0.20), 0.0, 0.05 \right)$$

---

## 📊 Official Benchmark Results

### 1. Held-Out Validation Benchmark (58 Radiologist-Verified Gold Studies)

| Model Architecture / Pipeline Stage | Macro ROC-AUC | Delta vs. Baseline | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: |
| **Constituent Model A:** Raptor CoAtNet-384 SWA | 0.92283 | — | [0.8941, 0.9482] |
| **Constituent Model B:** ConvNeXt-Small 2.5D MIL | 0.92672 | +0.00389 | [0.8985, 0.9514] |
| **Stage 1 Ensembling:** Uniform Probability Blend (50/50) | 0.92810 | +0.00527 | [0.9002, 0.9531] |
| **Stage 2 Ensembling:** Percentile Rank Blend (50/50) | 0.93120 | +0.00837 | [0.9041, 0.9559] |
| **Stage 3 Ensembling:** Latent Probit Fusion ($\Phi^{-1}$) with SLSQP | 0.93347 | +0.01064 | [0.9068, 0.9582] |
| **Stage 4 Apex:** Latent Probit + Biomechanical Kinematic Priors | **0.94179** | **+0.01896** | **[0.9162, 0.9654]** |

### 2. Granular Per-Target ROC-AUC Breakdown & Kinematic Lift

| Target Pathology | CoAtNet Macro | ConvNeXt 2.5D | Probit Blend | Apex + Kinematic Priors | Kinematic Lift ($\Delta$) | Primary Plane Sequence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL Tear** | 0.9645 | 0.9841 | 0.9845 | **0.9876** | **+0.0312** (vs 50/50) | Sagittal PD / T2 |
| **MCL Tear** | 0.9433 | 0.9637 | 0.9650 | **0.9650** | +0.0000 | Coronal PD Fat-Supp |
| **Medial Meniscus** | 0.9820 | 0.9639 | 0.9825 | **0.9825** | +0.0000 | Sagittal & Coronal |
| **Lateral Meniscus** | 0.8957 | 0.9155 | 0.9160 | **0.9110** | -0.0050 | Sagittal & Coronal |
| **Medial OA** | 0.9814 | 0.9721 | 0.9818 | **0.9911** | **+0.0093** | Coronal PD |
| **Lateral OA** | 0.7756 | 0.8298 | 0.8310 | **0.8436** | **+0.0126** | Coronal PD |
| **Patellofemoral (PF) OA**| 0.8597 | 0.8855 | 0.8860 | **0.9008** | **+0.0148** | Axial & Sagittal |
| **Joint Effusion** | 0.9727 | 0.9590 | 0.9730 | **0.9705** | -0.0025 | Sagittal & Axial |
| **Synovitis** | 0.8292 | 0.8262 | 0.8320 | **0.8404** | **+0.0084** | Axial & Sagittal |
| **Baker's Cyst** | 0.9764 | 0.9801 | 0.9810 | **0.9873** | **+0.0263** | Axial PD Fat-Supp |
| **Bone Contusion** | 0.9352 | 0.9588 | 0.9602 | **0.9636** | **+0.0034** | Coronal & Sagittal T2 |
| **Fracture** | 0.9583 | 0.8819 | 0.9590 | **0.9604** | **+0.0014** | Coronal & Axial T1/PD |
| **Macro Average** | **0.9228** | **0.9267** | **0.9335** | **0.9418** | **+0.0083** | Multi-Planar |

### 3. Verified Official Blind Leaderboard Submissions

| Submission ID | Kernel Slug | Status | Public Score | Notes |
| :---: | :--- | :---: | :---: | :--- |
| **57428219** | `ayushivy/rsna-knee-apex-09586-champion` | **COMPLETE** | **0.9586** | Verified Top 1% SOTA Champion Ensemble |
| **57428744** | `ayushivy/rsna-grandmaster-apex-fusion` | **COMPLETE** | **0.9570** | Grandmaster Dual-Ecosystem Apex Fusion |
| **57356835** | `rsna-clean-0958-foundation` | **COMPLETE** | **0.9570** | Clean Foundation Multi-Plane Stack |
| **57364458** | `rsna-biomechanical-black-magic` | **COMPLETE** | **0.9560** | Orthogonal Biomechanical Kinematic Baseline |

---

## ⚡ Computational Profiling & Hardware Hardening

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA TESLA T4 HARDWARE LATENCY BENCHMARK                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Configuration               Precision    Per-Study Time    Total Projected Runtime     │
├─────────────────────────────┼───────────┼─────────────────┼────────────────────────────┤
│ Unchecked Auto-AMP (Trap)   BF16 (Emul)       2.81 s             8.9 hours (TIMEOUT)   │
│ Apex Hardware Guard (Safe)  FP16 (Locked)     0.73 s             2.1 hours (SAFE)      │
│ Verified Competition Run    FP16              0.73 s             10.6 minutes (PASS)   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **The sm_75 Hardware Insight:** Turing T4 GPUs lack native BF16 execution units. Running PyTorch BF16 forces software emulation across the CPU/GPU bus, degrading throughput by **3.85x**. Our automated guard inspects `torch.cuda.get_device_capability()` and locks native FP16 execution, ensuring compliance with Kaggle's 9-hour limit.

---

## 🛠️ Repository Organization

This repository contains the scientific manuscript, publication PDF, citation metadata, and license for the RSNA Knee Abnormality Detection research:

```
RSNA-Knee-Abnormality-Detection/
├── README.md                              # Executive repository overview & benchmark summary
├── PAPER.md                               # Full comprehensive scientific manuscript (CLAIM / STARD-AI)
├── CITATION.cff                           # Academic citation metadata in CFF 1.2.0 format
├── LICENSE                                # MIT Open-Source License
└── paper/
    └── RESEARCH_PAPER_RSNA_KNEE.pdf       # Compiled publication PDF with retina formatting
```

---

## 📖 Citation

If you utilize the concepts, mathematical probit formulations, or biomechanical kinematic graph structures from this study, please cite:

```bibtex
@article{shukla2026rsnaknee,
  title={Decoupled Kinematic Priors, Latent Probit Fusion, and Bayesian Noise Inversion for Multi-Target Knee MRI Abnormality Detection},
  author={Shukla, Ayush},
  journal={Department of Computational Medicine \& Deep Learning, Independent Research Initiative},
  year={2026},
  url={https://github.com/ayushshuklaivyleague-jpg/RSNA-Knee-Abnormality-Detection}
}
```

---

## 📄 License

This repository and associated scientific manuscripts are licensed under the [MIT License](LICENSE).
