# Grokking Delay — Experimental Validation Suite

Reproducible experiments for **"Why Grokking Takes So Long: A First-Principles Theory of Representational Phase Transitions"** (Truong X.K., Truong Q.H., Luu D.T., March 2026).

## Overview

8 training scripts validate the paper's main theorem:

$$T_{\text{grok}} - T_{\text{mem}} = \Theta\!\left(\frac{1}{\eta\lambda}\log\frac{\|\theta_{\text{mem}}\|^2}{\|\theta_{\text{post}}\|^2}\right)$$

Two figure scripts produce publication-quality plots.

## Quick Start (Google Colab + T4)

1. Upload all `.py` files to `/content/` on Colab
2. Select **GPU → T4** runtime
3. Run scripts in order:

```python
# Core experiments (from paper)
%run s1_lyapunov_v2.py      # ~3 min
%run s2_lambda_sweep_v2.py   # ~25 min
%run s3_modulus_sweep_v2.py   # ~25 min
%run s4_spectral_v2.py       # ~8 min
%run s5_eta_sweep_v2.py      # ~20 min

# Supplementary experiments (new)
%run s6_sgd_vs_adamw.py      # ~8 min
%run s7_hires_fourier.py     # ~15 min
%run s8_weight_decay_convention.py  # ~12 min

# Generate figures
%run make_figures.py                  # Fig 1–5
%run make_figures_supplementary.py    # Fig S1–S3
```

All results save automatically to `Google Drive/grokking_results/`.

## Scripts

### Core Experiments (S1–S5)

| Script | Validates | Section | Key Result |
|--------|-----------|---------|------------|
| `s1_lyapunov_v2.py` | Exponential norm contraction | 3.2–3.3 | Fitted rate 0.99860, R²=0.9991 |
| `s2_lambda_sweep_v2.py` | T∝1/λ, three regimes | 4.2–4.3 | Slope R²=0.971, CI [1082,1271] |
| `s3_modulus_sweep_v2.py` | Norm-ratio formula vs p | 4.4 | Pearson r=0.91, slope 1.08 |
| `s4_spectral_v2.py` | R(f_θ) collapse, gap∝R | 4.6 | OLS R²=0.77, RANSAC R²=0.99 |
| `s5_eta_sweep_v2.py` | T∝1/η, joint ηλ scaling | 4.5 | R²=0.92, mean T·ηλ=9.58 |

### Supplementary Experiments (S6–S8)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `s6_sgd_vs_adamw.py` | SGD vs AdamW ablation — verifies theory matches SGD exactly, quantifies AdamW gap | ~8 min |
| `s7_hires_fourier.py` | High-res Fourier (n_b=p, n_c=p) vs low-res (n_b=3, n_c=5) — checks measurement noise | ~15 min |
| `s8_weight_decay_convention.py` | Factor-of-2 verification: SGD(wd=2λ) vs SGD(wd=λ) vs AdamW(wd=λ) | ~12 min |

### Figure Scripts

| Script | Output |
|--------|--------|
| `make_figures.py` | `fig1_lyapunov.pdf` through `fig5_spectral.pdf` |
| `make_figures_supplementary.py` | `figS1_sgd_vs_adamw.pdf`, `figS2_hires_fourier.pdf`, `figS3_wd_convention.pdf` |

## File Structure

```
grokking_results/           # on Google Drive
├── script1_v2/
│   ├── full_results.json   # all data + logs (for figures)
│   ├── summary.json        # lightweight (no logs)
│   └── figs/summary.png    # diagnostic plot
├── script2_v2/
│   └── ...
├── ...
├── script6_sgd_vs_adamw/
│   ├── full_results.json
│   ├── summary.json
│   └── figs/sgd_vs_adamw.png
├── script7_hires_fourier/
│   └── ...
├── script8_wd_convention/
│   └── ...
└── paper_figures/
    ├── fig1_lyapunov.pdf
    ├── ...
    ├── fig5_spectral.pdf
    ├── figS1_sgd_vs_adamw.pdf
    ├── figS2_hires_fourier.pdf
    └── figS3_wd_convention.pdf
```

## Shared Infrastructure

`shared_v2.py` provides:

- **Model**: 1-layer transformer (d=128, 4 heads, 512 FFN) matching paper Section 4.1
- **Training**: `train_run()` with configurable optimizer, Fourier measurement, early stopping
- **Measurements**: V(θ)=‖θ‖², Fourier R(f_θ), exponential decay fitting, logit bounds
- **I/O**: Google Drive mount, JSON save with full logs, summary export

## Dependencies

Standard Colab environment (no extra installs needed):

```
torch, numpy, scipy, sklearn, matplotlib
```

## What the Supplementary Scripts Address

**S6 (SGD vs AdamW)**: The paper's theory is proved for SGD with weight decay, but all main experiments use AdamW. S6 runs the identical setup with both optimizers, showing that SGD's contraction rate matches the theoretical prediction (1−2ηλ) more closely than AdamW, while both exhibit the same qualitative behavior. This strengthens Remark 3.3 in the paper.

**S7 (High-res Fourier)**: S4's Fourier measurement uses a fast approximation (n_b=3, n_c=5 out of p=97). S7 re-runs with full resolution (n_b=p, n_c=p) on the same seeds, allowing direct comparison. This validates that the OLS R² for the gap∝R relationship is robust to measurement resolution, and provides precise K* values.

**S8 (Weight decay convention)**: PyTorch's `SGD(weight_decay=w)` adds w·θ to the gradient, while the paper writes the regularization term as 2λθ. S8 runs three configurations — SGD(wd=2λ), SGD(wd=λ), and AdamW(wd=λ) — to verify which convention matches the theoretical rate. This resolves the factor-of-2 ambiguity.

## Citation

```bibtex
@article{truong2026grokking,
  title={Why Grokking Takes So Long: A First-Principles Theory of Representational Phase Transitions},
  author={Truong, Xuan Khanh and Truong, Quynh Hoa and Luu, Duc Trung},
  year={2026},
  institution={H\&K Research Studio, Clevix LLC}
}
```
