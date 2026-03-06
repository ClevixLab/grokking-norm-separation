# Data Directory

## Full results (with logs, for figure generation)

| File | Script | Runs | Size |
|------|--------|------|------|
| `full_results1.json` | S1: Lyapunov | 10 | ~32 KB |
| `full_results2.json` | S2: λ sweep | 80 | ~615 KB |
| `full_results3.json` | S3: Modulus sweep | 49 | ~161 KB |
| `full_results4.json` | S4: Spectral | 5 | ~286 KB |
| `full_results5.json` | S5: η sweep | 70 | ~1.4 MB |
| `full_results6.json` | S6: SGD vs AdamW | 10 | derived from S8 |
| `full_results7.json` | S7: Hi-res Fourier | 10 | ~568 KB |
| `full_results8.json` | S8: WD convention | 15 | ~369 KB |
| `full_results9.json` | S9: Multiplication | 20 | flat list |
| `full_results10.json` | S10: Parity | 15 | flat list |

**Note:** `full_results6.json` is derived from `full_results8.json` (which contains a superset: SGD(wd=2λ), SGD(wd=λ), and AdamW(wd=λ)). To regenerate, run `s6_sgd_vs_adamw.py`.

## Summary files (lightweight, no logs)

Summary files strip the `logs`, `fourier_logs`, and `logit_logs` fields to keep
file sizes small. Used for quick analysis without loading full trajectories.
