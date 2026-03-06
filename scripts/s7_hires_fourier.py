"""
Script 7: HIGH-RESOLUTION FOURIER SPECTRAL ANALYSIS
=====================================================
Re-does the spectral separation analysis (S4) with FULL Fourier sampling
(n_b=p, n_c=p) instead of the default n_b=3, n_c=5.

Purpose:
  - Check whether low-res sampling in S4 introduced measurement noise
  - Get precise R values for the validation gap theorem
  - Validate that OLS R2 for gap~R improves with better measurement

Colab setup:
  - Place shared_v2.py in the same directory (or /content/)
  - Mounts Google Drive automatically via setup_drive()
  - Saves results to /content/drive/MyDrive/grokking_results/script7_hires_fourier/

Runtime: ~15 min on T4 (hi-res Fourier is slower, 5+5 seeds)
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor

RUN_NAME = "script7_hires_fourier"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("=" * 70)
log("  SCRIPT 7: HIGH-RESOLUTION FOURIER SPECTRAL ANALYSIS")
log(f"  p=97, eta=1e-3, lam=1.0, 5 seeds, FULL Fourier (n_b=p, n_c=p)")
log(f"  Device: {DEVICE}")
log(f"  Output: {os.path.dirname(FULL_JSON)}")
log("=" * 70)

# ── Config ──
P = 97; ETA = 1e-3; LAM = 1.0
SEEDS = [42, 43, 44, 45, 46]
MAX_STEPS = 50000; FOURIER_EVERY = 500; POST_EXTEND = 5000
R_THRESHOLD = 0.03; PRE_MARGIN = 500; POST_MARGIN = 1000
BOOTSTRAP_N = 2000

# ── TRAIN with high-res Fourier ──
log("\n[PHASE 1] Training with HIGH-RES Fourier (n_b=p, n_c=p)")
runs_hi = []
for seed in SEEDS:
    log(f"\n── Seed {seed} (high-res) ──")
    t0 = time.time()
    r = train_run(
        p=P, eta=ETA, lam=LAM, seed=seed,
        max_steps=MAX_STEPS,
        measure_fourier_every=FOURIER_EVERY,
        measure_logit=True,
        post_grok_extend=POST_EXTEND,
        fourier_nb=P, fourier_nc=P,
    )
    dt = time.time() - t0
    log(f"  T_grok={r['T_grok']} R_final={r['R_final']:.6f} K_final={r['K_final']} ({dt:.0f}s)")
    runs_hi.append(r)

# ── TRAIN with low-res Fourier (same seeds, for direct comparison) ──
log("\n[PHASE 2] Training with LOW-RES Fourier (n_b=3, n_c=5)")
runs_lo = []
for seed in SEEDS:
    log(f"  Low-res seed {seed}")
    t0 = time.time()
    r = train_run(
        p=P, eta=ETA, lam=LAM, seed=seed,
        max_steps=MAX_STEPS,
        measure_fourier_every=FOURIER_EVERY,
        measure_logit=False,
        post_grok_extend=POST_EXTEND,
        fourier_nb=3, fourier_nc=5,
    )
    dt = time.time() - t0
    log(f"  T_grok={r['T_grok']} R_final={r['R_final']:.6f} ({dt:.0f}s)")
    runs_lo.append(r)

# ── BUILD K* from high-res post-grok spectra ──
half = P // 2 + 1

def build_K_star(runs_list, label=""):
    post_specs = []
    for r in runs_list:
        Tg = r.get("T_grok", 0)
        fl = r.get("fourier_logs", {})
        for st, spec in zip(fl.get("steps", []), fl.get("spectra", [])):
            if st > Tg + POST_MARGIN:
                post_specs.append(np.array(spec[:half]))
    K = []
    ms = np.zeros(half)
    if post_specs:
        ms = np.mean(post_specs, axis=0)
        order = np.argsort(ms)[::-1]
        cum = 0; total = np.sum(ms)
        for idx in order:
            K.append(int(idx))
            cum += ms[idx] / total
            if cum >= 0.99:
                break
        log(f"  {label} K*: |K*| = {len(K)} (99% cumulative energy)")
    return K, ms

K_star_hi, mean_spec_hi = build_K_star(runs_hi, "HIGH-RES")
K_star_lo, mean_spec_lo = build_K_star(runs_lo, "LOW-RES")

# ── COMPUTE per-point R + GAP ──
def compute_scatter(runs_list, K_used):
    R_all, gap_all, meta = [], [], []
    for r in runs_list:
        Tg = r.get("T_grok", 0)
        fl = r.get("fourier_logs", {})
        vs = r.get("logs", {}).get("steps", [])
        vl_list = r.get("logs", {}).get("val_loss", [])
        bl_pts = [vl for s, vl in zip(vs, vl_list) if s > Tg + 2000][-5:]
        bl = np.mean(bl_pts) if bl_pts else 0.0
        for st, spec in zip(fl.get("steps", []), fl.get("spectra", [])):
            if st >= Tg - PRE_MARGIN:
                continue
            spec = np.array(spec[:half])
            te = np.sum(spec)
            if te < 1e-15:
                continue
            inK = np.sum(spec[K_used]) if K_used else 0
            R = (te - inK) / te
            if R < R_THRESHOLD:
                continue
            vl = next((v for s, v in zip(vs, vl_list) if s >= st), None)
            if vl is None:
                continue
            gap = vl - bl
            R_all.append(float(R)); gap_all.append(float(gap))
            meta.append({"seed": r["seed"], "step": st, "R": float(R), "gap": float(gap)})
    return R_all, gap_all, meta

R_hi, gap_hi, meta_hi = compute_scatter(runs_hi, K_star_hi)
R_lo, gap_lo, meta_lo = compute_scatter(runs_lo, K_star_lo if K_star_lo else K_star_hi)

# ── OLS & RANSAC ──
def fit_gap_model(R_list, gap_list, label=""):
    result = {"slope_ols": 0, "R2_ols": 0, "slope_ransac": 0, "R2_ransac": 0,
              "ci_low": 0, "ci_high": 0, "n_points": len(R_list)}
    if len(R_list) < 5:
        return result
    sl, ic, rv, _, _ = linregress(R_list, gap_list)
    result.update(slope_ols=float(sl), intercept_ols=float(ic), R2_ols=float(rv ** 2))
    log(f"  {label} OLS: slope={sl:.3f}, intercept={ic:.3f}, R2={rv**2:.4f} (n={len(R_list)})")

    if len(R_list) >= 10:
        X = np.array(R_list).reshape(-1, 1)
        y = np.array(gap_list)
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        sl_r = ransac.estimator_.coef_[0]
        ic_r = ransac.estimator_.intercept_
        im = ransac.inlier_mask_
        ip = np.mean(im) * 100
        yp = sl_r * X[im, 0] + ic_r
        ssr = np.sum((y[im] - yp) ** 2)
        sst = np.sum((y[im] - np.mean(y[im])) ** 2)
        r2r = 1 - ssr / sst if sst > 0 else 0
        result.update(slope_ransac=float(sl_r), R2_ransac=float(r2r), inlier_pct=float(ip))
        log(f"  {label} RANSAC: slope={sl_r:.3f}, R2={r2r:.4f}, inliers={ip:.1f}%")

        # Bootstrap CI
        boot_slopes = []
        for _ in range(BOOTSTRAP_N):
            idx = np.random.choice(len(R_list), len(R_list), replace=True)
            bs, _, _, _, _ = linregress([R_list[i] for i in idx], [gap_list[i] for i in idx])
            boot_slopes.append(bs)
        result["ci_low"] = float(np.percentile(boot_slopes, 2.5))
        result["ci_high"] = float(np.percentile(boot_slopes, 97.5))
        log(f"  {label} Bootstrap 95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    return result

log("\n" + "=" * 70)
log("GAP vs R ANALYSIS")
log("=" * 70)
fit_hi = fit_gap_model(R_hi, gap_hi, "HIGH-RES")
fit_lo = fit_gap_model(R_lo, gap_lo, "LOW-RES")

# ── Compare R at matched steps ──
log("\n── R VALUE COMPARISON (same seeds, matched steps) ──")
r_diffs = []
for rh, rl in zip(runs_hi, runs_lo):
    hd = dict(zip(rh.get("fourier_logs", {}).get("steps", []),
                   rh.get("fourier_logs", {}).get("R", [])))
    ld = dict(zip(rl.get("fourier_logs", {}).get("steps", []),
                   rl.get("fourier_logs", {}).get("R", [])))
    for s in hd:
        if s in ld:
            r_diffs.append(abs(hd[s] - ld[s]))

if r_diffs:
    log(f"  Mean |R_hi - R_lo|: {np.mean(r_diffs):.4f}")
    log(f"  Max  |R_hi - R_lo|: {np.max(r_diffs):.4f}")

log("\n── PER-SEED K and R_final ──")
for rh, rl in zip(runs_hi, runs_lo):
    log(f"  Seed {rh['seed']}: K_hi={rh['K_final']:3d} K_lo={rl['K_final']:3d} | "
        f"R_hi={rh['R_final']:.6f} R_lo={rl['R_final']:.6f}")

# ── SAVE to Google Drive ──
analysis = {
    "K_star_hires": K_star_hi, "K_star_lowres": K_star_lo,
    "K_star_size_hires": len(K_star_hi), "K_star_size_lowres": len(K_star_lo),
    "fit_hires": fit_hi, "fit_lowres": fit_lo,
    "scatter_hires": meta_hi, "scatter_lowres": meta_lo,
    "mean_post_spectrum_hires": mean_spec_hi.tolist(),
    "R_diff_stats": {
        "mean": float(np.mean(r_diffs)) if r_diffs else None,
        "max": float(np.max(r_diffs)) if r_diffs else None,
    },
}
# Tag low-res runs so make_figures can separate them
for r in runs_lo:
    r["_lowres"] = True
all_runs = runs_hi + runs_lo
save_full_results(all_runs, FULL_JSON, extra=analysis)
save_summary_json(runs_hi, SUMMARY_JSON, extra=analysis)

# ── Diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for r in runs_hi:
    fl = r.get("fourier_logs", {})
    if fl:
        axes[0, 0].plot(fl["steps"], fl["R"], alpha=0.6, lw=1.2, label=f's{r["seed"]}')
        if r["T_grok"]:
            axes[0, 0].axvline(r["T_grok"], color='red', lw=1.2, alpha=0.4, ls='--')
axes[0, 0].set_ylabel("R(f)"); axes[0, 0].set_xlabel("Step")
axes[0, 0].set_title("(a) High-res R(f)"); axes[0, 0].legend(fontsize=8)

if R_hi:
    axes[0, 1].scatter(R_hi, gap_hi, alpha=0.4, s=25, c='royalblue', edgecolor='none')
    xl = np.linspace(0, max(R_hi), 100)
    axes[0, 1].plot(xl, fit_hi["slope_ols"] * xl + fit_hi.get("intercept_ols", 0),
                     'r--', lw=2, label=f'R2={fit_hi["R2_ols"]:.3f}')
    axes[0, 1].legend()
axes[0, 1].set_xlabel("R"); axes[0, 1].set_ylabel("Val gap")
axes[0, 1].set_title("(b) Gap vs R (high-res)")

if R_lo:
    axes[0, 2].scatter(R_lo, gap_lo, alpha=0.4, s=25, c='orange', edgecolor='none')
    xl = np.linspace(0, max(R_lo), 100)
    axes[0, 2].plot(xl, fit_lo["slope_ols"] * xl + fit_lo.get("intercept_ols", 0),
                     'r--', lw=2, label=f'R2={fit_lo["R2_ols"]:.3f}')
    axes[0, 2].legend()
axes[0, 2].set_xlabel("R"); axes[0, 2].set_ylabel("Val gap")
axes[0, 2].set_title("(c) Gap vs R (low-res)")

# (d) R hi vs lo scatter
hi_m, lo_m = [], []
for rh, rl in zip(runs_hi, runs_lo):
    hd = dict(zip(rh.get("fourier_logs", {}).get("steps", []),
                   rh.get("fourier_logs", {}).get("R", [])))
    ld = dict(zip(rl.get("fourier_logs", {}).get("steps", []),
                   rl.get("fourier_logs", {}).get("R", [])))
    for s in hd:
        if s in ld:
            hi_m.append(hd[s]); lo_m.append(ld[s])
if hi_m:
    axes[1, 0].scatter(hi_m, lo_m, alpha=0.4, s=20, c='green')
    lim = max(max(hi_m), max(lo_m)) * 1.1
    axes[1, 0].plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
axes[1, 0].set_xlabel("R (hi-res)"); axes[1, 0].set_ylabel("R (lo-res)")
axes[1, 0].set_title("(d) R: high vs low resolution")

# (e) Cumulative energy
sorted_e = np.sort(mean_spec_hi)[::-1]
cum = np.cumsum(sorted_e) / (np.sum(sorted_e) + 1e-15)
axes[1, 1].plot(cum[:50], 'b-', lw=2.5)
axes[1, 1].axhline(0.99, color='red', ls='--')
axes[1, 1].axvline(len(K_star_hi), color='green', ls='--', label=f'K*={len(K_star_hi)}')
axes[1, 1].set_xlabel("# frequencies"); axes[1, 1].set_ylabel("Cumulative energy")
axes[1, 1].set_title("(e) K* detection (hi-res)"); axes[1, 1].legend()

# (f) Summary
axes[1, 2].axis('off')
txt = (f"HIGH-RES (n_b=p, n_c=p):\n"
       f"  K* = {len(K_star_hi)}\n"
       f"  OLS R2 = {fit_hi['R2_ols']:.4f}\n"
       f"  Slope  = {fit_hi['slope_ols']:.2f}\n\n"
       f"LOW-RES (n_b=3, n_c=5):\n"
       f"  K* = {len(K_star_lo)}\n"
       f"  OLS R2 = {fit_lo['R2_ols']:.4f}\n"
       f"  Slope  = {fit_lo['slope_ols']:.2f}\n")
if r_diffs:
    txt += f"\nMean |dR| = {np.mean(r_diffs):.4f}"
axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
axes[1, 2].set_title("(f) Summary")

plt.suptitle(f"Script 7: High-Res Fourier — p={P}", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "hires_fourier.png"), dpi=150)
plt.close()

log(f"\nDiagnostic plot: {OUT}/hires_fourier.png")
log(f"Full results: {FULL_JSON}")
log("SCRIPT 7 COMPLETE!")
