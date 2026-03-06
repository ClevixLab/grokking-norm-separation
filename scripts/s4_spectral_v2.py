"""
Script 4 v2: SPECTRAL SEPARATION — FULL DATA SAVE
Validates: Sec 3.4, 4.6 (R(f_θ) collapse, validation gap ∝ R)
Runtime: ~8 min on T4

Changes from v1 (v8):
  - FIXED: ci_low/ci_high NameError (now computed via bootstrap)
  - Saves FULL data: per-point (R, gap) scatter data + R trajectories
  - Saves fourier_logs and logit_logs per run
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor

RUN_NAME = "script4_v2"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("="*80)
log("SCRIPT 4 v2: SPECTRAL SEPARATION — FULL DATA SAVE")
log("="*80)

# ── Config ──
SEEDS = [42, 43, 44, 45, 46]
P = 97; ETA = 1e-3; LAM = 1.0
MAX_STEPS = 50000; FOURIER_EVERY = 500; POST_EXTEND = 5000
PRE_MARGIN = 500; POST_MARGIN = 1000; R_THRESHOLD = 0.03
BOOTSTRAP_N = 2000

# ── TRAIN ──
runs = []
for seed in SEEDS:
    log(f"\n── Seed {seed} ──")
    r = train_run(
        p=P, eta=ETA, lam=LAM, seed=seed,
        max_steps=MAX_STEPS,
        measure_fourier_every=FOURIER_EVERY,
        measure_logit=True,
        post_grok_extend=POST_EXTEND,
    )
    runs.append(r)

# ── BUILD FIXED K* ──
half = P // 2 + 1
post_specs = []
for r in runs:
    Tg = r.get("T_grok", 0)
    fl = r.get("fourier_logs", {})
    for st, spec in zip(fl.get("steps", []), fl.get("spectra", [])):
        if st > Tg + POST_MARGIN:
            post_specs.append(np.array(spec[:half]))

K_star = []
mean_spec = np.zeros(half)
if post_specs:
    mean_spec = np.mean(post_specs, axis=0)
    order = np.argsort(mean_spec)[::-1]
    cum = 0; total = np.sum(mean_spec)
    for idx in order:
        K_star.append(int(idx))
        cum += mean_spec[idx] / total
        if cum >= 0.99: break
    log(f"\nFixed K*: |K*| = {len(K_star)} (99% cumulative energy)")

# ── COMPUTE per-point R + GAP ──
R_all = []; gap_all = []; scatter_meta = []
for ri, r in enumerate(runs):
    Tg = r.get("T_grok", 0)
    fl = r.get("fourier_logs", {})
    val_steps = r.get("logs", {}).get("steps", [])
    val_loss = r.get("logs", {}).get("val_loss", [])
    baseline = np.mean(
        [vl for s, vl in zip(val_steps, val_loss) if s > Tg + 2000][-5:]
    ) if any(s > Tg + 2000 for s in val_steps) else 0.0

    for st, spec in zip(fl.get("steps", []), fl.get("spectra", [])):
        if st >= Tg - PRE_MARGIN: continue
        spec = np.array(spec[:half])
        total = np.sum(spec)
        if total < 1e-15: continue
        inK = np.sum(spec[K_star]) if K_star else 0
        R = (total - inK) / total
        if R < R_THRESHOLD: continue
        vl = next((v for s, v in zip(val_steps, val_loss) if s >= st), None)
        if vl is None: continue
        gap = vl - baseline
        R_all.append(float(R)); gap_all.append(float(gap))
        scatter_meta.append({"seed": r["seed"], "step": st, "R": float(R), "gap": float(gap)})

# ── OLS & RANSAC ──
slope_ols = intercept_ols = r2_ols = 0.0
slope_r = intercept_r = r2_r = inlier_pct = 0.0
ci_low = ci_high = 0.0

if len(R_all) >= 5:
    slope_ols, intercept_ols, r_val, _, _ = linregress(R_all, gap_all)
    r2_ols = r_val ** 2
    log(f"\nOLS fit: slope={slope_ols:.3f}, R²={r2_ols:.3f}")

if len(R_all) >= 10:
    X = np.array(R_all).reshape(-1, 1)
    y = np.array(gap_all)
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    slope_r = ransac.estimator_.coef_[0]
    intercept_r = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    inlier_pct = np.mean(inlier_mask) * 100
    y_pred_in = slope_r * X[inlier_mask, 0] + intercept_r
    ss_res = np.sum((y[inlier_mask] - y_pred_in)**2)
    ss_tot = np.sum((y[inlier_mask] - np.mean(y[inlier_mask]))**2)
    r2_r = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    log(f"RANSAC: slope={slope_r:.3f}, R²(inliers)={r2_r:.3f}, inliers={inlier_pct:.1f}%")

# Bootstrap CI for OLS slope
if len(R_all) >= 10:
    boot_slopes = []
    for _ in range(BOOTSTRAP_N):
        idx = np.random.choice(len(R_all), len(R_all), replace=True)
        bR = [R_all[i] for i in idx]
        bg = [gap_all[i] for i in idx]
        bs, _, _, _, _ = linregress(bR, bg)
        boot_slopes.append(bs)
    ci_low, ci_high = float(np.percentile(boot_slopes, 2.5)), float(np.percentile(boot_slopes, 97.5))
    log(f"Bootstrap 95% CI for OLS slope: [{ci_low:.2f}, {ci_high:.2f}]")

viol_0 = sum(1 for g in gap_all if g < 0) / len(gap_all) if gap_all else 0
viol_fit = sum(1 for R,g in zip(R_all,gap_all) if g < slope_r*R + intercept_r) / len(R_all) if R_all else 0

# ── SAVE FULL DATA ──
analysis = {
    "K_star": K_star, "K_star_size": len(K_star),
    "slope_ols": float(slope_ols), "R2_ols": float(r2_ols),
    "slope_ransac": float(slope_r), "R2_ransac": float(r2_r),
    "inlier_pct": float(inlier_pct),
    "ci_low": float(ci_low), "ci_high": float(ci_high),
    "violation_rate_zero": float(viol_0),
    "violation_rate_fit": float(viol_fit),
    "scatter_data": scatter_meta,  # per-point R, gap for scatter plots
    "mean_post_spectrum": mean_spec.tolist(),
}
save_full_results(runs, FULL_JSON, extra=analysis)
save_summary_json(runs, SUMMARY_JSON, extra={
    "K_star_size": len(K_star), "slope_ols": float(slope_ols), "R2_ols": float(r2_ols),
    "slope_ransac": float(slope_r), "R2_ransac": float(r2_r),
    "ci_low": float(ci_low), "ci_high": float(ci_high),
})

# ── Quick diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for r in runs:
    fl = r.get("fourier_logs", {})
    if fl:
        axes[0,0].plot(fl["steps"], fl["R"], alpha=0.5, lw=1.2, label=f'seed {r["seed"]}')
        if r["T_grok"]: axes[0,0].axvline(r["T_grok"], color='red', lw=1.5, alpha=0.4, ls='--')
axes[0,0].set_ylabel("R(f_θ)"); axes[0,0].set_title("(a) Non-Fourier energy R")
axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

if R_all:
    axes[0,1].scatter(R_all, gap_all, alpha=0.4, s=25, c='royalblue', edgecolor='navy', lw=0.5)
    xl = np.linspace(0, max(R_all), 100)
    axes[0,1].plot(xl, slope_ols*xl + intercept_ols, 'r--', lw=2, label=f'OLS R²={r2_ols:.3f}')
    axes[0,1].legend(fontsize=9)
axes[0,1].set_xlabel("R(f_θ)"); axes[0,1].set_ylabel("Val gap"); axes[0,1].set_title("(b) Gap vs R")

# (c) Irreversibility
post_R_dict = {}
for r in runs:
    fl = r.get("fourier_logs", {})
    if fl and r["T_grok"]:
        for s, rv in zip(fl["steps"], fl["R"]):
            if s >= r["T_grok"]:
                rel = s - r["T_grok"]
                post_R_dict.setdefault(rel, []).append(rv)
if post_R_dict:
    ss = sorted(post_R_dict.keys())
    mm = [np.mean(post_R_dict[s]) for s in ss]
    axes[0,2].plot(ss, mm, 'b-', lw=2); axes[0,2].set_yscale('log')
axes[0,2].set_xlabel("Steps after grokking"); axes[0,2].set_title("(c) Irreversibility")

# (d-f) placeholders
axes[1,0].set_title("(d) Spectral heatmap"); axes[1,1].set_title("(e) Logit bound")
sorted_energy = np.sort(mean_spec)[::-1]
cum_energy = np.cumsum(sorted_energy) / (np.sum(sorted_energy) + 1e-15)
axes[1,2].plot(cum_energy[:50], 'b-', lw=2.5)
axes[1,2].axhline(0.99, color='red', ls='--'); axes[1,2].axvline(len(K_star), color='green', ls='--')
axes[1,2].set_title(f"(f) K* = {len(K_star)}")

plt.suptitle("Script 4 v2: Spectral Separation", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "summary.png"), dpi=150); plt.close()

log("SCRIPT 4 v2 COMPLETE!")
