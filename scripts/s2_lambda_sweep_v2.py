"""
Script 2 v2: λ SWEEP — FIXED + FULL DATA SAVE
Validates: Sec 4.2 (T_grok ∝ 1/λ, three regimes), Sec 3.7 (main theorem)
Runtime: ~18-28 min on T4

Changes from v1 (v9):
  - FIXED: removed redundant drive.mount before shared import
  - FIXED: uses shared_v2 helpers (setup_drive, get_out_dir, etc.)
  - NEW: saves full_results.json with ALL logs for figure plotting
  - NEW: bootstrap CI for slope
  - Identical training logic
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor

RUN_NAME = "script2_v2"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")
CAPTION_PATH = get_json_path(RUN_NAME, "figure_caption.txt")

log("="*80)
log("SCRIPT 2 v2: λ SWEEP — FIXED + FULL DATA SAVE")
log(f"Output folder: {os.path.dirname(FULL_JSON)}")
log("="*80)

# ── Config ──
LAMBDAS = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
REGIME_II = [0.1, 0.3, 0.5, 1.0]
SEEDS = list(range(42, 52))   # 10 seeds
ETA = 1e-3
P = 97

# ── TRAIN ──
results = []
for lam in LAMBDAS:
    max_s = 15000 if lam <= 0.01 else 80000
    for seed in SEEDS:
        log(f"  λ={lam:.3f}, seed={seed}, max_steps={max_s}")
        try:
            r = train_run(
                p=P, eta=ETA, lam=lam, seed=seed,
                max_steps=max_s, log_every=500
            )
            log(f"    → T_grok={r.get('T_grok')}, T·λ={r.get('T_grok_x_lam')}")
            results.append(r)
        except Exception as e:
            log(f"    → ERROR: {e}")

# ── ANALYSIS ──
log("\n" + "="*70)
log("RESULTS TABLE & THEORY COMPARISON")
log("="*70)

by_lam = {}
for r in results:
    by_lam.setdefault(r.get("lam"), []).append(r)

for lam in LAMBDAS:
    runs = by_lam.get(lam, [])
    grokked = [r for r in runs if r.get("T_grok") is not None]
    if grokked:
        tg = [r["T_grok"] for r in grokked]
        tp = [r.get("T_grok_x_lam") for r in grokked if r.get("T_grok_x_lam") is not None]
        dm = [r.get("delta_min") for r in grokked if r.get("delta_min") is not None]
        mean_dm = np.mean(dm) if dm else np.nan
        log(f"  λ={lam:>6.3f} | T_grok={np.mean(tg):>6.0f}±{np.std(tg):>4.0f} | "
              f"T·λ={np.mean(tp):>6.0f} | Δ_min={mean_dm:>6.3f} | n_grok={len(grokked)}/{len(runs)}")
    else:
        log(f"  λ={lam:>6.3f} | no grokking | — | — | 0/{len(runs)}")

# Linear fit (Regime II)
fit_x = [1/r["lam"] for r in results if r.get("lam") in REGIME_II and r.get("T_grok")]
fit_y = [r["T_grok"] for r in results if r.get("lam") in REGIME_II and r.get("T_grok")]
slope = intercept = R2 = 0.0
th_slope = 0.0
if len(fit_x) >= 5:
    slope, intercept, r_val, _, _ = linregress(fit_x, fit_y)
    R2 = r_val**2
    th_slope = math.log(P) / (2*ETA)
    log(f"\nLINEAR FIT (Regime II): T = {slope:.0f}/λ + {intercept:.0f}, R²={R2:.4f}")
    log(f"Theory slope: {th_slope:.0f} | Ratio: {slope/th_slope:.2f}")

# Bootstrap CI for slope
ci_low = ci_high = 0.0
BOOTSTRAP_N = 2000
if len(fit_x) >= 5:
    boot_slopes = []
    for _ in range(BOOTSTRAP_N):
        idx = np.random.choice(len(fit_x), len(fit_x), replace=True)
        bx = [fit_x[i] for i in idx]
        by = [fit_y[i] for i in idx]
        bs, _, _, _, _ = linregress(bx, by)
        boot_slopes.append(bs)
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
    log(f"Bootstrap 95% CI for slope: [{ci_low:.0f}, {ci_high:.0f}]")

# T·λ product
products = [r.get("T_grok_x_lam") for r in results
            if r.get("lam") in REGIME_II and r.get("T_grok_x_lam") is not None]
mean_product = np.mean(products) if products else np.nan
cv_product = np.std(products)/np.mean(products) if products else np.nan
if products:
    log(f"T·λ PRODUCT: mean={mean_product:.0f}, CV={cv_product:.3f}")

# ── SAVE FULL DATA ──
analysis = {
    "lambdas": LAMBDAS,
    "regime_II": REGIME_II,
    "linear_fit": {"slope": float(slope), "intercept": float(intercept), "R2": float(R2),
                   "theory_slope": float(th_slope), "ci_low": float(ci_low), "ci_high": float(ci_high)},
    "T_lambda_product": {"mean": float(mean_product), "cv": float(cv_product)},
}
save_full_results(results, FULL_JSON, extra=analysis)
save_summary_json(results, SUMMARY_JSON, extra=analysis)

# ── Quick diagnostic plot ──
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
colors = {'I': 'gray', 'II': 'royalblue', 'III': 'crimson'}

for r in results:
    regime = 'I' if r["lam"]<=0.01 else ('III' if r["lam"]>=2.0 else 'II')
    if r.get("T_grok"):
        axes[0,0].scatter(1/r["lam"], r["T_grok"], c=colors[regime], alpha=0.45, s=28,
                          edgecolor='black', linewidth=0.3)
    else:
        axes[0,0].scatter(1/r["lam"], 80000, c=colors[regime], marker='x', alpha=0.4, s=35)
if len(fit_x) >= 5:
    xl = np.linspace(0, max(fit_x)*1.1, 100)
    axes[0,0].plot(xl, slope*xl + intercept, 'k--', lw=2, label=f'OLS fit R²={R2:.3f}')
    axes[0,0].legend(fontsize=10)
axes[0,0].set_xlabel("1/λ"); axes[0,0].set_ylabel("T_grok")
axes[0,0].set_title("(a) T_grok ∝ 1/λ"); axes[0,0].grid(True, alpha=0.3)

for r in results:
    if r.get("T_grok_x_lam"):
        regime = 'I' if r["lam"]<=0.01 else ('III' if r["lam"]>=2.0 else 'II')
        axes[0,1].scatter(r["lam"], r["T_grok_x_lam"], c=colors[regime], alpha=0.45, s=28,
                          edgecolor='black', linewidth=0.3)
if products:
    axes[0,1].axhline(mean_product, color='red', linestyle='--', lw=2,
                       label=f'Mean = {mean_product:.0f}')
    axes[0,1].legend(fontsize=10)
axes[0,1].set_xlabel("λ"); axes[0,1].set_ylabel("T·λ")
axes[0,1].set_xscale('log'); axes[0,1].set_title("(b) T·λ product stability")

for regime_lam, color, title in [(0.01,'gray','Regime I'), (0.5,'royalblue','Regime II'),
                                  (5.0,'crimson','Regime III')]:
    for r in results:
        if r["lam"] == regime_lam and r["seed"] == 42:
            axes[1,0].plot(r["logs"]["steps"], r["logs"]["val_acc"], color=color, lw=2,
                           label=title, alpha=0.85)
axes[1,0].set_xlabel("Step"); axes[1,0].set_ylabel("Val acc")
axes[1,0].set_title("(c) Three regimes"); axes[1,0].legend(fontsize=10)

lam_delta, dm_means, dm_stds = [], [], []
for lam in REGIME_II:
    dms = [r.get("delta_min") for r in by_lam.get(lam, []) if r.get("delta_min") is not None]
    if dms:
        lam_delta.append(lam); dm_means.append(np.mean(dms)); dm_stds.append(np.std(dms))
if lam_delta:
    axes[1,1].errorbar(lam_delta, dm_means, yerr=dm_stds, fmt='o', capsize=5, color='royalblue')
axes[1,1].set_xlabel("λ"); axes[1,1].set_ylabel("Δ_min")
axes[1,1].set_title("(d) Validation gap vs λ")

plt.suptitle(f"Script 2 v2: λ Sweep — p={P}, η={ETA}", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "lambda_sweep_v2.png"), dpi=150); plt.close()

log(f"\nDiagnostic plot saved to: {OUT}")
log("SCRIPT 2 v2 COMPLETE!")
