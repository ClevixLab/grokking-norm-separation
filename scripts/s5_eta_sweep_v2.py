"""
Script 5 v2: ETA SWEEP — FULL DATA SAVE
Validates: Sec 4.5 (T_grok ∝ 1/η, joint η×λ universality)
Runtime: ~20 min on T4

Changes from v1 (v8.1):
  - Saves FULL per-seed data (was only aggregated means)
  - Saves logs for representative curves
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress

RUN_NAME = "script5_v2"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("="*80)
log("SCRIPT 5 v2: ETA SWEEP — FULL DATA SAVE")
log("="*80)

# ── Config ──
P = 97
ETAS_A = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
LAM_A = 1.0
SEEDS_A = [42, 43, 44, 45, 46]

ETAS_B = [2e-3, 1e-3, 5e-4]
LAMS_B = [0.5, 1.0, 2.0]
SEEDS_B = [42, 43, 44, 45, 46]

MAX_STEPS_BASE = 100000
LOG_EVERY = 200

# ── TRAIN — Experiment A ──
log("\nEXPERIMENT A: η SWEEP (fixed λ=1.0)")
results_A = []
for eta in ETAS_A:
    max_steps = min(int(MAX_STEPS_BASE * (1e-3 / eta)), 200000)
    for seed in SEEDS_A:
        log(f"  η={eta:.1e}, seed={seed}, max_steps={max_steps}")
        r = train_run(p=P, eta=eta, lam=LAM_A, seed=seed,
                      max_steps=max_steps, log_every=LOG_EVERY)
        results_A.append(r)

# ── TRAIN — Experiment B ──
log("\nEXPERIMENT B: JOINT η×λ GRID")
results_B = []
for eta in ETAS_B:
    for lam in LAMS_B:
        max_steps = min(int(MAX_STEPS_BASE * (1e-3 / eta)), 200000)
        for seed in SEEDS_B:
            log(f"  η={eta:.1e}, λ={lam}, seed={seed}")
            r = train_run(p=P, eta=eta, lam=lam, seed=seed,
                          max_steps=max_steps, log_every=LOG_EVERY)
            results_B.append(r)

# ── ANALYSIS ──
log("\n" + "="*70)
log("ANALYSIS — Section 4.5")
log("="*70)

by_eta = {}
for r in results_A:
    by_eta.setdefault(r["eta"], []).append(r)

# Per-eta stats
eta_stats = []
for eta in sorted(by_eta.keys(), reverse=True):
    grokked = [r for r in by_eta[eta] if r.get("T_grok")]
    if grokked:
        tg = [r["T_grok"] for r in grokked]
        tp = [r["T_grok"] * eta for r in grokked]
        stat = {
            "eta": eta,
            "mean_T_grok": float(np.mean(tg)), "std_T_grok": float(np.std(tg)),
            "mean_T_eta": float(np.mean(tp)), "std_T_eta": float(np.std(tp)),
            "n_grokked": len(grokked), "n_total": len(by_eta[eta]),
        }
        eta_stats.append(stat)
        log(f"  η={eta:.1e}: T_grok={stat['mean_T_grok']:.0f}±{stat['std_T_grok']:.0f}, "
            f"T·η={stat['mean_T_eta']:.2f}±{stat['std_T_eta']:.2f}, "
            f"n={stat['n_grokked']}/{stat['n_total']}")

# Doubling test
doubling_ratios = []
for i in range(len(ETAS_A)-1):
    hi, lo = ETAS_A[i], ETAS_A[i+1]
    tg_hi = np.mean([r["T_grok"] for r in by_eta.get(hi, []) if r.get("T_grok")])
    tg_lo = np.mean([r["T_grok"] for r in by_eta.get(lo, []) if r.get("T_grok")])
    ratio = tg_lo / tg_hi if tg_hi > 0 else float('nan')
    doubling_ratios.append(float(ratio))
    log(f"  η {hi:.1e} → {lo:.1e}: T ratio = {ratio:.2f}×")

# Linear fit
fit_x = [1/r["eta"] for r in results_A if r.get("T_grok")]
fit_y = [r["T_grok"] for r in results_A if r.get("T_grok")]
slope = intercept = R2 = 0.0
if len(fit_x) >= 5:
    slope, intercept, r_val, _, _ = linregress(fit_x, fit_y)
    R2 = r_val ** 2
    log(f"\nLinear fit: slope={slope:.0f}, intercept={intercept:.0f}, R²={R2:.4f}")

# Joint η×λ
products = [r["T_grok"]*r["eta"]*r["lam"] for r in results_B if r.get("T_grok")]
mean_prod = float(np.mean(products)) if products else 0
cv_prod = float(np.std(products)/np.mean(products)) if products else 0
log(f"Joint T·(ηλ): mean={mean_prod:.3f}, CV={cv_prod:.3f}")

# ── SAVE ──
all_results = results_A + results_B
analysis = {
    "experiment": "eta_sweep",
    "eta_stats": eta_stats,
    "doubling_ratios": doubling_ratios,
    "linear_fit": {"slope": float(slope), "intercept": float(intercept), "R2": float(R2)},
    "joint_eta_lambda": {"mean_T_eta_lambda": mean_prod, "cv": cv_prod},
    "config_A": {"etas": ETAS_A, "lam": LAM_A, "seeds": SEEDS_A},
    "config_B": {"etas": ETAS_B, "lams": LAMS_B, "seeds": SEEDS_B},
    "n_results_A": len(results_A),
    "n_results_B": len(results_B),
}
save_full_results(all_results, FULL_JSON, extra=analysis)
save_summary_json(all_results, SUMMARY_JSON, extra=analysis)

# ── Quick diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for r in results_A:
    if r.get("T_grok"):
        axes[0,0].scatter(1/r["eta"], r["T_grok"], alpha=0.5, s=30, c='blue', edgecolor='navy')
if len(fit_x) >= 5:
    xl = np.linspace(0, max(fit_x)*1.1, 100)
    axes[0,0].plot(xl, slope*xl + intercept, 'r--', lw=2, label=f'R²={R2:.3f}')
    axes[0,0].legend()
axes[0,0].set_xlabel("1/η"); axes[0,0].set_ylabel("T_grok"); axes[0,0].set_title("(a) T_grok ∝ 1/η")

for eta in sorted(by_eta.keys()):
    tp = [r["T_grok"]*r["eta"] for r in by_eta[eta] if r.get("T_grok")]
    if tp:
        axes[0,1].errorbar([eta], [np.mean(tp)], yerr=[np.std(tp)], fmt='o', capsize=5, c='blue')
axes[0,1].set_xscale('log'); axes[0,1].set_xlabel("η"); axes[0,1].set_ylabel("T·η")
axes[0,1].set_title("(b) T·η product")

grid = np.full((len(LAMS_B), len(ETAS_B)), np.nan)
for r in results_B:
    if r.get("T_grok") and r["lam"] in LAMS_B and r["eta"] in ETAS_B:
        i = LAMS_B.index(r["lam"]); j = ETAS_B.index(r["eta"])
        if np.isnan(grid[i,j]):
            grid[i,j] = r["T_grok"] * r["eta"] * r["lam"]
im = axes[0,2].imshow(grid, aspect='auto', origin='lower', cmap='viridis')
axes[0,2].set_xticks(range(len(ETAS_B))); axes[0,2].set_xticklabels([f'{e:.0e}' for e in ETAS_B])
axes[0,2].set_yticks(range(len(LAMS_B))); axes[0,2].set_yticklabels([f'{l:.1f}' for l in LAMS_B])
plt.colorbar(im, ax=axes[0,2], label='T·(ηλ)')
axes[0,2].set_title("(c) T·(ηλ) universality")

# (d) Doubling test
axes[1,0].bar(range(len(doubling_ratios)), doubling_ratios, color='steelblue')
axes[1,0].axhline(2.0, color='red', ls='--', label='Expected ×2')
axes[1,0].set_title("(d) Doubling test"); axes[1,0].legend()

# (e) T_escape: measured vs theory
t_esc_m = np.array([r.get("T_escape") for r in results_A if r.get("T_escape")], dtype=float)
t_esc_t = np.array([r.get("T_escape_theory") for r in results_A if r.get("T_escape")], dtype=float)
if len(t_esc_m) > 0:
    axes[1,1].scatter(t_esc_t, t_esc_m, alpha=0.6, s=40, c='blue')
    lim = max(np.max(t_esc_m), np.max(t_esc_t)) * 1.1
    axes[1,1].plot([0, lim], [0, lim], 'k--', label='y = x')
    axes[1,1].legend()
axes[1,1].set_xlabel("T_escape (theory)"); axes[1,1].set_ylabel("T_escape (measured)")
axes[1,1].set_title("(e) Formula validation")

plt.suptitle("Script 5 v2: Learning Rate Sensitivity", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "eta_sweep_v2.png"), dpi=150); plt.close()

log("SCRIPT 5 v2 COMPLETE!")
