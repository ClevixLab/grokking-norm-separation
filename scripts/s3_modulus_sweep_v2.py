"""
Script 3 v2: MODULUS SWEEP — FULL DATA SAVE
Validates: Sec 4.4 (T_grok vs p, norm-ratio formula)
Runtime: ~25 min on T4

Changes from v1:
  - Saves FULL logs via save_full_results() (was stripping logs)
  - Identical training logic
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress

RUN_NAME = "script3_v2"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("="*60)
log("  SCRIPT 3 v2: MODULUS SWEEP — FULL DATA SAVE")
log("="*60)

# ── Config ──
PRIMES = [53, 67, 89, 97, 101, 113, 127]
SEEDS = list(range(42, 49))  # 7 seeds
ETA = 1e-3
LAM = 1.0

# ── TRAIN ──
results = []
for p in PRIMES:
    for seed in SEEDS:
        log(f"  p={p}, seed={seed}")
        t0 = time.time()
        r = train_run(p=p, eta=ETA, lam=LAM, seed=seed, max_steps=50000, log_every=200)
        dt = time.time()-t0
        log(f"    T_grok={r['T_grok']} V_mem={r.get('V_at_mem','N/A')} ({dt:.0f}s)")
        results.append(r)

# ── ANALYSIS ──
log("\n" + "="*70)
log("MODULUS SWEEP ANALYSIS")
log("="*70)

by_p = {}
for r in results:
    by_p.setdefault(r["p"], []).append(r)

# Per-p aggregates
p_stats = []
for p in PRIMES:
    runs = by_p.get(p, [])
    grokked = [r for r in runs if r.get("T_grok")]
    if grokked:
        tg = [r["T_grok"] for r in grokked]
        tm = [r["T_mem"] for r in grokked if r["T_mem"]]
        vm = [r["V_at_mem"] for r in grokked if r["V_at_mem"]]
        vf = [r["V_final"] for r in grokked]
        te = [r["T_escape_theory"] for r in grokked if r["T_escape_theory"]]
        delay = [r["T_grok"]-r["T_mem"] for r in grokked if r["T_mem"]]
        rates = [r["fit"]["rate_fit"] for r in grokked if r.get("fit")]
        
        stat = {
            "p": p,
            "T_grok_mean": float(np.mean(tg)), "T_grok_std": float(np.std(tg)),
            "T_mem_mean": float(np.mean(tm)) if tm else None,
            "delay_mean": float(np.mean(delay)) if delay else None,
            "V_mem_mean": float(np.mean(vm)) if vm else None,
            "V_mem_std": float(np.std(vm)) if vm else None,
            "V_final_mean": float(np.mean(vf)),
            "T_esc_theory_mean": float(np.mean(te)) if te else None,
            "rate_fit_mean": float(np.mean(rates)) if rates else None,
            "n_grokked": len(grokked), "n_total": len(runs),
        }
        p_stats.append(stat)
        log(f"  p={p:3d} Tmem={stat['T_mem_mean']:6.0f} Tgrok={stat['T_grok_mean']:6.0f} "
            f"Delay={stat['delay_mean']:6.0f} Vmem={stat['V_mem_mean']:7.0f} "
            f"Vfin={stat['V_final_mean']:6.0f} Tesc_th={stat['T_esc_theory_mean']:7.0f}")

# Correlation: measured delay vs theory escape time
delays_all = [r["T_grok"]-r["T_mem"] for r in results if r.get("T_grok") and r.get("T_mem")]
tesc_all = [r["T_escape_theory"] for r in results if r.get("T_grok") and r.get("T_mem") and r.get("T_escape_theory")]
corr_r = corr_slope = corr_intercept = 0.0
if len(delays_all) >= 5 and len(tesc_all) >= 5:
    corr_slope, corr_intercept, corr_r_val, _, _ = linregress(tesc_all[:len(delays_all)], delays_all)
    corr_r = corr_r_val
    log(f"\nDelay vs T_esc_theory: slope={corr_slope:.2f} r={corr_r:.3f}")

# ── SAVE ──
analysis = {
    "primes": PRIMES,
    "p_stats": p_stats,
    "delay_vs_theory": {
        "slope": float(corr_slope),
        "intercept": float(corr_intercept),
        "pearson_r": float(corr_r),
    }
}
save_full_results(results, FULL_JSON, extra=analysis)
save_summary_json(results, SUMMARY_JSON, extra=analysis)

# ── Quick diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) T_grok vs log(p)
for r in results:
    if r.get("T_grok"):
        axes[0,0].scatter(np.log(r["p"]), r["T_grok"], alpha=0.4, s=25, c='steelblue')
for s in p_stats:
    axes[0,0].errorbar(np.log(s["p"]), s["T_grok_mean"], yerr=s["T_grok_std"],
                        fmt='o', capsize=4, c='red', ms=8)
axes[0,0].set_xlabel("log(p)"); axes[0,0].set_ylabel("T_grok")
axes[0,0].set_title("(a) T_grok vs log(p)")

# (b) Delay vs log(p)
for s in p_stats:
    if s["delay_mean"]:
        axes[0,1].errorbar(np.log(s["p"]), s["delay_mean"],
                            fmt='o', capsize=4, c='red', ms=8)
axes[0,1].set_xlabel("log(p)"); axes[0,1].set_ylabel("T_grok − T_mem")
axes[0,1].set_title("(b) Delay vs log(p)")

# (c) V_mem vs p
for r in results:
    if r.get("V_at_mem"):
        axes[0,2].scatter(r["p"], r["V_at_mem"], alpha=0.4, s=25, c='green')
axes[0,2].set_xlabel("p"); axes[0,2].set_ylabel("V_mem")
axes[0,2].set_title("(c) V_mem vs p")

# (d) K vs p
for s in p_stats:
    ks = [r["K_final"] for r in by_p[s["p"]] if r.get("K_final")]
    if ks:
        axes[1,0].errorbar(s["p"], np.mean(ks), yerr=np.std(ks), fmt='o', capsize=4, c='blue', ms=8)
axes[1,0].set_xlabel("p"); axes[1,0].set_ylabel("K")
axes[1,0].set_title("(d) Fourier support K vs p")

# (e) T_mem vs p
for s in p_stats:
    if s["T_mem_mean"]:
        axes[1,1].plot(s["p"], s["T_mem_mean"], 'go', ms=8)
axes[1,1].set_xlabel("p"); axes[1,1].set_ylabel("T_mem")
axes[1,1].set_title("(e) Memorisation time vs p")

# (f) Grokking curves for p=53, 97, 127
for pp, color in [(53,'blue'), (97,'green'), (127,'red')]:
    for r in results:
        if r["p"]==pp and r["seed"]==42 and r.get("T_grok"):
            axes[1,2].plot(r["logs"]["steps"], r["logs"]["val_acc"], color=color,
                           lw=2, label=f'p={pp} (T={r["T_grok"]})')
axes[1,2].set_xlabel("Step"); axes[1,2].set_ylabel("Val acc")
axes[1,2].set_title("(f) Grokking curves"); axes[1,2].legend(fontsize=9)

plt.suptitle("Script 3 v2: Modulus Sweep", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "summary.png"), dpi=150); plt.close()

log(f"\nDiagnostic plot: {OUT}/summary.png")
log("SCRIPT 3 v2 COMPLETE!")
