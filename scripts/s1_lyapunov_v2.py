"""
Script 1 v2: LYAPUNOV ESCAPE VALIDATION
Validates: Sec 3.2-3.3, Theorem 2 (exponential decay)
Runtime: ~3 min on T4

Changes from v1:
  - Saves FULL logs (V_t trajectory per step) via save_full_results()
  - Also saves lightweight summary for backward compat
  - Identical training logic
"""
from shared_v2 import *
setup_drive()

RUN_NAME = "script1_v2"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")
SEEDS = list(range(42, 52))  # 10 seeds

log("="*60)
log("  SCRIPT 1 v2: LYAPUNOV ESCAPE VALIDATION")
log(f"  p=97, η=1e-3, λ=1.0, {len(SEEDS)} seeds")
log("="*60)

results = []
for seed in SEEDS:
    log(f"\n── Seed {seed} ──")
    t0 = time.time()
    r = train_run(p=97, eta=1e-3, lam=1.0, seed=seed, max_steps=50000, log_every=200)
    dt = time.time()-t0
    log(f"  T_mem={r['T_mem']} T_grok={r['T_grok']} V_mem={r['V_at_mem']:.0f} "
        f"V_final={r['V_final']:.0f} ({dt:.0f}s)")
    if r['fit']:
        log(f"  Fit: rate={r['fit']['rate_fit']:.6f} theory={r['fit']['rate_theory']:.6f} "
            f"R²={r['fit']['R2']:.4f}")
    results.append(r)

# ── Aggregate ──
T_groks = [r["T_grok"] for r in results if r["T_grok"]]
T_mems = [r["T_mem"] for r in results if r["T_mem"]]
fits = [r["fit"] for r in results if r["fit"]]

extra = {
    "T_grok_mean": float(np.mean(T_groks)),
    "T_grok_std": float(np.std(T_groks)),
    "T_mem_mean": float(np.mean(T_mems)),
    "mean_rate_fit": float(np.mean([f["rate_fit"] for f in fits])),
    "std_rate_fit": float(np.std([f["rate_fit"] for f in fits])),
    "mean_R2": float(np.mean([f["R2"] for f in fits])),
}

# ── Save FULL data (with logs) + summary ──
save_full_results(results, FULL_JSON, extra=extra)
save_summary_json(results, SUMMARY_JSON, extra={"aggregate": extra})

# ── Quick diagnostic plot (low-res, for console check) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for r in results:
    axes[0,0].plot(r["logs"]["steps"], r["logs"]["V_t"], alpha=0.6, label=f's{r["seed"]}')
    if r["T_grok"]: axes[0,0].axvline(r["T_grok"], color='red', alpha=0.1)
axes[0,0].set_yscale('log'); axes[0,0].set_xlabel("Step"); axes[0,0].set_ylabel("||θ||²")
axes[0,0].set_title("(a) Lyapunov V_t trajectories"); axes[0,0].legend(fontsize=6, ncol=2)

for r in results:
    axes[0,1].plot(r["logs"]["steps"], r["logs"]["val_acc"], alpha=0.5, color='red')
    axes[0,1].plot(r["logs"]["steps"], r["logs"]["train_acc"], alpha=0.3, color='blue')
axes[0,1].set_xlabel("Step"); axes[0,1].set_ylabel("Accuracy")
axes[0,1].set_title("(b) Train (blue) / Val (red)")

r0 = results[0]
r_theory = 1 - 2*1e-3*1.0
if r0["T_mem"] and r0["fit"]:
    mask = [s >= r0["T_mem"] for s in r0["logs"]["steps"]]
    t_plot = np.array([s for s,m in zip(r0["logs"]["steps"],mask) if m])
    v_plot = np.array([v for v,m in zip(r0["logs"]["V_t"],mask) if m])
    axes[1,0].plot(t_plot, v_plot, 'b-', linewidth=1.5, label='Measured V_t')
    t_rel = t_plot - t_plot[0]
    v_th = v_plot[0] * np.exp(np.log(r_theory) * t_rel)
    axes[1,0].plot(t_plot, v_th, 'r--', linewidth=2, label=f'Theory: (1-2ηλ)^t')
    v_fit = r0["fit"]["A"] * np.exp(np.log(r0["fit"]["rate_fit"]) * t_rel) + r0["fit"]["C"]
    axes[1,0].plot(t_plot, v_fit, 'g:', linewidth=2, label=f'Fit: R²={r0["fit"]["R2"]:.3f}')
    axes[1,0].set_yscale('log'); axes[1,0].set_xlabel("Step"); axes[1,0].set_ylabel("||θ||²")
    axes[1,0].set_title("(c) V_t: measured vs theory (seed 42)"); axes[1,0].legend(fontsize=9)

axes[1,1].hist(T_groks, bins=8, color='steelblue', alpha=0.7, edgecolor='black')
axes[1,1].axvline(np.mean(T_groks), color='red', linestyle='--',
                   label=f'Mean={np.mean(T_groks):.0f}±{np.std(T_groks):.0f}')
axes[1,1].set_xlabel("T_grok"); axes[1,1].set_ylabel("Count")
axes[1,1].set_title("(d) Grokking time distribution"); axes[1,1].legend()

plt.suptitle(f"Script 1 v2: Lyapunov Escape — p=97, η=1e-3, λ=1.0, {len(SEEDS)} seeds",
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, "summary.png"), dpi=150); plt.close()

log(f"\nDiagnostic plot: {OUT}/summary.png")
log("SCRIPT 1 v2 COMPLETE!")
