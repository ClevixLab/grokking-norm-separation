"""
Script 6: SGD vs AdamW ABLATION
================================
Closes the gap between theory (SGD + weight decay) and experiments (AdamW).

Validates:
  - Theorem 3.2 contraction rate matches SGD exactly
  - AdamW effective rate differs by a measurable, consistent factor
  - Grokking delay formula holds for BOTH optimizers

Colab setup:
  - Place shared_v2.py in the same directory (or /content/)
  - Mounts Google Drive automatically via setup_drive()
  - Saves results to /content/drive/MyDrive/grokking_results/script6_sgd_vs_adamw/

Runtime: ~8 min on T4 (2 optimizers x 5 seeds x 50k steps)
"""
from shared_v2 import *
setup_drive()

from scipy.stats import linregress
from torch.optim import SGD as TorchSGD

RUN_NAME = "script6_sgd_vs_adamw"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("=" * 70)
log("  SCRIPT 6: SGD vs AdamW ABLATION")
log(f"  p=97, eta=1e-3, lam=1.0, 5 seeds x 2 optimizers")
log(f"  Device: {DEVICE}")
log(f"  Output: {os.path.dirname(FULL_JSON)}")
log("=" * 70)

# ── Config ──
P = 97
ETA = 1e-3
LAM = 1.0
SEEDS = [42, 43, 44, 45, 46]
MAX_STEPS = 50000
LOG_EVERY = 200


# ── Custom training loop supporting both optimizers ──
def train_run_optimizer(p, eta, lam, seed, optimizer_type="adamw",
                        max_steps=50000, log_every=200, batch_size=512,
                        grok_thresh=0.99):
    """
    Training loop with selectable optimizer.
    optimizer_type: 'adamw' or 'sgd'

    SGD convention:
      Paper: theta_{t+1} = theta_t - eta(grad_L + 2*lambda*theta_t)
      PyTorch SGD(weight_decay=w): grad -> grad + w*theta
        so update is theta - eta*(grad + w*theta) = theta*(1-eta*w) - eta*grad
      To match paper: set w = 2*lambda
        -> theta*(1 - 2*eta*lambda) on manifold (grad=0)

    AdamW convention:
      PyTorch AdamW(weight_decay=w): decoupled, theta -> (1-eta*w)*theta
      With w=lambda: theta*(1 - eta*lambda) from WD alone
    """
    set_seed(seed)
    (xt, yt), (xv, yv) = build_data(p, 0.5, seed)
    model = Model(p).to(DEVICE)

    if optimizer_type == "adamw":
        opt = AdamW(model.parameters(), lr=eta, weight_decay=lam)
    elif optimizer_type == "sgd":
        # w=2*lam to match paper convention theta - eta(grad + 2*lam*theta)
        opt = TorchSGD(model.parameters(), lr=eta, weight_decay=2 * lam, momentum=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    n_train = xt.size(0)
    logs = {"steps": [], "V_t": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []}

    T_mem = T_grok = None
    V_at_mem = None

    for step in range(max_steps):
        # Early stop: no need to keep training long after grokking
        if T_grok is not None and step > T_grok + 3000:
            break

        model.train()
        opt.zero_grad()
        idx = torch.randint(0, n_train, (min(batch_size, n_train),), device=DEVICE)
        loss = F.cross_entropy(model(xt[idx]), yt[idx])
        loss.backward()
        opt.step()

        if step % log_every == 0:
            Vt = V(model)
            tl, ta = eval_model(model, xt, yt)
            vl, va = eval_model(model, xv, yv)
            logs["steps"].append(step)
            logs["V_t"].append(Vt)
            logs["train_loss"].append(tl)
            logs["train_acc"].append(ta)
            logs["val_loss"].append(vl)
            logs["val_acc"].append(va)

            if T_mem is None and ta >= grok_thresh:
                T_mem = step
                V_at_mem = Vt
            if T_grok is None and va >= grok_thresh:
                T_grok = step

    V_final = V(model)
    fit = fit_exp_decay(logs["steps"], logs["V_t"], T_mem, eta, lam)
    T_esc_th = T_escape_theory(V_at_mem, V_final, eta, lam)

    return {
        "seed": seed, "p": p, "eta": eta, "lam": lam,
        "optimizer": optimizer_type,
        "T_mem": T_mem, "T_grok": T_grok,
        "V_at_mem": V_at_mem, "V_final": V_final,
        "fit": fit,
        "T_escape_theory": T_esc_th,
        "logs": logs,
    }


# ── TRAIN ──
results = []
for opt_type in ["adamw", "sgd"]:
    log(f"\n{'─' * 50}")
    log(f"  Optimizer: {opt_type.upper()}")
    log(f"{'─' * 50}")
    for seed in SEEDS:
        t0 = time.time()
        r = train_run_optimizer(
            p=P, eta=ETA, lam=LAM, seed=seed,
            optimizer_type=opt_type, max_steps=MAX_STEPS, log_every=LOG_EVERY
        )
        dt = time.time() - t0
        grok_str = f"T_grok={r['T_grok']}" if r['T_grok'] else "NO GROK"
        fit_str = f"rate={r['fit']['rate_fit']:.6f} R2={r['fit']['R2']:.4f}" if r['fit'] else "no fit"
        log(f"  [{opt_type}] seed={seed}: T_mem={r['T_mem']} {grok_str} | {fit_str} ({dt:.0f}s)")
        results.append(r)

# ── ANALYSIS ──
log("\n" + "=" * 70)
log("COMPARISON: SGD vs AdamW")
log("=" * 70)

by_opt = {"adamw": [], "sgd": []}
for r in results:
    by_opt[r["optimizer"]].append(r)

comparison = {}
for opt_type in ["adamw", "sgd"]:
    runs = by_opt[opt_type]
    grokked = [r for r in runs if r.get("T_grok")]
    fits = [r["fit"] for r in runs if r.get("fit")]
    rates = [f["rate_fit"] for f in fits]
    R2s = [f["R2"] for f in fits]
    T_groks = [r["T_grok"] for r in grokked]
    delays = [r["T_grok"] - r["T_mem"] for r in grokked if r["T_mem"]]

    stat = {
        "n_grokked": len(grokked), "n_total": len(runs),
        "mean_rate": float(np.mean(rates)) if rates else None,
        "std_rate": float(np.std(rates)) if rates else None,
        "mean_R2": float(np.mean(R2s)) if R2s else None,
        "mean_T_grok": float(np.mean(T_groks)) if T_groks else None,
        "std_T_grok": float(np.std(T_groks)) if T_groks else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
        "contraction_gamma": float(1 - np.mean(rates)) if rates else None,
    }
    comparison[opt_type] = stat

    log(f"\n  {opt_type.upper()}:")
    log(f"    Grokked: {stat['n_grokked']}/{stat['n_total']}")
    if stat['mean_rate']:
        log(f"    Contraction rate: {stat['mean_rate']:.6f} +/- {stat['std_rate']:.6f}")
        log(f"    Contraction gamma: {stat['contraction_gamma']:.6f}")
        log(f"    Mean R2: {stat['mean_R2']:.4f}")
    if stat['mean_T_grok']:
        log(f"    T_grok: {stat['mean_T_grok']:.0f} +/- {stat['std_T_grok']:.0f}")
        log(f"    Delay:  {stat['mean_delay']:.0f}")

# Theory comparison
r_theory_paper = 1 - 2 * ETA * LAM  # 0.998
log(f"\n  THEORY: 1-2*eta*lam = {r_theory_paper:.6f}")
log(f"  SGD measured:   {comparison['sgd'].get('mean_rate', 'N/A')}")
log(f"  AdamW measured: {comparison['adamw'].get('mean_rate', 'N/A')}")

if comparison['sgd'].get('contraction_gamma') and comparison['adamw'].get('contraction_gamma'):
    ratio = comparison['adamw']['contraction_gamma'] / comparison['sgd']['contraction_gamma']
    log(f"  gamma_AdamW / gamma_SGD = {ratio:.3f}")
    comparison["gamma_ratio"] = float(ratio)

# ── SAVE to Google Drive ──
analysis = {
    "config": {"p": P, "eta": ETA, "lam": LAM, "seeds": SEEDS},
    "comparison": comparison,
    "theory_rate_paper": float(r_theory_paper),
}
save_full_results(results, FULL_JSON, extra=analysis)
save_summary_json(results, SUMMARY_JSON, extra=analysis)

# ── Diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) V_t trajectories
for r in by_opt["adamw"]:
    axes[0, 0].plot(r["logs"]["steps"], r["logs"]["V_t"], 'b-', alpha=0.4, lw=1)
for r in by_opt["sgd"]:
    axes[0, 0].plot(r["logs"]["steps"], r["logs"]["V_t"], 'r-', alpha=0.4, lw=1)
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("||theta||^2")
axes[0, 0].set_title("(a) V_t: AdamW (blue) vs SGD (red)")

# (b) Fitted rates per seed
adamw_rates = [(r["seed"], r["fit"]["rate_fit"]) for r in by_opt["adamw"] if r.get("fit")]
sgd_rates = [(r["seed"], r["fit"]["rate_fit"]) for r in by_opt["sgd"] if r.get("fit")]
if adamw_rates:
    s, v = zip(*adamw_rates)
    axes[0, 1].scatter(s, v, c='blue', s=60, label='AdamW', zorder=3)
if sgd_rates:
    s, v = zip(*sgd_rates)
    axes[0, 1].scatter(s, v, c='red', s=60, marker='s', label='SGD', zorder=3)
axes[0, 1].axhline(r_theory_paper, color='green', ls='--', lw=2,
                     label=f'Theory (1-2eta*lam)={r_theory_paper:.4f}')
axes[0, 1].set_xlabel("Seed"); axes[0, 1].set_ylabel("Fitted rate")
axes[0, 1].set_title("(b) Contraction rates"); axes[0, 1].legend(fontsize=8)

# (c) Val accuracy (seed 42)
for r in by_opt["adamw"]:
    if r["seed"] == 42:
        axes[0, 2].plot(r["logs"]["steps"], r["logs"]["val_acc"], 'b-', lw=2, label='AdamW')
for r in by_opt["sgd"]:
    if r["seed"] == 42:
        axes[0, 2].plot(r["logs"]["steps"], r["logs"]["val_acc"], 'r-', lw=2, label='SGD')
axes[0, 2].set_xlabel("Step"); axes[0, 2].set_ylabel("Val acc")
axes[0, 2].set_title("(c) Grokking curves (seed 42)"); axes[0, 2].legend()

# (d) T_grok boxplot
adamw_tg = [r["T_grok"] for r in by_opt["adamw"] if r.get("T_grok")]
sgd_tg = [r["T_grok"] for r in by_opt["sgd"] if r.get("T_grok")]
box_data = []
box_labels = []
if adamw_tg:
    box_data.append(adamw_tg); box_labels.append("AdamW")
if sgd_tg:
    box_data.append(sgd_tg); box_labels.append("SGD")
if box_data:
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
    colors_box = ['lightblue', 'lightsalmon']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_box[i])
axes[1, 0].set_ylabel("T_grok"); axes[1, 0].set_title("(d) T_grok distribution")

# (e) Escape: measured vs theory
for opt_type, color, marker in [("adamw", "blue", "o"), ("sgd", "red", "s")]:
    t_m = [r["T_grok"] - r["T_mem"] for r in by_opt[opt_type]
           if r.get("T_grok") and r.get("T_mem")]
    t_t = [r["T_escape_theory"] for r in by_opt[opt_type]
           if r.get("T_grok") and r.get("T_mem") and r.get("T_escape_theory")]
    n = min(len(t_m), len(t_t))
    if n > 0:
        axes[1, 1].scatter(t_t[:n], t_m[:n], c=color, marker=marker, s=50,
                           alpha=0.7, label=opt_type.upper())
lim_vals = ([r.get("T_escape_theory", 0) for r in results if r.get("T_escape_theory")] +
            [r["T_grok"] - r["T_mem"] for r in results if r.get("T_grok") and r.get("T_mem")])
if lim_vals:
    lim = max(v for v in lim_vals if v) * 1.1
    axes[1, 1].plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
axes[1, 1].set_xlabel("T_esc (theory)"); axes[1, 1].set_ylabel("Delay (measured)")
axes[1, 1].set_title("(e) Formula validation"); axes[1, 1].legend()

# (f) Summary text
axes[1, 2].axis('off')
txt = f"Theory: 1-2*eta*lam = {r_theory_paper:.6f}\n\n"
for opt_type in ["sgd", "adamw"]:
    s = comparison.get(opt_type, {})
    txt += f"{opt_type.upper()}:\n"
    if s.get('mean_rate'):
        txt += f"  Rate: {s['mean_rate']:.6f} +/- {s.get('std_rate',0):.6f}\n"
        txt += f"  Gap:  {abs(s['mean_rate'] - r_theory_paper):.6f}\n"
    if s.get('mean_T_grok'):
        txt += f"  T_grok: {s['mean_T_grok']:.0f}\n"
    txt += "\n"
axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
axes[1, 2].set_title("(f) Summary")

plt.suptitle(f"Script 6: SGD vs AdamW — p={P}, eta={ETA}, lam={LAM}",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "sgd_vs_adamw.png"), dpi=150)
plt.close()

log(f"\nDiagnostic plot: {OUT}/sgd_vs_adamw.png")
log(f"Full results: {FULL_JSON}")
log("SCRIPT 6 COMPLETE!")
