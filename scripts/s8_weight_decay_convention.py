"""
Script 8: WEIGHT DECAY CONVENTION VERIFICATION
================================================
Clarifies the factor-of-2 discrepancy between paper and PyTorch.

Paper dynamics:   theta_{t+1} = theta_t - eta(grad_L + 2*lambda*theta_t)
  -> On manifold: theta_{t+1} = (1 - 2*eta*lambda) * theta_t

PyTorch SGD(weight_decay=w):
  -> grad -> grad + w*theta, then theta = theta - eta*(grad + w*theta)
  -> On manifold: theta_{t+1} = (1 - eta*w) * theta_t
  -> With w=2*lambda: matches paper exactly (1 - 2*eta*lambda)
  -> With w=lambda:   gives (1 - eta*lambda), factor-of-2 off

PyTorch AdamW(weight_decay=w):
  -> Decoupled WD: theta = (1 - eta*w)*theta - eta*adam_step
  -> With w=lambda: WD contributes (1 - eta*lambda)

This script runs 3 configs to verify which matches theory best.

Colab setup:
  - Place shared_v2.py in the same directory (or /content/)
  - Mounts Google Drive automatically via setup_drive()
  - Saves to /content/drive/MyDrive/grokking_results/script8_wd_convention/

Runtime: ~12 min on T4 (3 configs x 5 seeds x 50k steps)
"""
from shared_v2 import *
setup_drive()

from torch.optim import SGD as TorchSGD

RUN_NAME = "script8_wd_convention"
OUT = get_out_dir(RUN_NAME)
FULL_JSON = get_json_path(RUN_NAME, "full_results.json")
SUMMARY_JSON = get_json_path(RUN_NAME, "summary.json")

log("=" * 70)
log("  SCRIPT 8: WEIGHT DECAY CONVENTION VERIFICATION")
log(f"  3 configs: SGD(wd=2lam), SGD(wd=lam), AdamW(wd=lam)")
log(f"  p=97, eta=1e-3, lam=1.0, 5 seeds each")
log(f"  Device: {DEVICE}")
log(f"  Output: {os.path.dirname(FULL_JSON)}")
log("=" * 70)

# ── Config ──
P = 97; ETA = 1e-3; LAM = 1.0
SEEDS = [42, 43, 44, 45, 46]
MAX_STEPS = 50000; LOG_EVERY = 200

CONFIGS = [
    {
        "name": "sgd_2lam",
        "label": "SGD(wd=2*lam)",
        "description": "Matches paper: theta - eta(grad + 2*lam*theta)",
        "theory_rate": 1 - 2 * ETA * LAM,  # 0.998
    },
    {
        "name": "sgd_1lam",
        "label": "SGD(wd=lam)",
        "description": "Half the paper WD: theta - eta(grad + lam*theta)",
        "theory_rate": 1 - ETA * LAM,  # 0.999
    },
    {
        "name": "adamw_1lam",
        "label": "AdamW(wd=lam)",
        "description": "Decoupled WD: (1-eta*lam)*theta - eta*adam_step",
        "theory_rate": 1 - ETA * LAM,  # 0.999 from WD term alone
    },
]


def train_with_config(p, eta, lam, seed, config, max_steps=50000,
                      log_every=200, batch_size=512, grok_thresh=0.99):
    """Train with a specific optimizer config."""
    set_seed(seed)
    (xt, yt), (xv, yv) = build_data(p, 0.5, seed)
    model = Model(p).to(DEVICE)

    cfg_name = config["name"]
    if cfg_name == "sgd_2lam":
        opt = TorchSGD(model.parameters(), lr=eta, weight_decay=2 * lam, momentum=0.0)
    elif cfg_name == "sgd_1lam":
        opt = TorchSGD(model.parameters(), lr=eta, weight_decay=lam, momentum=0.0)
    elif cfg_name == "adamw_1lam":
        opt = AdamW(model.parameters(), lr=eta, weight_decay=lam)
    else:
        raise ValueError(cfg_name)

    n_train = xt.size(0)
    logs = {"steps": [], "V_t": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []}
    T_mem = T_grok = None; V_at_mem = None

    for step in range(max_steps):
        if T_grok is not None and step > T_grok + 3000:
            break

        model.train(); opt.zero_grad()
        idx = torch.randint(0, n_train, (min(batch_size, n_train),), device=DEVICE)
        loss = F.cross_entropy(model(xt[idx]), yt[idx])
        loss.backward(); opt.step()

        if step % log_every == 0:
            Vt = V(model)
            tl, ta = eval_model(model, xt, yt)
            vl, va = eval_model(model, xv, yv)
            logs["steps"].append(step); logs["V_t"].append(Vt)
            logs["train_loss"].append(tl); logs["train_acc"].append(ta)
            logs["val_loss"].append(vl); logs["val_acc"].append(va)
            if T_mem is None and ta >= grok_thresh:
                T_mem = step; V_at_mem = Vt
            if T_grok is None and va >= grok_thresh:
                T_grok = step

    V_final = V(model)
    fit = fit_exp_decay(logs["steps"], logs["V_t"], T_mem, eta, lam)

    return {
        "seed": seed, "p": p, "eta": eta, "lam": lam,
        "config": cfg_name,
        "config_label": config["label"],
        "theory_rate": config["theory_rate"],
        "T_mem": T_mem, "T_grok": T_grok,
        "V_at_mem": V_at_mem, "V_final": V_final,
        "fit": fit, "logs": logs,
    }


# ── TRAIN ──
results = []
for cfg in CONFIGS:
    log(f"\n{'─' * 50}")
    log(f"  Config: {cfg['label']} — {cfg['description']}")
    log(f"  Theory rate: {cfg['theory_rate']:.6f}")
    log(f"{'─' * 50}")
    for seed in SEEDS:
        t0 = time.time()
        r = train_with_config(P, ETA, LAM, seed, cfg, MAX_STEPS, LOG_EVERY)
        dt = time.time() - t0
        fit_str = f"rate={r['fit']['rate_fit']:.6f} R2={r['fit']['R2']:.4f}" if r['fit'] else "no fit"
        grok_str = f"T_grok={r['T_grok']}" if r['T_grok'] else "NO GROK"
        log(f"  [{cfg['name']}] s={seed}: {grok_str} | {fit_str} ({dt:.0f}s)")
        results.append(r)

# ── ANALYSIS ──
log("\n" + "=" * 70)
log("CONVENTION COMPARISON")
log("=" * 70)

by_cfg = {}
for r in results:
    by_cfg.setdefault(r["config"], []).append(r)

comparison = {}
for cfg in CONFIGS:
    name = cfg["name"]
    runs = by_cfg.get(name, [])
    fits = [r["fit"] for r in runs if r.get("fit")]
    grokked = [r for r in runs if r.get("T_grok")]
    rates = [f["rate_fit"] for f in fits]
    R2s = [f["R2"] for f in fits]

    stat = {
        "label": cfg["label"],
        "theory_rate": float(cfg["theory_rate"]),
        "mean_rate_fit": float(np.mean(rates)) if rates else None,
        "std_rate_fit": float(np.std(rates)) if rates else None,
        "mean_R2": float(np.mean(R2s)) if R2s else None,
        "gap_from_theory": float(abs(np.mean(rates) - cfg["theory_rate"])) if rates else None,
        "n_grokked": len(grokked), "n_total": len(runs),
        "mean_T_grok": float(np.mean([r["T_grok"] for r in grokked])) if grokked else None,
        "std_T_grok": float(np.std([r["T_grok"] for r in grokked])) if grokked else None,
    }
    comparison[name] = stat

    log(f"\n  {cfg['label']}:")
    log(f"    Theory rate:  {stat['theory_rate']:.6f}")
    if stat['mean_rate_fit']:
        log(f"    Fitted rate:  {stat['mean_rate_fit']:.6f} +/- {stat['std_rate_fit']:.6f}")
        log(f"    |Gap|:        {stat['gap_from_theory']:.6f}")
        log(f"    R2:           {stat['mean_R2']:.4f}")
    log(f"    Grokked:      {stat['n_grokked']}/{stat['n_total']}")
    if stat['mean_T_grok']:
        log(f"    T_grok:       {stat['mean_T_grok']:.0f} +/- {stat['std_T_grok']:.0f}")

# Key finding
log("\n  KEY FINDING:")
sgd2 = comparison.get("sgd_2lam", {})
sgd1 = comparison.get("sgd_1lam", {})
adam = comparison.get("adamw_1lam", {})
if sgd2.get("gap_from_theory") is not None and sgd1.get("gap_from_theory") is not None:
    if sgd2["gap_from_theory"] < sgd1["gap_from_theory"]:
        log("    SGD(wd=2*lam) matches paper theory (1-2*eta*lam) BEST")
        log("    -> Paper's factor of 2 is CORRECT for SGD L2 regularisation")
    else:
        log("    SGD(wd=lam) matches better — paper convention needs review")
if adam.get("gap_from_theory") is not None and sgd2.get("gap_from_theory") is not None:
    log(f"    AdamW gap: {adam['gap_from_theory']:.6f} vs SGD(2lam) gap: {sgd2['gap_from_theory']:.6f}")

# ── SAVE to Google Drive ──
analysis = {
    "comparison": comparison,
    "configs": [c["name"] for c in CONFIGS],
    "config_details": CONFIGS,
}
save_full_results(results, FULL_JSON, extra=analysis)
save_summary_json(results, SUMMARY_JSON, extra=analysis)

# ── Diagnostic plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors_cfg = {"sgd_2lam": "red", "sgd_1lam": "green", "adamw_1lam": "blue"}
labels_cfg = {"sgd_2lam": "SGD(wd=2lam)", "sgd_1lam": "SGD(wd=lam)", "adamw_1lam": "AdamW(wd=lam)"}

# (a) V_t trajectories (seed 42)
for cfg in CONFIGS:
    name = cfg["name"]
    for r in by_cfg.get(name, []):
        if r["seed"] == 42:
            axes[0, 0].plot(r["logs"]["steps"], r["logs"]["V_t"],
                            color=colors_cfg[name], lw=1.5, label=labels_cfg[name])
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("||theta||^2")
axes[0, 0].set_title("(a) Norm decay (seed 42)"); axes[0, 0].legend(fontsize=8)

# (b) Fitted rates per seed with theory lines
for cfg in CONFIGS:
    name = cfg["name"]
    rates = [(r["seed"], r["fit"]["rate_fit"]) for r in by_cfg.get(name, []) if r.get("fit")]
    if rates:
        seeds, rs = zip(*rates)
        dx = {"sgd_2lam": -0.3, "sgd_1lam": 0, "adamw_1lam": 0.3}[name]
        axes[0, 1].scatter([s + dx for s in seeds], rs, c=colors_cfg[name], s=50,
                           label=labels_cfg[name], zorder=3)
    axes[0, 1].axhline(cfg["theory_rate"], color=colors_cfg[name], ls='--', lw=1, alpha=0.5)
axes[0, 1].set_xlabel("Seed"); axes[0, 1].set_ylabel("Fitted rate")
axes[0, 1].set_title("(b) Rates (dots) vs theory (dashed)"); axes[0, 1].legend(fontsize=7)

# (c) Gap from theory bar chart
gap_vals = [comparison.get(c["name"], {}).get("gap_from_theory", 0) or 0 for c in CONFIGS]
bars = axes[0, 2].bar(range(len(CONFIGS)), gap_vals,
                       color=[colors_cfg[c["name"]] for c in CONFIGS],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0, 2].set_xticks(range(len(CONFIGS)))
axes[0, 2].set_xticklabels([labels_cfg[c["name"]] for c in CONFIGS], fontsize=7, rotation=10)
axes[0, 2].set_ylabel("|fitted - theory|")
axes[0, 2].set_title("(c) Distance from theory")

# (d) Val accuracy (seed 42)
for cfg in CONFIGS:
    name = cfg["name"]
    for r in by_cfg.get(name, []):
        if r["seed"] == 42:
            axes[1, 0].plot(r["logs"]["steps"], r["logs"]["val_acc"],
                            color=colors_cfg[name], lw=1.5, label=labels_cfg[name])
axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("Val acc")
axes[1, 0].set_title("(d) Grokking curves (seed 42)"); axes[1, 0].legend(fontsize=8)

# (e) T_grok boxplot
box_data = []
box_labels_list = []
box_colors_list = []
for cfg in CONFIGS:
    name = cfg["name"]
    tg = [r["T_grok"] for r in by_cfg.get(name, []) if r.get("T_grok")]
    if tg:
        box_data.append(tg)
        box_labels_list.append(labels_cfg[name])
        box_colors_list.append(colors_cfg[name])
if box_data:
    bp = axes[1, 1].boxplot(box_data, labels=box_labels_list, patch_artist=True, widths=0.5)
    for patch, col in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(col); patch.set_alpha(0.35)
axes[1, 1].set_ylabel("T_grok"); axes[1, 1].set_title("(e) T_grok comparison")
axes[1, 1].tick_params(axis='x', labelsize=7)

# (f) Summary
axes[1, 2].axis('off')
txt = "CONVENTION VERIFICATION\n\n"
for cfg in CONFIGS:
    name = cfg["name"]
    s = comparison.get(name, {})
    txt += f"{labels_cfg[name]}:\n"
    txt += f"  Theory: {s.get('theory_rate', 'N/A'):.6f}\n"
    if s.get('mean_rate_fit'):
        txt += f"  Fitted: {s['mean_rate_fit']:.6f}\n"
        txt += f"  |Gap|:  {s.get('gap_from_theory', 0):.6f}\n"
    txt += "\n"
axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
axes[1, 2].set_title("(f) Summary")

plt.suptitle(f"Script 8: Weight Decay Convention — p={P}, eta={ETA}, lam={LAM}",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "wd_convention.png"), dpi=150)
plt.close()

log(f"\nDiagnostic plot: {OUT}/wd_convention.png")
log(f"Full results: {FULL_JSON}")
log("SCRIPT 8 COMPLETE!")
