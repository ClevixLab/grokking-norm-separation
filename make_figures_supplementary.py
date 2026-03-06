"""
Supplementary Figure Script -- Publication figures for S6, S7, S8
=================================================================
Reads full_results.json from S6-S8 and produces 3 supplementary figures.
Complements the main make_figures.py (which handles S1-S5).

Usage:
    python make_figures_supplementary.py [--base_dir /path/to/grokking_results]
"""
import os, json, sys, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = "/content/drive/MyDrive/grokking_results"
if "--base_dir" in sys.argv:
    BASE_DIR = sys.argv[sys.argv.index("--base_dir")+1]

if os.path.isdir(os.path.join(REPO_ROOT, "data")):
    DATA_DIR = os.path.join(REPO_ROOT, "data")
    FIGS_OUT = os.path.join(REPO_ROOT, "figures")
else:
    DATA_DIR = BASE_DIR
    FIGS_OUT = os.path.join(BASE_DIR, "paper_figures")
os.makedirs(FIGS_OUT, exist_ok=True)

_REPO_FILE_MAP = {
    "script6_sgd_vs_adamw": "full_results6.json",
    "script7_hires_fourier": "full_results7.json",
    "script8_wd_convention": "full_results8.json",
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'lines.linewidth': 1.5, 'axes.linewidth': 0.8,
})

C = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
    'gray': '#999999', 'black': '#000000',
}

def load_json(script_name, filename="full_results.json"):
    if os.path.isdir(os.path.join(REPO_ROOT, "data")):
        mapped = _REPO_FILE_MAP.get(script_name, filename)
        path = os.path.join(DATA_DIR, mapped)
    else:
        path = os.path.join(DATA_DIR, script_name, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- skipping")
        return None
    with open(path) as f:
        return json.load(f)

def savefig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGS_OUT, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGS_OUT}/{name}.pdf + .png")


# ================================================================
#  FIGURE S1: SGD vs AdamW Ablation (from S6)
# ================================================================
def make_fig_s1():
    data = load_json("script6_sgd_vs_adamw")
    if not data:
        return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    comp = analysis.get("comparison", {})
    theory_rate = analysis.get("theory_rate_paper", 0.998)

    by_opt = {"adamw": [], "sgd": []}
    for r in runs:
        by_opt.setdefault(r.get("optimizer", "adamw"), []).append(r)

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)

    # (a) V_t trajectories -- both optimizers, seed 42
    ax = fig.add_subplot(gs[0, 0])
    for r in by_opt.get("adamw", []):
        if r["seed"] == 42:
            ax.plot(r["logs"]["steps"], r["logs"]["V_t"], color=C['blue'], lw=1.5, label='AdamW')
    for r in by_opt.get("sgd", []):
        if r["seed"] == 42:
            ax.plot(r["logs"]["steps"], r["logs"]["V_t"], color=C['red'], lw=1.5, label='SGD')
    ax.set_yscale('log')
    ax.set_xlabel("Training step"); ax.set_ylabel(r"$\|\theta_t\|^2$")
    ax.set_title("(a) Norm decay (seed 42)")
    ax.legend()

    # (b) V_t trajectories -- all seeds, both optimizers
    ax = fig.add_subplot(gs[0, 1])
    for r in by_opt.get("adamw", []):
        ax.plot(r["logs"]["steps"], r["logs"]["V_t"], color=C['blue'], alpha=0.35, lw=0.8)
    for r in by_opt.get("sgd", []):
        ax.plot(r["logs"]["steps"], r["logs"]["V_t"], color=C['red'], alpha=0.35, lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel("Training step"); ax.set_ylabel(r"$\|\theta_t\|^2$")
    ax.set_title("(b) All seeds: AdamW (blue) / SGD (red)")

    # (c) Fitted contraction rates per seed
    ax = fig.add_subplot(gs[0, 2])
    for opt_name, color, marker, dx in [("adamw", C['blue'], 'o', -0.15), ("sgd", C['red'], 's', 0.15)]:
        rates = [(r["seed"], r["fit"]["rate_fit"]) for r in by_opt.get(opt_name, []) if r.get("fit")]
        if rates:
            seeds, rs = zip(*rates)
            ax.scatter([s + dx for s in seeds], rs, c=color, s=50, marker=marker,
                       label=opt_name.upper(), zorder=3, edgecolor='white', lw=0.5)
    ax.axhline(theory_rate, color=C['green'], ls='--', lw=1.5,
               label=f'Theory $1-2\\eta\\lambda={theory_rate:.4f}$')
    # Add mean lines
    for opt_name, color in [("adamw", C['blue']), ("sgd", C['red'])]:
        s = comp.get(opt_name, {})
        if s.get("mean_rate"):
            ax.axhline(s["mean_rate"], color=color, ls=':', lw=1.0, alpha=0.7)
    ax.set_xlabel("Seed"); ax.set_ylabel("Fitted contraction rate")
    ax.set_title("(c) Rate comparison")
    ax.legend(loc='lower left', fontsize=7)

    # (d) Grokking curves (val acc), seed 42
    ax = fig.add_subplot(gs[1, 0])
    for opt_name, color in [("adamw", C['blue']), ("sgd", C['red'])]:
        for r in by_opt.get(opt_name, []):
            if r["seed"] == 42:
                ax.plot(r["logs"]["steps"], r["logs"]["val_acc"], color=color, lw=1.5,
                        label=opt_name.upper())
    ax.set_xlabel("Step"); ax.set_ylabel("Val accuracy")
    ax.set_title("(d) Grokking curves (seed 42)")
    ax.legend()

    # (e) T_grok boxplot
    ax = fig.add_subplot(gs[1, 1])
    tg_adamw = [r["T_grok"] for r in by_opt.get("adamw", []) if r.get("T_grok")]
    tg_sgd = [r["T_grok"] for r in by_opt.get("sgd", []) if r.get("T_grok")]
    box_data = []
    box_labels = []
    if tg_adamw:
        box_data.append(tg_adamw); box_labels.append("AdamW")
    if tg_sgd:
        box_data.append(tg_sgd); box_labels.append("SGD")
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
        box_colors = [C['blue'], C['red']][:len(box_data)]
        for patch, col in zip(bp['boxes'], box_colors):
            patch.set_facecolor(col); patch.set_alpha(0.35)
    ax.set_ylabel(r"$T_{\mathrm{grok}}$")
    ax.set_title("(e) Grokking time")

    # (f) Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    lines = ["Optimizer Comparison Summary\n"]
    for opt_name in ["sgd", "adamw"]:
        s = comp.get(opt_name, {})
        lines.append(f"{opt_name.upper()}:")
        if s.get("mean_rate"):
            lines.append(f"  Rate: {s['mean_rate']:.6f} +/- {s.get('std_rate',0):.6f}")
            lines.append(f"  Gamma: {s.get('contraction_gamma',0):.6f}")
        if s.get("mean_T_grok"):
            lines.append(f"  T_grok: {s['mean_T_grok']:.0f} +/- {s.get('std_T_grok',0):.0f}")
        lines.append(f"  Grokked: {s.get('n_grokked',0)}/{s.get('n_total',0)}")
        lines.append("")
    lines.append(f"Theory (1-2eta*lam): {theory_rate:.6f}")
    if comp.get("adamw", {}).get("contraction_gamma") and comp.get("sgd", {}).get("contraction_gamma"):
        ratio = comp["adamw"]["contraction_gamma"] / comp["sgd"]["contraction_gamma"]
        lines.append(f"Gamma_AdamW / Gamma_SGD: {ratio:.3f}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.set_title("(f) Summary")

    fig.suptitle("Figure S1: SGD vs AdamW Ablation ($p=97$, $\\eta=10^{-3}$, $\\lambda=1.0$)",
                 fontsize=11, fontweight='bold', y=1.01)
    savefig(fig, "figS1_sgd_vs_adamw")
    plt.close()


# ================================================================
#  FIGURE S2: High-Resolution Fourier (from S7)
# ================================================================
def make_fig_s2():
    data = load_json("script7_hires_fourier")
    if not data:
        return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    fit_hi = analysis.get("fit_hires", {})
    fit_lo = analysis.get("fit_lowres", {})
    K_star = analysis.get("K_star_hires", [])
    K_star_lo = analysis.get("K_star_lowres", [])
    mean_spec = np.array(analysis.get("mean_post_spectrum_hires", []))
    scatter_hi = analysis.get("scatter_hires", [])
    scatter_lo = analysis.get("scatter_lowres", [])
    rdiff = analysis.get("R_diff_stats", {})

    # Separate hi-res and lo-res runs
    runs_hi = [r for r in runs if not r.get("_lowres")]
    runs_lo = [r for r in runs if r.get("_lowres")]

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)

    # (a) R trajectories -- high-res
    ax = fig.add_subplot(gs[0, 0])
    for r in runs_hi:
        fl = r.get("fourier_logs", {})
        if fl and fl.get("steps"):
            ax.plot(fl["steps"], fl["R"], alpha=0.6, lw=1.0, label=f's{r["seed"]}')
            if r.get("T_grok"):
                ax.axvline(r["T_grok"], color=C['red'], lw=0.8, alpha=0.3, ls='--')
    ax.set_ylabel(r"$\mathcal{R}(f_\theta)$"); ax.set_xlabel("Step")
    ax.set_title("(a) High-res $\\mathcal{R}$ trajectories")
    ax.legend(fontsize=6, ncol=2)

    # (b) Gap vs R -- high-res
    ax = fig.add_subplot(gs[0, 1])
    if scatter_hi:
        Rs = [s["R"] for s in scatter_hi]
        gaps = [s["gap"] for s in scatter_hi]
        ax.scatter(Rs, gaps, alpha=0.35, s=18, c=C['blue'], edgecolor='none')
        if len(Rs) >= 5:
            sl, ic, rv, _, _ = linregress(Rs, gaps)
            xl = np.linspace(0, max(Rs), 100)
            ax.plot(xl, sl * xl + ic, '--', color=C['red'], lw=1.5,
                    label=f'OLS $R^2={rv**2:.3f}$')
            ax.legend()
    ax.set_xlabel(r"$\mathcal{R}(f_\theta)$"); ax.set_ylabel("Validation gap")
    ax.set_title(f"(b) Gap vs $\\mathcal{{R}}$ (high-res, $n$={fit_hi.get('n',0)})")

    # (c) Gap vs R -- low-res (comparison)
    ax = fig.add_subplot(gs[0, 2])
    if scatter_lo:
        Rs = [s["R"] for s in scatter_lo]
        gaps = [s["gap"] for s in scatter_lo]
        ax.scatter(Rs, gaps, alpha=0.35, s=18, c=C['orange'], edgecolor='none')
        if len(Rs) >= 5:
            sl, ic, rv, _, _ = linregress(Rs, gaps)
            xl = np.linspace(0, max(Rs), 100)
            ax.plot(xl, sl * xl + ic, '--', color=C['red'], lw=1.5,
                    label=f'OLS $R^2={rv**2:.3f}$')
            ax.legend()
    ax.set_xlabel(r"$\mathcal{R}(f_\theta)$"); ax.set_ylabel("Validation gap")
    ax.set_title(f"(c) Gap vs $\\mathcal{{R}}$ (low-res, $n$={fit_lo.get('n',0)})")

    # (d) R high-res vs R low-res scatter
    ax = fig.add_subplot(gs[1, 0])
    hi_R_matched, lo_R_matched = [], []
    for rh, rl in zip(runs_hi, runs_lo):
        fl_hi = rh.get("fourier_logs", {})
        fl_lo = rl.get("fourier_logs", {})
        if fl_hi and fl_lo:
            hd = dict(zip(fl_hi.get("steps", []), fl_hi.get("R", [])))
            ld = dict(zip(fl_lo.get("steps", []), fl_lo.get("R", [])))
            for s in hd:
                if s in ld:
                    hi_R_matched.append(hd[s]); lo_R_matched.append(ld[s])
    if hi_R_matched:
        ax.scatter(hi_R_matched, lo_R_matched, alpha=0.35, s=15, c=C['green'], edgecolor='none')
        lim = max(max(hi_R_matched), max(lo_R_matched)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='$y=x$')
        ax.legend()
    ax.set_xlabel("$\\mathcal{R}$ (high-res)"); ax.set_ylabel("$\\mathcal{R}$ (low-res)")
    ax.set_title("(d) Resolution comparison")

    # (e) Cumulative energy -> K*
    ax = fig.add_subplot(gs[1, 1])
    if len(mean_spec) > 0:
        sorted_e = np.sort(mean_spec)[::-1]
        cum = np.cumsum(sorted_e) / (np.sum(sorted_e) + 1e-15)
        ax.plot(cum[:50], color=C['blue'], lw=2.0)
        ax.axhline(0.99, color=C['red'], ls='--', lw=1.0, label='99\\% threshold')
        ax.axvline(len(K_star), color=C['green'], ls='--', lw=1.0,
                   label=f'$K^*={len(K_star)}$ (high-res)')
        if K_star_lo:
            ax.axvline(len(K_star_lo), color=C['orange'], ls=':', lw=1.0,
                       label=f'$K^*={len(K_star_lo)}$ (low-res)')
        ax.legend(fontsize=7)
    ax.set_xlabel("Number of frequencies"); ax.set_ylabel("Cumulative energy")
    ax.set_title("(e) $K^*$ detection")

    # (f) Summary comparison
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    lines = [
        "Resolution Comparison\n",
        f"HIGH-RES (n_b=p, n_c=p):",
        f"  K* = {len(K_star)}",
        f"  OLS slope = {fit_hi.get('slope_ols', 0):.2f}",
        f"  OLS R2 = {fit_hi.get('R2_ols', 0):.4f}",
        f"  RANSAC R2 = {fit_hi.get('R2_ransac', 0):.4f}",
        f"  CI = [{fit_hi.get('ci_low',0):.1f}, {fit_hi.get('ci_high',0):.1f}]",
        "",
        f"LOW-RES (n_b=3, n_c=5):",
        f"  K* = {len(K_star_lo)}",
        f"  OLS slope = {fit_lo.get('slope_ols', 0):.2f}",
        f"  OLS R2 = {fit_lo.get('R2_ols', 0):.4f}",
        f"  RANSAC R2 = {fit_lo.get('R2_ransac', 0):.4f}",
        "",
        f"Mean |Delta R| = {rdiff.get('mean', 'N/A')}",
        f"Max  |Delta R| = {rdiff.get('max', 'N/A')}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.set_title("(f) Summary")

    fig.suptitle("Figure S2: High-Resolution Fourier Analysis ($p=97$, $K^*=%d$)" % len(K_star),
                 fontsize=11, fontweight='bold', y=1.01)
    savefig(fig, "figS2_hires_fourier")
    plt.close()


# ================================================================
#  FIGURE S3: Weight Decay Convention (from S8)
# ================================================================
def make_fig_s3():
    data = load_json("script8_wd_convention")
    if not data:
        return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    comp = analysis.get("comparison", {})

    CONFIGS = analysis.get("configs", ["sgd_2lam", "sgd_1lam", "adamw_1lam"])
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(r.get("config", "unknown"), []).append(r)

    colors_cfg = {"sgd_2lam": C['red'], "sgd_1lam": C['green'], "adamw_1lam": C['blue']}
    labels_cfg = {
        "sgd_2lam": r"SGD($w{=}2\lambda$)",
        "sgd_1lam": r"SGD($w{=}\lambda$)",
        "adamw_1lam": r"AdamW($w{=}\lambda$)",
    }
    markers_cfg = {"sgd_2lam": "s", "sgd_1lam": "^", "adamw_1lam": "o"}

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)

    # (a) V_t trajectories (seed 42)
    ax = fig.add_subplot(gs[0, 0])
    for cfg in CONFIGS:
        for r in by_cfg.get(cfg, []):
            if r["seed"] == 42:
                ax.plot(r["logs"]["steps"], r["logs"]["V_t"],
                        color=colors_cfg[cfg], lw=1.5, label=labels_cfg[cfg])
    ax.set_yscale('log')
    ax.set_xlabel("Step"); ax.set_ylabel(r"$\|\theta_t\|^2$")
    ax.set_title("(a) Norm decay (seed 42)")
    ax.legend(fontsize=7)

    # (b) Fitted rates per seed, with theory lines
    ax = fig.add_subplot(gs[0, 1])
    for cfg in CONFIGS:
        rates = [(r["seed"], r["fit"]["rate_fit"]) for r in by_cfg.get(cfg, []) if r.get("fit")]
        theory = comp.get(cfg, {}).get("theory_rate", 0)
        if rates:
            seeds, rs = zip(*rates)
            dx = {"sgd_2lam": -0.3, "sgd_1lam": 0, "adamw_1lam": 0.3}.get(cfg, 0)
            ax.scatter([s + dx for s in seeds], rs, c=colors_cfg[cfg], s=45,
                       marker=markers_cfg[cfg], label=labels_cfg[cfg], zorder=3,
                       edgecolor='white', lw=0.4)
        if theory > 0:
            ax.axhline(theory, color=colors_cfg[cfg], ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel("Seed"); ax.set_ylabel("Fitted rate")
    ax.set_title("(b) Rates (dots) vs theory (dashed)")
    ax.legend(fontsize=6, loc='lower left')

    # (c) Gap from theory (bar chart)
    ax = fig.add_subplot(gs[0, 2])
    gaps = []
    bar_colors = []
    bar_labels = []
    for cfg in CONFIGS:
        g = comp.get(cfg, {}).get("gap_from_theory", 0)
        gaps.append(g if g else 0)
        bar_colors.append(colors_cfg[cfg])
        bar_labels.append(labels_cfg[cfg])
    bars = ax.bar(range(len(CONFIGS)), gaps, color=bar_colors, alpha=0.7,
                  edgecolor=C['black'], linewidth=0.5)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=15)
    ax.set_ylabel("|fitted $-$ theory|")
    ax.set_title("(c) Distance from theory")

    # (d) Val accuracy curves (seed 42)
    ax = fig.add_subplot(gs[1, 0])
    for cfg in CONFIGS:
        for r in by_cfg.get(cfg, []):
            if r["seed"] == 42:
                ax.plot(r["logs"]["steps"], r["logs"]["val_acc"],
                        color=colors_cfg[cfg], lw=1.5, label=labels_cfg[cfg])
    ax.set_xlabel("Step"); ax.set_ylabel("Val accuracy")
    ax.set_title("(d) Grokking curves (seed 42)")
    ax.legend(fontsize=7)

    # (e) T_grok boxplot
    ax = fig.add_subplot(gs[1, 1])
    box_data = []
    box_labels_list = []
    box_colors_list = []
    for cfg in CONFIGS:
        tg = [r["T_grok"] for r in by_cfg.get(cfg, []) if r.get("T_grok")]
        if tg:
            box_data.append(tg)
            box_labels_list.append(labels_cfg[cfg])
            box_colors_list.append(colors_cfg[cfg])
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels_list, patch_artist=True, widths=0.5)
        for patch, col in zip(bp['boxes'], box_colors_list):
            patch.set_facecolor(col); patch.set_alpha(0.35)
    ax.set_ylabel(r"$T_{\mathrm{grok}}$")
    ax.set_title("(e) Grokking time comparison")
    ax.tick_params(axis='x', labelsize=6)

    # (f) Summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    lines = ["Weight Decay Convention\n"]
    for cfg in CONFIGS:
        s = comp.get(cfg, {})
        lines.append(f"{labels_cfg.get(cfg, cfg)}:")
        lines.append(f"  Theory:  {s.get('theory_rate', 'N/A')}")
        if s.get("mean_rate_fit"):
            lines.append(f"  Fitted:  {s['mean_rate_fit']:.6f}")
            lines.append(f"  |Gap|:   {s.get('gap_from_theory',0):.6f}")
        if s.get("mean_T_grok"):
            lines.append(f"  T_grok:  {s['mean_T_grok']:.0f}")
        lines.append(f"  Grokked: {s.get('n_grokked',0)}/{s.get('n_total',0)}")
        lines.append("")

    # Highlight winner
    sgd2_gap = comp.get("sgd_2lam", {}).get("gap_from_theory")
    sgd1_gap = comp.get("sgd_1lam", {}).get("gap_from_theory")
    if sgd2_gap is not None and sgd1_gap is not None:
        if sgd2_gap < sgd1_gap:
            lines.append("=> SGD(w=2lam) matches paper")
            lines.append("   convention BETTER")
        else:
            lines.append("=> SGD(w=lam) matches better")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.set_title("(f) Summary")

    fig.suptitle("Figure S3: Weight Decay Convention ($p=97$, $\\eta=10^{-3}$, $\\lambda=1.0$)",
                 fontsize=11, fontweight='bold', y=1.01)
    savefig(fig, "figS3_wd_convention")
    plt.close()


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SUPPLEMENTARY FIGURE SCRIPT -- S6, S7, S8")
    print(f"  Reading from: {BASE_DIR}")
    print(f"  Writing to:   {FIGS_OUT}")
    print("=" * 60)

    make_fig_s1()
    make_fig_s2()
    make_fig_s3()

    print(f"\nAll supplementary figures saved to: {FIGS_OUT}")
    print("Done!")
