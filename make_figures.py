"""
Master Figure Script — Publication-Quality Figures for Grokking Paper
=====================================================================
Reads full_results.json from S1–S5 and produces 5 unified figures.

Usage:
    python make_figures.py [--base_dir /path/to/grokking_results]

Output: figs/fig1_lyapunov.pdf ... figs/fig5_spectral.pdf

Dependencies: numpy, matplotlib, scipy, sklearn (same as training scripts)
"""
import os, json, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from scipy.optimize import curve_fit

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
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

# ── Publication style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Color palette (colorblind-friendly, Okabe-Ito based)
C = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'gray':    '#999999',
    'black':   '#000000',
}

_REPO_FILE_MAP = {
    "script1_v2": "full_results1.json",
    "script2_v2": "full_results2.json",
    "script3_v2": "full_results3.json",
    "script4_v2": "full_results4.json",
    "script5_v2": "full_results5.json",
    "script6_sgd_vs_adamw": "full_results6.json",
    "script7_hires_fourier": "full_results7.json",
    "script8_wd_convention": "full_results8.json",
}

def load_json(script_name, filename="full_results.json"):
    if os.path.isdir(os.path.join(REPO_ROOT, "data")):
        mapped = _REPO_FILE_MAP.get(script_name, filename)
        path = os.path.join(DATA_DIR, mapped)
    else:
        path = os.path.join(DATA_DIR, script_name, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping")
        return None
    with open(path) as f:
        return json.load(f)

def savefig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGS_OUT, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGS_OUT}/{name}.pdf + .png")

# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: Lyapunov Escape (S1)
# ═══════════════════════════════════════════════════════════════
def make_fig1():
    data = load_json("script1_v2")
    if not data: return
    runs = data["runs"]
    
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.32)
    
    # (a) V_t trajectories (all seeds)
    ax = fig.add_subplot(gs[0, 0])
    for r in runs:
        steps = r["logs"]["steps"]; Vt = r["logs"]["V_t"]
        ax.plot(steps, Vt, alpha=0.55, lw=1.0)
        if r["T_grok"]:
            ax.axvline(r["T_grok"], color=C['red'], alpha=0.12, lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel("Training step"); ax.set_ylabel(r"$\|\theta_t\|^2$")
    ax.set_title("(a) Norm trajectories (10 seeds)")
    
    # (b) Measured vs fitted (seed 42)
    ax = fig.add_subplot(gs[0, 1])
    r0 = runs[0]
    if r0["T_mem"] and r0["fit"]:
        mask = [s >= r0["T_mem"] for s in r0["logs"]["steps"]]
        t_plot = np.array([s for s,m in zip(r0["logs"]["steps"],mask) if m])
        v_plot = np.array([v for v,m in zip(r0["logs"]["V_t"],mask) if m])
        t_rel = t_plot - t_plot[0]
        
        ax.plot(t_plot, v_plot, color=C['blue'], lw=1.5, label='Measured', zorder=3)
        
        # Theory: (1-2ηλ)^t
        r_th = r0["fit"]["rate_theory"]
        v_th = v_plot[0] * np.exp(np.log(r_th) * t_rel)
        ax.plot(t_plot, v_th, '--', color=C['red'], lw=1.5, label=r'Theory $(1{-}2\eta\lambda)^t$')
        
        # Fit: A·r^t + C
        rf = r0["fit"]["rate_fit"]
        v_fit = r0["fit"]["A"] * np.exp(np.log(rf) * t_rel) + r0["fit"]["C"]
        ax.plot(t_plot, v_fit, ':', color=C['green'], lw=2.0,
                label=f'Fit: $R^2={r0["fit"]["R2"]:.3f}$')
        
        ax.set_yscale('log')
        ax.set_xlabel("Training step"); ax.set_ylabel(r"$\|\theta_t\|^2$")
        ax.set_title("(b) Measured vs theory (seed 42)")
        ax.legend(loc='upper right')
    
    # (c) Fitted rate vs theory
    ax = fig.add_subplot(gs[1, 0])
    rates = [r["fit"]["rate_fit"] for r in runs if r.get("fit")]
    seeds = [r["seed"] for r in runs if r.get("fit")]
    r_theory = runs[0]["fit"]["rate_theory"] if runs[0].get("fit") else 0.998
    ax.scatter(seeds, rates, color=C['blue'], s=40, zorder=3, edgecolor='white', lw=0.5)
    ax.axhline(r_theory, color=C['red'], ls='--', lw=1.2, label=f'Theory: $1-2\\eta\\lambda={r_theory:.4f}$')
    ax.axhline(np.mean(rates), color=C['green'], ls=':', lw=1.2,
               label=f'Mean fit: ${np.mean(rates):.5f}$')
    ax.set_xlabel("Seed"); ax.set_ylabel("Fitted contraction rate")
    ax.set_title("(c) Contraction rate per seed")
    ax.legend(loc='lower left')
    
    # (d) T_grok distribution
    ax = fig.add_subplot(gs[1, 1])
    T_groks = [r["T_grok"] for r in runs if r["T_grok"]]
    ax.hist(T_groks, bins=8, color=C['cyan'], alpha=0.7, edgecolor=C['black'], lw=0.5)
    ax.axvline(np.mean(T_groks), color=C['red'], ls='--', lw=1.5,
               label=f'Mean$={np.mean(T_groks):.0f}\\pm{np.std(T_groks):.0f}$')
    ax.set_xlabel(r"$T_{\mathrm{grok}}$"); ax.set_ylabel("Count")
    ax.set_title("(d) Grokking time distribution")
    ax.legend()
    
    fig.suptitle("Figure 1: Lyapunov Escape Validation ($p=97$, $\\eta=10^{-3}$, $\\lambda=1.0$)",
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, "fig1_lyapunov")
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Weight Decay Sweep (S2)
# ═══════════════════════════════════════════════════════════════
def make_fig2():
    data = load_json("script2_v2")
    if not data: return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    
    REGIME_II = analysis.get("regime_II", [0.1, 0.3, 0.5, 1.0])
    
    # (a) T_grok vs 1/λ
    for r in runs:
        regime = 'gray' if r["lam"]<=0.01 else (C['red'] if r["lam"]>=2.0 else C['blue'])
        if r.get("T_grok"):
            axes[0].scatter(1/r["lam"], r["T_grok"], c=regime, alpha=0.4, s=20,
                           edgecolor='k', linewidth=0.3, zorder=2)
    fit = analysis.get("linear_fit", {})
    if fit.get("R2", 0) > 0:
        xl = np.linspace(0, 12, 100)
        axes[0].plot(xl, fit["slope"]*xl + fit["intercept"], '--', color=C['red'], lw=1.5,
                     label=f'OLS $R^2={fit["R2"]:.3f}$')
        axes[0].legend()
    axes[0].set_xlabel("$1/\\lambda$"); axes[0].set_ylabel("$T_{\\mathrm{grok}}$")
    axes[0].set_title("(a) $T_{\\mathrm{grok}} \\propto 1/\\lambda$")
    
    # (b) T·λ product
    by_lam = {}
    for r in runs:
        if r.get("T_grok_x_lam"):
            by_lam.setdefault(r["lam"], []).append(r["T_grok_x_lam"])
    for lam in sorted(by_lam.keys()):
        vals = by_lam[lam]
        color = C['gray'] if lam<=0.01 else (C['red'] if lam>=2.0 else C['blue'])
        axes[1].errorbar([lam], [np.mean(vals)], yerr=[np.std(vals)],
                         fmt='o', capsize=4, color=color, ms=5)
    axes[1].set_xscale('log')
    axes[1].set_xlabel("$\\lambda$"); axes[1].set_ylabel("$T \\cdot \\lambda$")
    axes[1].set_title("(b) Product stability")
    
    # (c) Representative curves
    for lam_val, color, label in [(0.01, C['gray'], 'Regime I'),
                                   (0.5, C['blue'], 'Regime II'),
                                   (5.0, C['red'], 'Regime III')]:
        for r in runs:
            if r["lam"] == lam_val and r["seed"] == 42:
                axes[2].plot(r["logs"]["steps"], r["logs"]["val_acc"],
                            color=color, lw=1.5, label=label)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Val accuracy")
    axes[2].set_title("(c) Three regimes"); axes[2].legend()
    
    fig.suptitle("Figure 2: Weight Decay Sensitivity ($p=97$, $\\eta=10^{-3}$)",
                 fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    savefig(fig, "fig2_lambda_sweep")
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  FIGURE 3: Modulus Dependence (S3)
# ═══════════════════════════════════════════════════════════════
def make_fig3():
    data = load_json("script3_v2")
    if not data: return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    p_stats = analysis.get("p_stats", [])
    
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)
    
    # (a) Delay vs p
    ax = fig.add_subplot(gs[0, 0])
    for r in runs:
        if r.get("T_grok") and r.get("T_mem"):
            ax.scatter(r["p"], r["T_grok"]-r["T_mem"], alpha=0.3, s=15, c=C['cyan'], edgecolor='none')
    for s in p_stats:
        if s.get("delay_mean"):
            ax.plot(s["p"], s["delay_mean"], 'o', color=C['red'], ms=6, zorder=3)
    ax.set_xlabel("$p$"); ax.set_ylabel("Delay $T_{\\mathrm{grok}}-T_{\\mathrm{mem}}$")
    ax.set_title("(a) Grokking delay vs $p$")
    
    # (b) V_mem vs p  
    ax = fig.add_subplot(gs[0, 1])
    for r in runs:
        if r.get("V_at_mem"):
            ax.scatter(r["p"], r["V_at_mem"], alpha=0.3, s=15, c=C['green'], edgecolor='none')
    for s in p_stats:
        if s.get("V_mem_mean"):
            ax.plot(s["p"], s["V_mem_mean"], 'o', color=C['red'], ms=6, zorder=3)
    # Linear fit
    ps = [s["p"] for s in p_stats if s.get("V_mem_mean")]
    vs = [s["V_mem_mean"] for s in p_stats if s.get("V_mem_mean")]
    if len(ps) >= 3:
        sl, ic, _, _, _ = linregress(ps, vs)
        xl = np.linspace(min(ps), max(ps), 50)
        ax.plot(xl, sl*xl+ic, '--', color=C['red'], lw=1.2, label=f'slope={sl:.0f}')
        ax.legend()
    ax.set_xlabel("$p$"); ax.set_ylabel(r"$\|\theta_{\mathrm{mem}}\|^2$")
    ax.set_title(r"(b) $\|\theta_{\mathrm{mem}}\|^2$ vs $p$")
    
    # (c) Norm ratio validation
    ax = fig.add_subplot(gs[0, 2])
    delays = []; tesc_th = []
    for r in runs:
        if r.get("T_grok") and r.get("T_mem") and r.get("T_escape_theory"):
            delays.append(r["T_grok"]-r["T_mem"])
            tesc_th.append(r["T_escape_theory"])
    if delays:
        ax.scatter(tesc_th, delays, alpha=0.4, s=20, c=C['blue'], edgecolor='none')
        lim = max(max(delays), max(tesc_th)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='$y=x$')
        if len(delays) >= 5:
            sl, ic, rv, _, _ = linregress(tesc_th, delays)
            ax.plot([0, lim], [ic, sl*lim+ic], '--', color=C['red'], lw=1.2,
                    label=f'Fit: slope={sl:.2f}, $r$={rv:.2f}')
        ax.legend()
    ax.set_xlabel(r"$T_{\mathrm{esc}}^{\mathrm{th}}$")
    ax.set_ylabel("Measured delay")
    ax.set_title("(c) Norm-ratio validation")
    
    # (d) K vs p
    ax = fig.add_subplot(gs[1, 0])
    by_p = {}
    for r in runs: by_p.setdefault(r["p"], []).append(r)
    for p in sorted(by_p.keys()):
        ks = [r["K_final"] for r in by_p[p] if r.get("K_final")]
        if ks:
            ax.errorbar(p, np.mean(ks), yerr=np.std(ks), fmt='o', capsize=3,
                        color=C['blue'], ms=5)
    ax.set_xlabel("$p$"); ax.set_ylabel("$K$")
    ax.set_title("(d) Fourier support $K$ vs $p$")
    
    # (e) T_mem vs p
    ax = fig.add_subplot(gs[1, 1])
    for s in p_stats:
        if s.get("T_mem_mean"):
            ax.plot(s["p"], s["T_mem_mean"], 'o', color=C['green'], ms=6)
    ax.set_xlabel("$p$"); ax.set_ylabel("$T_{\\mathrm{mem}}$")
    ax.set_title("(e) Memorisation time vs $p$")
    
    # (f) Representative curves
    ax = fig.add_subplot(gs[1, 2])
    for pp, color, ls in [(53, C['blue'], '-'), (97, C['green'], '-'), (127, C['red'], '-')]:
        for r in runs:
            if r["p"]==pp and r["seed"]==42 and r.get("T_grok"):
                ax.plot(r["logs"]["steps"], r["logs"]["val_acc"], color=color, lw=1.5,
                        label=f'$p={pp}$')
                break
    ax.set_xlabel("Step"); ax.set_ylabel("Val accuracy")
    ax.set_title("(f) Grokking curves"); ax.legend()
    
    fig.suptitle("Figure 3: Modulus Dependence ($\\eta=10^{-3}$, $\\lambda=1.0$)",
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, "fig3_modulus_sweep")
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  FIGURE 4: Learning Rate Sensitivity (S5)
# ═══════════════════════════════════════════════════════════════
def make_fig4():
    data = load_json("script5_v2")
    if not data: return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    
    n_A = analysis.get("n_results_A", len(runs))
    runs_A = runs[:n_A]; runs_B = runs[n_A:]
    
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)
    
    # (a) T_grok vs 1/η
    ax = fig.add_subplot(gs[0, 0])
    for r in runs_A:
        if r.get("T_grok"):
            ax.scatter(1/r["eta"], r["T_grok"], alpha=0.5, s=25, c=C['blue'],
                      edgecolor='white', lw=0.3, zorder=2)
    fit = analysis.get("linear_fit", {})
    if fit.get("R2", 0) > 0:
        xl = np.linspace(0, 12000, 100)
        ax.plot(xl, fit["slope"]*xl + fit["intercept"], '--', color=C['red'], lw=1.5,
                label=f'$R^2={fit["R2"]:.3f}$')
        ax.legend()
    ax.set_xlabel("$1/\\eta$"); ax.set_ylabel("$T_{\\mathrm{grok}}$")
    ax.set_title("(a) $T_{\\mathrm{grok}} \\propto 1/\\eta$")
    
    # (b) T·η product
    ax = fig.add_subplot(gs[0, 1])
    by_eta = {}
    for r in runs_A:
        if r.get("T_grok"): by_eta.setdefault(r["eta"], []).append(r["T_grok"]*r["eta"])
    for eta in sorted(by_eta.keys()):
        vals = by_eta[eta]
        ax.errorbar([eta], [np.mean(vals)], yerr=[np.std(vals)],
                    fmt='o', capsize=4, color=C['blue'], ms=5)
    ax.set_xscale('log')
    ax.set_xlabel("$\\eta$"); ax.set_ylabel("$T \\cdot \\eta$")
    ax.set_title("(b) $T \\cdot \\eta$ product")
    
    # (c) Joint η×λ heatmap
    ax = fig.add_subplot(gs[0, 2])
    config_B = analysis.get("config_B", {})
    etas_B = config_B.get("etas", [2e-3, 1e-3, 5e-4])
    lams_B = config_B.get("lams", [0.5, 1.0, 2.0])
    grid = np.full((len(lams_B), len(etas_B)), np.nan)
    counts = np.zeros_like(grid)
    for r in runs_B:
        if r.get("T_grok") and r["eta"] in etas_B and r["lam"] in lams_B:
            i = lams_B.index(r["lam"]); j = etas_B.index(r["eta"])
            val = r["T_grok"] * r["eta"] * r["lam"]
            if np.isnan(grid[i,j]):
                grid[i,j] = val; counts[i,j] = 1
            else:
                grid[i,j] = (grid[i,j]*counts[i,j] + val) / (counts[i,j]+1)
                counts[i,j] += 1
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(etas_B))); ax.set_xticklabels([f'{e:.0e}' for e in etas_B])
    ax.set_yticks(range(len(lams_B))); ax.set_yticklabels([f'{l:.1f}' for l in lams_B])
    plt.colorbar(im, ax=ax, label='$T \\cdot \\eta\\lambda$', fraction=0.046)
    ax.set_xlabel("$\\eta$"); ax.set_ylabel("$\\lambda$")
    ax.set_title("(c) $T \\cdot \\eta\\lambda$ universality")
    
    # (d) Doubling test
    ax = fig.add_subplot(gs[1, 0])
    ratios = analysis.get("doubling_ratios", [])
    if ratios:
        ax.bar(range(len(ratios)), ratios, color=C['cyan'], edgecolor=C['black'], lw=0.5)
        ax.axhline(2.0, color=C['red'], ls='--', lw=1.2, label='Expected $\\times 2$')
        ax.legend()
    ax.set_ylabel("$T_{\\mathrm{lo}} / T_{\\mathrm{hi}}$")
    ax.set_title("(d) Doubling test")
    
    # (e) T_escape: measured vs theory
    ax = fig.add_subplot(gs[1, 1])
    t_m = [r.get("T_escape") for r in runs_A if r.get("T_escape") and r.get("T_escape_theory")]
    t_t = [r.get("T_escape_theory") for r in runs_A if r.get("T_escape") and r.get("T_escape_theory")]
    if t_m:
        ax.scatter(t_t, t_m, alpha=0.5, s=25, c=C['blue'], edgecolor='white', lw=0.3)
        lim = max(max(t_m), max(t_t)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='$y=x$')
        ax.legend()
    ax.set_xlabel("$T_{\\mathrm{esc}}$ (theory)"); ax.set_ylabel("$T_{\\mathrm{esc}}$ (measured)")
    ax.set_title("(e) Formula validation")
    
    # (f) Representative curves at different η
    ax = fig.add_subplot(gs[1, 2])
    for eta_val, color in [(2e-3, C['blue']), (1e-3, C['green']), (1e-4, C['red'])]:
        for r in runs_A:
            if r["eta"]==eta_val and r["seed"]==42:
                ax.plot(r["logs"]["steps"], r["logs"]["val_acc"], color=color, lw=1.5,
                        label=f'$\\eta={eta_val:.0e}$')
                break
    ax.set_xlabel("Step"); ax.set_ylabel("Val accuracy")
    ax.set_title("(f) Effect of $\\eta$"); ax.legend()
    
    fig.suptitle("Figure 4: Learning Rate Sensitivity ($p=97$)",
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, "fig4_eta_sweep")
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  FIGURE 5: Spectral Separation (S4)
# ═══════════════════════════════════════════════════════════════
def make_fig5():
    data = load_json("script4_v2")
    if not data: return
    runs = data["runs"]
    analysis = data.get("analysis", {})
    
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)
    P = 97; half = P // 2 + 1
    K_star = analysis.get("K_star", [])
    
    # (a) R(f_θ) trajectories
    ax = fig.add_subplot(gs[0, 0])
    for r in runs:
        fl = r.get("fourier_logs", {})
        if fl and fl.get("steps"):
            ax.plot(fl["steps"], fl["R"], alpha=0.6, lw=1.0, label=f's{r["seed"]}')
            if r["T_grok"]:
                ax.axvline(r["T_grok"], color=C['red'], lw=0.8, alpha=0.3, ls='--')
    ax.set_ylabel("$\\mathcal{R}(f_\\theta)$"); ax.set_xlabel("Step")
    ax.set_title("(a) Non-Fourier energy")
    ax.legend(fontsize=6, ncol=2)
    
    # (b) Val gap vs R scatter
    ax = fig.add_subplot(gs[0, 1])
    scatter = analysis.get("scatter_data", [])
    if scatter:
        Rs = [s["R"] for s in scatter]
        gaps = [s["gap"] for s in scatter]
        ax.scatter(Rs, gaps, alpha=0.35, s=18, c=C['blue'], edgecolor='none')
        if len(Rs) >= 5:
            sl, ic, rv, _, _ = linregress(Rs, gaps)
            xl = np.linspace(0, max(Rs), 100)
            ax.plot(xl, sl*xl+ic, '--', color=C['red'], lw=1.5,
                    label=f'OLS $c={sl:.1f}$, $R^2={rv**2:.3f}$')
            ax.legend()
    ax.set_xlabel("$\\mathcal{R}(f_\\theta)$"); ax.set_ylabel("Validation gap")
    ax.set_title("(b) Gap $\\propto \\mathcal{R}$")
    
    # (c) Irreversibility
    ax = fig.add_subplot(gs[0, 2])
    post_R = {}
    for r in runs:
        fl = r.get("fourier_logs", {})
        if fl and r.get("T_grok"):
            for s, rv in zip(fl.get("steps",[]), fl.get("R",[])):
                if s >= r["T_grok"]:
                    rel = s - r["T_grok"]
                    post_R.setdefault(rel, []).append(rv)
    if post_R:
        ss = sorted(post_R.keys())
        mm = [np.mean(post_R[s]) for s in ss]
        sd = [np.std(post_R[s]) for s in ss]
        ax.plot(ss, mm, color=C['blue'], lw=1.5)
        ax.fill_between(ss, np.array(mm)-np.array(sd), np.array(mm)+np.array(sd),
                        color=C['blue'], alpha=0.15)
        ax.set_yscale('log')
    ax.set_xlabel("Steps after grokking"); ax.set_ylabel("$\\mathcal{R}$")
    ax.set_title("(c) Irreversibility")
    
    # (d) Spectral heatmap
    ax = fig.add_subplot(gs[1, 0])
    for r in runs:
        fl = r.get("fourier_logs", {})
        if fl and fl.get("spectra"):
            spectra = np.array(fl["spectra"])[:, :half]
            total = spectra.sum(axis=1, keepdims=True) + 1e-15
            spectra_norm = spectra / total
            im = ax.imshow(spectra_norm.T, aspect='auto', origin='lower',
                          extent=[fl["steps"][0], fl["steps"][-1], 0, half],
                          norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno')
            if r["T_grok"]:
                ax.axvline(r["T_grok"], color='cyan', lw=2.0, alpha=0.8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            break
    ax.set_xlabel("Step"); ax.set_ylabel("Frequency $k$")
    ax.set_title("(d) Spectral evolution")
    
    # (e) Logit bound
    ax = fig.add_subplot(gs[1, 1])
    for r in runs:
        ll = r.get("logit_logs", {})
        if ll and ll.get("steps"):
            ax.plot(ll["steps"], ll["B"], alpha=0.6, lw=1.0)
    ax.set_xlabel("Step"); ax.set_ylabel("$\\|z\\|_\\infty$")
    ax.set_title("(e) Logit bound")
    
    # (f) Cumulative energy → K*
    ax = fig.add_subplot(gs[1, 2])
    mean_spec = np.array(analysis.get("mean_post_spectrum", []))
    if len(mean_spec) > 0:
        sorted_e = np.sort(mean_spec)[::-1]
        cum = np.cumsum(sorted_e) / (np.sum(sorted_e) + 1e-15)
        ax.plot(cum[:50], color=C['blue'], lw=2.0)
        ax.axhline(0.99, color=C['red'], ls='--', lw=1.0, label='99% threshold')
        ax.axvline(len(K_star), color=C['green'], ls='--', lw=1.0,
                   label=f'$K^*={len(K_star)}$')
        ax.legend()
    ax.set_xlabel("Number of frequencies"); ax.set_ylabel("Cumulative energy")
    ax.set_title("(f) $K^*$ detection")
    
    fig.suptitle("Figure 5: Spectral Separation ($p=97$, $K^*=%d$)" % len(K_star),
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, "fig5_spectral")
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  MASTER FIGURE SCRIPT — Publication Figures for Paper")
    print(f"  Reading from: {BASE_DIR}")
    print(f"  Writing to:   {FIGS_OUT}")
    print("="*60)
    
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    
    print(f"\nAll figures saved to: {FIGS_OUT}")
    print("Done!")
