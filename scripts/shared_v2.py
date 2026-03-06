"""
shared_v2.py — Enhanced shared infrastructure for all validation scripts.
Changes from v1:
  - save_full_results(): saves ALL data including logs (for figure plotting)
  - save_summary_json(): saves lightweight summary (backward compatible)
  - No behavioral changes to train_run() — fully backward compatible
"""
import os, json, random, math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Google Drive setup ──
DRIVE_BASE = "/content/drive/MyDrive/grokking_results"

def setup_drive():
    """Mount Google Drive and create results directory."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except:
        pass  # Not on Colab or already mounted
    os.makedirs(DRIVE_BASE, exist_ok=True)
    log(f"Results will be saved to: {DRIVE_BASE}")

def get_out_dir(script_name):
    """Create and return output directory on Google Drive."""
    d = os.path.join(DRIVE_BASE, script_name, "figs")
    os.makedirs(d, exist_ok=True)
    return d

def get_json_path(script_name, filename="summary.json"):
    """Return path for JSON output on Google Drive."""
    d = os.path.join(DRIVE_BASE, script_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)

# ── Logging ──
def log(msg, **kwargs): print(msg, flush=True, **kwargs)

# ── Reproducibility ──
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

# ── Dataset ──
def build_data(p, frac, seed):
    """Build (a+b) mod p dataset with deterministic split."""
    rng = random.Random(seed)
    pairs = [(a,b) for a in range(p) for b in range(p)]
    rng.shuffle(pairs); sp = int(len(pairs)*frac)
    def tt(pl):
        x = torch.tensor(pl, dtype=torch.long).to(DEVICE)
        y = torch.tensor([(a+b)%p for a,b in pl], dtype=torch.long).to(DEVICE)
        return x, y
    return tt(pairs[:sp]), tt(pairs[sp:])

# ── Model ──
class Model(nn.Module):
    """1-layer transformer matching paper Sec 4.1."""
    def __init__(self, p, d=128, heads=4, ff=512):
        super().__init__()
        self.tok = nn.Embedding(p, d); self.pos = nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, 0.0, activation="gelu", batch_first=True)
        self.tf = nn.TransformerEncoder(enc, 1)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, p)
    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(2, device=x.device).unsqueeze(0).expand(x.size(0),-1))
        return self.head(self.norm(self.tf(h)[:,-1,:]))

# ── Measurements ──

def V(model):
    """||θ||² — Lyapunov function."""
    return sum(p.data.pow(2).sum().item() for p in model.parameters())

@torch.no_grad()
def eval_model(model, x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1)==y).float().mean().item()
    return loss, acc

@torch.no_grad()
def fourier_R_K(model, p, n_b=3, n_c=5):
    """
    Fourier measurement: R(f_θ) and K.
    Fast default: n_b=3, n_c=5. Accurate: n_b=20, n_c=p.
    """
    model.eval()
    spectra = []
    b_values = list(range(min(n_b, p)))
    for b in b_values:
        xs = torch.arange(p, device=DEVICE)
        pairs = torch.stack([xs, torch.full_like(xs, b)], dim=1)
        logits = model(pairs)
        classes = list(range(min(n_c, p))) if n_c >= p else random.sample(range(p), n_c)
        for c in classes:
            fc = logits[:, c].cpu().numpy().astype(np.float64)
            spectra.append(np.abs(np.fft.fft(fc)/p)**2)
    ms = np.mean(spectra, axis=0)
    total = ms.sum()
    if total < 1e-15: return 1.0, 0, ms
    norm = ms/total; peak = norm.max()
    K = int(np.sum(norm > peak*0.01))
    dominant = set(np.argsort(norm)[::-1][:max(K,1)])
    R = float(1.0 - sum(norm[k] for k in dominant))
    return R, K, ms

@torch.no_grad()
def logit_bound(model, p):
    """||z_θ(x)||_∞ — max logit across all inputs."""
    model.eval()
    mx = 0.0
    for a0 in range(0, p, 97):
        aa = torch.arange(a0, min(a0+97, p), device=DEVICE)
        bb = torch.arange(p, device=DEVICE)
        ga, gb = torch.meshgrid(aa, bb, indexing='ij')
        pairs = torch.stack([ga.flatten(), gb.flatten()], dim=1)
        mx = max(mx, model(pairs).abs().max().item())
    return mx

def fit_exp_decay(steps, Vs, T_mem, eta, lam):
    """Fit V_t = A·r^t + C post-memorisation."""
    if T_mem is None: return None
    mask = [s >= T_mem for s in steps]
    t = np.array([s for s,m in zip(steps,mask) if m], dtype=float)
    v = np.array([vv for vv,m in zip(Vs,mask) if m], dtype=float)
    if len(t) < 5: return None
    t -= t[0]; r_th = 1 - 2*eta*lam
    try:
        def f(t,A,lr,C): return A*np.exp(lr*t)+C
        popt,_ = curve_fit(f, t, v,
            p0=[max(v[0]-v[-1]*0.9,1), np.log(max(r_th,0.001)), v[-1]*0.9],
            maxfev=10000, bounds=([0,-1,0],[1e7,0,v[0]*1.1]))
        r_fit = np.exp(popt[1])
        vp = f(t,*popt)
        ss_res = np.sum((v-vp)**2); ss_tot = np.sum((v-np.mean(v))**2)
        R2 = 1-ss_res/ss_tot if ss_tot > 0 else 0
        return {"rate_fit":float(r_fit), "rate_theory":float(r_th),
                "R2":float(R2), "A":float(popt[0]), "C":float(popt[2])}
    except: return None

def T_escape_theory(V_mem, V_final, eta, lam):
    if V_mem and V_final > 0 and V_mem > V_final:
        return (1/(2*eta*lam)) * math.log(V_mem/V_final)
    return None

# ── Generic training loop ──

def train_run(p, eta, lam, seed, max_steps=50000, log_every=200,
              measure_fourier_every=0, measure_logit=False,
              post_grok_extend=0, batch_size=512, grok_thresh=0.99,
              fourier_nb=3, fourier_nc=5):
    """
    Unified training function. Returns dict with all measurements.
    measure_fourier_every=0 means no Fourier measurement (fast).
    """
    set_seed(seed)
    (xt,yt),(xv,yv) = build_data(p, 0.5, seed)
    model = Model(p).to(DEVICE)
    opt = AdamW(model.parameters(), lr=eta, weight_decay=lam)
    n_train = xt.size(0)

    logs = {"steps":[], "V_t":[], "train_loss":[], "train_acc":[],
            "val_loss":[], "val_acc":[]}
    fourier_logs = {"steps":[], "R":[], "K":[], "spectra":[]}
    logit_logs = {"steps":[], "B":[]}

    T_mem = T_grok = None; V_at_mem = val_loss_at_mem = None
    effective_max = max_steps

    for step in range(effective_max + post_grok_extend + 5000):
        if T_grok is None and step >= effective_max: break
        if T_grok is not None and step > T_grok + post_grok_extend + 2000: break

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
                T_mem = step; V_at_mem = Vt; val_loss_at_mem = vl
            if T_grok is None and va >= grok_thresh:
                T_grok = step

        if measure_fourier_every > 0 and step % measure_fourier_every == 0:
            R, K, spec = fourier_R_K(model, p, n_b=fourier_nb, n_c=fourier_nc)
            fourier_logs["steps"].append(step)
            fourier_logs["R"].append(R); fourier_logs["K"].append(K)
            fourier_logs["spectra"].append(spec.tolist())

        if measure_logit and measure_fourier_every > 0 and step % measure_fourier_every == 0:
            logit_logs["steps"].append(step)
            logit_logs["B"].append(logit_bound(model, p))

    V_final = V(model)
    R_final, K_final, _ = fourier_R_K(model, p, n_b=fourier_nb, n_c=fourier_nc)

    T_esc = None
    if T_mem is not None:
        thresh = V_final * 1.5
        for s, v in zip(logs["steps"], logs["V_t"]):
            if s >= T_mem and v <= thresh:
                T_esc = s; break

    delta_min = None
    if T_mem is not None and T_grok is not None:
        pre = [vl for s,vl in zip(logs["steps"],logs["val_loss"]) if T_mem<=s<T_grok]
        post = [vl for s,vl in zip(logs["steps"],logs["val_loss"]) if s>=T_grok]
        if pre and post: delta_min = float(np.mean(pre) - np.mean(post))

    fit = fit_exp_decay(logs["steps"], logs["V_t"], T_mem, eta, lam)
    T_esc_th = T_escape_theory(V_at_mem, V_final, eta, lam)

    return {
        "seed": seed, "p": p, "eta": eta, "lam": lam,
        "T_mem": T_mem, "T_grok": T_grok, "T_escape": T_esc,
        "T_escape_theory": T_esc_th,
        "V_at_mem": V_at_mem, "V_final": V_final,
        "K_final": K_final, "R_final": R_final,
        "delta_min": delta_min, "val_loss_at_mem": val_loss_at_mem,
        "fit": fit,
        "T_grok_x_lam": (T_grok*lam) if T_grok else None,
        "T_grok_x_eta": (T_grok*eta) if T_grok else None,
        "T_grok_x_eta_lam": (T_grok*eta*lam) if T_grok else None,
        "logs": logs,
        "fourier_logs": fourier_logs if measure_fourier_every > 0 else None,
        "logit_logs": logit_logs if measure_logit else None,
    }

# ═══════════════════════════════════════════════════════════════
#  NEW in v2: Full-data JSON save utilities
# ═══════════════════════════════════════════════════════════════

def _make_serializable(obj):
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def save_full_results(results, path, extra=None):
    """
    Save ALL data from train_run results, INCLUDING logs.
    This is what the master figure script reads.
    
    Args:
        results: list of dicts from train_run()
        path: output JSON path  
        extra: optional dict of additional analysis data to include
    """
    payload = {
        "version": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": _make_serializable(results),
    }
    if extra:
        payload["analysis"] = _make_serializable(extra)
    
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    
    size_mb = os.path.getsize(path) / (1024*1024)
    log(f"Full results saved: {path} ({size_mb:.1f} MB, {len(results)} runs)")


def save_summary_json(results, path, extra=None):
    """
    Save lightweight summary (backward compatible with v1).
    Strips logs to keep file small.
    """
    stripped = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ("logs", "fourier_logs", "logit_logs")}
        stripped.append(s)
    
    payload = {"results": _make_serializable(stripped)}
    if extra:
        payload.update(_make_serializable(extra))
    
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    
    log(f"Summary saved: {path}")
