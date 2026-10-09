#!/usr/bin/env python3
"""
Partial R² residualization probe (ref/target2.txt protocol)
============================================================

Protocol core: [NOT] subtracting x_t's R² from each layer's R² (that is the R²
difference), but [first residualize the target against x_t, then explain the
residual with the layer features]. The FWL theorem guarantees the residual
method is numerically identical to the partial R²:

    R²_reduced        = R²(x_t -> y)              # pass-through baseline, y ∈ {x0, x0_LF, x0_HF, ε}
    r = y − x_t·W_c                                # W_c = OLS(x_t -> y), 16→16, overdetermined
    partial R²(ℓ)     = R²(z_ℓ -> r)              # 1152→16, overdetermined, closed-form
    R²_full           = 1 − (1−R²_reduced)(1−partial)   # back-solved via the identity; no need to run separately

Partial R² = (R²_full − R²_reduced)/(1 − R²_reduced): the fraction of the
residual variance that x_t cannot explain which is explained by z_ℓ. When the
pass-through baseline is high (0.67+ from the start), the R² difference
systematically compresses the visibility of "active processing"; the partial
R² must be used (protocol §1).

Steps (protocol §2):
  Step 1  Data pairing (same forward, same noise realization!): x_t / x0 / ε /
          frequency bands extracted in the same pass as per-layer
          representations; also t(α,σ) metadata. x_t is 16-dim per token
          (4ch×2×2) — a new extraction point must be added to the extraction
          pipeline and saved to disk.
  Step 2  Reduced regression (pass-through baseline) + theoretical sanity
          check (per-dim white-noise corruption):
          R²_reduced(x_t→x0) ≈ α², R²_reduced(x_t→ε) ≈ σ² (SNR=2 → 2/3 and 1/3).
          A clear deviation in practice (>0.02) → first check latent
          normalization and patch alignment.
          Byproduct: the regression R² of z_ℓ→x_t = the pass-through rate
          τ_ℓ profile (a measurement missing from the protocol's early
          rounds).
  Step 3  Residual regression (per layer): partial R²(ℓ) = R²(M_xt z_ℓ → r),
          layer-independent, pure post-processing.
          The strict FWL form requires z_ℓ to also be residualized against
          x_t (M_xt: z − x_t·W_zt, W_zt = OLS(x_t→z_ℓ); consistent with the
          protocol §1 partial R² definition and §4 "controlling covariate ≡
          regression after projection"). The protocol's literal R²(z→r)
          equals it only when z⊥x_t — when τ_ℓ is high in practice (z
          strongly correlated with x_t), the literal form [underestimates]
          partial and breaks the identity.
          Default closed-form OLS (same family as reduced → the FWL identity
          is exact);
          --solver ridge switches to standardized ridge regression (input is
          also the residualized z; the identity is only approximate).
  Step 4  Report (R²_reduced, partial R²) + numerical self-check: pick 2-3
          layers and actually run the joint regression [z ‖ x_t] (1168-dim)
          to verify the back-solved R²_full (the self-check always uses OLS
          to verify the identity).
  Step 5  Optional --sweep_sizes: fit-subset sample-size sweep (especially
          important on the ε side, protocol §3).

Data organization (same as probe_features.py; this repo's DiT extraction data
uses --input_layout chw):

    feature_root/
        {class}_{image_name}/
            {per-layer repr}.npy  # (1152, gh, gw) chw
            {xt}.npy              # x_t 16-dim per token: (4, 2gh, 2gw) or (16, gh, gw)
            {x0,eps}.npy          # regression targets, same layout (auto patchify alignment)
            {x0_LF,x0_HF}.npy     # band decomposition (optional; when absent, derived
                                  # offline from x0 by Gaussian low-pass, aligned with the
                                  # pipe definition gaussian_blur(k=5, σ=1.0))
            (other modules ignored) # irrelevant points in the directory only get their
                                  # shapes printed and are skipped; they are not
                                  # required to align with the token grid

Held-out protocol: W_c and each layer's W are learned on fit images; R² is
reported on held-out images; residuals on held-out images are recomputed with
the held-out images' own x_t (not reusing train residuals).

Usage:
  python probe_partial_r2.py --feature_root out/dit-eps_snr2 --snr 2 \
      --targets x0 x0_LF x0_HF eps --result_path partial_eps_snr2.json

  Pairwise mode (multiple --xt_module; the pairing is decided by --xt_pair_by,
  with auto deciding by unique count match):
  layer:  the i-th x_t pairs with the i-th per-layer representation, each pair
          computes all targets in full (controlling-variable source-layer
          sweep; e.g. 9 after-attn x_t paired with 9 mlp layers, 3 targets):
  python probe_partial_r2.py --feature_root out/dit-eps_snr2 --snr 2 \
      --targets x0 eps --xt_pair_by layer --xt_module xt_a xt_b \
      --layers blk0 blk1 --result_path pairwise_layer.json
  target: the i-th x_t pairs with the i-th target, across all layers:
  python probe_partial_r2.py --feature_root out/dit-eps_snr2 --snr 2 \
      --targets x0 eps --xt_module xt_x0 xt_eps --xt_pair_by target \
      --result_path pairwise_target.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import probe_features as pf


# =========================================================================== #
#              Module discovery / shape scanning / tokenization                #
# =========================================================================== #

TARGET_ALIASES = {
    'x0': ['x0'], 'eps': ['eps'],
    'x0_LF': ['x0_LF', 'x0-low', 'x0_low'], 'x0_HF': ['x0_HF', 'x0-high', 'x0_high'],
    'xt': ['xt', 'x_t'],
}


def scan_module_shapes(folders, modules):
    """[Lazy scan] read only the npy header of each module's first image
    (folders[0]) — startup cost = one file open per module, independent of
    the image count. Returns {module: first-image storage shape}; cross-image
    shape consistency is verified at load time instead (LayerLoader /
    preload_tokens take the reference shape)."""
    f0 = folders[0]
    shapes = {}
    for m in modules:
        a = np.load(f0 / f'{m}.npy', mmap_mode='r')
        shapes[m] = tuple(int(v) for v in a.shape)
    return shapes


def check_shape(arr, module, folder_name, ref_shape):
    """On-load check: this image's npy storage shape must match the first
    image's reference (companion self-check of the lazy scan; the original
    full pre-check moved here)."""
    s = tuple(int(v) for v in arr.shape)
    if s != tuple(ref_shape):
        raise RuntimeError(f'{folder_name}/{module}.npy shape {s} mismatches '
                           f'the first image {tuple(ref_shape)}; check the feature files')


def chw_of(shape, layout):
    if len(shape) == 2:
        return None
    if layout == 'chw':
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    return (int(shape[2]), int(shape[0]), int(shape[1]))


def resolve_layers(modules, shapes, layout, explicit=None):
    """Per-layer representation modules + token grid; auto rule: all 3-dim
    modules with the largest channel dim."""
    if explicit:
        layers = list(dict.fromkeys(explicit))
        missing = [m for m in layers if m not in modules]
        if missing:
            raise RuntimeError(f'--layers modules not found: {missing}, candidates: {modules}')
        grids = {chw_of(shapes[m], layout)[1:] for m in layers
                 if chw_of(shapes[m], layout) is not None}
        if len(grids) != 1:
            raise RuntimeError(f'per-layer representation spatial grids differ: {grids}')
        return layers, next(iter(grids))
    cands = {m: chw_of(shapes[m], layout) for m in modules
             if chw_of(shapes[m], layout) is not None}
    if not cands:
        raise RuntimeError('no 3-dim feature modules, cannot determine the token grid (use explicit --layers)')
    c_max = max(c for c, _, _ in cands.values())
    grids = {(h, w) for c, h, w in cands.values() if c == c_max}
    if len(grids) > 1:
        print(f'[warn] modules with max channel dim {c_max} have inconsistent spatial grids {sorted(grids)}')
    layers = sorted(m for m, (c, _, _) in cands.items() if c == c_max)
    grid = (cands[layers[0]][1], cands[layers[0]][2])
    layers = [m for m in layers if cands[m][1:] == grid] or [layers[0]]
    return layers, grid


def build_module_plan(shape, layout, grid, name='?'):
    """Module npy storage shape -> tokenization plan; name is only for error localization."""
    if len(shape) == 2:
        gh, gw = grid
        if int(shape[0]) != gh * gw:
            raise RuntimeError(f'2-dim module {name} token count {int(shape[0])} '
                               f'mismatches grid {grid}')
        return ('2d', (int(shape[1]),))
    c, h, w = chw_of(shape, layout)
    gh, gw = grid
    if (h, w) == (gh, gw):
        return ('grid', (c, gh, gw, 1, 1))
    if (h, w) == (2 * gh, 2 * gw):
        p = h // gh
        return ('patch', (c, gh, gw, p, p))
    raise RuntimeError(f'module {name} spatial size {(h, w)} mismatches the token '
                       f'grid {grid} (first-image storage shape {tuple(int(v) for v in shape)})')


def tok_dim(plan):
    kind, g = plan
    if kind == '2d':
        return g[0]
    c, gh, gw, p, _ = g
    return c if kind == 'grid' else c * p * p


def to_tokens(arr, plan):
    """npy array -> (T, d_tok) fp32; token order (h, w) row-major, in-channel order same as patchify."""
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    kind, g = plan
    if kind == '2d':
        return x.contiguous()
    c, gh, gw, p, _ = g
    if kind == 'grid':
        return x.reshape(c, gh * gw).T.contiguous()
    x = x.reshape(c, gh, p, gw, p).permute(1, 3, 0, 2, 4)
    return x.reshape(gh * gw, c * p * p).contiguous()


# =========================================================================== #
#              Band decomposition (offline reproduction of the pipe)           #
# =========================================================================== #

def unpatchify_tokens(tok, plan, grid, patch_size):
    """(T, d_tok) -> latent (c, gh*p, gw*p). The 'grid' layout needs patch_size (default 2)."""
    kind, g = plan
    gh, gw = grid
    if kind == 'patch':
        c, _, _, p, _ = g
    else:
        p = patch_size
        c = tok.shape[1] // (p * p)
    x = tok.reshape(gh, gw, c, p, p).permute(2, 0, 3, 1, 4)
    return x.reshape(c, gh * p, gw * p).contiguous()


def patchify_latent(lat, grid):
    gh, gw = grid
    c, H, W = lat.shape
    p = H // gh
    x = lat.reshape(c, gh, p, gw, p).permute(1, 3, 0, 2, 4)
    return x.reshape(gh * gw, c * p * p).contiguous()


def gaussian_blur2d(x, k, sigma):
    """(c, H, W) Gaussian low-pass with reflect padding; aligned with the pipe band-split definition k=5, σ=1.0."""
    assert k % 2 == 1, 'blur kernel must be odd'
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    ker = torch.outer(g, g).expand(x.shape[0], 1, k, k)
    xp = F.pad(x.unsqueeze(0), (k // 2,) * 4, mode='reflect')
    return F.conv2d(xp, ker, groups=x.shape[0])[0]


def build_band_offline(x0_map, plan_x0, grid, args):
    """x0 tokens -> (LF, HF) tokens; x0 = LF + HF (latent-space gaussian_blur)."""
    low, high = {}, {}
    for fname, tok in x0_map.items():
        lat = unpatchify_tokens(tok, plan_x0, grid, args.patch_size)
        lf = gaussian_blur2d(lat, args.blur_kernel, args.blur_sigma)
        low[fname] = patchify_latent(lf, grid)
        high[fname] = patchify_latent(lat - lf, grid)
    return low, high


# =========================================================================== #
#              α/σ metadata parsing (theoretical sanity check)                 #
# =========================================================================== #

def lam_to_alpha_sigma(lam, path):
    """logSNR -> (α, σ); same formula as lscp.path_coefficients (VP: α²=sigmoid(λ))."""
    lam = max(min(lam, 20.0), -20.0)
    if path == 'fm':
        a = 1.0 / (1.0 + math.exp(-lam / 2))
        return a, 1.0 - a
    a2 = 1.0 / (1.0 + math.exp(-lam))
    return math.sqrt(a2), math.sqrt(1.0 - a2)


def resolve_alpha_sigma(args):
    """Priority --meta_json > --lam > --snr; returns (None, None) and skips the theoretical check when all absent."""
    if args.meta_json:
        with open(args.meta_json) as fh:
            d = json.load(fh)
        if 'alpha' in d and 'sigma' in d:
            return float(d['alpha']), float(d['sigma'])
        if 'lam' in d:
            return lam_to_alpha_sigma(float(d['lam']), args.path)
        raise RuntimeError('--meta_json needs an alpha/sigma or lam field')
    if args.lam is not None:
        return lam_to_alpha_sigma(args.lam, args.path)
    if args.snr is not None:
        a2 = args.snr / (1.0 + args.snr)      # VP normalization α²+σ²=1
        return math.sqrt(a2), math.sqrt(1.0 - a2)
    return None, None


# =========================================================================== #
#           Split / OLS / evaluation (isomorphic to the projection script)     #
# =========================================================================== #

def stratified_fit_val_holdout(folders, holdout_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)
    groups = {}
    for f in folders:
        groups.setdefault(pf.class_of(f), []).append(f)
    fit, val, hold = [], [], []
    for c in sorted(groups):
        mem = sorted(groups[c])
        rng.shuffle(mem)
        k = len(mem)
        n_ho = int(round(k * holdout_ratio)) if k >= 2 else 0
        n_va = int(round((k - n_ho) * val_ratio)) if k - n_ho >= 1 else 0
        hold += mem[:n_ho]
        val += mem[n_ho:n_ho + n_va]
        fit += mem[n_ho + n_va:]
    degen = not hold or not val
    if degen:
        print(f'[warn] too few images, degenerate split: fit={len(fit)} val={len(val)} '
              f'holdout={len(hold)} (empty sets fall back to fit; metrics are reference-only)')
    if not hold:
        hold = list(fit)
    return {'fit': fit, 'val': val, 'holdout': hold}, degen


class GramOLS:
    """Streamed OLS sufficient statistics (centered, CPU float64 accumulation,
    device fp32 per-image computation).
    solve returns the minimum-norm OLS solution (the unique solution when
    overdetermined)."""

    def __init__(self, d_in, d_out, device):
        self.device, self.d_in, self.d_out = device, d_in, d_out
        self.G = torch.zeros(d_in, d_in, dtype=torch.float64)
        self.B = torch.zeros(d_in, d_out, dtype=torch.float64)
        self.sx = torch.zeros(d_in, dtype=torch.float64)
        self.sy = torch.zeros(d_out, dtype=torch.float64)
        self.n = 0

    @torch.no_grad()
    def update(self, X, Y):
        X = X.reshape(-1, self.d_in).float()
        Y = Y.reshape(-1, self.d_out).float()
        if self.device.type == 'cpu':
            Xd, Yd = X.double(), Y.double()
            self.G += Xd.T @ Xd
            self.B += Xd.T @ Yd
        else:
            Xd, Yd = X.to(self.device), Y.to(self.device)
            self.G += (Xd.T @ Xd).cpu().double()
            self.B += (Xd.T @ Yd).cpu().double()
        self.sx += X.sum(0).double()
        self.sy += Y.sum(0).double()
        self.n += X.shape[0]

    def solve(self, rcond=1e-10):
        n = max(self.n, 1)
        mu_x = self.sx / n
        mu_y = self.sy / n
        Gc = self.G - n * torch.outer(mu_x, mu_x)
        Bc = self.B - n * torch.outer(mu_x, mu_y)
        evals, evecs = torch.linalg.eigh(Gc)
        emax = float(evals.max().clamp_min(1e-30))
        keep = evals > emax * rcond
        inv = torch.zeros_like(evals)
        inv[keep] = 1.0 / evals[keep]
        W = evecs @ (inv[:, None] * (evecs.T @ Bc))
        cond = emax / float(evals[keep].min()) if bool(keep.any()) else float('inf')
        return W, mu_x, mu_y, cond, int(keep.sum())


class OLSModel:
    """Centered OLS: ŷ = (x − μ_x) @ W + μ_y (CPU float64)."""

    def __init__(self, W, mu_x, mu_y):
        self.W, self.mu_x, self.mu_y = W, mu_x, mu_y

    def __call__(self, X):
        X = X.reshape(-1, self.W.shape[0]).double()
        return (X - self.mu_x) @ self.W + self.mu_y


class RidgeModel:
    """Standardized ridge regression model (reuses probe_features' Standardizer convention)."""

    def __init__(self, W, stdz, device):
        self.W, self.stdz, self.device = W, stdz.to(device), device

    def __call__(self, X):
        Xs = self.stdz.apply_x(X.to(self.device))
        return self.stdz.unapply_y(Xs @ self.W).cpu()


def new_stats(d_out):
    zeros = lambda: torch.zeros(d_out, dtype=torch.float64)
    return {'sum_p': zeros(), 'sum_p2': zeros(), 'sum_y': zeros(),
            'sum_y2': zeros(), 'sum_py': zeros(), 'sum_cos': 0.0, 'n': 0}


def acc_stats(st, P, Y):
    d_out = st['sum_p'].numel()
    p = P.reshape(-1, d_out).double()
    t = Y.reshape(-1, d_out).double()
    st['sum_p'] += p.sum(0)
    st['sum_p2'] += p.pow(2).sum(0)
    st['sum_y'] += t.sum(0)
    st['sum_y2'] += t.pow(2).sum(0)
    st['sum_py'] += (p * t).sum(0)
    cos = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)
    st['sum_cos'] += cos.sum().item()
    st['n'] += p.shape[0]


def stream_stats(pairs, predict_fn, d_out):
    """Stream-accumulate sufficient statistics; pairs is passed as a generator to avoid whole layers staying in memory."""
    st = new_stats(d_out)
    for X, Y in pairs:
        acc_stats(st, predict_fn(X), Y)
    return st


def channel_metrics(st):
    """Per-channel R² averaged is the primary metric; also returns the per-channel R² array (used per-channel by the FWL identity)."""
    n = st['n']
    mu_y = st['sum_y'] / max(n, 1)
    ss_res = st['sum_p2'] - 2 * st['sum_py'] + st['sum_y2']
    ss_tot = st['sum_y2'] - n * mu_y ** 2
    tot_max = float(ss_tot.max()) if n > 0 else 0.0
    valid = ss_tot > tot_max * 1e-9 if tot_max > 0 else torch.zeros_like(ss_tot, dtype=torch.bool)
    r2_c = torch.full_like(ss_tot, float('nan'))
    if bool(valid.any()):
        r2_c[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
        r2_avg = float(r2_c[valid].mean())
        r2_var = float(1.0 - ss_res[valid].sum() / ss_tot[valid].sum())
        mse = float(ss_res[valid].sum() / (n * int(valid.sum())))
    else:
        r2_avg = r2_var = mse = float('nan')
    return {'r2_avg': r2_avg, 'r2_var': r2_var, 'mse': mse,
            'cos_sim': float(st['sum_cos'] / max(n, 1)), 'n_test_loc': int(n),
            'n_valid_channels': int(valid.sum()),
            'r2_per_channel': [float(v) for v in r2_c]}


def identity_full_avg(red_m, part_m):
    """R²_full = 1 − (1−R²_reduced)(1−partial), computed per channel then averaged
    (valid channels = R² valid on both sides; pure algebraic identity, no
    orthogonality needed)."""
    red = torch.tensor(red_m['r2_per_channel'], dtype=torch.float64)
    par = torch.tensor(part_m['r2_per_channel'], dtype=torch.float64)
    v = torch.isfinite(red) & torch.isfinite(par)
    if not bool(v.any()):
        return float('nan')
    return float((1.0 - (1.0 - red[v]) * (1.0 - par[v])).mean())


def preload_tokens(folders, module, plan, shape=None):
    """Preload small target modules: {folder_name: (T, d_tok)} fp32 CPU.
    shape: first-image shape given by scan_module_shapes, for on-load
    cross-image consistency checks."""
    out = {}
    for f in folders:
        arr = np.load(f / f'{module}.npy')
        if shape is not None:
            check_shape(arr, module, f.name, shape)
        out[f.name] = to_tokens(arr, plan)
    return out


class LayerLoader:
    """Lazy per-layer feature loading + optional whole-layer RAM cache (fp32,
    (T, d)); discarded when the layer is done.
    shape: first-image reference shape, verified for cross-image consistency
    on every load."""

    def __init__(self, folders, module, plan, use_cache, shape=None):
        self.by_name = {f.name: f for f in folders}
        self.module, self.plan, self.use_cache = module, plan, use_cache
        self.shape = tuple(shape) if shape is not None else None
        self.cache = {}

    def __call__(self, folder):
        key = folder.name
        if self.use_cache and key in self.cache:
            return self.cache[key]
        arr = np.load(self.by_name[key] / f'{self.module}.npy')
        if self.shape is not None:
            check_shape(arr, self.module, key, self.shape)
        x = to_tokens(arr, self.plan)
        if self.use_cache:
            self.cache[key] = x
        return x


# =========================================================================== #
#              One full profile (reduced + per-layer partial)                  #
# =========================================================================== #

def run_partial_profile(fit_folders, val_folders, eval_folders, y_list, ys, pairs,
                        d_y, d_by, layers, loader_factory, args, device,
                        spot_layers, with_tau):
    """Protocol Steps 2-4: reduced pass-through baseline + per-layer residual
    regression + τ_ℓ + spot self-check.
    All fitting uses only fit_folders; R² is reported on eval_folders.
    pairs: [{'key', 'xt_name', 'xt_map', 'd_xt', 'layers', 'targets'}] — each
    pair uses its own x_t as the control variable for its targets, effective
    only on its own layers:
      shared: 1 pair, all layers × all targets (single shared x_t);
      target: each pair (x_t_i, [target_i]) across all layers (x_t↔target
              pairwise);
      layer:  each pair (x_t_i, [layer_i]) with all targets (x_t↔layer
              pairwise: the i-th layer is residualized by the i-th x_t, all
              targets computed in full — controlling-variable source-layer
              sweep).
    Output is nested by pair.key: reduced[pair_key][y] / red_cond[pair_key][y] /
    layers[lm]['partial'][pair_key][y] / layers[lm]['tau'][pair_key], etc.
    Equivalence: metrics of any (layer, x_t, target) combination ≡ running
    that combination alone in single-x_t mode (same data, same pipeline, only
    a different organization)."""
    layer_pairs = {lm: [pi for pi, p in enumerate(pairs) if lm in p['layers']]
                   for lm in layers}

    # ---- Step 2: reduced regression (x_t -> y), overdetermined OLS, each pair uses its own x_t ----
    red_grams = {}
    for pi, p in enumerate(pairs):
        for y in p['targets']:
            red_grams[(pi, y)] = GramOLS(p['d_xt'], d_y[y], device)
    for f in fit_folders:
        for pi, p in enumerate(pairs):
            xt_f = p['xt_map'][f.name]
            for y in p['targets']:
                red_grams[(pi, y)].update(xt_f, ys[y][f.name])
    Wc, red_cond = {}, {}
    for (pi, y), g in red_grams.items():
        W, mx, my, cond, _ = g.solve(args.ols_rcond)
        Wc[(pi, y)] = (W, mx, my)
        red_cond.setdefault(pairs[pi]['key'], {})[y] = cond

    def residual(pi, y, f):
        """r = y − x_t·W_c (the pair's own W_c; eval images recompute with their own x_t, protocol Step 4)."""
        W, mx, my = Wc[(pi, y)]
        return (ys[y][f.name].double()
                - (pairs[pi]['xt_map'][f.name].double() - mx) @ W - my).float()

    red_metrics = {}
    for (pi, y) in red_grams:            # dict order-preserving: group output by pair
        model = OLSModel(*Wc[(pi, y)])
        red_metrics.setdefault(pairs[pi]['key'], {})[y] = channel_metrics(
            stream_stats(((pairs[pi]['xt_map'][f.name], ys[y][f.name])
                          for f in eval_folders), model, d_y[y]))

    # ---- Step 3: per-layer residual regression (partial R²) + τ_ℓ + spot joint self-check ----
    layer_res = {}
    for lm in layers:
        t0 = time.time()
        loader = loader_factory(lm)
        is_spot = lm in spot_layers
        need_ols = (args.solver == 'ols') or is_spot
        owner_pis = layer_pairs[lm]
        d_lm = d_by[lm]
        own = (f", x_t={pairs[owner_pis[0]]['xt_name']}"
               if len(owner_pis) == 1 else f", {len(owner_pis)} x_t sharing this layer")
        print(f"  [{lm}] accumulating Grams (fit={len(fit_folders)} images{own})...",
              flush=True)

        # M_xt operator (per pair): W_zt = OLS(x_t → z_ℓ). The strict FWL form
        # requires z and y to be residualized against [the same] x_t (protocol
        # §1 partial R² definition / §4 "controlling covariate ≡ regression after
        # projection"): the protocol's literal R²(z→r) equals it only when z⊥x_t,
        # and underestimates partial when τ_ℓ is high.
        zt_grams = {pi: GramOLS(pairs[pi]['d_xt'], d_lm, device)
                    for pi in owner_pis}
        part_grams = ({(pi, y): GramOLS(d_lm, d_y[y], device)
                       for pi in owner_pis for y in pairs[pi]['targets']}
                      if need_ols else {})
        tau_grams = ({pi: GramOLS(d_lm, pairs[pi]['d_xt'], device)
                      for pi in owner_pis} if with_tau else {})
        joint_grams = ({(pi, y): GramOLS(d_lm + pairs[pi]['d_xt'], d_y[y], device)
                        for pi in owner_pis for y in pairs[pi]['targets']}
                       if is_spot else {})
        ridges = ({(pi, y): pf.ClosedFormRidge(d_lm, d_y[y], device, torch.float32)
                   for pi in owner_pis for y in pairs[pi]['targets']}
                  if args.solver == 'ridge' else {})

        # Pass A0 (fit): M_xt Grams + τ Grams (per pair) + spot joint Grams
        for f in fit_folders:
            z = loader(f)
            for pi in owner_pis:
                xt_f = pairs[pi]['xt_map'][f.name]
                zt_grams[pi].update(xt_f, z)
                if with_tau:
                    tau_grams[pi].update(z, xt_f)
                if is_spot:
                    zx = torch.cat([z, xt_f], dim=1)
                    for y in pairs[pi]['targets']:
                        joint_grams[(pi, y)].update(zx, ys[y][f.name])
        W_zt, mx_zt, mz_zt = {}, {}, {}
        for pi in owner_pis:
            W, mx, mz, _, _ = zt_grams[pi].solve(args.ols_rcond)
            W_zt[pi], mx_zt[pi], mz_zt[pi] = W, mx, mz

        def z_resid(pi, f):
            """Residualize z_ℓ against pair pi's x_t (M_xt operator; eval images recompute with their own x_t)."""
            return (loader(f).double()
                    - (pairs[pi]['xt_map'][f.name].double() - mx_zt[pi]) @ W_zt[pi]
                    - mz_zt[pi]).float()

        # Pass A1 (fit): residual Grams + ridge Grams (input = M_xt z; free when cache is on)
        for f in fit_folders:
            for pi in owner_pis:
                zr = z_resid(pi, f)
                for y in pairs[pi]['targets']:
                    r = residual(pi, y, f)
                    if need_ols:
                        part_grams[(pi, y)].update(zr, r)
                    if args.solver == 'ridge':
                        ridges[(pi, y)].update(zr, r)

        # OLS partial models (default solver family, same family as reduced -> the FWL
        # identity is exact; kept in ridge mode too, for the spot self-check)
        part_ols, part_cond = {}, {}
        if need_ols:
            for (pi, y), g in part_grams.items():
                W, mx, my, cond, _ = g.solve(args.ols_rcond)
                part_ols[(pi, y)] = OLSModel(W, mx, my)
                part_cond[(pi, y)] = cond

        # ridge mode: stream-select γ on val (min MSE in standardized space); grid median when no val
        gamma_used = {}
        if args.solver == 'ridge':
            part_models = {}
            va = list(val_folders)
            for (pi, y) in part_grams:
                Ws, stdz = ridges[(pi, y)].solve(list(args.gamma_grid))
                if va:
                    stdz_dev = stdz.to(device)
                    sse = {g: 0.0 for g in args.gamma_grid}
                    n_tok = 0
                    for f in va:
                        Xs = stdz_dev.apply_x(z_resid(pi, f).to(device))
                        Ys = stdz_dev.apply_y(residual(pi, y, f).to(device))
                        n_tok += Xs.shape[0]
                        for g in args.gamma_grid:
                            diff = Xs @ Ws[g] - Ys
                            sse[g] += float(diff.pow(2).sum())
                    gamma = min(sse, key=lambda g: sse[g] / max(n_tok, 1))
                else:
                    gamma = list(args.gamma_grid)[len(args.gamma_grid) // 2]
                gamma_used[(pi, y)] = gamma
                part_models[(pi, y)] = RidgeModel(Ws[gamma], stdz, device)
        else:
            part_models = part_ols

        # τ_ℓ model (z_ℓ -> x_t pass-through rate, OLS family), solved/evaluated once per pair
        tau_models, tau_conds = {}, {}
        if with_tau:
            for pi in owner_pis:
                W, mx, my, cond, _ = tau_grams[pi].solve(args.ols_rcond)
                tau_models[pi] = OLSModel(W, mx, my)
                tau_conds[pi] = cond

        # Pass C (eval): streamed reporting (generator arguments; whole layers never stay in memory)
        entry = {'partial': {}, 'cond': {}, 'spotcheck': {},
                 'tau': {}, 'tau_cond': {}}
        if args.solver == 'ridge':
            entry['gamma'] = {}
        for pi in owner_pis:
            pkey = pairs[pi]['key']
            entry['partial'][pkey] = {}
            entry['cond'][pkey] = {}
            if args.solver == 'ridge':
                entry['gamma'][pkey] = {}
            for y in pairs[pi]['targets']:
                part_m = channel_metrics(stream_stats(
                    ((z_resid(pi, f), residual(pi, y, f)) for f in eval_folders),
                    part_models[(pi, y)], d_y[y]))
                entry['partial'][pkey][y] = part_m
                entry['cond'][pkey][y] = part_cond[(pi, y)]
                if args.solver == 'ridge':
                    entry['gamma'][pkey][y] = gamma_used[(pi, y)]
                if is_spot:
                    # FWL self-check always uses OLS (the identity is exact for OLS; ridge is only approximate)
                    if args.solver == 'ols':
                        ident_part = part_m
                    else:
                        ident_part = channel_metrics(stream_stats(
                            ((z_resid(pi, f), residual(pi, y, f))
                             for f in eval_folders), part_ols[(pi, y)], d_y[y]))
                    jW, jmx, jmy, jcond, _ = joint_grams[(pi, y)].solve(args.ols_rcond)
                    joint_m = channel_metrics(stream_stats(
                        ((torch.cat([loader(f), pairs[pi]['xt_map'][f.name]], dim=1),
                          ys[y][f.name]) for f in eval_folders),
                        OLSModel(jW, jmx, jmy), d_y[y]))
                    full_id = identity_full_avg(red_metrics[pkey][y], ident_part)
                    entry['spotcheck'].setdefault(pkey, {})[y] = {
                        'r2_full_identity': full_id,
                        'r2_full_direct': joint_m['r2_avg'],
                        'abs_diff': abs(full_id - joint_m['r2_avg']),
                        'joint_cond': jcond,
                    }
            if with_tau:
                entry['tau'][pkey] = channel_metrics(stream_stats(
                    ((loader(f), pairs[pi]['xt_map'][f.name]) for f in eval_folders),
                    tau_models[pi], pairs[pi]['d_xt']))
                entry['tau_cond'][pkey] = tau_conds[pi]
        layer_res[lm] = entry

        items = [(pairs[pi]['key'], y) for pi in owner_pis
                 for y in pairs[pi]['targets']]
        msg = '  '.join(f"{y}_pR2={entry['partial'][pk][y]['r2_avg']:.4f}"
                        for pk, y in items)
        if with_tau:
            if len(owner_pis) == 1:
                msg += f"  tau={entry['tau'][pairs[owner_pis[0]]['key']]['r2_avg']:.4f}"
            else:
                msg += '  ' + '  '.join(
                    f"{pairs[pi]['key']}_tau={entry['tau'][pairs[pi]['key']]['r2_avg']:.4f}"
                    for pi in owner_pis)
        if entry['spotcheck']:
            fwl = max(sc['abs_diff']
                      for ym in entry['spotcheck'].values() for sc in ym.values())
            msg += f"  FWL|Δfull|={fwl:.2e}"
        print(f"  [{lm}] {msg}  ({time.time() - t0:.1f}s)")
        del loader
    return {'reduced': red_metrics, 'red_cond': red_cond, 'layers': layer_res}


# =========================================================================== #
#                                   Main                                      #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='partial R² residualization probe (target2 protocol): reduced '
                    'regression x_t→y -> residual r = y − x_t·W_c -> partial '
                    'R²(ℓ) = R²(z_ℓ→r) + τ_ℓ profile + theoretical sanity '
                    'check + FWL identity self-check',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--feature_root', type=str, required=True,
                   help='feature root directory containing {class}_{image_name}/ folders '
                        '(single cell × single SNR)')
    p.add_argument('--xt_module', type=str, nargs='+', default=None,
                   help='x_t (progressively noised sample) module name, 16-dim per token; '
                        'passing 1 = shared by all layers×targets (auto-discovered by '
                        'alias ' + '/'.join(TARGET_ALIASES['xt']) +
                        ' by default); passing multiple = paired with per-layer '
                        'representations or targets via --xt_pair_by')
    p.add_argument('--xt_pair_by', type=str, default='auto',
                   choices=['auto', 'layer', 'target'],
                   help='pairing mode when multiple --xt_module are given: layer=the '
                        'i-th x_t pairs with the i-th per-layer representation, each '
                        'pair computes all targets in full (controlling-variable '
                        'source-layer sweep); '
                        'target=the i-th x_t pairs with the i-th target, across all '
                        'layers; auto=decide by unique count match (n_xt==n_layers→'
                        'layer, n_xt==n_targets→target, ambiguous double match '
                        'raises)')
    p.add_argument('--targets', type=str, nargs='+',
                   default=['x0', 'x0_LF', 'x0_HF', 'eps'],
                   help='regression targets y (ground-truth signals); x0_LF/x0_HF are '
                        'generated offline from x0 when missing on disk')
    p.add_argument('--band_split', type=str, default='auto',
                   choices=['auto', 'none', 'blur'],
                   help='frequency-band target source: auto=use disk if present, '
                        'else generate from x0 by Gaussian low-pass; none=skip bands; '
                        'blur=force offline recompute from x0 (alignment self-check)')
    p.add_argument('--blur_kernel', type=int, default=5,
                   help='offline band-split Gaussian kernel width (aligned with pipe: k=5)')
    p.add_argument('--blur_sigma', type=float, default=1.0,
                   help='offline band-split Gaussian σ (aligned with pipe: σ=1.0)')
    p.add_argument('--patch_size', type=int, default=2,
                   help='implied patch side of grid-layout x0 (for unpatchify)')
    p.add_argument('--layers', type=str, nargs='+', default=None,
                   help='per-layer representation module names; default auto-selects all '
                        '3-dim modules with the largest channel dim')
    p.add_argument('--input_layout', type=str, default='chw', choices=['hwd', 'chw'],
                   help='npy layout: chw=(c, h, w) channels-first (DiT extraction '
                        'data of this repo); hwd=(h, w, d) channels-last')
    p.add_argument('--max_images', type=int, default=None,
                   help='use only the first N images after sorting by folder name')
    # Held-out protocol
    p.add_argument('--holdout_ratio', type=float, default=0.2,
                   help='held-out image fraction (all R² reported on held-out images)')
    p.add_argument('--val_ratio', type=float, default=0.125,
                   help='validation fraction of (1-holdout) (for ridge γ selection)')
    p.add_argument('--split_seed', type=int, default=0)
    # Solvers
    p.add_argument('--solver', type=str, default='ols', choices=['ols', 'ridge'],
                   help='partial regression solver: ols=closed-form minimum-norm OLS '
                        '(same family as reduced; the FWL identity is exact); '
                        'ridge=standardized ridge regression (same protocol as the '
                        'original ridge profile; the identity is only approximate)')
    p.add_argument('--gamma_grid', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0],
                   help='ridge-mode regularization grid (γ chosen on val)')
    p.add_argument('--ols_rcond', type=float, default=1e-10,
                   help='relative eigenvalue truncation threshold for the OLS Gram')
    # Self-check / byproducts
    p.add_argument('--spotcheck_layers', type=str, nargs='*', default=None,
                   help='layers that run the joint regression [z‖x_t] FWL self-check; '
                        'default auto = one layer each at 1/4, 1/2, 3/4 depth; '
                        'pass none to disable')
    p.add_argument('--no_tau', action='store_true',
                   help='disable the τ_ℓ = R²(z_ℓ→x_t) pass-through-rate profile')
    p.add_argument('--sweep_sizes', type=int, nargs='*', default=[],
                   help='fit-subset sample-size sweep (protocol §3, especially important '
                        'on the ε side); eval/val split stays fixed')
    # α/σ metadata (theoretical sanity check)
    p.add_argument('--snr', type=float, default=None,
                   help='SNR=α²/σ² (VP normalization α²=snr/(1+snr)); for the theoretical check')
    p.add_argument('--lam', type=float, default=None,
                   help='logSNR λ (converted to α/σ with the same formula as lscp.path_coefficients)')
    p.add_argument('--path', type=str, default='vp', choices=['vp', 'fm'],
                   help='diffusion path family (for --lam conversion)')
    p.add_argument('--meta_json', type=str, default=None,
                   help='α/σ metadata json (with an alpha/sigma or lam field), highest priority')
    # Performance
    p.add_argument('--cache_layer', type=str, default='auto',
                   choices=['auto', 'on', 'off'],
                   help='whole-layer feature RAM cache (large speedup when each layer is '
                        'streamed multiple times)')
    p.add_argument('--cache_gb', type=float, default=8.0,
                   help='auto mode enables the cache when a single layer stays under this value (GB)')
    # Misc
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or a specific cuda/mps/cpu')
    p.add_argument('--result_path', type=str,
                   default='probe_partial_r2_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.feature_root)
    if not root.is_dir():
        raise RuntimeError(f'feature root directory does not exist: {root}')

    folders_all = pf.list_image_folders(root)
    if args.max_images is not None:
        folders_all = folders_all[:max(args.max_images, 0)]
        print(f'--max_images={args.max_images}: kept the first {len(folders_all)} images after sorting')
    if not folders_all:
        raise RuntimeError('no usable images after truncation')

    modules = pf.discover_modules(folders_all)
    shapes = scan_module_shapes(folders_all, modules)
    layers, grid = resolve_layers(modules, shapes, args.input_layout, args.layers)
    T = grid[0] * grid[1]
    print(f'images: {len(folders_all)} | modules found: {len(modules)} | '
          f'layout: {args.input_layout} | token grid: {grid} (T={T})')
    print(f'per-layer representations ({len(layers)} layers): {layers}')

    # ---- x_t: the control variable to residualize against (hard protocol requirement, 16-dim per token) ----
    # 1 name = shared by all layers×targets (auto-discovered by alias by default);
    # multiple names = paired with per-layer representations or regression targets via
    # --xt_pair_by (the decision is made after the target list is final; see the pairs
    # construction below)
    if args.xt_module:
        xt_names = []
        for n in args.xt_module:
            if n not in modules:
                raise RuntimeError(f'--xt_module {n} not among candidates: {sorted(modules)}')
            xt_names.append(n)
    else:
        found = next((a for a in TARGET_ALIASES['xt'] if a in modules), None)
        if found is None:
            raise RuntimeError(
                'cannot find the x_t (progressively noised sample) representation module '
                '— a hard requirement of this protocol.\n'
                '  Add an x_t extraction point to the extraction pipeline (same forward, '
                'same noise realization as the per-layer representations/eps/x0/pred: '
                'x_t = α·x0 + σ·ε, 16-dim per token), save it, and specify the module '
                'name with --xt_module; you can also pass multiple module names paired '
                'via --xt_pair_by.')
        xt_names = [found]

    # ---- Regression targets y (ground-truth signals; bands can be generated offline) ----
    y_list, made_bands = [], []
    for t in list(dict.fromkeys(args.targets)):
        if t in ('xt', 'x_t'):
            print(f'[note] target {t} is the control variable being residualized itself; skipped')
            continue
        hit = next((a for a in TARGET_ALIASES.get(t, []) if a in modules), None)
        if t in modules:
            if t in ('x0_LF', 'x0_HF') and args.band_split == 'blur':
                made_bands.append(t)          # force offline recompute (alignment self-check)
            else:
                y_list.append(t)
        elif hit:
            y_list.append(hit)
        elif t in ('x0_LF', 'x0_HF'):
            if args.band_split == 'none':
                print(f'[note] target {t}: --band_split=none, skipped')
            elif 'x0' in modules:
                made_bands.append(t)
            else:
                print(f'[note] target {t}: no x0 module found, cannot generate the band offline; skipped')
        else:
            print(f'[note] target {t} not among candidates: {sorted(modules)}; skipped')
    if not y_list and not made_bands:
        raise RuntimeError(f'no usable regression targets (candidates: {sorted(modules)})')
    # ---- Tokenization plans: build only for modules participating in this protocol
    # (layers + x_t + targets; offline band generation additionally needs x0). Other
    # modules in feature_root (other points / reference features, e.g. 37x37 DINOv2
    # references) may have different spatial grids and must not break the whole
    # protocol — skip them and print their first-image shapes for verification,
    # avoiding unrelated modules failing with "mismatches the token grid".
    used = list(dict.fromkeys(layers + xt_names + y_list))
    if made_bands and 'x0' in modules:
        used.append('x0')
    plans = {}
    for m in used:
        plans[m] = build_module_plan(shapes[m], args.input_layout, grid, name=m)
    d_by = {m: tok_dim(plans[m]) for m in used}
    skipped = [m for m in modules if m not in used]
    if skipped:
        print(f'{len(skipped)} modules not participating in this protocol (tokenization skipped, first-image shapes):')
        for m in skipped:
            print(f'  {m}: {shapes[m]}')
    print('module token dims: ' + '  '.join(f'{m}={d_by[m]}' for m in used
                                        if m not in layers))
    d_y = {t: d_by[t] for t in y_list}
    if made_bands:
        print(f"band targets {made_bands}: generated offline from x0 by "
              f"gaussian_blur(k={args.blur_kernel}, σ={args.blur_sigma}) "
              f"(x0 = LF + HF, aligned with the pipe band-split definition)")
    print(f'regression targets: {y_list + made_bands} | x_t modules: {xt_names}')

    # Preload small targets (resident in memory: T*16*4B/image ≈ 65KB, ~65MB for 1000 images)
    t0 = time.time()
    ys = {t: preload_tokens(folders_all, t, plans[t], shapes[t]) for t in y_list}
    if made_bands:
        x0_map = ys['x0'] if 'x0' in ys else preload_tokens(
            folders_all, 'x0', plans['x0'], shapes['x0'])
        low, high = build_band_offline(x0_map, plans['x0'], grid, args)
        for t, m in (('x0_LF', low), ('x0_HF', high)):
            if t in made_bands:
                ys[t] = m
                d_y[t] = d_by['x0']
                y_list.append(t)
    print(f'small-target preload done ({time.time() - t0:.1f}s)')

    # ---- x_t preload and pairing (three modes) ----
    # shared: 1 shared x_t, all layers × all targets;
    # layer:  the i-th x_t pairs with the i-th per-layer representation, each pair
    #         computes all targets in full (controlling-variable source-layer sweep:
    #         each layer's z is residualized only by its own x_t);
    # target: the i-th x_t pairs with the i-th target, across all layers (each target
    #         uses its own control variable).
    xt_cache = {}

    def xt_map_of(name):
        if name not in xt_cache:
            xt_cache[name] = preload_tokens(folders_all, name, plans[name],
                                            shapes[name])
        return xt_cache[name]

    n_xt, n_l, n_y = len(xt_names), len(layers), len(y_list)
    if n_xt == 1:
        if args.xt_pair_by != 'auto':
            raise RuntimeError('--xt_pair_by only matters when multiple --xt_module are '
                               'given (a single x_t is always the shared mode)')
        pair_mode = 'shared'
    elif args.xt_pair_by == 'auto':
        if n_xt == n_l and n_xt != n_y:
            pair_mode = 'layer'
        elif n_xt == n_y and n_xt != n_l:
            pair_mode = 'target'
        elif n_xt == n_y == n_l:
            raise RuntimeError(
                f'--xt_module count {n_xt} equals both the layer count {n_l} and the '
                f'target count {n_y}; the pairing mode is ambiguous, specify '
                f'--xt_pair_by layer|target explicitly '
                f'(layer=the i-th x_t pairs with the i-th layer; target=the i-th x_t '
                f'pairs with the i-th target)')
        else:
            raise RuntimeError(
                f'--xt_module given {n_xt} modules {xt_names}, matching neither the '
                f'per-layer representation count {n_l} {layers} nor the regression '
                f'target count {n_y} {y_list}; cannot pair.\n'
                f'  layer mode requires n_xt == n_layers; target mode requires '
                f'n_xt == n_targets (note xt/x_t in --targets are skipped and missing '
                f'bands are generated offline and appended), or pass 1 x_t shared by '
                f'all layers×targets.')
    else:
        pair_mode = args.xt_pair_by
        if pair_mode == 'layer' and n_xt != n_l:
            raise RuntimeError(f'--xt_pair_by layer requires the --xt_module count ({n_xt}) '
                               f'to match the per-layer representation count ({n_l})')
        if pair_mode == 'target' and n_xt != n_y:
            raise RuntimeError(f'--xt_pair_by target requires the --xt_module count ({n_xt}) '
                               f'to match the regression target count ({n_y}): {y_list}')

    if pair_mode == 'shared':
        pairs = [{'key': 'shared', 'xt_name': xt_names[0],
                  'xt_map': xt_map_of(xt_names[0]), 'd_xt': d_by[xt_names[0]],
                  'layers': list(layers), 'targets': list(y_list)}]
        print(f"control variable: {xt_names[0]} (d={d_by[xt_names[0]]}, "
              f"shared by all layers × all targets)")
    elif pair_mode == 'layer':
        print('pairwise mode (layer): the i-th x_t pairs with the i-th per-layer '
              'representation, each pair computing all targets in full '
              '(independent reduced/residualization/partial/τ each):')
        pairs = []
        for name, lm in zip(xt_names, layers):
            pairs.append({'key': lm, 'xt_name': name, 'xt_map': xt_map_of(name),
                          'd_xt': d_by[name], 'layers': [lm],
                          'targets': list(y_list)})
            print(f'  {lm} <- {name} (d={d_by[name]})')
    else:
        print('pairwise mode (target): each target uses its own x_t control variable, '
              'across all layers (independent reduced/residualization/partial/τ each):')
        pairs = []
        for name, y in zip(xt_names, y_list):
            pairs.append({'key': y, 'xt_name': name, 'xt_map': xt_map_of(name),
                          'd_xt': d_by[name], 'layers': list(layers),
                          'targets': [y]})
            print(f'  {y:<8} <- {name} (d={d_by[name]})')

    # ---- α/σ metadata (theoretical sanity check) ----
    alpha, sigma = resolve_alpha_sigma(args)
    if alpha is not None:
        print(f'path coefficients: α={alpha:.4f} (α²={alpha ** 2:.4f})  σ={sigma:.4f} '
              f'(σ²={sigma ** 2:.4f})  <- theoretical sanity-check baseline')
    else:
        print('[note] no α/σ given (--snr/--lam/--meta_json); skipping the theoretical sanity check')

    # ---- Split / device / cache ----
    splits, degen = stratified_fit_val_holdout(folders_all, args.holdout_ratio,
                                               args.val_ratio, args.split_seed)
    print(f"split: fit={len(splits['fit'])}  val={len(splits['val'])}  "
          f"holdout={len(splits['holdout'])}")
    device = pf.pick_device(args.device)
    print(f'device: {device}')
    d_max = max(d_by[m] for m in layers)
    est_gb = len(folders_all) * T * d_max * 4 / 1024 ** 3
    use_cache = (args.cache_layer == 'on' or
                 (args.cache_layer == 'auto' and est_gb <= args.cache_gb))
    print(f'whole-layer RAM cache: {"on" if use_cache else "off"} '
          f'(single-layer estimate {est_gb:.2f} GB, threshold {args.cache_gb} GB)')

    # spot layers: default auto = one layer each at 1/4, 1/2, 3/4 depth
    if args.spotcheck_layers:
        if args.spotcheck_layers == ['none']:
            spot_layers = []
        else:
            spot_layers = [m for m in dict.fromkeys(args.spotcheck_layers)
                           if m in layers]
            missing = [m for m in args.spotcheck_layers
                       if m not in layers and m != 'none']
            if missing:
                raise RuntimeError(f'--spotcheck_layers not in the per-layer list: {missing}')
    else:
        idx = sorted({len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4})
        spot_layers = [layers[i] for i in idx]
    with_tau = not args.no_tau
    print(f'FWL self-check spot layers: {spot_layers} | τ_ℓ profile: {"on" if with_tau else "off"}')

    loader_factory = lambda lm: LayerLoader(folders_all, lm, plans[lm],
                                            use_cache, shapes[lm])
    width = max([len(m) for m in layers] + [8]) + 2

    # ---- Steps 2-4: reduced baseline + per-layer partial R² ----
    print(f"\n==== per-layer profile (solver={args.solver}, R² reported on holdout) ====")
    profile = run_partial_profile(splits['fit'], splits['val'], splits['holdout'],
                                  y_list, ys, pairs, d_y, d_by, layers,
                                  loader_factory, args, device, spot_layers,
                                  with_tau)

    # ---- reduced baseline + theoretical sanity check (protocol Step 2) ----
    print('\n===== reduced pass-through baseline (x_t→y, holdout) and theoretical check =====')
    for pkey, ym in profile['reduced'].items():
        xt_label = next(p['xt_name'] for p in pairs if p['key'] == pkey)
        for y, r in ym.items():
            msg = (f"  {xt_label}→{y:<7} R2_avg={r['r2_avg']:.4f}  "
                   f"R2_var={r['r2_var']:.4f}  "
                   f"cond={profile['red_cond'][pkey][y]:.2e}")
            theory = ('x0', alpha ** 2) if (y == 'x0' and alpha is not None) else \
                     ('eps', sigma ** 2) if (y == 'eps' and sigma is not None) else None
            if theory:
                dev = abs(r['r2_avg'] - theory[1])
                flag = ', [clear deviation >0.02] check latent normalization/patch alignment first!' \
                    if dev > 0.02 else ''
                msg += f"  [theory {theory[0]}²={theory[1]:.4f}, |Δ|={dev:.4f}{flag}]"
            print(msg)

    # ---- Per-layer summary table ----
    print('\n===== per-layer summary (holdout; pR2=partial R², tau=pass-through rate τ_ℓ) =====')
    if pair_mode == 'target':
        # Each layer spans all pairs (pair.key == target name): one pR2/tau column pair per (target)
        head = '  ' + 'layer'.ljust(width) + ''.join(
            f'{y}:pR2'.rjust(13) + f'{y}:tau'.rjust(13) for y in y_list)
        print(head)
        for lm in layers:
            e = profile['layers'][lm]
            row = '  ' + lm.ljust(width)
            for y in y_list:
                row += f"{e['partial'][y][y]['r2_avg']:.4f}".rjust(13)
                row += f"{e['tau'][y]['r2_avg']:.4f}".rjust(13)
            print(row)
    else:
        # shared/layer: each layer has exactly 1 pair (layer mode pkey=the paired layer name)
        head = '  ' + 'layer'.ljust(width) + ''.join(y.rjust(12) for y in y_list)
        if with_tau:
            head += 'tau'.rjust(11)
        print(head)
        for lm in layers:
            e = profile['layers'][lm]
            pkey = next(iter(e['partial']))
            row = '  ' + lm.ljust(width)
            for y in y_list:
                row += (f"{e['partial'][pkey][y]['r2_avg']:.4f}").rjust(12)
            if with_tau:
                row += (f"{e['tau'][pkey]['r2_avg']:.4f}").rjust(11)
            print(row)
    for lm, e in profile['layers'].items():
        for pkey, ym in e['spotcheck'].items():
            for y, sc in ym.items():
                print(f"  [FWL check {lm}/{pkey}/{y}] R²_full identity back-solve="
                      f"{sc['r2_full_identity']:.4f}  "
                      f"joint regression direct={sc['r2_full_direct']:.4f}  "
                      f"|Δ|={sc['abs_diff']:.2e}")

    # ---- Optional: fit sample-size sweep (protocol §3) ----
    sweep_res = {}
    if args.sweep_sizes:
        rng_sweep = np.random.RandomState(args.split_seed + 1000)
        fit_names = [f.name for f in splits['fit']]
        perm = rng_sweep.permutation(len(fit_names))
        for s in args.sweep_sizes:
            s_eff = min(s, len(fit_names))
            if s_eff < s:
                print(f'[warn] sweep {s} exceeds the fit set {len(fit_names)}; running with {s_eff}')
            take = {fit_names[i] for i in perm[:s_eff]}
            sub = [f for f in splits['fit'] if f.name in take]
            print(f"\n==== sample-size sweep n_fit={s_eff} (eval/val unchanged) ====")
            sub_prof = run_partial_profile(sub, splits['val'], splits['holdout'],
                                           y_list, ys, pairs, d_y, d_by, layers,
                                           loader_factory, args, device,
                                           spot_layers, with_tau)
            # shared/layer has 1 pair per layer; target's pkey==y: keys degenerate to target names
            sweep_res[str(s)] = {
                lm: {y: m['r2_avg']
                     for pk, ym in e['partial'].items() for y, m in ym.items()}
                for lm, e in sub_prof['layers'].items()}
            for lm in layers:
                row = '  ' + lm.ljust(width)
                for y in y_list:
                    row += (f"{sweep_res[str(s)][lm][y]:.4f}").rjust(12)
                print(row)

    # ---- Save ----
    out = {
        'protocol': 'target2: partial R2 residualization (probe_partial_r2.py)',
        'feature_root': str(root),
        'n_images': len(folders_all),
        'grid': list(grid),
        'layers': layers,
        'pair_mode': pair_mode,
        'xt_module': pairs[0]['xt_name'] if pair_mode == 'shared' else None,
        'xt_modules': xt_names,
        'pairs': [{'key': p['key'], 'xt_module': p['xt_name'],
                   'layers': p['layers'], 'targets': p['targets']}
                  for p in pairs],
        'targets': y_list,
        'bands_generated_offline': made_bands,
        'module_token_dims': d_by,
        'alpha_sigma': None if alpha is None else
                       {'alpha': alpha, 'sigma': sigma},
        'splits': {k: [f.name for f in v] for k, v in splits.items()},
        'degenerate_split': degen,
        'args': vars(args),
        'profile': profile,
        'sweep_sizes': sweep_res or None,
    }
    with open(args.result_path, 'w') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f'\nresults saved to: {args.result_path}')


if __name__ == '__main__':
    main()
