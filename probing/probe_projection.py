#!/usr/bin/env python3
"""
Output-code projection probe (P_⊥ projection protocol, ref/target1.txt)
=======================================================================

One experiment answers the three questions of protocol §0:
  Q1  is the late-stage corres drop caused by output codes crowding the
      nearest-neighbor metric -> probe recovery magnitude after projection
  Q2  premise of the "alignment" explanation (new codes polluting the
      metric) -> consistent recovery across three cells (VP+eps/x0/u)
  Q3  two-component independence + quantitative decomposition of the
      pass-through part -> residual level of x0-linear after projection

Protocol essentials (per-layer independent, protocol §2):
  Step 1  W = OLS(Z_ℓ -> code): unregularized [minimum-norm] solution (the
          unique OLS solution when overdetermined n≫d; ridge forbidden —
          shrinkage leaves output-code residue in z_⊥, pitfall 1), centered
          fit (pitfall 2), float64 Gram accumulation (pitfall 3).
          S = orth(W) = orthonormal basis of the output-code readout column
          space (SVD truncates near-zero singular directions, pitfall 6);
          fit independently per layer, never reused across layers (pitfall 4).
  Step 2  P_⊥ = I − S Sᵀ,  z_⊥ = z_ℓ @ P_⊥ (dimension unchanged, per token).
          [Offline equivalence] the projection is a per-token channel-wise
          linear map, applied directly to the [on-disk] npy features;
          mathematically equivalent to "injecting P_⊥ in a hook during a
          second forward" (all probes are per-token regressions or GAP
          aggregation) — hence this script costs zero forwards, layer by layer.
  Step 3  Sanity checks (mandatory):
          SC1  refit OLS z_⊥ -> code on the held-out set; R² ≈ 0
               (< --sc1_threshold); if it fails, --expand_topk can re-verify
               with the subspace expanded by [W + top residual-SVD directions]
          SC2  projected energy profile ||SSᵀz||²/||z||² reported per layer
               (expected ≤ a few percent)
          SC3  --random_control: project an equal number of [random]
               orthonormal directions as a control (rules out the trivial
               "removing any 16 dims helps" effect; the corres-side random
               control is done in the external pipeline)
  Step 4  Re-run probes after projection (same protocol as the original
          linear probes):
          x0-linear / eps-linear : standardized ridge closed-form solution
                                   + γ selected on the validation set
          classification         : GAP + multi-class ridge regression
                                   (reusing probe_classification)
          corres                 : not in this script — use --save_basis to
          export each layer's S (npz), or --export_dir to export projected
          feature npy for the external corres pipeline (the same post-
          processing must be used before/after projection, protocol §2 Step 4).

Data organization (same as probe_features.py; this repo's DiT extraction
data uses --input_layout chw):

    feature_root/
        {class}_{image_name}/
            {per-layer repr}.npy  # (1152, gh, gw) chw, token sequence from rearrange
            {code}.npy        # model final-layer-out (pred/eps_hat/x0_hat/...),
                              # (4, 2gh, 2gw) latent or (16, gh, gw) token grid
            {x0,eps}.npy      # linear-probe targets, same layout (auto patchify alignment)

Held-out protocol (§2 Step 1): class-stratified three-way split fit / val /
holdout (default 0.7/0.1/0.2, 1000 images -> 700/100/200). W and the SC1
readout are fit on fit+val (the protocol's 800 fitting images); linear probes
fit on fit, select γ on val, report on holdout — all probe numbers come from
held-out images that never participated in fitting W (pitfall 5).

Usage (protocol §7 pilot: single cell × SNR≈9 × all 28 layers, half a day to
verify the SC1-SC3 full chain):
  python probe_projection.py --feature_root out/dit-eps_snr9 \
      --code_target pred --targets x0 eps --result_path proj_eps_snr9.json

target4 dual-code mode (--code_target with 2 codes):
  python probe_projection.py --feature_root out/dit-x0_snr9 \
      --code_target x0_hat eps_hat --targets x0 eps \
      --result_path proj_target4.json
  # Full projection evaluation for each code separately, plus: per-layer
  # principal-angle spectrum (svd(Q_aᵀQ_b), mean cos² vs the random baseline
  # max(ra,rb)/d ≈ 16/1152) + the 2×2 cross-ablation
  # ({drop-code-a-subspace, drop-code-b-subspace} × {measure-a, measure-b},
  # regression target = the code itself, unprojected readability as the
  # reference row): entangled latent stage = off-diagonal cells drop along
  # with the diagonal; decoupled = diagonal zeroed, off-diagonal cells stay
  # at baseline.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import probe_features as pf
import probe_classification as pc


# =========================================================================== #
#              Module discovery / shape scanning / tokenization                #
# =========================================================================== #

# code target = the model's own output code (final-layer-out);
# parameterization-named aliases allowed
CODE_ALIASES = ['pred', 'out', 'eps_hat', 'x0_hat', 'u_hat', 'v_hat', 'final_out']
# linear-probe targets = ground-truth signals; only exact names / underscore
# variants accepted (never confuse x0 with x0_hat etc.)
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
    preload_tokens take the reference shape and raise on mismatch)."""
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
    """Storage shape -> (c, h, w) semantics (3-dim) or None (2-dim (T, d) is already a token sequence)."""
    if len(shape) == 2:
        return None
    if layout == 'chw':
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    return (int(shape[2]), int(shape[0]), int(shape[1]))


def resolve_layers(modules, shapes, layout, explicit=None):
    """Determine per-layer representation modules and the token grid (gh, gw).
    Auto rule: 3-dim modules whose channel dim equals the global max channel
    dim (DiT: 1152 ≫ 16/4)."""
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
    cands = {}
    for m in modules:
        c_hw = chw_of(shapes[m], layout)
        if c_hw is not None:
            cands[m] = c_hw
    if not cands:
        raise RuntimeError('no 3-dim feature modules, cannot determine the token grid (use explicit --layers)')
    c_max = max(c for c, _, _ in cands.values())
    grids = {(h, w) for c, h, w in cands.values() if c == c_max}
    if len(grids) > 1:
        print(f'[warn] modules with max channel dim {c_max} have inconsistent '
              f'spatial grids {sorted(grids)}; using the grid of the first module in the union')
    layers = sorted(m for m, (c, _, _) in cands.items() if c == c_max)
    grid = (cands[layers[0]][1], cands[layers[0]][2])
    layers = [m for m in layers if cands[m][1:] == grid] or [layers[0]]
    return layers, grid


def build_module_plan(shape, layout, grid, name='?'):
    """Storage shape -> tokenization plan ('grid'|'patch'|'2d', params). The
    '2d' param is (d,).
    'grid' : spatial size == token grid, d=c per token;
    'patch': spatial size == 2×token grid (latent), d=c*p*p per token (same
             token order and in-channel order as lscp.patchify). name is only
             for error localization."""
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
    """npy array -> (T, d_tok) fp32; token order (h, w) row-major, same as patchify."""
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    kind, g = plan
    if kind == '2d':
        return x.contiguous()
    c, gh, gw, p, _ = g
    if kind == 'grid':
        return x.reshape(c, gh * gw).T.contiguous()          # (T, c)
    x = x.reshape(c, gh, p, gw, p).permute(1, 3, 0, 2, 4)    # (gh, gw, c, p, p)
    return x.reshape(gh * gw, c * p * p).contiguous()        # (T, c*p*p)


def resolve_code_target(name, modules):
    """Resolve the code target: exact name first, then the model-output alias class."""
    if name in modules:
        return name
    for a in CODE_ALIASES:
        if a in modules:
            print(f'[note] code target {name} not found, using {a}')
            return a
    raise RuntimeError(f'code target {name} not among candidates: {modules} '
                       f'(use --code_target to name the model output-code module)')


def resolve_target(name, modules):
    """Resolve the linear-probe target (ground-truth signal; no aliasing to model estimates)."""
    if name in modules:
        return name
    for a in TARGET_ALIASES.get(name, []):
        if a in modules:
            return a
    raise RuntimeError(f'probe target {name} not among candidates: {modules}')


# =========================================================================== #
#                       Split (fit/val/holdout)                                #
# =========================================================================== #

def stratified_fit_val_holdout(folders, holdout_ratio, val_ratio, seed):
    """Class-stratified three-way split: cut holdout first, then val from the
    remainder; val takes val_ratio of what remains.
    Returns ({'fit': [...], 'val': [...], 'holdout': [...]}, whether degenerate)."""
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
              f'holdout={len(hold)} (empty sets fall back to fit; metrics are reference only)')
    if not hold:
        hold = list(fit)
    return {'fit': fit, 'val': val, 'holdout': hold}, degen


# =========================================================================== #
#             OLS (min-norm) and projection basis (protocol pitfalls 1/2/3)     #
# =========================================================================== #

class GramOLS:
    """Streamed OLS sufficient statistics (centered, float64 accumulation).
    Per-image Grams are computed in fp32 on the device then merged into a CPU
    float64 accumulator (protocol pitfall 3: read in fp16/fp32, accumulate in
    float64; MPS does not support fp64, and the per-image merge rounding error
    is a relative ~1e-6, negligible). solve returns the [minimum-norm] OLS
    solution: unique when overdetermined; near-zero eigenvalues truncated when
    ill-conditioned (pitfall 1: never add ridge — shrinkage leakage leaves
    output-code correlation after projection)."""

    def __init__(self, d_in, d_out, device):
        self.device, self.d_in, self.d_out = device, d_in, d_out
        self.G = torch.zeros(d_in, d_in, dtype=torch.float64)   # Σ xᵀx (CPU fp64)
        self.B = torch.zeros(d_in, d_out, dtype=torch.float64)  # Σ xᵀy
        self.sx = torch.zeros(d_in, dtype=torch.float64)        # Σ x
        self.sy = torch.zeros(d_out, dtype=torch.float64)       # Σ y
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
        """Centering identity: G_c = G − n·μμᵀ, B_c = B − n·μ_x μ_yᵀ.
        Returns (W, mu_x, mu_y, condition number, effective rank)."""
        n = max(self.n, 1)
        mu_x = self.sx / n
        mu_y = self.sy / n
        Gc = self.G - n * torch.outer(mu_x, mu_x)
        Bc = self.B - n * torch.outer(mu_x, mu_y)
        evals, evecs = torch.linalg.eigh(Gc)                    # CPU float64
        emax = float(evals.max().clamp_min(1e-30))
        keep = evals > emax * rcond
        inv = torch.zeros_like(evals)
        inv[keep] = 1.0 / evals[keep]
        W = evecs @ (inv[:, None] * (evecs.T @ Bc))             # min-norm OLS
        cond = emax / float(evals[keep].min()) if bool(keep.any()) else float('inf')
        return W, mu_x, mu_y, cond, int(keep.sum())

    def cross_cov(self):
        """Centered cross-covariance B_c = Σxᵀy − n·μ_x μ_yᵀ (CPU float64).
        Projection-nulling criterion: Cov(z_⊥, y) = (I−SSᵀ)B_c, hence
        S = orth(B_c) nulls it exactly.
        (Previously S = orth(OLS W = Sxx⁻¹B_c) was used: under an ill-conditioned
        spectrum, whitening by G rotates the subspace; in low-variance
        directions W's estimation noise ∝1/√λ dominates the column space,
        leaving residual cross-covariance after projection -> SC1 fails.
        The failure was reproduced on synthetic data (known ground-truth
        column space); orth(B_c) nulls it in one round.)"""
        n = max(self.n, 1)
        return self.B - n * torch.outer(self.sx / n, self.sy / n)


def orth_basis(W, rcond=1e-6):
    """Orthonormal basis S (d, r) of the column space of W (d, k) + singular
    values. SVD truncates near-zero singular directions (protocol pitfall 6:
    automatic rank reduction when individual output-code dims have very low
    variance)."""
    W = W.detach().to(torch.float64).cpu()
    if W.numel() == 0:
        return W, torch.zeros(0, dtype=torch.float64)
    U, s, _ = torch.linalg.svd(W, full_matrices=False)
    r = int((s > float(s[0]) * rcond).sum())
    return U[:, :r].contiguous(), s[:r]


def project_tokens(z, S32, device):
    """z (T, d) -> (z_⊥ (T, d) cpu, projected energy, total energy).
    With S32=None (the unprojected baseline of the target4 2×2), returns
    as-is with energy 0."""
    zd = z.to(device)
    if S32 is None:
        return z, 0.0, float(zd.pow(2).sum())
    zs = zd @ S32
    zp = zd - zs @ S32.T
    return zp.cpu(), float(zs.pow(2).sum()), float(zd.pow(2).sum())


def principal_angles(Sa, Sb):
    """Principal-angle cosine spectrum of two subspace bases (target4): the
    singular values of svd(SaᵀSb), length min(ra, rb); each value = the
    cosine of one principal angle. Random baseline of mean cos² =
    max(ra, rb)/d (E[Σcos²] = ra·rb/d divided by the min(ra,rb) principal
    angles; when ra=rb=k it is k/d — two 16-dim subspaces in 1152 dims give
    16/1152 ≈ 0.014). Only a spectrum clearly above the baseline counts as
    “shared directions”. Returns (cos spectrum fp64 CPU, mean cos²)."""
    M = (Sa.detach().double().cpu().T
         @ Sb.detach().double().cpu())          # (ra, rb)
    s = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    return s, float((s ** 2).mean())


class OLSModel:
    """Centered OLS linear readout: ŷ = (x − μ_x) @ W + μ_y (CPU float64)."""

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


# =========================================================================== #
#                  Streamed evaluation (per-channel R² + cosine)               #
# =========================================================================== #

def new_stats(d_out):
    """Raw-scale sufficient-statistics container (isomorphic to pf.evaluate)."""
    zeros = lambda: torch.zeros(d_out, dtype=torch.float64)
    return {'sum_p': zeros(), 'sum_p2': zeros(), 'sum_y': zeros(),
            'sum_y2': zeros(), 'sum_py': zeros(), 'sum_cos': 0.0, 'n': 0}


def acc_stats(st, P, Y):
    """Merge one batch of predictions/targets into the sufficient statistics
    (P/Y: (n, d_out) or reshapable)."""
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


def channel_metrics(st):
    """Recover metrics from the sufficient statistics; semantics match
    pf.metrics_from_stats (mean per-channel R² is the primary metric); also
    returns the per-channel R² array for per-channel use in the partial-R²
    identity."""
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


# =========================================================================== #
#              Data loading (small targets resident + layer cache)             #
# =========================================================================== #

def preload_tokens(folders, module, plan, shape=None):
    """Preload small-target modules: {folder_name: (T, d_tok)} fp32 CPU.
    shape: first-image shape from scan_module_shapes, for on-load cross-image
    consistency checks."""
    out = {}
    for f in folders:
        arr = np.load(f / f'{module}.npy')
        if shape is not None:
            check_shape(arr, module, f.name, shape)
        out[f.name] = to_tokens(arr, plan)
    return out


class LayerLoader:
    """Lazy loading of single-layer features + optional whole-layer in-memory
    cache (fp32, (T, d)); discarded after each layer is processed.
    shape: first-image reference shape, checked for cross-image consistency
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
#                    GAP classification (reuses the classification probe)      #
# =========================================================================== #

def gap_classify(X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes, gamma_grid, device):
    """GAP features -> closed-form multi-class ridge regression (primal/dual
    adaptive), γ selected on validation top-1."""
    X_tr_s = torch.stack(X_tr)
    mu, inv_std = pc.standardize_fit(X_tr_s)
    dim = X_tr_s.shape[1]

    def std(X):
        if not len(X):
            return torch.zeros(0, dim)
        Xs = X if isinstance(X, torch.Tensor) else torch.stack(X)
        return (Xs - mu) * inv_std

    Xs_tr, Xs_va, Xs_te = std(X_tr), std(X_va), std(X_te)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    mode = 'primal' if X_tr_s.shape[1] <= X_tr_s.shape[0] else 'dual'
    ridge = pc.MultiClassRidge(Xs_tr, pc.onehot(y_tr, n_classes).float(), mode)
    Ws = ridge.solve(gamma_grid, device)
    gamma = pc.select_gamma_clf(Ws, ridge.intercept, Xs_va, y_va_t, device)
    scores = pc.scores_of(Xs_te, Ws[gamma], ridge.intercept, device)
    acc = pc.acc_topk(scores, y_te_t)
    return {'solver': f'closed({mode}, γ={gamma:g})', 'gamma': float(gamma),
            'n_test': int(len(y_te)), **acc}


# =========================================================================== #
#                 Per-layer evaluation flow for a single basis                  #
# =========================================================================== #

def fit_eval_basis(loader, S64, splits, code, ys, targets, labels, n_classes,
                   use_cls, want_probes, args, device, d, d_code, d_y):
    """For a given orthonormal basis S: streamed fit on fit/val (SC1 OLS +
    ridge probes + GAP rows) -> γ selection on val -> holdout evaluation,
    while accumulating the SC2 energy profile. Returns the result dict.
    With S64=None (the unprojected baseline row of the target4 dual-code
    mode), no projection is applied and energy is recorded as 0."""
    Sd = None if S64 is None else S64.to(torch.float32).to(device)
    ener = {'fit': [0.0, 0.0], 'val': [0.0, 0.0], 'holdout': [0.0, 0.0]}
    n_tok = {'fit': 0, 'val': 0, 'holdout': 0}     # token counts (denominator for absolute quantities)

    def project(z):
        """z (T, d) -> (z_⊥ (T, d) cpu, projected energy, total energy)."""
        return project_tokens(z, Sd, device)

    # ---- Pass B (fit): SC1 Gram + probe Grams + GAP rows ----
    gram_sc1 = GramOLS(d, d_code, device)
    ridges = {t: pf.ClosedFormRidge(d, d_y[t], device, torch.float32)
              for t in targets} if want_probes else {}
    gap_tr, lab_tr = [], []
    for f in splits['fit']:
        zp, en, den = project(loader(f))
        ener['fit'][0] += en
        ener['fit'][1] += den
        n_tok['fit'] += zp.shape[0]
        gram_sc1.update(zp, code[f.name])
        if want_probes:      # with the SC3 sc1 control (want_probes=False), no probes are built; skip updates
            for t in targets:
                ridges[t].update(zp, ys[t][f.name])
        if use_cls:
            gap_tr.append(zp.mean(0))
            lab_tr.append(labels[f.name])

    # Solve ridge probes in one shot (γ grid in parallel)
    solved = {t: ridges[t].solve(args.gamma_grid) for t in targets} \
        if want_probes else {}

    # ---- Pass B2 (val): γ selection (validation MSE) + SC1 Gram supplement (fit+val protocol) ----
    sse = {t: {g: 0.0 for g in args.gamma_grid} for t in targets}
    cnt = {t: 0 for t in targets}
    gap_va, lab_va = [], []
    for f in splits['val']:
        zp, en, den = project(loader(f))
        ener['val'][0] += en
        ener['val'][1] += den
        n_tok['val'] += zp.shape[0]
        gram_sc1.update(zp, code[f.name])
        if want_probes:
            for t in targets:
                Ws, stdz = solved[t]
                stdz_d = stdz.to(device)
                Xs = stdz_d.apply_x(zp.to(device))
                Ys = stdz_d.apply_y(ys[t][f.name].to(device))
                for g in args.gamma_grid:
                    sse[t][g] += (Xs @ Ws[g] - Ys).pow(2).sum().item()
                cnt[t] += Ys.numel()
        if use_cls:
            gap_va.append(zp.mean(0))
            lab_va.append(labels[f.name])

    W_sc1, mx, my, _, _ = gram_sc1.solve(args.ols_rcond)   # fit+val (the protocol's 800 images)
    sc1_model = OLSModel(W_sc1, mx, my)

    gamma_pick = {}
    for t in targets:
        if want_probes and splits['val']:
            gamma_pick[t] = min(sse[t], key=sse[t].get)
        else:  # no validation set: grid median
            gamma_pick[t] = sorted(args.gamma_grid)[len(args.gamma_grid) // 2]
    models = {t: RidgeModel(solved[t][0][gamma_pick[t]], solved[t][1], device)
              for t in targets} if want_probes else {}

    # ---- Pass C (holdout): one streamed scan feeds SC1 + all probes + GAP rows at once ----
    eval_folders = splits['holdout'] if splits['holdout'] else splits['fit']
    degen_eval = not splits['holdout']
    st_sc1 = new_stats(d_code)
    st_probe = {t: new_stats(d_y[t]) for t in targets} if want_probes else {}
    gap_te, lab_te = [], []
    for f in eval_folders:
        zp, en, den = project(loader(f))
        ener['holdout'][0] += en
        ener['holdout'][1] += den
        n_tok['holdout'] += zp.shape[0]
        acc_stats(st_sc1, sc1_model(zp), code[f.name])
        if want_probes:      # same as Pass B: no probe models in the sc1-control case, skip
            for t in targets:
                acc_stats(st_probe[t], models[t](zp), ys[t][f.name])
        if use_cls:
            gap_te.append(zp.mean(0))
            lab_te.append(labels[f.name])

    out = {'rank': int(d if S64 is None else S64.shape[1])}
    out['sc1'] = channel_metrics(st_sc1)
    out['sc1']['solver'] = 'ols(min-norm)'
    out['_W_sc1'] = W_sc1
    if want_probes:
        out['probes'] = {}
        for t in targets:
            r = channel_metrics(st_probe[t])
            r['gamma'] = float(gamma_pick[t])
            r['solver'] = f'closed(ridge, γ={gamma_pick[t]:g})'
            out['probes'][t] = r
    if use_cls:
        out['classification'] = gap_classify(
            gap_tr, lab_tr, gap_va, lab_va, gap_te, lab_te,
            n_classes, args.gamma_grid, device)
    out['energy_fit_pct'] = 100.0 * ener['fit'][0] / max(ener['fit'][1], 1e-30)
    out['energy_val_pct'] = 100.0 * ener['val'][0] / max(ener['val'][1], 1e-30)
    out['energy_holdout_pct'] = 100.0 * ener['holdout'][0] / max(ener['holdout'][1], 1e-30)
    # Protocol §0 (target3): SC2 absolute energy profile — squared-norm scale
    # averaged per token (raw norm, mean not removed; strictly the same
    # denominator as the energy% above, hence self-consistent).
    # Interpretation: share drops + absolute constant = passive dilution from
    # residual-stream norm growth;
    #                  share drops + absolute drops too = real compression
    # (the compression window overlapping the corres drop -> mechanism-level candidate).
    out['abs_code_energy'] = {s: ener[s][0] / max(n_tok[s], 1) for s in ener}
    out['abs_total_energy'] = {s: ener[s][1] / max(n_tok[s], 1) for s in ener}
    out['degenerate_eval'] = degen_eval
    return out


def sc1_eval(loader, S64, splits, code, args, device, d, d_code):
    """target4 2×2 cross-cell evaluation: post-projection OLS readability of
    z⊥ -> code (a lightweight single-cell version of the SC1 mechanism; no
    probes/GAP). Same protocol as the SC1 of fit_eval_basis: GramOLS fit on
    fit+val (unregularized min-norm, pitfall 1 forbids ridge), evaluation on
    holdout. S64=None is the unprojected baseline. Returns a channel_metrics
    dict."""
    S32 = None if S64 is None else S64.to(torch.float32).to(device)
    gram = GramOLS(d, d_code, device)
    for f in splits['fit'] + splits['val']:
        zp, _, _ = project_tokens(loader(f), S32, device)
        gram.update(zp, code[f.name])
    W, mx, my, _, _ = gram.solve(args.ols_rcond)
    model = OLSModel(W, mx, my)
    st = new_stats(d_code)
    eval_folders = splits['holdout'] if splits['holdout'] else splits['fit']
    for f in eval_folders:
        zp, _, _ = project_tokens(loader(f), S32, device)
        acc_stats(st, model(zp), code[f.name])
    return channel_metrics(st)


def random_basis(d, rank, seed):
    """SC3 random control: rank random orthonormal directions in d dims (QR of randn)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, rank, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :rank].contiguous()


def layer_seed(name, seed):
    h = hashlib.sha256(name.encode()).hexdigest()
    return (int(h[:12], 16) + seed) % (2 ** 31)


# =========================================================================== #
#                                 Export                                      #
# =========================================================================== #

@torch.no_grad()
def export_projected(loader, S64, folders, module, out_dir, layout):
    """Export the [fully projected] z_⊥ = z − (z@S)Sᵀ as npy following the
    original directory structure (shape/layout identical to the input,
    fp32), for the external corres pipeline to re-run nearest-neighbor probes
    on the projected features. Note the export is the dimension-preserving
    z_⊥, not the low-dim coordinates z@S, so features before/after projection
    stay interchangeable (protocol §2 Step 4: the same post-processing must
    be used before/after projection)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = S64.to(torch.float32)
    kind, g = loader.plan
    n = 0
    for f in folders:
        z = loader(f)
        zp = z - (z @ S) @ S.T          # (T, d_tok) full projection z_⊥, dimension unchanged
        if kind == '2d':
            arr = zp.numpy()
        else:
            c, gh, gw, p, _ = g
            # token memory order = (gh, gw, c, p, p) (p=1 for grid), inverse of to_tokens
            arr = (zp.reshape(gh, gw, c, p, p)
                   .permute(2, 0, 3, 1, 4).reshape(c, gh * p, gw * p).numpy())
            if layout == 'hwd':
                arr = arr.transpose(1, 2, 0)
        dst = out / f.name
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / f'{module}.npy', arr.astype(np.float32))
        n += 1
    print(f'    exported projected features (z_⊥, dimension unchanged): {module} x {n} images -> {out}')


def save_basis_npz(path, layers, bases, code_names, grid):
    """bases[lm] = S64 (single code) or {code: S64} (dual code, target4).
    Keys: single code S__{lm} (same as the old version), dual code S__{code}__{lm}."""
    payload = {'__layers__': np.array(layers, dtype=object),
               '__code_targets__': np.array(code_names, dtype=object),
               '__grid__': np.array(grid, dtype=np.int64)}
    for lm in layers:
        b = bases[lm]
        if isinstance(b, dict):
            for cn, S in b.items():
                payload[f'S__{cn}__{lm}'] = S.to(torch.float32).numpy()
        else:
            payload[f'S__{lm}'] = b.to(torch.float32).numpy()
    np.savez_compressed(path, **payload)
    print(f'per-layer orthonormal bases S saved to: {path} (external corres pipeline: z_⊥ = z − (z@S)@Sᵀ)')


# =========================================================================== #
#                                   Main                                      #
# =========================================================================== #

def _selftest():
    """target4 principal-angle spectrum math selftest (no data dependency):
    synthetic subspaces with known principal angles -> svd spectrum recovery;
    identity/orthogonal boundary cases; random-baseline formula vs measured
    random subspaces."""
    print('== probe_projection selftest: principal-angle spectrum (target4) ==')
    torch.manual_seed(0)
    d, r = 200, 12
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    theta = torch.linspace(0.05, 1.50, r, dtype=torch.float64)
    U1 = Q[:, :r]
    U2 = Q[:, :r] * theta.cos() + Q[:, r:2 * r] * theta.sin()
    s, mc = principal_angles(U1, U2)
    err = float((s - theta.cos()).abs().max())
    assert err < 1e-9, f'principal-angle cosine spectrum recovery failed: max err {err:.2e}'
    s1, mc1 = principal_angles(U1, U1)              # same subspace -> all 1
    assert float((s1 - 1).abs().max()) < 1e-9 and mc1 > 1 - 1e-9
    s0, mc0 = principal_angles(U1, Q[:, r:2 * r])   # orthogonal subspace -> all 0
    assert float(s0.abs().max()) < 1e-9 and mc0 < 1e-12
    base = max(r, r) / d                            # E[mean cos²] = r/d
    S_ra, S_rb = random_basis(d, r, 0), random_basis(d, r, 1)
    _, mc_rand = principal_angles(S_ra, S_rb)
    assert 0 < mc_rand < 4 * base, mc_rand          # measured random close to baseline
    print(f'  [ok] specified principal-angle spectrum recovered (max err {err:.1e}); identity=all 1 / orthogonal=all 0')
    print(f'  [ok] random {r}-dim subspace mean cos²={mc_rand:.4f} close to baseline '
          f'{base:.4f} (E[Σcos²]=ra·rb/d)')
    print('ALL PASS')


def parse_args():
    p = argparse.ArgumentParser(
        description='output-code projection probe (P_⊥ protocol): per-layer OLS readout of the output code '
                    '-> orthonormal basis S -> project z_⊥ = z − (z@S)Sᵀ -> SC1/SC2/SC3 + re-run linear probes/classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--feature_root', type=str, default=None,
                   help='feature root directory, containing {class}_{image_name}/ folders '
                        '(single cell × single SNR; may be omitted with --selftest)')
    p.add_argument('--code_target', type=str, nargs='+', default=['pred'],
                   help='output-code module names, 1-2 of them (2 = target4 dual-code mode: additionally '
                        'computes per-layer principal-angle spectra of the two subspaces + the 2×2 cross-ablation); '
                        'automatic alias fallback: ' + '/'.join(CODE_ALIASES))
    p.add_argument('--layers', type=str, nargs='+', default=None,
                   help='per-layer representation module names; default auto-selects all 3-dim modules with the largest channel dim')
    p.add_argument('--targets', '--target', dest='targets', type=str,
                   nargs='+', default=['x0', 'eps'],
                   help='linear-probe target modules to re-run after projection (ground-truth signals; --target is an alias)')
    p.add_argument('--input_layout', type=str, default='chw', choices=['hwd', 'chw'],
                   help='npy layout: chw=(c, h, w) channels first (DiT extraction data of this repo); '
                        'hwd=(h, w, d) channels last')
    p.add_argument('--max_images', type=int, default=None,
                   help='use only the first N images in folder-name order')
    # Held-out protocol
    p.add_argument('--holdout_ratio', type=float, default=0.2,
                   help='held-out image ratio (protocol: 1000 images -> 200 evaluation images)')
    p.add_argument('--val_ratio', type=float, default=0.125,
                   help='validation images as a fraction of (1-holdout) (for γ selection; 1000 images -> 100)')
    p.add_argument('--split_seed', type=int, default=0)
    # OLS / projection
    p.add_argument('--ols_rcond', type=float, default=1e-10,
                   help='OLS Gram eigenvalue truncation relative threshold (min-norm, ridge forbidden)')
    p.add_argument('--orth_rcond', type=float, default=1e-6,
                   help='orth(W) singular-value truncation relative threshold (auto rank reduction for low-variance dims, pitfall 6)')
    p.add_argument('--sc1_threshold', type=float, default=0.05,
                   help='SC1 pass threshold: post-projection code readout R² should be below this')
    p.add_argument('--expand_topk', type=int, default=0,
                   help='when SC1 fails, re-verify with the subspace expanded by [W + top-k residual-SVD directions]')
    p.add_argument('--random_control', type=str, default='sc1',
                   choices=['none', 'sc1', 'full'],
                   help='SC3 random-direction control: sc1=re-verify only the code readout R²; '
                        'full=re-run the linear probes as well (roughly double the cost)')
    # Probes
    p.add_argument('--gamma_grid', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0], help='ridge regularization grid')
    p.add_argument('--no_classification', action='store_true',
                   help='disable the GAP classification probe (the classification row of protocol Step 4)')
    # Performance / export
    p.add_argument('--cache_layer', type=str, default='auto',
                   choices=['auto', 'on', 'off'],
                   help='whole-layer in-memory feature cache (major speedup when a layer is streamed multiple times)')
    p.add_argument('--cache_gb', type=float, default=8.0,
                   help='in auto mode, enable when the single-layer cache stays within this value (GB)')
    p.add_argument('--save_basis', type=str, default=None,
                   help='save per-layer orthonormal bases S to npz (for the external corres pipeline to apply the projection)')
    p.add_argument('--export_dir', type=str, default=None,
                   help='root directory for exporting projected-feature npy (directory structure matches feature_root)')
    p.add_argument('--export_layers', type=str, nargs='+', default=None,
                   help='export only the specified layers (default: all layers; disk usage comparable to the original features)')
    # Misc
    p.add_argument('--selftest', action='store_true',
                   help='principal-angle spectrum math selftest (target4, no data dependency)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or a specific cuda/mps/cpu')
    p.add_argument('--result_path', type=str, default='probe_projection_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.feature_root:
        raise RuntimeError('--feature_root is required (or use --selftest instead)')
    root = Path(args.feature_root)
    if not root.is_dir():
        raise RuntimeError(f'feature root directory does not exist: {root}')

    folders_all = pf.list_image_folders(root)
    if args.max_images is not None:
        folders_all = folders_all[:max(args.max_images, 0)]
        print(f'--max_images={args.max_images}: keeping the first {len(folders_all)} images in sorted order')
    if not folders_all:
        raise RuntimeError('no usable images after truncation')

    modules = pf.discover_modules(folders_all)
    shapes = scan_module_shapes(folders_all, modules)
    layers, grid = resolve_layers(modules, shapes, args.input_layout, args.layers)

    # Resolve the actual module names of code/probe targets first (fixes the
    # participating module set, used to filter plan construction)
    if len(args.code_target) > 2:
        raise RuntimeError('--code_target takes at most 2 values (dual code = target4 mode)')
    code_names = [resolve_code_target(t, modules) for t in args.code_target]
    if len(set(code_names)) != len(code_names):
        raise RuntimeError(f'--code_target duplicated: {code_names}')
    dual = len(code_names) == 2
    targets = [resolve_target(t, modules) for t in
               list(dict.fromkeys(args.targets)) if t not in code_names]

    # ---- tokenization plan: built only for modules participating in this
    # protocol (layers + output codes + probe targets). Other modules in
    # feature_root (other extraction points/reference features, e.g. the 37x37
    # DINOv2 reference) may have different spatial grids and must not abort
    # the whole protocol — skip them and print their first-image shapes for
    # review, avoiding irrelevant modules raising "mismatches the token grid".
    used = list(dict.fromkeys(layers + code_names + targets))
    plans = {}
    for m in used:
        plans[m] = build_module_plan(shapes[m], args.input_layout, grid, name=m)
    d_by = {m: tok_dim(plans[m]) for m in used}
    T = grid[0] * grid[1]
    skipped = [m for m in modules if m not in used]
    if skipped:
        print(f'{len(skipped)} module(s) not participating in this protocol (tokenization skipped, first-image shapes):')
        for m in skipped:
            print(f'  {m}: {shapes[m]}')
    print(f'images: {len(folders_all)} | modules: {len(modules)} | '
          f'layout: {args.input_layout} | token grid: {grid} (T={T})')
    print(f'per-layer representations ({len(layers)} layers): {layers}')
    print('module token dims: ' + '  '.join(f'{m}={d_by[m]}' for m in used
                                            if m not in layers))
    d = d_by[layers[0]]
    d_code = {c: d_by[c] for c in code_names}
    d_y = {t: d_by[t] for t in targets}
    print('code targets: ' + '  '.join(f'{c} (d={d_code[c]})' for c in code_names)
          + f' | probe targets: { {t: d_y[t] for t in targets} }')
    if dual:
        print('[target4] dual-code mode: full projection evaluation per code, plus per-layer principal-angle '
              'spectra (svd(Q_aᵀQ_b), mean cos² vs random baseline max(ra,rb)/d) and '
              'the 2×2 cross-ablation ({drop-a, drop-b} × {measure-a, measure-b}, regression target = '
              'the code itself, unprojected readability as reference; independent of --targets)')

    # Preload small targets + labels
    t0 = time.time()
    codes_data = {c: preload_tokens(folders_all, c, plans[c], shapes[c])
                  for c in code_names}
    ys = {t: preload_tokens(folders_all, t, plans[t], shapes[t])
          for t in targets}
    print(f'small-target preload done ({time.time() - t0:.1f}s)')

    classes = sorted({pf.class_of(f) for f in folders_all})
    use_cls = (not args.no_classification) and len(classes) >= 2
    if use_cls:
        idx = {c: i for i, c in enumerate(classes)}
        labels = {f.name: idx[pf.class_of(f)] for f in folders_all}
        print(f'GAP classification probe: on ({len(classes)} classes)')
    else:
        labels = {}
        print('GAP classification probe: off' + ('' if args.no_classification
                                                 else ' (< 2 classes)'))

    # Split (class-stratified)
    splits, degen = stratified_fit_val_holdout(folders_all, args.holdout_ratio,
                                               args.val_ratio, args.split_seed)
    print(f"split: fit={len(splits['fit'])}  val={len(splits['val'])}  "
          f"holdout={len(splits['holdout'])}  (W/SC1 fit set = fit+val = "
          f"{len(splits['fit']) + len(splits['val'])} images)")

    device = pf.pick_device(args.device)
    print(f'device: {device}')
    est_gb = len(folders_all) * T * d * 4 / 1024 ** 3
    use_cache = (args.cache_layer == 'on' or
                 (args.cache_layer == 'auto' and est_gb <= args.cache_gb))
    print(f'whole-layer memory cache: {"on" if use_cache else "off"} '
          f'(per-layer estimate {est_gb:.2f} GB, threshold {args.cache_gb} GB)')

    # ---------------- Per-layer processing ----------------
    results, bases = {}, {}
    for lm in layers:
        t0 = time.time()
        loader = LayerLoader(folders_all, lm, plans[lm], use_cache, shapes[lm])
        single = len(code_names) == 1

        entries = {}
        ols_fit = {}                        # code -> unprojected OLS fit (W, mx, my)
        for cn in code_names:
            # Pass A: B_c = Cov(Z, code) cross-covariance, fit set = fit+val (the protocol's 800 images)
            gram = GramOLS(d, d_code[cn], device)
            for f in splits['fit'] + splits['val']:
                gram.update(loader(f), codes_data[cn][f.name])
            W64, mx_a, my_a, cond, rank_g = gram.solve(args.ols_rcond)
            ols_fit[cn] = (W64, mx_a, my_a)  # the 2×2 baseline cell reuses this same fit (consistent scale)
            S64, svals = orth_basis(gram.cross_cov(), args.orth_rcond)
            if single:
                bases[lm] = S64   # single code: keep the old bare-basis structure (npz key compatibility)
            else:
                bases.setdefault(lm, {})[cn] = S64

            main_res = fit_eval_basis(loader, S64, splits, codes_data[cn], ys,
                                      targets, labels, len(classes), use_cls,
                                      True, args, device, d, d_code[cn], d_y)
            sc1 = main_res['sc1']
            sc1_pass = sc1['r2_avg'] < args.sc1_threshold

            # SC3: random-direction control (seed key: same as the old version for single code, plus the code name for dual)
            rand_res = None
            if args.random_control != 'none':
                seed_key = lm if single else f'{lm}/{cn}'
                S_rand = random_basis(d, S64.shape[1],
                                      layer_seed(seed_key, args.seed))
                rand_res = fit_eval_basis(loader, S_rand, splits,
                                          codes_data[cn], ys, targets, labels,
                                          len(classes), use_cls,
                                          want_probes=(args.random_control
                                                       == 'full'),
                                          args=args, device=device, d=d,
                                          d_code=d_code[cn], d_y=d_y)

            # SC1 fail -> expand the subspace and re-verify ([W + top residual-SVD directions])
            exp_res = None
            if not sc1_pass and args.expand_topk > 0:
                U_res, _, _ = torch.linalg.svd(main_res['_W_sc1'],
                                               full_matrices=False)
                S_cat = torch.cat([S64, U_res[:, :args.expand_topk]], dim=1)
                S_exp, _ = orth_basis(S_cat, args.orth_rcond)
                print(f'    [SC1 fail] expanding the projection subspace: {S64.shape[1]} -> '
                      f'{S_exp.shape[1]} dims, re-verifying...')
                exp_res = fit_eval_basis(loader, S_exp, splits, codes_data[cn],
                                         ys, targets, labels, len(classes),
                                         use_cls, True, args, device, d,
                                         d_code[cn], d_y)
                if single:
                    bases[lm] = S_exp
                else:
                    bases[lm][cn] = S_exp

            # Report (prefix the code name in dual-code mode)
            tag = '' if single else f'{cn} '
            print(f"\n=== {tag}[{lm}] Gram rank={rank_g} cond={cond:.2e} | "
                  f"orth(B_c) rank={S64.shape[1]}/{d_code[cn]} | svals top3="
                  f"{[round(float(v), 3) for v in svals[:3]]} ===")
            print(f"    SC2  energy fit/val/holdout = "
                  f"{main_res['energy_fit_pct']:.3f}%"
                  f" / {main_res['energy_val_pct']:.3f}%"
                  f" / {main_res['energy_holdout_pct']:.3f}%")
            ae, at = main_res['abs_code_energy'], main_res['abs_total_energy']
            print(f"    SC2  absolute/token  code subspace = {ae['fit']:.4g}/"
                  f"{ae['val']:.4g}/{ae['holdout']:.4g}   z total norm² = "
                  f"{at['fit']:.4g}/{at['val']:.4g}/{at['holdout']:.4g}")
            mark = 'PASS' if sc1_pass else 'FAIL (<--sc1_threshold)'
            print(f"    SC1  z_⊥→{cn}  R2_avg={sc1['r2_avg']:.4f}  [{mark}]"
                  f"  ({sc1['solver']})")
            for t, r in main_res.get('probes', {}).items():
                print(f"    probe {t:<8} R2_avg={r['r2_avg']:.4f}  R2_var="
                      f"{r['r2_var']:.4f}  cos={r['cos_sim']:.4f}  "
                      f"({r['solver']})")
            if use_cls:
                c = main_res['classification']
                t5 = f"  top5={c['top5']:.4f}" if 'top5' in c else ''
                print(f"    classification GAP   top1={c['top1']:.4f}{t5}  ({c['solver']})")
            if rand_res is not None:
                rs = rand_res['sc1']['r2_avg']
                verdict = (
                    f'control normal: random projection does not remove code readability (SC1={rs:.3f}); '
                    f'the main projection SC1≈0 is code-specific'
                    if rs >= args.sc1_threshold else
                    f'[note] random-projection SC1 is also ≈0 ({rs:.3f}): the random directions overlap the code '
                    f'subspace too much; the SC1 pass is not specific evidence')
                print(f"    SC3  random {S64.shape[1]}-dim control  SC1 R2_avg={rs:.4f}"
                      f"  ({verdict})")
            if exp_res is not None:
                print(f"    SC1' expanded R2_avg={exp_res['sc1']['r2_avg']:.4f} "
                      f"(threshold {args.sc1_threshold})")

            keep = {k: v for k, v in main_res.items() if not k.startswith('_')}
            e = {'rank': S64.shape[1], 'gram_rank': rank_g,
                 'gram_cond': cond,
                 'singular_values': [float(v) for v in svals], **keep}
            if rand_res is not None:
                e['random_control'] = {k: v for k, v in rand_res.items()
                                       if not k.startswith('_')}
            if exp_res is not None:
                e['expanded'] = {k: v for k, v in exp_res.items()
                                 if not k.startswith('_')}
            e['sc1_pass'] = bool(sc1_pass)
            entries[cn] = e

        if not dual:
            results[lm] = entries[code_names[0]]
        else:
            # -------- target4: principal-angle spectrum + 2×2 cross-ablation --------
            # The 2×2 regression target = the two codes themselves (protocol:
            # x₀/ε are both the readout subspace source and the measurement
            # target), independent of --targets (ground-truth probes, still
            # re-run per code as in target1).
            c0, c1 = code_names
            Sa, Sb = bases[lm][c0], bases[lm][c1]
            s_cos, mean_cos2 = principal_angles(Sa, Sb)
            ra, rb = int(Sa.shape[1]), int(Sb.shape[1])
            base_line = max(ra, rb) / d          # random baseline E[mean cos²]
            # 2×2 four cells: row = the code subspace being dropped, column = the OLS R² of the
            # regression-target code.
            # Diagonal cells (drop c, measure c) = the SC1 of that code itself (≈0 by construction);
            # off-diagonal cells (drop row code, measure column code) computed fresh; baseline row =
            # unprojected readability (evaluated on holdout directly with the same Pass A OLS fit,
            # consistent scale).
            cells = {f'ablate_{r}__regress_{c}': entries[r]['sc1']['r2_avg']
                     for r, c in ((c0, c0), (c1, c1))}
            for r, c in ((c0, c1), (c1, c0)):    # off-diagonal cells
                cells[f'ablate_{r}__regress_{c}'] = sc1_eval(
                    loader, bases[lm][r], splits, codes_data[c],
                    args, device, d, d_code[c])['r2_avg']
            baseline = {}
            eval_folders = (splits['holdout'] if splits['holdout']
                            else splits['fit'])
            for c in (c0, c1):
                W, mx, my = ols_fit[c]
                st = new_stats(d_code[c])
                for f in eval_folders:
                    acc_stats(st, OLSModel(W, mx, my)(loader(f)),
                              codes_data[c][f.name])
                baseline[c] = channel_metrics(st)['r2_avg']
            keeps = {}
            for r, c in ((c0, c1), (c1, c0)):    # off-diagonal keep ratio (interpretation basis)
                if baseline[c] > 1e-9:
                    keeps[r] = max(cells[f'ablate_{r}__regress_{c}'],
                                   0.0) / baseline[c]
            results[lm] = {
                'codes': entries,
                'principal_angles': {
                    'ranks': [ra, rb],
                    'cos2_spectrum': [float(v * v) for v in s_cos],
                    'mean_cos2': mean_cos2,
                    'random_baseline_mean_cos2': base_line,
                    'shared': bool(mean_cos2 > 3 * base_line),
                },
                'cross_ablation_2x2': {
                    'regression_targets': [c0, c1],
                    'note': ('row = the code subspace being dropped, column = the OLS R² of the '
                             'regression-target code (regression target = --code_target itself); '
                             'diagonal = the SC1 of each code itself (≈0 by construction), '
                             'baseline = unprojected readability'),
                    'baseline_r2_avg': {c: float(baseline[c])
                                        for c in (c0, c1)},
                    'cells_r2_avg': {k: float(v)
                                     for k, v in cells.items()},
                    'cross_keep_ratio': {r: (None if v is None
                                             else float(v))
                                         for r, v in keeps.items()},
                },
            }
            # Report
            spec = [round(float(v * v), 3) for v in s_cos[:6]]
            print(f"\n    [target4] principal-angle spectrum  cos² (top {min(6, len(s_cos))}/"
                  f"{len(s_cos)}) = {spec}")
            shared_txt = ('subspaces significantly shared (geometric entanglement)'
                          if mean_cos2 > 3 * base_line else 'close to random (geometrically independent)')
            print(f"             mean cos²={mean_cos2:.4f}  vs random baseline "
                  f"{base_line:.4f} -> {shared_txt}")
            print(f"    [target4] 2×2 cross-ablation (row = the code subspace being dropped, column = the regression-"
                  f"target code OLS R²; target = the code itself, reference = unprojected)")
            print('             unprojected baseline    ' + '  '.join(
                f"{c}={baseline[c]:.3f}" for c in (c0, c1)))
            for r in (c0, c1):
                row = {c: cells[f'ablate_{r}__regress_{c}']
                       for c in (c0, c1)}
                other = c1 if r == c0 else c0
                k = (f"  (keep_{other}={keeps[r]:.0%})"
                     if keeps.get(r) is not None else '')
                print(f"             drop {r:<12} " + '  '.join(
                    f"{c}={row[c]:.3f}" for c in (c0, c1)) + k)
            valid = [v for v in keeps.values() if v is not None]
            if len(valid) == 2:
                km = min(valid)
                if km < 0.5:
                    verdict = ('functionally entangled: dropping one code subspace also damages the readout '
                               'of the other (off-diagonal drops along with the diagonal)')
                elif km >= 0.8:
                    verdict = ('decoupled: dropping each subspace only zeroes its own readout, off-diagonal '
                               'cells stay at baseline')
                else:
                    verdict = 'transition state (partial entanglement)'
                print(f'    [target4] verdict: {verdict}')
        print(f"    ({time.time() - t0:.1f}s)")
        del loader

    # Export
    if args.save_basis:
        save_basis_npz(args.save_basis, layers, bases, code_names, grid)
    if args.export_dir:
        export_set = layers if not args.export_layers \
            else [m for m in args.export_layers if m in layers]
        for lm in export_set:
            loader = LayerLoader(folders_all, lm, plans[lm], False, shapes[lm])
            export_projected(loader, bases[lm], folders_all, lm,
                             args.export_dir, args.input_layout)
            del loader

    # ---------------- Summary table ----------------
    print('\n========== Summary (after projection, holdout metrics) ==========')
    width = max([len(m) for m in layers] + [8]) + 2
    n_fail = 0
    if not dual:
        cols = ['SC1_R2'] + [f'{t}_R2' for t in targets] + \
               (['cls_top1'] if use_cls else [])
        print('  ' + 'layer'.ljust(width) + 'energy%'.rjust(10)
              + ''.join(c.rjust(11) for c in cols))
        for lm in layers:
            r = results[lm]
            row = lm.ljust(width) + f"{r['energy_holdout_pct']:.3f}".rjust(10)
            row += f"{r['sc1']['r2_avg']:.4f}".rjust(11)
            for t in targets:
                row += f"{r['probes'][t]['r2_avg']:.4f}".rjust(11)
            if use_cls:
                row += f"{r['classification']['top1']:.4f}".rjust(11)
            print('  ' + row)
        n_fail = sum(1 for r in results.values() if not r['sc1_pass'])
    else:
        # target4 dual code: mean cos² (principal angles) + per-code SC1/probes
        heads = (['mean_cos2'] + [f'{c}/SC1' for c in code_names]
                 + [f'{c}/{t}' for c in code_names for t in targets])
        print('  ' + 'layer'.ljust(width)
              + ''.join(h.rjust(13) for h in heads))
        for lm in layers:
            r = results[lm]
            row = lm.ljust(width)
            row += f"{r['principal_angles']['mean_cos2']:.4f}".rjust(13)
            for c in code_names:
                row += f"{r['codes'][c]['sc1']['r2_avg']:.4f}".rjust(13)
            for c in code_names:
                for t in targets:
                    row += (f"{r['codes'][c]['probes'][t]['r2_avg']:.4f}"
                            .rjust(13))
            n_fail += sum(1 for c in code_names
                          if not r['codes'][c]['sc1_pass'])
            print('  ' + row)
    n_tot = len(results) * (len(code_names) if dual else 1)
    print(f"\nSC1 verdict: {n_tot - n_fail}/{n_tot} layer×code PASS "
          f"(threshold {args.sc1_threshold})"
          + ('' if n_fail == 0 else f'; {n_fail} failed, consider --expand_topk re-verification'))

    out = {
        'protocol': ('target1+target4: P_perp projection & principal angles '
                     '(probe_projection.py)'),
        'feature_root': str(root),
        'n_images': len(folders_all),
        'grid': list(grid),
        'layers': layers,
        'code_target': code_names[0],
        'code_targets': code_names,
        'dual_code_mode': dual,
        'targets': targets,
        'module_token_dims': d_by,
        'classes': classes if use_cls else None,
        'splits': {k: [f.name for f in v] for k, v in splits.items()},
        'degenerate_split': degen,
        'args': vars(args),
        'results': results,
    }
    with open(args.result_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nresults saved to: {args.result_path}')


if __name__ == '__main__':
    main()
