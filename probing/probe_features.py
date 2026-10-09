#!/usr/bin/env python3
"""
2D feature probing
==================

Data layout convention::

    feature_root/
        {class}_{image_name}/     # the part before the first underscore in
                                  # the folder name is the image class
            {module_name}.npy     # features extracted by that module, 2D
                                  # feature map (h, w, d), or channels-first
                                  # (c, h, w) (use --input_layout chw)

Task: each run designates one [target module] and several (usually ~28)
[input modules], predicts the target features from the input features, and
reports prediction quality (R2 / MSE / per-position cosine similarity) on
the held-out test set. The linear probe additionally reports the energy
gain gain = Σ||W·x_std||² / Σ||x_std||², where W is the fitted linear map
(no intercept) and x_std the standardized input, measuring how much input
energy survives / is amplified through W; per-input-module probing then
yields each input module's own energy gain.

Implementation notes
--------------------
0. Dim layout: npy is interpreted per --input_layout and unified to internal
   (d, h, w): 'hwd' = (h, w, d) channels last (default); 'chw' = (c, h, w)
   channels first (torch (C, H, W) conv feature maps, and ViT features saved
   from token sequences via rearrange 'b (h w) c -> b c h w', are of this
   kind). A wrong layout treats the channel dim as a spatial dim; typical
   symptom: every input module's probe metric is ≈0 and all nearly identical.
1. Spatial alignment: modules may differ in spatial size; all features are
   bilinearly interpolated to one grid (--align_to): auto defaults to the
   input grid when [all input-module grids agree] (only the target needs one
   interpolation, inputs keep native resolution), otherwise falls back to
   the target grid; target / inputs can also be forced. Within one module
   the channel count d must be identical across images (error otherwise);
   images of different spatial sizes are aligned to the unified grid.
   --pair_mode patch disables interpolation (M2 zero-resize measurement):
   inputs and target keep native grids, the target is blocked by the input
   token's receptive field (kh×kw), and each token regresses a
   d_out*kh*kw-dim target (flat probes linear/mlp only).
2. Per-position semantics: "flatten to vectors" means concatenating all
   input feature channels into one vector at each spatial position (each
   position is an independent regression sample, the 1x1-conv view);
   "keep 2D" preserves the (D, H, W) feature map so a 3x3 conv can use
   neighborhood context. Four probes:
     linear  : flatten -> single Linear            (per-position linear, no spatial context)
     mlp     : flatten -> Linear-act-Linear        (per-position nonlinear, no spatial context)
     conv    : 2D -> single 3x3 Conv               (linear + 3x3 spatial context)
     convmlp : 2D -> 3x3 Conv-act-3x3 Conv         (nonlinear + 3x3 spatial context)
3. Solver strategy:
     - linear has a closed-form solution (standardized ridge regression):
       stream-accumulate Gram statistics, then one eigendecomposition solves
       the whole γ grid in parallel; γ is picked on the validation set.
       Enabled by default when D_in <= --closed_form_max_dim
       (--linear_solver can force closed / adam).
     - Other probes have no closed form: Adam + cosine annealing, few
       epochs, keep the best weights by validation loss (output layer
       zero-initialized so the initial prediction equals the target mean).
4. Standardization: inputs and targets are standardized per channel by
   train-set mean/variance (the closed form does the equivalent inside Gram
   space); all metrics are computed on the [original scale]. R2 is computed
   per channel then averaged (channels whose out-of-train variance ≈ 0 are
   dropped).
5. Splits: image (folder)-level train/val/test split, stratified by class
   (the prefix before the first underscore of the folder name);
   --max_images first takes the first N images in sorted order, then splits.
6. Results: by default each input module probes the target separately
   (--per_input_probes; the default linear closed form is nearly free); the
   [all-inputs-concatenated] probe is off by default and needs --probe.
   Per-line console results + summary table + JSON (--result_path) for easy
   sorting and comparison.
7. Two-target cross analysis: when --target names exactly two modules, an
   extra [per-layer depth profile] is computed — for each common input
   layer, two linear predictors (layer→target_a / layer→target_b; prefer
   reusing the linear predictors already trained by the per-input probes,
   retrain when missing) compute the Pearson correlation corr(a, b) of the
   predictor outputs on the test set; one value per layer forms the profile
   curve (JSON top-level cross_predictor_corr: per_layer detail + ordered
   pearson_profile array).
8. Lazy loading: startup reads only each module's npy header on the first
   image (cost = one file open per module, independent of image count);
   feature bodies and cross-image consistency checks are deferred to actual
   loading; grid detection likewise looks only at the first image, and
   odd-sized images fall back to interpolation. No more long startup waits
   on network storage.
9. Saving deletion directions (ref/target5.txt cross-subspace deletion):
   enabled by supplying --save_deletion_dirs. Default
   --deletion_mode fullspace = the SC1 protocol (aligned with
   probe_projection): the initial deletion basis = orth(B_t) (the column
   space of the standardized-space cross-covariance Cov(Xs,Ys)); after
   deletion Cov(z_⊥,y) is strictly zeroed and refitting min-norm OLS gives
   train R²≈0, passing in one round (criterion R² ≤ deletion_r2_target; on
   numeric residue, expand with the top directions of the residual-readout
   SVD and re-verify). Note [you must NOT use the column space of the OLS
   readout W as the initial basis]: under an ill-conditioned spectrum W's
   low-variance directions are dominated by 1/√λ estimation noise, leaving
   cross-covariance residue after projection (the failure mode recorded in
   the probe_projection.cross_cov comments; in practice 20 iterations
   cannot clean it). Directions are an orthonormal basis in standardized
   space; the npz carries mu/inv_std, so external deletion must standardize
   first: z=(x−mu)⊙inv_std → z−=(z@dirsᵀ)dirs → x'=z⊙std+mu
   (delete_direction.py branches automatically on the entry's space field).
   Selective deletion (--deletion_protect, fullspace): before deleting a
   target's directions, first subtract the subspace of the protected
   targets (oblique projection: P ← orth((I−S_proS_proᵀ)·orth(B_t))); the
   protected signal is strictly untouched (covariance components unchanged,
   measured protected-R² loss ≈ 0); the cost is that target components
   overlapping the protection are kept (deletion incomplete, residue =
   overlap). The npz entry carries
   protect/r2_del_before/r2_pro_before/r2_pro_after fields; pass2 and
   online deletion need no change (directions pre-subtracted). Meaning of
   the dual metrics: the target's residual R² is not a failure but the
   geometric lower bound given by the overlap with the protected subspace.
   linear must use the closed form (Adam, lacking Gram statistics, degrades
   to a single projection with a warning); alternatives r2_target (spectral
   back-solve, diagonal-covariance approximation, distorted when channels
   are strongly correlated) and topk (original space, k given directly).
   Recommend --gamma_grid 1e-8 to approach unregularized OLS.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# probe name -> data layout: flat = per-position vectors (N, D); map = 2D feature maps (B, D, H, W)
PROBE_LAYOUTS = {'linear': 'flat', 'mlp': 'flat', 'conv': 'map', 'convmlp': 'map'}
PROBE_NAMES = list(PROBE_LAYOUTS)


# =========================================================================== #
#                        Data discovery & scanning                            #
# =========================================================================== #

def list_image_folders(root: Path):
    """Return all image feature folders sorted by name (requires at least
    one .npy inside; unrelated subdirectories such as raw-image dirs under
    feature_root are ignored automatically)."""
    folders = sorted((p for p in root.iterdir() if p.is_dir()
                      and next(p.glob('*.npy'), None) is not None),
                     key=lambda p: p.name)
    if not folders:
        raise RuntimeError(f'no image feature folders found under {root}')
    return folders


def class_of(folder: Path) -> str:
    """The part before the first underscore of the folder name = class."""
    return folder.name.split('_')[0]


def discover_modules(folders):
    """Candidate modules = the intersection of .npy filenames (extension
    stripped) across all folders."""
    per_folder = [{p.stem for p in f.glob('*.npy')} for f in folders]
    common = set.intersection(*per_folder)
    if not common:
        raise RuntimeError('no intersection of .npy module names across '
                           'folders, check the data organization')
    union = set.union(*per_folder)
    for m in sorted(union - common):
        owners = [f.name for f, mods in zip(folders, per_folder) if m in mods][:3]
        print(f'[warn] module {m} appears only in some folders (e.g. '
              f'{owners}...), removed from the candidates')
    return sorted(common)


def scan_shapes(folders, modules, layout: str = 'hwd'):
    """[Lazy scan] reads only each module's npy header on the first image
    (folders[0]) for shape and channel count — startup cost = one file open
    per module, independent of image count (a full scan of all images on
    network storage can take minutes). Returned shapes follow the internal
    (h, w, d) semantics (chw layout already transposed); shapes contains
    only (folders[0].name, m) keys.
    Cross-image channel-count consistency is now checked at load time:
    see FeatureBank.load.
    """
    f0 = folders[0]
    shapes, d_by_module = {}, {}
    for m in modules:
        arr = np.load(f0 / f'{m}.npy', mmap_mode='r')
        if arr.ndim != 3:
            raise RuntimeError(f'{f0}/{m}.npy shape {tuple(arr.shape)} is not a 3D feature map')
        if layout == 'chw':
            shape = (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[0]))
        else:
            shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        shapes[(f0.name, m)] = shape
        d_by_module[m] = shape[2]
    return shapes, d_by_module


# =========================================================================== #
#                        Feature loading & datasets                           #
# =========================================================================== #

class FeatureBank:
    """Lazily loads features by (folder, module), with optional in-memory
    cache (float32, (d, h, w)).
    layout: npy storage layout 'hwd'=(h, w, d) / 'chw'=(c, h, w), unified to
    (d, h, w).
    Records a module's channel count on first sight and verifies consistency
    on every later load (the load-time self-check of the lazy scan)."""

    def __init__(self, use_cache: bool, layout: str = 'hwd'):
        self.use_cache = use_cache
        self.layout = layout
        self._cache = {}
        self._d_seen = {}   # module -> channel count seen on first load

    def load(self, folder: Path, module: str) -> torch.Tensor:
        key = (folder.name, module)
        if self.use_cache and key in self._cache:
            return self._cache[key]
        arr = np.load(folder / f'{module}.npy')
        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if self.layout == 'hwd':
            x = x.permute(2, 0, 1).contiguous()  # (h, w, d) -> (d, h, w)
        # 'chw': the npy is already (c, h, w) = internal (d, h, w), no permute
        # Load-time check (the full pre-check formerly in scan_shapes moved
        # here): a module's channel count must be identical across images;
        # error immediately instead of letting later matrix ops crash deep
        d = x.shape[0]
        if self._d_seen.setdefault(module, d) != d:
            raise RuntimeError(
                f'module {module} has channel count {d} in {folder.name} but '
                f'{self._d_seen[module]} in earlier images, check the '
                f'feature files')
        if self.use_cache:
            self._cache[key] = x
        return x


def resize_spatial(x: torch.Tensor, hw) -> torch.Tensor:
    """(d, h, w) -> bilinear-aligned to (h_t, w_t)."""
    if (x.shape[1], x.shape[2]) == tuple(hw):
        return x
    return F.interpolate(x.unsqueeze(0), size=tuple(hw),
                         mode='bilinear', align_corners=False)[0]


class ProbeDataset(Dataset):
    """One image = one sample. Returns:
        flat: X (H*W, D_in), y (H*W, d_out)   # each row is one spatial position
        map : X (D_in, H, W), y (d_out, H, W)
    All input features are aligned to ref_hw (the target reference grid) and
    concatenated along channels.
    """

    def __init__(self, folders, bank: FeatureBank, input_modules, target_module,
                 layout: str, ref_hw):
        self.folders = list(folders)
        self.bank = bank
        self.input_modules = list(input_modules)
        self.target_module = target_module
        self.layout = layout
        self.ref_hw = tuple(ref_hw)

    def __len__(self):
        return len(self.folders)

    def _load_pair(self, idx):
        folder = self.folders[idx]
        y = resize_spatial(self.bank.load(folder, self.target_module), self.ref_hw)
        xs = [resize_spatial(self.bank.load(folder, m), self.ref_hw)
              for m in self.input_modules]
        return torch.cat(xs, dim=0), y  # (D_in, H, W), (d_out, H, W)

    def __getitem__(self, idx):
        X, y = self._load_pair(idx)
        if self.layout == 'map':
            return X, y
        d_in, h, w = X.shape
        d_out = y.shape[0]
        return (X.permute(1, 2, 0).reshape(h * w, d_in),
                y.permute(1, 2, 0).reshape(h * w, d_out))


class PatchPairDataset(Dataset):
    """M2 patch-pairing dataset (zero resize): inputs and target both keep
    native grids, no interpolation at all. The target is blocked by the
    input token's receptive field — each token (i, j) is paired with the
    flattened vector (d_out*kh*kw,) of all target values in its kh×kw
    neighborhood: [c0's kh*kw positions, c1's, ...] (channels first,
    in-patch positions row-major). Only flat probes (linear/mlp) can use it:
    returns X (hi*wi, D_in), Y (hi*wi, d_out*kh*kw).
    Any cross-image/cross-module grid mismatch with the first image raises
    immediately (patch pairing refuses interpolation).
    Background: a per-position linear probe can only use the input vector of
    a single position; upsampling would spread one token's information over
    kh×kw output positions (iid targets get diluted by squared weights);
    patch pairing lets each token's original values pair directly with the
    raw target inside its receptive field — full information, no dilution."""

    def __init__(self, folders, bank, input_modules, target_module,
                 in_hw, patch_hw, tgt_hw):
        self.folders = list(folders)
        self.bank = bank
        self.input_modules = list(input_modules)
        self.target_module = target_module
        self.in_hw = tuple(int(v) for v in in_hw)
        self.patch_hw = tuple(int(v) for v in patch_hw)
        self.tgt_hw = tuple(int(v) for v in tgt_hw)
        kh, kw = self.patch_hw
        if self.tgt_hw[0] != self.in_hw[0] * kh \
                or self.tgt_hw[1] != self.in_hw[1] * kw:
            raise RuntimeError(f'patch pairing requires target grid {self.tgt_hw} == '
                               f'input grid {self.in_hw} × patch ({kh}, {kw})')

    def __len__(self):
        return len(self.folders)

    def _load_pair(self, idx):
        folder = self.folders[idx]
        hi, wi = self.in_hw
        xs = []
        for m in self.input_modules:
            x = self.bank.load(folder, m)
            if (x.shape[1], x.shape[2]) != self.in_hw:
                raise RuntimeError(
                    f'patch pairing: module {m} grid {(x.shape[1], x.shape[2])} '
                    f'in {folder.name} differs from first image {self.in_hw}, '
                    f'refusing interpolated alignment (zero resize)')
            xs.append(x)
        y = self.bank.load(folder, self.target_module)
        if (y.shape[1], y.shape[2]) != self.tgt_hw:
            raise RuntimeError(
                f'patch pairing: target {self.target_module} grid '
                f'{(y.shape[1], y.shape[2])} in {folder.name} differs from '
                f'first image {self.tgt_hw}, refusing interpolated alignment '
                f'(zero resize)')
        d_out = y.shape[0]
        kh, kw = self.patch_hw
        # target blocking: (d, ho, wo) -> (hi*wi, d*kh*kw), token (i, j) row-major
        Y = (y.reshape(d_out, hi, kh, wi, kw)
              .permute(1, 3, 0, 2, 4)
              .reshape(hi * wi, d_out * kh * kw))
        X = (torch.cat(xs, dim=0)
              .permute(1, 2, 0)
              .reshape(hi * wi, -1))
        return X, Y

    def __getitem__(self, idx):
        return self._load_pair(idx)


def make_collate(layout):
    if layout == 'map':
        def collate(batch):
            return (torch.stack([b[0] for b in batch], 0),
                    torch.stack([b[1] for b in batch], 0))
    else:
        def collate(batch):
            return (torch.cat([b[0] for b in batch], 0),
                    torch.cat([b[1] for b in batch], 0))
    return collate


# =========================================================================== #
#                                Standardization                              #
# =========================================================================== #

class Standardizer:
    """Per-channel standardization params (train-set statistics), supports
    (N, D) / (D, H, W) / (B, D, H, W)."""

    def __init__(self, mu_x, inv_std_x, mu_y, std_y):
        self.mu_x, self.inv_std_x = mu_x, inv_std_x
        self.mu_y, self.std_y = mu_y, std_y

    def to(self, device):
        return Standardizer(self.mu_x.to(device), self.inv_std_x.to(device),
                            self.mu_y.to(device), self.std_y.to(device))

    def _bcast(self, v, x):
        if x.dim() == 4:      # (B, D, H, W)
            return v[None, :, None, None]
        if x.dim() == 3:      # (D, H, W)
            return v[:, None, None]
        return v[None, :]     # (N, D)

    def apply_x(self, x):
        return (x - self._bcast(self.mu_x, x)) * self._bcast(self.inv_std_x, x)

    def apply_y(self, y):
        return (y - self._bcast(self.mu_y, y)) / self._bcast(self.std_y, y)

    def unapply_y(self, y):
        return y * self._bcast(self.std_y, y) + self._bcast(self.mu_y, y)


@torch.no_grad()
def compute_standardizer(dataset: ProbeDataset, args) -> Standardizer:
    """Stream-accumulates per-channel mean/variance on the given (train)
    dataset."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate(dataset.layout),
                        num_workers=args.num_workers)
    sx = sx2 = sy = sy2 = None
    n = 0
    for X, y in loader:
        if X.dim() == 4:
            cnt = X.shape[0] * X.shape[2] * X.shape[3]
            xs_, xs2_ = X.sum((0, 2, 3)).double(), (X * X).sum((0, 2, 3)).double()
            ys_, ys2_ = y.sum((0, 2, 3)).double(), (y * y).sum((0, 2, 3)).double()
        else:
            cnt = X.shape[0]
            xs_, xs2_ = X.sum(0).double(), (X * X).sum(0).double()
            ys_, ys2_ = y.sum(0).double(), (y * y).sum(0).double()
        if sx is None:
            sx, sx2 = torch.zeros_like(xs_), torch.zeros_like(xs2_)
            sy, sy2 = torch.zeros_like(ys_), torch.zeros_like(ys2_)
        sx += xs_; sx2 += xs2_; sy += ys_; sy2 += ys2_; n += cnt
    mu_x = sx / n
    mu_y = sy / n
    var_x = (sx2 / n - mu_x ** 2).clamp_min(1e-12)
    var_y = (sy2 / n - mu_y ** 2).clamp_min(1e-12)
    return Standardizer(mu_x.float(), var_x.rsqrt().float(),
                        mu_y.float(), var_y.sqrt().float())


# =========================================================================== #
#                                 Probe models                                #
# =========================================================================== #

def make_act(name):
    return nn.GELU() if name == 'gelu' else nn.ReLU()


class LinearProbe(nn.Module):
    """Probe 1: flatten vectors -> single Linear (per-position linear)."""

    def __init__(self, d_in, d_out, hidden=None, act_name=None):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    """Probe 2: flatten vectors -> Linear-act-Linear."""

    def __init__(self, d_in, d_out, hidden, act_name):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = make_act(act_name)
        self.fc2 = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvProbe(nn.Module):
    """Probe 3: keep 2D -> single 3x3 Conv (padding=1 keeps resolution)."""

    def __init__(self, d_in, d_out, hidden=None, act_name=None):
        super().__init__()
        self.conv = nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ConvMLPProbe(nn.Module):
    """Probe 4: keep 2D -> 3x3 Conv-act-3x3 Conv."""

    def __init__(self, d_in, d_out, hidden, act_name):
        super().__init__()
        self.conv1 = nn.Conv2d(d_in, hidden, kernel_size=3, padding=1)
        self.act = make_act(act_name)
        self.conv2 = nn.Conv2d(hidden, d_out, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


PROBE_CLASSES = {'linear': LinearProbe, 'mlp': MLPProbe,
                 'conv': ConvProbe, 'convmlp': ConvMLPProbe}


# =========================================================================== #
#                       Adam training & generic evaluation                    #
# =========================================================================== #

@torch.no_grad()
def _split_mse(model, loader, stdz, device):
    """MSE in standardized space (for epoch selection)."""
    model.eval()
    tot, cnt = 0.0, 0
    for X, y in loader:
        Xs = stdz.apply_x(X.to(device))
        Ys = stdz.apply_y(y.to(device))
        tot += F.mse_loss(model(Xs), Ys, reduction='sum').item()
        cnt += Ys.numel()
    return tot / max(cnt, 1)


def train_probe(model, tr_loader, va_loader, stdz, device, args, seed):
    """Adam + cosine annealing, few epochs; keeps the best weights by
    validation loss.
    For flat layout the image-level batch is further cut into multiple
    optimizer steps of --batch_rows positions each, decoupling step count
    from image count (small datasets still get enough updates).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_val, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for X, y in tr_loader:
            Xs = stdz.apply_x(X.to(device))
            Ys = stdz.apply_y(y.to(device))
            # flat layout: step in blocks of positions; map layout: one step per image
            starts = (range(0, Xs.shape[0], args.batch_rows)
                      if Xs.dim() == 2 and Xs.shape[0] > args.batch_rows else [None])
            for lo in starts:
                sl = slice(lo, None if lo is None else lo + args.batch_rows)
                loss = F.mse_loss(model(Xs[sl]), Ys[sl])
                opt.zero_grad()
                loss.backward()
                opt.step()
                k = Ys[sl].numel()
                run += loss.item() * k
                seen += k
        sched.step()
        msg = f'      epoch {ep:3d}/{args.epochs} | train_mse {run / max(seen, 1):.6f}'
        if va_loader is not None:
            vl = _split_mse(model, va_loader, stdz, device)
            msg += f' | val_mse {vl:.6f}'
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, stdz, device):
    """Stream-accumulates sufficient statistics (original scale): per-channel
    R2/MSE + per-position cosine similarity."""
    d_out = stdz.mu_y.numel()
    zeros = lambda: torch.zeros(d_out, dtype=torch.float64)
    sum_p, sum_p2, sum_y, sum_y2, sum_py = zeros(), zeros(), zeros(), zeros(), zeros()
    sum_cos, n_loc = 0.0, 0
    model.eval()
    for X, y in loader:
        Xs = stdz.apply_x(X.to(device))
        P = model(Xs)
        P = stdz.unapply_y(P)  # de-standardize prediction to original scale, aligned with raw y
        if P.dim() == 4:  # (B, d, H, W) -> (B*H*W, d)
            P = P.permute(0, 2, 3, 1).reshape(-1, d_out)
            y = y.permute(0, 2, 3, 1).reshape(-1, d_out)
        p = P.detach().to('cpu').double()
        t = y.reshape(-1, d_out).double()
        sum_p += p.sum(0)
        sum_p2 += p.pow(2).sum(0)
        sum_y += t.sum(0)
        sum_y2 += t.pow(2).sum(0)
        sum_py += (p * t).sum(0)
        cos = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)
        sum_cos += cos.sum().item()
        n_loc += p.shape[0]
    return {'sum_p': sum_p, 'sum_p2': sum_p2, 'sum_y': sum_y,
            'sum_y2': sum_y2, 'sum_py': sum_py, 'sum_cos': sum_cos, 'n': n_loc}


def metrics_from_stats(st):
    """Recovers metrics from sufficient statistics. R2 is computed per
    channel then averaged (channels with variance≈0 dropped)."""
    n = st['n']
    mu_y = st['sum_y'] / max(n, 1)
    ss_res = st['sum_p2'] - 2 * st['sum_py'] + st['sum_y2']   # Σ(p-y)²
    ss_tot = st['sum_y2'] - n * mu_y ** 2                     # Σ(y-ȳ)²
    tot_max = float(ss_tot.max()) if n > 0 else 0.0
    valid = ss_tot > tot_max * 1e-9 if tot_max > 0 else torch.zeros_like(ss_tot, dtype=torch.bool)
    r2_c_full = torch.full((int(ss_tot.numel()),), float('nan'),
                           dtype=torch.float64)
    if bool(valid.any()):
        r2_c = 1.0 - ss_res[valid] / ss_tot[valid]
        r2_c_full[valid] = r2_c
        r2_avg = float(r2_c.mean())
        r2_var = float(1.0 - ss_res[valid].sum() / ss_tot[valid].sum())
        mse = float(ss_res[valid].sum() / (n * int(valid.sum())))
    else:
        r2_avg = r2_var = mse = float('nan')
    return {'r2_avg': r2_avg, 'r2_var': r2_var, 'mse': mse,
            'cos_sim': float(st['sum_cos'] / max(n, 1)), 'n_test_loc': int(n),
            'n_valid_channels': int(valid.sum()),
            # per-output-component R2 (NaN for invalid components). Under
            # position pairing length = target channel count; under patch
            # pairing length = d_out*kh*kw, reshape(d_out, kh*kw) then
            # aggregate per target channel (main loop writes
            # r2_channel_grouped)
            'r2_per_channel': r2_c_full.tolist()}


@torch.no_grad()
def energy_ratio(model, loader, stdz, device):
    """Energy gain of the linear probe: Σ||W·x_std||² / Σ||x_std||² (test
    set). W is the fitted linear map (no intercept, standardized space),
    measuring how much input energy survives / is amplified through W;
    computed on the same data as the main metrics for cross-input-module
    comparison."""
    W = model.fc.weight                      # (d_out, d_in), standardized space
    sum_wx2, sum_x2 = 0.0, 0.0
    for X, _ in loader:
        Xs = stdz.apply_x(X.to(device))
        wx = Xs @ W.T                        # W·x (per position)
        sum_wx2 += wx.pow(2).sum().item()
        sum_x2 += Xs.pow(2).sum().item()
    return sum_wx2 / max(sum_x2, 1e-12)


# =========================================================================== #
#                    Closed form: standardized ridge regression               #
# =========================================================================== #

class ClosedFormRidge:
    """Stream-accumulates [original-scale] Gram statistics, then a single
    eigendecomposition solves in standardized space:
        min_W ||Xs W - Ys||² + γ||W||²,  Xs/Ys are per-channel standardized
    solution W = (XsᵀXs + γI)⁻¹ XsᵀYs; no re-decomposition per γ on the
    grid.
    """

    def __init__(self, d_in, d_out, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.A = torch.zeros(d_in, d_in, dtype=dtype, device=device)   # XᵀX
        self.B = torch.zeros(d_in, d_out, dtype=dtype, device=device)  # XᵀY
        self.mx = torch.zeros(d_in, dtype=torch.float64)               # Σx
        self.my = torch.zeros(d_out, dtype=torch.float64)              # Σy
        self.sy2 = torch.zeros(d_out, dtype=torch.float64)             # Σy²
        self.n = 0

    @torch.no_grad()
    def update(self, X, Y):
        """X: (n, D_in) raw features; Y: (n, d_out) raw targets (CPU tensor)."""
        Xd = X.to(device=self.device, dtype=self.dtype)
        Yd = Y.to(device=self.device, dtype=self.dtype)
        self.A += Xd.T @ Xd
        self.B += Xd.T @ Yd
        self.mx += X.sum(0).double()
        self.my += Y.sum(0).double()
        self.sy2 += Y.pow(2).sum(0).double()
        self.n += X.shape[0]

    def solve(self, gamma_grid):
        """Returns ({γ: W (standardized space)}, Standardizer).
        Eigendecomposition on CPU float64."""
        n, dev, dt = self.n, self.device, self.dtype
        mu_x = (self.mx / n).to(device=dev, dtype=dt)
        mu_y = (self.my / n).to(device=dev, dtype=dt)
        var_y = (self.sy2 / n - self.my.pow(2) / n ** 2).clamp_min(1e-12)
        std_y = var_y.sqrt().to(device=dev, dtype=dt)
        # centering Gram identities: A_c = XᵀX - n·μμᵀ, B_c = XᵀY - n·μ_x μ_yᵀ
        A_c = self.A - n * torch.outer(mu_x, mu_x)
        B_c = self.B - n * torch.outer(mu_x, mu_y)
        var_x = torch.clamp(torch.diagonal(A_c) / n, min=1e-10)
        dinv = var_x.rsqrt()
        A_t = dinv[:, None] * A_c * dinv[None, :]           # XsᵀXs
        B_t = dinv[:, None] * B_c / std_y[None, :]          # XsᵀYs
        print('      closed form: eigendecomposition of the Gram matrix '
              '(CPU, float64)...', flush=True)
        # note: MPS has no float64; move back to CPU first, then upcast
        evals, evecs = torch.linalg.eigh(A_t.detach().to('cpu').to(torch.float64))
        evals = evals.to(device=dev, dtype=dt)
        evecs = evecs.to(device=dev, dtype=dt)
        evecs_tb = evecs.T @ B_t
        Ws = {g: evecs @ ((1.0 / (evals + g))[:, None] * evecs_tb)
              for g in gamma_grid}
        stdz = Standardizer(mu_x.detach().cpu().float(), dinv.detach().cpu().float(),
                            mu_y.detach().cpu().float(), std_y.detach().cpu().float())
        return Ws, stdz


@torch.no_grad()
def select_gamma(Ws, stdz, va_loader, device):
    """The γ with the smallest standardized-space MSE on the validation set."""
    if va_loader is None:
        return 1.0 if 1.0 in Ws else sorted(Ws)[len(Ws) // 2]
    sums = {g: 0.0 for g in Ws}
    cnt = 0
    for X, y in va_loader:
        Xs = stdz.apply_x(X.to(device))
        Ys = stdz.apply_y(y.to(device))
        for g, W in Ws.items():
            sums[g] += (Xs @ W - Ys).pow(2).sum().item()
        cnt += Ys.numel()
    return min(sums, key=lambda g: sums[g] / max(cnt, 1))


def ridge_to_linear(W):
    """Wraps the ridge solution W (x_std -> y_std) into a LinearProbe to
    reuse the unified evaluation."""
    model = LinearProbe(W.shape[0], W.shape[1]).to(W.device)
    with torch.no_grad():
        model.fc.weight.copy_(W.T)
        model.fc.bias.zero_()
    return model


# =========================================================================== #
#                               Single probe run                              #
# =========================================================================== #

def run_one(probe, target, inputs, d_in, d_out, ref_hw, bank, splits, args,
            device, seed, make_ds=None, label=None, return_model=False):
    """Runs train/solve + test evaluation for one (target, probe) pair and
    returns the metric dict.
    make_ds: optional dataset factory make_ds(folders) -> Dataset replacing
    the default ProbeDataset (e.g. a pixel-reconstruction probe uses raw
    images as regression targets); default behavior unchanged.
    label: optional label (e.g. input module name) printed before per-line
    results to distinguish origins.
    return_model: when True also returns (model, stdz) so cross analysis can
    reuse the predictor and standardization params.
    """
    tag = f'[{label}] ' if label else ''
    layout = PROBE_LAYOUTS[probe]
    if make_ds is None:
        make_ds = lambda fl: ProbeDataset(fl, bank, inputs, target, layout, ref_hw)
    tr_ds = make_ds(splits['train'])
    va_ds = make_ds(splits['val'])
    te_ds = make_ds(splits['test'])
    collate = make_collate(layout)

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=args.num_workers)

    tr_loader = mk(tr_ds, True)
    va_loader = mk(va_ds, False) if len(va_ds) else None
    te_ds_empty = len(te_ds) == 0
    te_loader = mk(te_ds, False) if not te_ds_empty else tr_loader  # degenerate: use the train set

    print(f'  [{probe}] {tag}layout={layout}, d_in={d_in} -> d_out={d_out}, '
          f'position counts train/val/test = '
          f'{len(tr_ds) * math.prod(ref_hw)}/{len(va_ds) * math.prod(ref_hw)}'
          f'/{len(te_ds) * math.prod(ref_hw)}', flush=True)

    t0 = time.time()
    use_closed = False
    if probe == 'linear':
        if args.linear_solver == 'closed':
            use_closed = True
        elif args.linear_solver == 'auto':
            if d_in <= args.closed_form_max_dim:
                use_closed = True
            else:
                print(f'      [note] D_in={d_in} > closed_form_max_dim='
                      f'{args.closed_form_max_dim}, linear falls back to '
                      f'Adam training (use --linear_solver closed to force '
                      f'the closed form)', flush=True)

    if use_closed:
        dtype = torch.float64 if args.gram_dtype == 'float64' else torch.float32
        ridge = ClosedFormRidge(d_in, d_out, device, dtype)
        for i, (X, Y) in enumerate(tr_loader):
            ridge.update(X, Y)
            if (i + 1) % 20 == 0:
                print(f'      Gram accumulation {i + 1} batches', flush=True)
        Ws, stdz = ridge.solve(args.gamma_grid)
        stdz = stdz.to(device)
        gamma = select_gamma(Ws, stdz, va_loader, device)
        model = ridge_to_linear(Ws[gamma])
        if getattr(args, 'save_deletion_dirs', None):
            # Cache standardized-space Gram statistics (sums of XsᵀXs /
            # XsᵀYs) for the SC1 iterative refit of fullspace deletion
            # directions; hung on a model attribute, not in state_dict,
            # invisible to other consumers.
            # note: ridge.A/B live on args.device (possibly cuda) while
            # mx/my/sy2 stay on CPU — move everything to CPU before
            # computing (same convention as solve's centering)
            n_r = ridge.n
            mu_x = ridge.mx / n_r                      # CPU float64
            mu_y = ridge.my / n_r
            A_dev = ridge.A.detach().to('cpu')
            B_dev = ridge.B.detach().to('cpu')
            A_c = A_dev - n_r * torch.outer(mu_x, mu_x)
            B_c = B_dev - n_r * torch.outer(mu_x, mu_y)
            var_x = torch.clamp(torch.diagonal(A_c) / n_r, min=1e-10)
            dinv = var_x.rsqrt()
            var_y = (ridge.sy2 / n_r - ridge.my.pow(2) / n_r ** 2).clamp_min(1e-12)
            std_y = var_y.sqrt()
            model._gram_stats_ = {
                'A_t': (dinv[:, None] * A_c * dinv[None, :]).to(torch.float64),
                'B_t': (dinv[:, None] * B_c / std_y[None, :]).to(torch.float64),
                'n': n_r,
            }
        solver_used = f'closed(ridge, γ={gamma:g})'
    else:
        stdz = compute_standardizer(tr_ds, args).to(device)
        hidden = (args.hidden_mlp if probe == 'mlp'
                  else args.hidden_conv if probe == 'convmlp' else 0)
        model = PROBE_CLASSES[probe](d_in, d_out, hidden, args.activation)
        model = train_probe(model, tr_loader, va_loader, stdz, device, args, seed)
        solver_used = f'adam({args.epochs}ep, lr={args.lr:g})'

    res = metrics_from_stats(evaluate(model, te_loader, stdz, device))
    if probe == 'linear':
        res['energy_ratio'] = energy_ratio(model, te_loader, stdz, device)
    res.update({'probe': probe, 'solver': solver_used,
                'seconds': round(time.time() - t0, 1),
                'd_in': d_in, 'd_out': d_out, 'ref_hw': list(ref_hw),
                'degenerate_test': te_ds_empty})
    gain_s = (f"  gain={res['energy_ratio']:.4f}"
              if 'energy_ratio' in res else '')
    print(f"    -> {tag}solver={res['solver']:<24} R2_avg={res['r2_avg']:.4f}  "
          f"R2_var={res['r2_var']:.4f}  MSE={res['mse']:.4g}  "
          f"cos={res['cos_sim']:.4f}{gain_s}  ({res['seconds']}s)", flush=True)
    if return_model:
        return res, model, stdz
    return res


# =========================================================================== #
#                              Splits & device                                #
# =========================================================================== #

@torch.no_grad()
def _corr_of_two_preds(model_a, stdz_a, model_b, stdz_b, d_out_a, d_out_b,
                       loader_a, loader_b, device):
    """Pearson correlation of two linear predictors' outputs on the same
    input sequence + each one's R2 statistics.
    loader_a / loader_b are strictly aligned batch by batch (same input
    layer, same grid, same split).
    Each spatial position's pair of output vectors is one sample point; all
    (position × channel) pairs are flattened and a single correlation is
    computed; predictions go through their own affine de-standardization,
    which does not affect the Pearson value.
    Pure evaluation (no training path), so gradient can be disabled
    globally."""
    zeros_a = lambda: torch.zeros(d_out_a, dtype=torch.float64)
    zeros_b = lambda: torch.zeros(d_out_b, dtype=torch.float64)
    (sum_pa, sum_pa2, sum_ya, sum_ya2, sum_pya) = (zeros_a(), zeros_a(),
                                                   zeros_a(), zeros_a(),
                                                   zeros_a())
    (sum_pb, sum_pb2, sum_yb, sum_yb2, sum_pyb) = (zeros_b(), zeros_b(),
                                                   zeros_b(), zeros_b(),
                                                   zeros_b())
    cross_ab, n_loc = 0.0, 0      # cross_ab = Σ p_a·p_b (position×channel flattened)
    model_a.eval()
    model_b.eval()
    for (X_a, y_a), (X_b, y_b) in zip(loader_a, loader_b):
        assert X_a.shape == X_b.shape, 'X must match exactly under the same input layer + grid'
        pa = stdz_a.unapply_y(model_a(stdz_a.apply_x(X_a.to(device))))
        pb = stdz_b.unapply_y(model_b(stdz_b.apply_x(X_b.to(device))))
        pa = pa.detach().to('cpu').double().reshape(-1, d_out_a)
        pb = pb.detach().to('cpu').double().reshape(-1, d_out_b)
        ta = y_a.reshape(-1, d_out_a).double()
        tb = y_b.reshape(-1, d_out_b).double()
        sum_pa += pa.sum(0)
        sum_pa2 += pa.pow(2).sum(0)
        sum_ya += ta.sum(0)
        sum_ya2 += ta.pow(2).sum(0)
        sum_pya += (pa * ta).sum(0)
        sum_pb += pb.sum(0)
        sum_pb2 += pb.pow(2).sum(0)
        sum_yb += tb.sum(0)
        sum_yb2 += tb.pow(2).sum(0)
        sum_pyb += (pb * tb).sum(0)
        cross_ab += float((pa * pb).sum())
        n_loc += pa.shape[0]

    # Pearson: (position, channel) pairs as samples, one dim each from the
    # two flattened output vectors
    n = n_loc * d_out_a
    Pa, Pa2 = float(sum_pa.sum()), float(sum_pa2.sum())
    Pb, Pb2 = float(sum_pb.sum()), float(sum_pb2.sum())
    num = n * cross_ab - Pa * Pb
    den = math.sqrt(max((n * Pa2 - Pa * Pa) * (n * Pb2 - Pb * Pb), 0.0))
    corr = float(min(max(num / den, -1.0), 1.0)) if den > 0 else float('nan')
    m_a = metrics_from_stats({'sum_p': sum_pa, 'sum_p2': sum_pa2,
                              'sum_y': sum_ya, 'sum_y2': sum_ya2,
                              'sum_py': sum_pya, 'sum_cos': 0.0,
                              'n': n_loc})
    m_b = metrics_from_stats({'sum_p': sum_pb, 'sum_p2': sum_pb2,
                              'sum_y': sum_yb, 'sum_y2': sum_yb2,
                              'sum_py': sum_pyb, 'sum_cos': 0.0,
                              'n': n_loc})
    return corr, m_a, m_b, n_loc


def cross_corr_profile(target_a, target_b, common_layers, d_by_module,
                       shapes, folders_all, f0, bank, splits, args, device,
                       per_models):
    """Two-target cross analysis (per-layer depth profile): for each common
    input layer m, two [linear] predictors (m→target_a and m→target_b)
    compute the Pearson correlation corr(a, b) of their outputs on the test
    set — one value per layer, forming the depth profile curve.
    Predictors prefer reusing the linear models already trained by the main
    loop's per-input probes (per_models); retrain when missing (e.g.
    --per_input_probes=none). Linear probes are per-position, independent of
    grid size, so reuse is unaffected by differing alignment grids; for each
    layer the two predictors forward on the same input features (that
    layer's features resized to its first-image grid), outputs are paired
    per position then flattened for the computation (requires equal target
    channel counts). High correlation = the two targets' linearly readable
    information overlaps heavily at that layer.
    note: this function must NOT be wrapped in @torch.no_grad() — the
    retraining path needs gradients (evaluation disables gradients locally
    inside _corr_of_two_preds).
    """
    profile = {'targets': [target_a, target_b], 'per_layer': {}}
    layout = PROBE_LAYOUTS['linear']
    for m in common_layers:
        entry = {'reused': True}
        models = {}
        for t in (target_a, target_b):
            cached = per_models.get(t, {}).get(m)
            if cached is not None:
                models[t] = cached
                continue
            entry['reused'] = False      # retrained (per-input probes disabled)
            ref_hw_m, _ = choose_ref_hw(shapes, folders_all, t, [m],
                                        args.align_to, f0)
            _, model_t, stdz_t = run_one(
                'linear', t, [m], d_by_module[m], d_by_module[t], ref_hw_m,
                bank, splits, args, device, seed=args.seed,
                label=f'corr·{t}', return_model=True)
            models[t] = (model_t, stdz_t)
        (model_a, stdz_a), (model_b, stdz_b) = (models[target_a],
                                                models[target_b])

        # unified eval grid: that layer's first-image input grid (both
        # models forward on the same X)
        ref_hw_c, _ = choose_ref_hw(shapes, folders_all, target_a, [m],
                                    'inputs', f0)

        def mk_loader(target):
            ds = ProbeDataset(splits['test'], bank, [m], target, layout,
                              ref_hw_c)
            if len(ds) == 0:  # same degenerate convention as run_one: empty test set falls back to train
                ds = ProbeDataset(splits['train'], bank, [m], target,
                                  layout, ref_hw_c)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=make_collate(layout),
                              num_workers=args.num_workers)

        corr, m_a, m_b, n_loc = _corr_of_two_preds(
            model_a, stdz_a, model_b, stdz_b, d_by_module[target_a],
            d_by_module[target_b], mk_loader(target_a), mk_loader(target_b),
            device)
        entry.update({'pearson': round(corr, 6),
                      'r2_a': m_a['r2_avg'], 'r2_b': m_b['r2_avg'],
                      'mse_a': m_a['mse'], 'mse_b': m_b['mse'],
                      'n_test_loc': int(n_loc)})
        profile['per_layer'][m] = entry
    profile['layers'] = list(common_layers)
    profile['pearson_profile'] = [profile['per_layer'][m]['pearson']
                                  for m in common_layers]
    return profile


# =========================================================================== #
#     Deletion-direction saving (cross-subspace deletion, ref/target5.txt)    #
# =========================================================================== #

@torch.no_grad()
def _sc1_iterate(stats, args, tag='', stats_pro_list=None):
    """SC1 protocol (aligned with probe_projection, standardized space) +
    optional protected-subspace subtraction.

    Initial deletion basis = orth(B_t) — the column space of the
    standardized-space cross-covariance Cov(Xs,Ys).
    Conclusion of probe_projection.cross_cov: using the column space of the
    OLS readout W as the initial basis, under an ill-conditioned spectrum
    W's low-variance directions are dominated by 1/√λ estimation noise,
    leaving cross-covariance residue that iterations cannot clean; orth(B)
    instead makes Cov(z_⊥,y)=(I−SSᵀ)B strictly zero, zeroed in one round.
    Criterion: refit min-norm OLS after deletion (pseudo-inverse truncated
    at ols_rcond, no ridge); its train R² must be <= --deletion_r2_target
    (aligned with projection --sc1_threshold); when numeric residue exceeds
    the threshold, expand with the [top directions of the residual-readout
    SVD] (expand_topk) and re-verify — normally converges in one round, the
    loop is only a backstop.

    Protect mode (stats_pro_list non-empty, --deletion_protect): before
    deletion, project the initial basis onto the orthogonal complement of
    the protected subspace (P ← orth((I−S_proS_proᵀ)·orth(B_t)), an oblique
    projection), so the deletion operator strictly never touches the
    protected targets' subspace (S_proᵀP=0 ⟹ deletion changes none of the
    protected signal's Cov components); the cost is that target components
    overlapping the protected subspace are kept (deletion incomplete). In
    this mode the R² threshold criterion does not apply (residue = overlap,
    not failure); one pass, no iteration; instead report the dual metrics
    [target residual R²] and [protected-signal R² before→after].

    stats: {'A_t': (d,d) sum of XsᵀXs, 'B_t': (d,d_out) sum of XsᵀYs,
            'n': sample count} — train-set standardized-space statistics,
            attached to model._gram_stats_ by run_one's closed-form path.
    stats_pro_list: list of the same statistics per protected target
            (may be empty).
    Returns (P (k,d) float32 row-orthonormal, iteration count,
          post-deletion refit train R² of the target, singular-value
          spectrum, protection metrics dict or None)."""
    C = stats['A_t'].to(torch.float64)
    B = stats['B_t'].to(torch.float64)
    n = float(stats['n'])
    d = C.shape[0]
    eye = torch.eye(d, dtype=C.dtype)
    # initial basis: orth(B_t) — left-singular truncation of the B_t
    # (d,d_out) column space (aligned with orth_basis)
    U0, s0, _ = torch.linalg.svd(B, full_matrices=False)
    keep0 = s0 > args.deletion_orth_rcond * float(s0[0])
    P = U0[:, keep0].T.contiguous()                        # (k0, d) rows = directions
    svals = s0[keep0]
    print(f'    [SC1] {tag} initial basis orth(B) rank={P.shape[0]}/{B.shape[1]}, '
          f'top-3 svals={[round(float(v), 3) for v in svals[:3]]}', flush=True)

    if stats_pro_list:
        return _selective_delete(C, B, P, n, d, eye, args, tag,
                                 stats_pro_list, svals)
    r2 = float('nan')
    for it in range(1, args.deletion_max_iter + 1):
        M = eye - P.T @ P
        C_del = M @ C @ M.T
        # refit min-norm OLS after deletion: W' = C_del⁺ (M B)
        # (ols_rcond truncation, no ridge)
        evals, evecs = torch.linalg.eigh(C_del)
        emax = float(evals.max().clamp_min(1e-30))
        keep = evals > args.deletion_rcond * emax
        C_inv = evecs @ torch.diag(
            torch.where(keep, 1.0 / evals.clamp_min(1e-300),
                        torch.zeros_like(evals))) @ evecs.T
        W_new = C_inv @ (M @ B)                            # (d, d_out), columns = readout directions
        # train R²_c = fitted variance in standardized space (train
        # Var(y_s)=1) → per-channel mean
        fit = torch.diagonal(W_new.T @ C_del @ W_new) / n
        r2 = float(fit.mean())
        verdict = 'PASS' if r2 <= args.deletion_r2_target else 'FAIL'
        print(f'    [SC1] {tag} iter {it}: k={P.shape[0]}, '
              f'post-deletion refit train R²={r2:.4f} '
              f'(threshold {args.deletion_r2_target:g}) [{verdict}]', flush=True)
        if r2 <= args.deletion_r2_target:
            return P.to(torch.float32), it, r2, svals, None
        # numeric residue → expand with top directions of the residual-readout
        # SVD (aligned with projection expand_topk)
        if args.deletion_expand_topk <= 0:
            print(f'    [SC1] {tag} expand_topk=0, stop expanding', flush=True)
            break
        U2, _, _ = torch.linalg.svd(W_new, full_matrices=False)
        m = min(args.deletion_expand_topk, U2.shape[1])
        S = U2[:, :m].T                                    # (m, d) rows = directions
        Uc, sc, Vhc = torch.linalg.svd(torch.cat([P, S], 0),
                                       full_matrices=False)
        P = Vhc[sc > args.deletion_orth_rcond * float(sc[0])]  # joint orthonormalization
        if P.shape[0] >= d:
            print(f'    [SC1] {tag} deletion set has reached d={d}, stopping early', flush=True)
            break
    print(f'    [SC1] {tag} threshold not reached (k={P.shape[0]}, train R²={r2:.4f}); '
          f'try smaller --deletion_orth_rcond or larger --deletion_expand_topk',
          flush=True)
    return P.to(torch.float32), args.deletion_max_iter, r2, svals, None


@torch.no_grad()
def _selective_delete(C, B, P0, n, d, eye, args, tag, stats_pro_list, svals0):
    """Protect mode (selective deletion: oblique projection + iteratively
    absorbing surrogate directions).

    The protected basis uses the readout-direction convention orth(C⁺B_p)
    (not orth(B_p)): (MCM)⁺=MC⁺M ⟹ when the readout direction w=C⁺b ⊥
    span(P), the refit readout after deletion satisfies
    bᵀ(MC⁺M)b=bᵀC⁺b, so the protected R² is strictly unchanged; whitening C
    would rotate span(B_p), and subtracting only span(B_p) cannot protect
    R².

    Why iteration is required: after partial deletion (M@B_t≠0, shared
    components kept), a refit readout on near-collinear features recovers
    the deleted part from surrogate directions (a single oblique projection
    leaves the target R² untouched in practice). Per round
    [delete → refit target readout → subtract protection from the new
    readout directions and merge], residue decreases monotonically and
    converges to the geometric lower bound = the target's components shared
    with the protected subspace (the price of "not aiming for complete
    deletion", determined by the two targets' subspace overlap, not a
    hyperparameter).

    Returns (P, iteration count, target residual R², initial-basis singular
    spectrum, protection metrics dict).

    The truncation threshold is anchored to the pre-deletion global spectrum
    (rcond×λmax(C)), not each matrix's own relative threshold: after
    deleting high-spectrum directions λmax(C_del) drops, the relative
    threshold drops with it, low-variance directions that were numerically
    unreachable come back, producing the illusion "R² rises after deletion"
    (the two sides' reachable sets then differ, incomparable). Anchoring
    gives the same ruler before and after deletion."""

    ev0 = torch.linalg.eigvalsh(C)
    tau_abs = args.deletion_rcond * float(ev0.max().clamp_min(1e-30))

    def gram_pinv(Mx):
        ev, evec = torch.linalg.eigh(Mx)
        keep = ev > tau_abs
        return evec @ torch.diag(torch.where(
            keep, 1.0 / ev.clamp_min(1e-300),
            torch.zeros_like(ev))) @ evec.T

    Ss = []
    for sp in stats_pro_list:
        Bp = sp['B_t'].to(torch.float64)
        Wp_ols = gram_pinv(C) @ Bp                      # protected-signal OLS readout directions
        Up, sup, _ = torch.linalg.svd(Wp_ols, full_matrices=False)
        Ss.append(Up[:, sup > args.deletion_orth_rcond * float(sup[0])])
    Sp = torch.linalg.qr(torch.cat(Ss, 1).contiguous())[0]  # (d, kp) joint orthonormal basis

    def project_out(rows):
        """Subtracts the protected subspace from a set of row vectors,
        returns the orthonormalized row basis"""
        r = rows.to(torch.float64) - (rows.to(torch.float64) @ Sp) @ Sp.T
        U, s, _ = torch.linalg.svd(r.T, full_matrices=False)
        if s.numel() == 0 or float(s[0]) <= 0:
            return torch.zeros(0, d, dtype=torch.float64)
        keep = s > args.deletion_orth_rcond * float(s[0])
        return U[:, keep].T

    def metrics(P_):
        """Under deletion state P_, the min-norm refit train R²_c of the
        target and each protected signal"""
        M = eye - P_.T @ P_
        C_del = M @ C @ M.T
        C_inv = gram_pinv(C_del)
        out = {}
        W = C_inv @ (M @ B)
        out['del'] = float((torch.diagonal(W.T @ C_del @ W) / n).mean())
        pros = []
        for sp in stats_pro_list:
            Bp = sp['B_t'].to(torch.float64)
            Wp = C_inv @ (M @ Bp)
            pros.append(float((torch.diagonal(Wp.T @ C_del @ Wp) / n).mean()))
        out['pro'] = sum(pros) / len(pros)
        return out

    P = project_out(P0)
    print(f'    [SC1] {tag} protection subtraction: protected subspace dim={Sp.shape[1]}, '
          f'k {P0.shape[0]}->{P.shape[0]}', flush=True)
    m0 = metrics(torch.zeros(0, d, dtype=torch.float64))    # before deletion
    if P.shape[0] == 0:
        print(f'    [SC1] {tag} initial basis lies entirely inside the '
              f'protected subspace, no deletable directions (k=0)',
              flush=True)
        pro_info = {'r2_del_before': m0['del'], 'r2_pro_before': m0['pro'],
                    'r2_pro_after': m0['pro']}
        return (torch.zeros(0, d, dtype=torch.float32), 1, m0['del'],
                torch.zeros(0), pro_info)

    m = metrics(P)
    r2_del = m['del']
    it_fin = 1
    print(f'    [SC1] {tag} iter 1: k={P.shape[0]}, target R² {m0["del"]:.4f}->'
          f'{r2_del:.4f}, protected R² {m0["pro"]:.4f}->{m["pro"]:.4f}', flush=True)
    for it in range(2, args.deletion_max_iter + 1):
        # surrogate readout directions → subtract protection → merge into
        # the deletion set (keeps P ⊥ span(C⁺B_p))
        M = eye - P.T @ P
        C_del = M @ C @ M.T
        C_inv = gram_pinv(C_del)
        W_new = C_inv @ (M @ B)                          # (d, d_out) surrogate readout
        U2, s2, _ = torch.linalg.svd(W_new, full_matrices=False)
        if not bool((s2 > args.deletion_orth_rcond * float(s2[0])).any()):
            break
        S_new = project_out(U2.T)                        # (m, d) rows, protection subtracted
        if S_new.shape[0] == 0:
            break
        Uc, sc, Vhc = torch.linalg.svd(torch.cat([P, S_new], 0),
                                       full_matrices=False)
        P = Vhc[sc > args.deletion_orth_rcond * float(sc[0])]
        m_new = metrics(P)
        improved = r2_del - m_new['del']
        it_fin = it
        print(f'    [SC1] {tag} iter {it}: k={P.shape[0]}, target R²->'
              f'{m_new["del"]:.4f}, protected R² {m_new["pro"]:.4f}', flush=True)
        if m_new['del'] <= args.deletion_r2_target:
            r2_del = m_new['del']
            break                                        # threshold met (possible when overlap is small)
        if improved <= 1e-4 * max(abs(r2_del), 1e-12):
            r2_del = m_new['del']
            break                                        # already at the shared lower bound
        r2_del = m_new['del']
    pro_info = {'r2_del_before': m0['del'], 'r2_pro_before': m0['pro'],
                'r2_pro_after': m['pro']}
    print(f'    [SC1] {tag} selective deletion converged: k={P.shape[0]}, target R² '
          f'{m0["del"]:.4f}->{r2_del:.4f} (residue=shared lower bound), protected R² '
          f'{m0["pro"]:.4f}->{m["pro"]:.4f} (loss '
          f'{m0["pro"] - m["pro"]:.4f})', flush=True)
    return P.to(torch.float32), it_fin, r2_del, svals0, pro_info


@torch.no_grad()
def _deletion_entries(per_models, results, targets, args):
    """Extracts the deletion-direction entry list from per-input linear
    probes.

    fullspace (default): the SC1 protocol (aligned with probe_projection);
    directions are a [standardized-space] orthonormal basis, initial set =
    orth(B_t) — the column space of the standardized-space cross-covariance
    Cov(Xs,Ys) (probe_projection's conclusion: the OLS readout W's column
    space is polluted by 1/√λ noise directions under an ill-conditioned
    spectrum, leaving cross-covariance residue after projection; orth(B_c)
    zeroes it in one round, train R²≈0);
    requires the Gram statistics cached by run_one's closed-form path
    (model._gram_stats_); when the Adam path has no statistics, degrades to
    a single projection (original-space W_raw) with a warning.
    r2_target/topk (controls): right-singular directions of
    W_raw = W_std·diag(inv_std_x) (original representation space) —
    r2_target relies on the diagonal-covariance approximation "each
    direction has data variance≈1", badly distorted when real ViT feature
    channels are strongly correlated (re-probing R² after deletion barely
    drops); a single column-space projection can also be bypassed by
    near-collinear surrogate directions — both are kept as controls only.
    Each entry's space field marks the direction space for pass2
    (delete_direction.py) branching.
    """
    entries = []
    for t in targets:
        for m, (model, stdz) in per_models.get(t, {}).items():
            res = results[t]['per_input'].get(m, {}).get('linear')
            if res is None:
                continue
            r2_full = float(res.get('r2_avg', float('nan')))
            W = model.fc.weight.detach().to('cpu').to(torch.float64)  # (d_out, d_in)
            inv = stdz.inv_std_x.detach().to('cpu').to(torch.float64)  # (d_in,)
            W_raw = W * inv[None, :]
            # SVD on CPU float64 (MPS has no float64; same convention as
            # the Gram solve)
            _, svals, Vh = torch.linalg.svd(W_raw, full_matrices=False)
            V = Vh.T                                        # (d_in, r) columns = right-singular directions
            r = int(svals.numel())
            cum = torch.cumsum(svals.pow(2) / svals.pow(2).sum(), 0)
            k = 0
            dirs_final = V[:, :k].T.contiguous()           # (k, d_in)
            entry_extra = {'space': 'original', 'iters': 0}
            svals_entry = svals                       # default: W_raw spectrum (matching dirs)
            protect_names = [p for p in (getattr(args, 'deletion_protect', None)
                                         or []) if p != t]
            if args.deletion_mode == 'fullspace':
                stats = getattr(model, '_gram_stats_', None)
                if stats is not None:
                    stats_pro_list, protect_used = [], []
                    for p in protect_names:
                        mp = per_models.get(p, {}).get(m)
                        sp = getattr(mp[0], '_gram_stats_', None) if mp else None
                        if sp is None:
                            print(f'[warn] {m}->{t}: protected target {p} has no '
                                  f'Gram statistics (not closed-form or '
                                  f'nonexistent), skip this protection')
                            continue
                        stats_pro_list.append(sp)
                        protect_used.append(p)
                    P, iters, r2_fin, svals_b, pro_info = _sc1_iterate(
                        stats, args, tag=f'{m}->{t}',
                        stats_pro_list=stats_pro_list)
                    dirs_final, k = P, int(P.shape[0])
                    entry_extra = {'space': 'standardized', 'iters': iters,
                                   'r2_sc1': float(r2_fin)}
                    svals_entry = svals_b             # orth(B_t) singular values (matching dirs)
                    if pro_info is not None:
                        entry_extra.update({
                            'protect': protect_used,
                            'r2_del_before': float(pro_info['r2_del_before']),
                            'r2_pro_before': float(pro_info['r2_pro_before']),
                            'r2_pro_after': float(pro_info['r2_pro_after'])})
                else:
                    print(f'[warn] {m}->{t}: no Gram statistics (linear not in '
                          f'closed form), fullspace degrades to a single '
                          f'column-space projection without SC1 verification '
                          f'— deletion may be incomplete on near-collinear '
                          f'features, prefer --linear_solver closed')
                    k = int((svals > args.deletion_rcond *
                             float(svals[0])).sum())
                    dirs_final = V[:, :k].T.contiguous()
            elif args.deletion_mode == 'topk':
                k = min(max(args.deletion_k, 0), r)
                dirs_final = V[:, :k].T.contiguous()
            elif r2_full == r2_full and r2_full > args.deletion_r2_target:
                need = 1.0 - args.deletion_r2_target / r2_full
                k = min(int((cum < need).sum()) + 1, r)
                dirs_final = V[:, :k].T.contiguous()
                if float(cum[-1]) < need:
                    print(f'[warn] {m}->{t}: the singular spectrum carries too '
                          f'little energy to push R² down to '
                          f'{args.deletion_r2_target:g} (deleting all only '
                          f'reaches {r2_full * float(1.0 - cum[-1]):.4f}), '
                          f'using k={k}')
            elif r2_full == r2_full and k == 0:
                print(f'[note] {m}->{t}: R²_full={r2_full:.4f} already not '
                      f'above target {args.deletion_r2_target:g}, k=0 '
                      f'(modulation is identity)')
            sc1 = entry_extra['space'] == 'standardized'
            r2_after = entry_extra.pop('r2_sc1', None) if sc1 else \
                ((r2_full * float(1.0 - cum[k - 1])) if k else
                 (r2_full if r2_full == r2_full else None))
            entries.append({
                'input': m, 'target': t, 'd': int(W_raw.shape[1]),
                'rank': r, 'k': int(k),
                'r2_full': r2_full if r2_full == r2_full else None,
                'r2_after': r2_after,
                'energy_frac_kept': (None if sc1 else
                                     (float(1.0 - cum[k - 1]) if k else 1.0)),
                'svals': svals_entry,
                'svals_full': svals,
                'dirs': dirs_final,
                **entry_extra,
            })
    return entries


@torch.no_grad()
def save_deletion_dirs(path, per_models, results, targets, args):
    """Saves the deletion directions as npz (key style matches
    probe_projection.save_basis_npz):
        dirs__{input}__{target}       (k, d) float32  deletion directions, one per row (orthonormal)
        svals__{input}__{target}      (k,) float32    matching singular values (that direction's readout strength)
        svals_full__{input}__{target} (r,) float32    full singular spectrum (reference for re-choosing k)
        mu__{input}__{target}         (d,) float32    input module's train-set per-channel mean (for reference)
        inv_std__{input}__{target}    (d,) float32    input module's train-set per-channel 1/std (for reference)
        __meta__                      JSON string     protocol params + per-entry detail/key names
    External modulation (after flattening the representation to (…, d) by
    channel): flat_⊥ = flat − (flat@dirs.T)@dirs.
    Disk strategy matches transform_network: write to the current directory
    first, then move to the target path (rename fallback across filesystems
    on network storage)."""
    entries = _deletion_entries(per_models, results, targets, args)
    if not entries:
        raise RuntimeError('no usable per-input linear predictors, cannot '
                           'save deletion directions')
    meta = {
        'protocol': 'target5 cross-subspace deletion: P_perp = I − V_kV_kᵀ '
                    'applied to the original representation; after '
                    'flattening the representation to (N, d) by channel, '
                    'flat_perp = flat − (flat @ dirs.T) @ dirs, shape '
                    'unchanged',
        'space': 'fullspace: standardized (orth(B_t) row-orthonormal basis; '
                 'standardize before deleting); '
                 'r2_target/topk: original (right-singular directions of '
                 'W_raw = W_std·diag(inv_std_x))',
        'feature_root': args.feature_root,
        'input_layout': args.input_layout,
        'deletion_mode': args.deletion_mode,
        'deletion_k': args.deletion_k,
        'deletion_r2_target': args.deletion_r2_target,
        'deletion_rcond': args.deletion_rcond,
        'deletion_orth_rcond': args.deletion_orth_rcond,
        'deletion_protect': getattr(args, 'deletion_protect', None),
        'deletion_max_iter': args.deletion_max_iter,
        'entries': [],
    }
    payload = {}
    for e in entries:
        keys = {name: f'{name}__{e["input"]}__{e["target"]}'
                for name in ('dirs', 'svals', 'svals_full', 'mu', 'inv_std')}
        stdz = per_models[e['target']][e['input']][1]
        payload[keys['dirs']] = e['dirs'].to(torch.float32).numpy()
        payload[keys['svals']] = e['svals'][:e['k']].to(torch.float32).numpy()
        payload[keys['svals_full']] = e['svals_full'].to(torch.float32).numpy()
        payload[keys['mu']] = stdz.mu_x.detach().cpu().numpy().astype(np.float32)
        payload[keys['inv_std']] = stdz.inv_std_x.detach().cpu().numpy().astype(np.float32)
        meta['entries'].append({**{kk: e.get(kk) for kk in
                                   ('input', 'target', 'd', 'rank', 'k',
                                    'r2_full', 'r2_after', 'energy_frac_kept',
                                    'space', 'iters', 'protect',
                                    'r2_del_before', 'r2_pro_before',
                                    'r2_pro_after')},
                                'keys': keys})
    payload['__meta__'] = json.dumps(meta, ensure_ascii=False)

    name = Path(path).name
    if not name.endswith('.npz'):
        name += '.npz'    # savez auto-appends when the name lacks .npz; add it explicitly to keep the filename deterministic
    dst = Path(path)
    if dst.suffix != '.npz':
        dst = dst.with_name(dst.name + '.npz')   # final filename deterministically ends with .npz
    np.savez_compressed(name, **payload)
    if Path(name).resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(name, str(dst))
    print(f'\ndeletion directions saved to: {path} (mode={args.deletion_mode}, '
          f'{len(entries)} input x target combinations in total)')
    for e in meta['entries']:
        if e.get('space') == 'standardized':
            if e.get('protect'):
                tail = (f", protect {','.join(e['protect'])}: target R² "
                        f"{e.get('r2_del_before', float('nan')):.4f}->"
                        f"{e['r2_after']:.4f} (residue=overlapping "
                        f"components), protected-signal R² "
                        f"{e.get('r2_pro_before', float('nan')):.4f}->"
                        f"{e.get('r2_pro_after', float('nan')):.4f}")
            else:
                tail = (f", SC1 {e.get('iters', '?')} iters, "
                        f"post-deletion refit train R² = {e['r2_after']:.4f}")
        else:
            r2a = e['r2_after']
            tail = (f", R² {e['r2_full']:.4f} -> "
                    f"~{'nan' if r2a is None else f'{r2a:.4f}'}")
        print(f"  {e['input']} -> {e['target']}: d={e['d']}, "
              f"k={e['k']}/{e['rank']}, space={e.get('space', 'original')}"
              f"{tail}")
    print('  external modulation: flat_⊥ = flat − (flat@dirs.T)@dirs'
          ' (the npz __meta__ carries the protocol and per-entry key names)')


def stratified_split(folders, ratios, seed):
    """Class-stratified image-level split; degrades safely when images
    are too few."""
    n = len(folders)
    if n < 3:
        print('[warn] fewer than 3 images, degrading to train=val=test full '
              'set, metrics are reference only')
        return ({'train': list(folders), 'val': list(folders), 'test': list(folders)},
                True)
    groups = {}
    for f in folders:
        groups.setdefault(class_of(f), []).append(f)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for c in sorted(groups):
        members = sorted(groups[c])
        rng.shuffle(members)
        k = len(members)
        n_va = min(int(round(k * ratios[1])), max(k - 1, 0))
        n_te = min(int(round(k * ratios[2])), max(k - 1 - n_va, 0))
        va += members[:n_va]
        te += members[n_va:n_va + n_te]
        tr += members[n_va + n_te:]
    if not va or not te:  # empty after stratification -> simple random split
        idx = list(range(n))
        rng.shuffle(idx)
        n_tr = int(round(n * ratios[0]))
        n_va = max(1, int(round(n * ratios[1])))
        tr = [folders[i] for i in idx[:n_tr]]
        va = [folders[i] for i in idx[n_tr:n_tr + n_va]]
        te = [folders[i] for i in idx[n_tr + n_va:]]
    return {'train': tr, 'val': va, 'test': te}, False


def pick_device(spec):
    if spec != 'auto':
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def estimate_cache_bytes(folders, modules, shapes):
    if not folders:
        return 0
    f0 = folders[0].name
    per_img = sum(math.prod(shapes[(f0, m)]) for m in modules) * 4  # float32
    return per_img * len(folders)


def choose_ref_hw(shapes, folders, target, inputs, mode, f0):
    """Chooses the unified alignment grid (per-position probes require
    inputs and target on the same grid):
    - target: unify to the target grid, all input modules interpolated to
      the target shape;
    - inputs: unify to the input grid, target interpolated to the input
      shape; when input grids disagree, take the first input's grid on the
      first image and warn;
    - auto  : equivalent to inputs when the input modules agree on the
      first image's grid (only the target is interpolated once, inputs keep
      native resolution); otherwise falls back to target.
    Returns (ref_hw, source description).
    Lazy-shape note: shapes contains only first-image (f0, m) keys; grid
    detection looks only at the first image; images whose grid differs from
    the first are uniformly interpolated by resize_spatial (does not affect
    correctness) — no more full scans of all images for detection.
    """
    tgt_hw = shapes[(f0, target)][:2]
    if mode == 'target':
        return tgt_hw, 'target'
    hw_set = {shapes[(f0, m)][:2] for m in inputs if (f0, m) in shapes}
    if len(hw_set) == 1:
        return next(iter(hw_set)), 'inputs'
    if mode == 'inputs':
        in_hw = shapes[(f0, inputs[0])][:2]
        print(f'[warn] --align_to=inputs but input-module grids disagree '
              f'{sorted(hw_set)}, unifying to the first input {inputs[0]} '
              f'grid {in_hw}; other modules will be interpolated to align')
        return in_hw, 'inputs-forced'
    print(f'[note] input-module grids disagree {sorted(hw_set)}, '
          f'unifying to the target {target} grid {tgt_hw} '
          f'(use --align_to inputs to force aligning to inputs)')
    return tgt_hw, 'target'


def build_pairing(bank, shapes, folders, f0, target, inputs, d_out, args):
    """Builds the input-target pairing scheme per --pair_mode, returns a
    dict:
    position: {'ref_hw', 'd_out_eff'=d_out, 'make_ds'=None, 'align_src'}
              -> run_one uses the default ProbeDataset (unified grid,
              per-position, behavior unchanged);
    patch:    {'ref_hw'=input grid, 'd_out_eff'=d_out*kh*kw,
               'make_ds'=PatchPairDataset factory, 'patch_hw'=(kh, kw),
               'tgt_hw'}
              -> M2 zero-resize measurement: inputs and target keep native
              grids, the target is blocked by the input token's receptive
              field, each token regresses all target values in its kh×kw
              neighborhood.
    Per-input probing allows different input grids (one input paired at a
    time); concatenated probing requires them to agree."""
    if args.pair_mode == 'position':
        ref_hw, align_src = choose_ref_hw(shapes, folders, target, inputs,
                                          args.align_to, f0)
        return {'ref_hw': ref_hw, 'd_out_eff': d_out, 'make_ds': None,
                'align_src': align_src}
    in_hws = {tuple(int(v) for v in shapes[(f0, m)][:2]) for m in inputs}
    if len(in_hws) != 1:
        raise RuntimeError(f'--pair_mode patch concatenated probing requires '
                           f'all input-module grids to agree, got '
                           f'{sorted(in_hws)}; probe inputs one by one instead')
    in_hw = next(iter(in_hws))
    tgt_hw = tuple(int(v) for v in shapes[(f0, target)][:2])
    kh, kw = tgt_hw[0] // in_hw[0], tgt_hw[1] // in_hw[1]
    if kh < 1 or kw < 1 or kh * in_hw[0] != tgt_hw[0] \
            or kw * in_hw[1] != tgt_hw[1]:
        raise RuntimeError(f'--pair_mode patch requires the target grid '
                           f'{tgt_hw} to be divisible by the input grid '
                           f'{in_hw}, actual patch = ({kh}, {kw})')

    def make_ds(fl):
        return PatchPairDataset(fl, bank, inputs, target, in_hw,
                                (kh, kw), tgt_hw)

    return {'ref_hw': in_hw, 'd_out_eff': d_out * kh * kw, 'make_ds': make_ds,
            'patch_hw': (kh, kw), 'tgt_hw': tgt_hw}


# =========================================================================== #
#                                   Main                                       #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='2D feature probes: predict target-module features from '
                    'multiple input-module features (linear/mlp/conv/convmlp '
                    'probes, closed-form ridge regression supported)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument('--feature_root', type=str, required=True,
                   help='feature root directory, containing '
                        '{class}_{image_name}/ folders')
    p.add_argument('--target', type=str, nargs='+', required=True,
                   help='target module name(s) (=npy filename without '
                        'extension); multiple targets are probed one by one')
    p.add_argument('--inputs', type=str, nargs='*', default=None,
                   help='input module name list; defaults to all modules '
                        'except the target')
    p.add_argument('--max_images', type=int, default=None,
                   help='only use features of the first N images in '
                        'folder-name sorted order')
    p.add_argument('--align_to', type=str, default='auto',
                   choices=['auto', 'target', 'inputs'],
                   help='unified alignment grid: auto=align to the input '
                        'grid when all input grids agree (the target is '
                        'interpolated over), otherwise align to the target '
                        'grid; target=always align to the target grid; '
                        'inputs=always align to the input grid (on '
                        'disagreement use the first input grid and warn)')
    p.add_argument('--pair_mode', type=str, default='position',
                   choices=['position', 'patch'],
                   help='input-target pairing: position=unified grid per '
                        'position (default, inputs and target mutually '
                        'interpolated to the reference grid); patch=M2 '
                        'zero-resize measurement, inputs and target keep '
                        'native grids, the target is blocked by the input '
                        'token receptive field (kh×kw), each token '
                        'regresses a d_out*kh*kw-dim target (linear/mlp '
                        'only; the target grid must be divisible by the '
                        'input grid; r2_channel_grouped in the results is '
                        'R² aggregated per target channel, r2_avg is the '
                        'mean over all in-patch components)')
    p.add_argument('--input_layout', type=str, default='hwd', choices=['hwd', 'chw'],
                   help='npy feature dim layout: hwd=(h, w, d) channels '
                        'last; chw=(c, h, w) channels first (torch (C,H,W) '
                        "feature maps, ViT features saved from token "
                        "sequences rearranged to 'b c h w', etc). A wrong "
                        'layout makes all module metrics ≈0 and nearly '
                        'identical')
    # probes
    p.add_argument('--probe', type=str, nargs='+', default=None,
                   choices=PROBE_NAMES,
                   help='probes for the [all-inputs-concatenated] run '
                        '(linear=flatten+Linear, mlp=flatten+Linear-act-'
                        'Linear, conv=3x3 conv, convmlp=3x3 conv-act-3x3 '
                        'conv); by default no concatenated run, only '
                        'per-input-module probes (see --per_input_probes)')
    p.add_argument('--per_input_probes', type=str, nargs='+', default=['linear'],
                   choices=PROBE_NAMES + ['none'],
                   help='probes used when each input module separately '
                        'probes the target (default linear=closed form, '
                        'nearly free); pass none to disable')
    p.add_argument('--hidden_mlp', type=int, default=1024, help='mlp hidden width')
    p.add_argument('--hidden_conv', type=int, default=256, help='convmlp hidden channel count')
    p.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    # training hyper-params (only for probes without a closed form)
    p.add_argument('--epochs', type=int, default=20,
                   help='Adam epochs (conv probes converge slower, consider --lr 3e-3)')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Adam lr; 3e-3 suggested for small data / conv probes')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=8,
                   help='images loaded per batch (map layout: images per optimizer step)')
    p.add_argument('--batch_rows', type=int, default=8192,
                   help='positions per optimizer step in flat layout (cuts the image-level batch further)')
    # closed form
    p.add_argument('--linear_solver', type=str, default='auto',
                   choices=['auto', 'closed', 'adam'],
                   help='linear probe solver: auto=closed form when dims allow')
    p.add_argument('--closed_form_max_dim', type=int, default=16384,
                   help='auto mode uses the closed form when D_in does not '
                        'exceed this (the Gram needs D_in²*4B memory, '
                        '16384 ≈ 1GB)')
    p.add_argument('--gamma_grid', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0], help='ridge regression regularization grid')
    p.add_argument('--gram_dtype', type=str, default='float32',
                   choices=['float32', 'float64'], help='Gram accumulation dtype')
    # misc
    p.add_argument('--split_ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   help='train/val/test image split ratios')
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0, help='training random seed')
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or a specific cuda/mps/cpu')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers (>0 duplicates the memory cache across processes)')
    p.add_argument('--cache_ram', type=str, default='auto', choices=['auto', 'on', 'off'],
                   help='feature memory cache: big speedup when scanning multiple targets')
    p.add_argument('--cache_ram_gb', type=float, default=8.0,
                   help='auto mode enables the cache when total data does not exceed this (GB)')
    # deletion-direction saving (ref/target5.txt cross-subspace deletion)
    p.add_argument('--save_deletion_dirs', type=str, default=None,
                   help='save path (supplying it enables saving): store '
                        'the right-singular directions of the per-input '
                        'linear probe readout matrix column space '
                        '(original representation space) as npz, for '
                        'externally modulating representations with '
                        'P⊥ = I − V_kV_kᵀ (target5 protocol)')
    p.add_argument('--deletion_mode', type=str, default='fullspace',
                   choices=['fullspace', 'r2_target', 'topk'],
                   help='how k is chosen: fullspace=SC1 protocol (aligned '
                        'with probe_projection; initial basis=orth(cross-'
                        'covariance B_t), Cov strictly zeroed after '
                        'deletion, refit train R²≈0 passes in one round; '
                        'standardized space; recommended); r2_target='
                        'spectral back-solved top-k (diagonal-covariance '
                        'approximation, control only); topk=take the '
                        'first deletion_k directions directly (original '
                        'space)')
    p.add_argument('--deletion_rcond', type=float, default=1e-10,
                   help='fullspace mode: eigenvalue truncation of the '
                        'post-deletion refit pseudo-inverse (aligned with '
                        'probe_projection --ols_rcond)')
    p.add_argument('--deletion_orth_rcond', type=float, default=1e-6,
                   help='fullspace mode: singular-value truncation of the '
                        'initial basis orth(B_t) (aligned with '
                        'probe_projection --orth_rcond)')
    p.add_argument('--deletion_expand_topk', type=int, default=4,
                   help='fullspace mode: per-round count of top residual-'
                        'readout SVD directions merged when SC1 misses '
                        'the threshold (0=no expansion; normally '
                        'converges in one round, backstop only)')
    p.add_argument('--deletion_protect', type=str, nargs='+', default=None,
                   help='fullspace selective deletion: protected target '
                        'name list (e.g. --deletion_protect pipe-x0 '
                        'pipe-eps means mutual protection) — before '
                        'deleting a target, first subtract the subspace '
                        'of the protected targets (oblique projection); '
                        'the protected signal is strictly untouched; the '
                        'cost is keeping target components overlapping '
                        'the protection (incomplete deletion, '
                        'residue=overlap)')
    p.add_argument('--deletion_max_iter', type=int, default=20,
                   help='fullspace mode: SC1 iteration cap (normally '
                        'converges in 1 round, only an expand backstop)')
    p.add_argument('--deletion_k', type=int, default=8,
                   help='number of directions kept in topk mode')
    p.add_argument('--deletion_r2_target', type=float, default=0.05,
                   help='fullspace mode: pass threshold of the '
                        'post-deletion refit R² (aligned with '
                        'probe_projection --sc1_threshold); in r2_target '
                        'mode, the target residual level')
    p.add_argument('--result_path', type=str, default='probe_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.feature_root)
    if not root.is_dir():
        raise RuntimeError(f'feature root directory does not exist: {root}')

    folders_all = list_image_folders(root)
    if args.max_images is not None:
        folders_all = folders_all[:max(args.max_images, 0)]
        print(f'--max_images={args.max_images}: keeping the first '
              f'{len(folders_all)} images in sorted order')
    if len(folders_all) == 0:
        raise RuntimeError('no usable images after truncation')

    modules = discover_modules(folders_all)
    shapes, d_by_module = scan_shapes(folders_all, modules, args.input_layout)
    f0 = folders_all[0].name
    print(f'images: {len(folders_all)} | modules: {len(modules)} | '
          f'input layout: {args.input_layout} -> internal (d, h, w)')
    print('module dims: ' + '  '.join(f'{m}(d={d_by_module[m]}, '
          f'example shape {shapes[(f0, m)]})' for m in modules))
    if args.input_layout == 'hwd':
        sus = [m for m in modules if shapes[(f0, m)][0] >= 8 * d_by_module[m]
               and shapes[(f0, m)][0] >= 256]
        if sus:
            print(f'[note] the first dim of module(s) {sus[:3]} '
                  f'({shapes[(f0, sus[0])][0]}) far exceeds the channel '
                  f'dim ({d_by_module[sus[0]]}); the (c, h, w) '
                  f'channels-first layout is probably misread, use '
                  f'--input_layout chw')

    # target & input validation
    targets = list(dict.fromkeys(args.target))
    for t in targets:
        if t not in modules:
            raise RuntimeError(f'target module {t} not among candidates: {modules}')
    if args.inputs:
        inputs_fixed = list(dict.fromkeys(args.inputs))
        for m in inputs_fixed:
            if m not in modules:
                raise RuntimeError(f'input module {m} not among candidates: {modules}')
        for t in targets:
            if t in inputs_fixed:
                raise RuntimeError(f'target {t} cannot also be an input')
    else:
        inputs_fixed = None  # per target, auto = "all other modules"

    # splits
    splits, degenerate = stratified_split(folders_all, args.split_ratios,
                                          args.split_seed)
    print(f"split: train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}  (class stratification "
          f"{'off, too little data' if degenerate else 'on'})")
    cls_dist = {}
    for f in folders_all:
        cls_dist[class_of(f)] = cls_dist.get(class_of(f), 0) + 1
    print(f'class distribution: {cls_dist}')

    # device & cache
    device = pick_device(args.device)
    print(f'device: {device}')
    est_gb = estimate_cache_bytes(folders_all, modules, shapes) / 1024 ** 3
    if args.cache_ram == 'on':
        use_cache = True
    elif args.cache_ram == 'off':
        use_cache = False
    else:
        use_cache = est_gb <= args.cache_ram_gb
    print(f'feature memory cache: {"on" if use_cache else "off"} '
          f'(estimated total {est_gb:.2f} GB, threshold '
          f'{args.cache_ram_gb} GB)')
    bank = FeatureBank(use_cache, args.input_layout)

    # check: at least one probing task
    joint_probes = list(args.probe) if args.probe else []
    per_probes = ([] if 'none' in args.per_input_probes
                  else list(args.per_input_probes))
    # with two targets, allow running only the cross profile (joint/per_input both off)
    if not joint_probes and not per_probes and len(targets) != 2:
        raise RuntimeError('no probing task at all: --probe not given '
                           'and --per_input_probes=none')

    # pre-validation for --pair_mode patch (M2 zero-resize measurement)
    if args.pair_mode == 'patch':
        bad = [q for q in joint_probes + per_probes
               if PROBE_LAYOUTS[q] != 'flat']
        if bad:
            raise RuntimeError(f'--pair_mode patch supports flat probes '
                               f'(linear/mlp) only, got {bad}; use position '
                               f'mode for conv probes')
        if args.save_deletion_dirs:
            raise RuntimeError('--save_deletion_dirs is incompatible with '
                               '--pair_mode patch (the patch readout '
                               'matrix output dim includes in-patch '
                               'positions, different from the per-channel '
                               'direction-deletion protocol)')

    # pre-validation for deletion-direction saving: depends on per-input
    # linear probe readout matrices
    if args.save_deletion_dirs:
        if 'linear' not in per_probes:
            if per_probes:
                per_probes.insert(0, 'linear')
                print('[note] --save_deletion_dirs needs per-input linear '
                      'probes; linear has been added to '
                      '--per_input_probes automatically')
            else:
                raise RuntimeError('--save_deletion_dirs needs per-input '
                                   'linear probes but '
                                   '--per_input_probes=none disabled them; '
                                   'remove none or drop '
                                   '--save_deletion_dirs')

    # per target × per probe
    results = {}
    per_models = {}   # target -> {input layer: (linear predictor, stdz)} for cross-analysis reuse
    for ti, target in enumerate(targets):
        inputs = inputs_fixed if inputs_fixed is not None \
            else [m for m in modules if m != target]
        if not inputs:
            raise RuntimeError(f'target {target} has no usable input modules')
        d_in = sum(d_by_module[m] for m in inputs)
        d_out = d_by_module[target]
        if args.pair_mode == 'patch':
            print(f'\n=== target {target} ({d_out} channels, --pair_mode '
                  f'patch (zero resize: inputs and target keep native '
                  f'grids, each token regresses all target values in its '
                  f'receptive field)) | inputs: {len(inputs)} modules'
                  + (f', concatenated D_in={d_in}' if joint_probes else '')
                  + ' ===', flush=True)
        else:
            ref_hw, align_src = choose_ref_hw(shapes, folders_all, target,
                                              inputs, args.align_to, f0)
            print(f'\n=== target {target} ({d_out} channels, reference '
                  f'grid {ref_hw} [aligned to {align_src}]) | inputs: '
                  f'{len(inputs)} modules'
                  + (f', concatenated D_in={d_in}' if joint_probes else '')
                  + ' ===', flush=True)
        joint_res = {}
        if joint_probes:
            pair_j = build_pairing(bank, shapes, folders_all, f0, target,
                                   inputs, d_out, args)
            for probe in joint_probes:
                joint_res[probe] = run_one(
                    probe, target, inputs, d_in, pair_j['d_out_eff'],
                    pair_j['ref_hw'], bank, splits, args, device,
                    seed=args.seed + ti, make_ds=pair_j['make_ds'])

        # per-input-module probes (single module -> target)
        per_res = {}
        if per_probes:
            print(f'  --- per-input-module probes (single module -> '
                  f'{target}) ---', flush=True)
            for m in inputs:
                d_m = d_by_module[m]
                if args.pair_mode == 'patch':
                    # rebuild patch pairing per input on its own grid
                    # (inputs may have different kh×kw)
                    pair_m = build_pairing(bank, shapes, folders_all, f0,
                                           target, [m], d_out, args)
                    ref_hw_m = pair_m['ref_hw']
                    make_ds_m, d_out_m = pair_m['make_ds'], pair_m['d_out_eff']
                else:
                    ref_hw_m, _ = choose_ref_hw(shapes, folders_all, target,
                                                [m], args.align_to, f0)
                    make_ds_m, d_out_m = None, d_out
                per_res[m] = {}
                for q in per_probes:
                    ret = run_one(
                        q, target, [m], d_m, d_out_m, ref_hw_m, bank,
                        splits, args, device, seed=args.seed + ti, label=m,
                        make_ds=make_ds_m, return_model=(q == 'linear'))
                    if make_ds_m is not None:
                        r0 = ret[0] if q == 'linear' else ret
                        r0.update(pair_mode='patch',
                                  patch_hw=list(pair_m['patch_hw']),
                                  tgt_hw=list(pair_m['tgt_hw']))
                        arr = np.asarray(r0['r2_per_channel'], dtype=float)
                        g = np.nanmean(arr.reshape(d_out, -1), axis=1)
                        r0['r2_channel_grouped'] = [float(v) for v in g]
                        print('      [patch] per-target-channel R² (mean '
                              'over in-patch positions): '
                              + np.array2string(g, precision=4), flush=True)
                    # keep the linear predictor for per-layer reuse in
                    # two-target cross analysis
                    if q == 'linear':
                        per_models.setdefault(target, {})[m] = ret[1:]
                        ret = ret[0]
                    per_res[m][q] = ret
            width_m = max(len(m) for m in inputs) + 2
            print(f'\n  --- per-input-module results (target {target}, '
                  f'test R2_avg / cos_sim / gain) ---')
            print('  ' + 'module'.ljust(width_m)
                  + ''.join(q.rjust(27) for q in per_probes))
            for m in inputs:
                row = m.ljust(width_m)
                for q in per_probes:
                    r = per_res[m][q]
                    g = (f" / {r['energy_ratio']:.4f}"
                         if 'energy_ratio' in r else ' / -')
                    row += (f"{r['r2_avg']:.4f} / {r['cos_sim']:.4f}"
                            f"{g}").rjust(27)
                print('  ' + row)
        results[target] = {'joint': joint_res, 'per_input': per_res}

    # deletion-direction saving (enabled by --save_deletion_dirs, target5 protocol)
    if args.save_deletion_dirs:
        save_deletion_dirs(args.save_deletion_dirs, per_models, results,
                           targets, args)

    # Two-target cross analysis: per-layer depth profile — one
    # corr(pred_a, pred_b) per common input layer. Predictors prefer
    # reusing the linear models trained by the per-input probes
    # (per_input), retraining when missing; each layer pairs the two
    # predictors' outputs per position on its own first-image grid, then
    # flattens for the Pearson computation.
    cross_res = None
    if len(targets) == 2 and args.pair_mode == 'patch':
        print('[note] under --pair_mode patch the linear predictor '
              'output dim includes in-patch positions, different from '
              'the two-target cross-analysis protocol; skipped')
    elif len(targets) == 2:
        ta_, tb_ = targets
        if d_by_module[ta_] != d_by_module[tb_]:
            print(f'[warn] two-target cross analysis requires equal '
                  f'target channel counts ({ta_}: {d_by_module[ta_]}, '
                  f'{tb_}: {d_by_module[tb_]}); skipped')
        else:
            ins_a = inputs_fixed if inputs_fixed is not None \
                else [m for m in modules if m != ta_]
            ins_b = inputs_fixed if inputs_fixed is not None \
                else [m for m in modules if m != tb_]
            common = [m for m in ins_a if m in ins_b]
            if not common:
                print('[warn] two-target cross analysis needs common '
                      'input layers, none here; skipped')
            else:
                print(f'\n=== cross analysis: per-layer corr(pred({ta_}), '
                      f'pred({tb_})) depth profile | {len(common)} common '
                      f'input layers ===', flush=True)
                cross_res = cross_corr_profile(
                    ta_, tb_, common, d_by_module, shapes, folders_all, f0,
                    bank, splits, args, device, per_models)
                for m in common:
                    e = cross_res['per_layer'][m]
                    print(f"  -> {m}: corr={e['pearson']:.4f}  "
                          f"(R² {ta_}={e['r2_a']:.4f} / "
                          f"{tb_}={e['r2_b']:.4f}"
                          f"{'' if e['reused'] else ', retrained'})")
    elif len(targets) > 2:
        print('[note] cross-predictor correlation supports exactly 2 --target; skipped')

    # summary table (concatenated probes)
    if joint_probes:
        print('\n========== summary (all inputs concatenated, '
              'test R2_avg / cos_sim / gain) ==========')
        width = max([len(t) for t in targets] + [8]) + 2
        header = 'target'.ljust(width) + ''.join(p.rjust(27) for p in joint_probes)
        print(header)
        for t in targets:
            row = t.ljust(width)
            for p in joint_probes:
                r = results[t]['joint'][p]
                g = (f" / {r['energy_ratio']:.4f}"
                     if 'energy_ratio' in r else ' / -')
                row += (f"{r['r2_avg']:.4f} / {r['cos_sim']:.4f}"
                        f"{g}").rjust(27)
            print(row)

    out = {
        'feature_root': str(root),
        'n_images': len(folders_all),
        'modules': modules,
        'module_dims': d_by_module,
        'inputs_per_target': {t: (inputs_fixed if inputs_fixed is not None
                                  else [m for m in modules if m != t])
                              for t in targets},
        'splits': {k: [f.name for f in v] for k, v in splits.items()},
        'degenerate_split': degenerate,
        'args': {k: v for k, v in vars(args).items()},
        'results': results,
    }
    if cross_res is not None:
        out['cross_predictor_corr'] = cross_res
    with open(args.result_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nresults saved to: {args.result_path}')


if __name__ == '__main__':
    main()
