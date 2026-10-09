#!/usr/bin/env python3
"""
Image classification probe (probe_classification)
==================================================

Task: predict image class from feature modules (class = the part of the folder
name before the first underscore); report top-1 / top-5 accuracy on a held-out
test set.

Data organization (same as probe_features.py)::

    feature_root/
        {class}_{image_name}/
            {module}.npy          # 2D feature map of shape (h, w, d)

Efficiency strategy
-------------------
1. Global average pooling (GAP): (h, w, d) -> (d,), one sample per image;
   GAP of multiple modules concatenated. Classification needs no spatial
   alignment, introduces no interpolation, and is unaffected by resolution
   differences across modules.
2. Closed-form multi-class ridge regression (one-hot targets + intercept,
   equivalent to least-squares classification with intercept):
   - D <= n (primal): eigendecomposition of the D×D Gram;
   - n <  D (dual)  : eigendecomposition of the n×n Gram, shrinking the
     decomposition size from D to n — when D is in the tens of thousands and
     the image count only hundreds, this turns infeasible into seconds
     (key speedup).
   Eigendecomposition is fixed on CPU float64; one decomposition solves the
   whole γ grid in parallel, choosing γ by validation top-1. Falls back to
   Adam+cross-entropy when min(n, D) > --closed_form_max_dim in auto mode.
3. Image (folder) level train/val/test split, stratified by class (reuses
   probe_features).
4. By default each module is GAP-ed and classified separately
   (--per_input_probes; the default linear closed-form solution is nearly
   free); the [all-modules-concatenated] probe is off by default and must be
   enabled explicitly with --probe.
5. Results: per-line console output + summary table + JSON for cross-module /
   cross-setting ranking comparisons.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import probe_features as pf


# =========================================================================== #
#                          GAP features and labels                             #
# =========================================================================== #

@torch.no_grad()
def gap_features(folders, bank, modules):
    """Per image: each module (d, h, w) -> spatial mean -> (d,); multi-module
    concat -> (D,). Returns (n, D) float32 CPU."""
    rows = []
    for folder in folders:
        parts = [bank.load(folder, m).mean(dim=(1, 2)) for m in modules]
        rows.append(torch.cat(parts, 0))
    return torch.stack(rows, 0).contiguous()


def build_labels(folders):
    """Folder-name prefix -> class index. Returns (y_idx (n,), classes list)."""
    classes = sorted({pf.class_of(f) for f in folders})
    idx = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([idx[pf.class_of(f)] for f in folders], dtype=torch.long)
    return y, classes


def onehot(y_idx, n_classes):
    Y = torch.zeros(len(y_idx), n_classes)
    Y[torch.arange(len(y_idx)), y_idx] = 1.0
    return Y


def standardize_fit(X):
    mu = X.mean(0)
    var = X.var(0, unbiased=False).clamp_min(1e-12)
    return mu, var.rsqrt()


# =========================================================================== #
#            Closed-form multi-class ridge (primal / dual adaptive)            #
# =========================================================================== #

class MultiClassRidge:
    """min_W 0.5||Xs W + 1·b - Y||² + 0.5γ||W||²,  Y one-hot.
    Xs is per-dim standardized features (mean 0); the optimal intercept is
    b = ȳ (class frequencies).
    The one-hot is centered first (Y_c = Y - 1·ȳ); after solving for W, the
    predicted scores = Xs W + ȳ.

    primal (D <= n): eigh(D×D),  W(γ) = (XsᵀXs + γI)⁻¹ XsᵀYc
    dual   (n <  D): eigh(n×n),  α(γ) = (XsXsᵀ + γI)⁻¹ Yc,  W(γ) = Xsᵀ α
    The eigh size is min(n, D) in both modes.
    """

    def __init__(self, X, Y, mode):
        self.mode = mode
        self.X = X                                   # (n, D) standardized float32
        self.intercept = Y.mean(0)                   # (C,) ȳ
        self.Yc = Y - self.intercept[None, :]        # centered one-hot
        if mode == 'primal':
            self.A = X.T @ X                         # (D, D)
            self.B = X.T @ self.Yc                   # (D, C)
        else:
            self.G = X @ X.T                         # (n, n)

    def solve(self, gamma_grid, device, dtype=torch.float32):
        M = self.A if self.mode == 'primal' else self.G
        print(f'      closed-form ({self.mode}): eigh {tuple(M.shape)} '
              f'(CPU, float64)...', flush=True)
        evals, evecs = torch.linalg.eigh(M.detach().to('cpu').to(torch.float64))
        evals = evals.to(device=device, dtype=dtype)
        evecs = evecs.to(device=device, dtype=dtype)
        if self.mode == 'primal':
            R = evecs.T @ self.B.to(device)
            Ws = {g: evecs @ ((1.0 / (evals + g))[:, None] * R)
                  for g in gamma_grid}
        else:
            R = evecs.T @ self.Yc.to(device)
            alphas = {g: evecs @ ((1.0 / (evals + g))[:, None] * R)
                      for g in gamma_grid}
            Ws = {g: self.X.to(device).T @ alphas[g] for g in gamma_grid}
        return Ws  # one (D, C) weight matrix per γ (standardized space)


@torch.no_grad()
def scores_of(Xs, W, intercept, device):
    return Xs.to(device) @ W.to(device) + intercept.to(device)


def acc_topk(scores, y_idx):
    """scores (n, C) / y_idx (n,) -> {'top1': .., 'top5': ..(when C>=5)}"""
    C = scores.shape[1]
    _, pred = scores.topk(min(5, C), dim=1)
    correct = pred.eq(y_idx[:, None].to(pred.device))
    out = {'top1': float(correct[:, 0].float().mean())}
    if C >= 5:
        out['top5'] = float(correct.any(dim=1).float().mean())
    return out


def select_gamma_clf(Ws, intercept, Xs_va, y_va, device):
    """The γ with the highest validation top-1; grid median when no validation set."""
    if len(y_va) == 0:
        return sorted(Ws)[len(Ws) // 2]
    best_g, best_acc = None, -1.0
    for g, W in Ws.items():
        acc = acc_topk(scores_of(Xs_va, W, intercept, device), y_va)['top1']
        print(f'      γ={g:g}: val top1={acc:.4f}')
        if acc > best_acc:
            best_acc, best_g = acc, g
    return best_g


# =========================================================================== #
#                       Adam fallback (CE loss)                                #
# =========================================================================== #

class ClfLinear(nn.Module):
    def __init__(self, d_in, n_classes, hidden=None, act_name=None):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.fc(x)


class ClfMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden, act_name):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = pf.make_act(act_name)
        self.fc2 = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def train_clf(model, Xs_tr, y_tr, Xs_va, y_va, args, device, seed):
    """Adam + cosine annealing + cross-entropy; keeps the best weights by validation top-1."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1))
    loader = DataLoader(TensorDataset(Xs_tr, y_tr), batch_size=args.batch_size,
                        shuffle=True)
    best_acc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for xb, yb in loader:
            logits = model(xb.to(device))
            loss = F.cross_entropy(logits, yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(yb)
            seen += len(yb)
        sched.step()
        msg = f'      epoch {ep:3d}/{args.epochs} | train_ce {run / max(seen, 1):.4f}'
        if len(y_va):
            model.eval()
            with torch.no_grad():
                acc = acc_topk(model(Xs_va.to(device)), y_va)['top1']
            msg += f' | val_top1 {acc:.4f}'
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =========================================================================== #
#              Single solve + evaluation (shared by concat/per-module)         #
# =========================================================================== #

def solve_and_eval(probe, Xs_tr, y_tr, Xs_va, y_va, Xs_te, y_te, C,
                   args, device, seed, label=None):
    """Train/solve + test-evaluate the given (sliced) features; return the result dict.
    label: optional label (e.g. the input module name), printed before the
    per-line result to identify the source.
    """
    tag = f'[{label}] ' if label else ''
    t0 = time.time()
    n_tr, D_ = Xs_tr.shape
    m = min(n_tr, D_)
    degenerate_test = len(y_te) == 0
    te_X = Xs_tr if degenerate_test else Xs_te
    te_y = y_tr if degenerate_test else y_te
    if probe == 'linear' and args.solver in ('auto', 'closed'):
        use_closed = True
        if args.solver == 'auto' and m > args.closed_form_max_dim:
            use_closed = False
            print(f'      [note] min(n,D)={m} > closed_form_max_dim='
                  f'{args.closed_form_max_dim}, linear falls back to Adam+CE')
    else:
        use_closed = False

    if use_closed:
        mode = 'primal' if D_ <= n_tr else 'dual'
        Y_oh = onehot(y_tr, C).float()
        ridge = MultiClassRidge(Xs_tr, Y_oh, mode)
        Ws = ridge.solve(args.gamma_grid, device)
        gamma = select_gamma_clf(Ws, ridge.intercept, Xs_va, y_va, device)
        scores = scores_of(te_X, Ws[gamma], ridge.intercept, device)
        solver_used = f'closed({mode}, γ={gamma:g})'
    else:
        if probe == 'mlp':
            model = ClfMLP(D_, C, args.hidden_mlp, args.activation)
        else:
            model = ClfLinear(D_, C)
        model = train_clf(model, Xs_tr, y_tr, Xs_va, y_va, args, device,
                          seed=seed)
        model.eval()
        with torch.no_grad():
            scores = model(te_X.to(device))
        solver_used = f'adam({args.epochs}ep, lr={args.lr:g})'

    accs = acc_topk(scores, te_y)
    res = {'probe': probe, 'solver': solver_used, 'n_classes': C,
           'n_test': int(len(te_y)), 'degenerate_test': degenerate_test,
           'seconds': round(time.time() - t0, 1), **accs}
    top5 = f"  top5={res['top5']:.4f}" if 'top5' in res else ''
    print(f"    [{probe}] -> {tag}solver={res['solver']:<26} "
          f"top1={res['top1']:.4f}{top5}  ({res['seconds']}s)", flush=True)
    return res


# =========================================================================== #
#                                   Main                                       #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='image classification probe: GAP features -> multi-class '
                    'ridge/MLP; reports top-1/top-5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--feature_root', type=str, required=True,
                   help='feature root directory containing {class}_{image_name}/ folders')
    p.add_argument('--modules', type=str, nargs='*', default=None,
                   help='module list to use (concatenated after GAP); default = all modules')
    p.add_argument('--input_layout', type=str, default='hwd', choices=['hwd', 'chw'],
                   help='npy feature dim layout: hwd=(h, w, d) channels-last; '
                        'chw=(c, h, w) channels-first (torch (C,H,W) feature '
                        "maps, ViT features saved as token sequences rearranged "
                        "to 'b c h w', etc.)")
    p.add_argument('--max_images', type=int, default=None,
                   help='use only the first N images after sorting by folder name')
    p.add_argument('--probe', type=str, nargs='+', default=None,
                   choices=['linear', 'mlp'],
                   help='probes for the [all-modules-concatenated] probe: '
                        'linear=GAP+linear classifier (closed-form solution '
                        'by default), mlp=GAP+two-layer MLP (Adam); by default '
                        'no concatenation is run, only per-module probes '
                        '(see --per_input_probes)')
    p.add_argument('--per_input_probes', type=str, nargs='+', default=['linear'],
                   choices=['linear', 'mlp', 'none'],
                   help='probes for per-module GAP classification '
                        '(default linear = closed-form solution, nearly free); '
                        'pass none to disable')
    p.add_argument('--solver', type=str, default='auto',
                   choices=['auto', 'closed', 'adam'],
                   help='linear solver: auto = closed-form when min(n,D)<=max_dim')
    p.add_argument('--closed_form_max_dim', type=int, default=20000,
                   help='fall back to Adam when min(n, D) exceeds this in auto mode')
    p.add_argument('--gamma_grid', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                   help='ridge regularization grid (chosen by validation top-1)')
    p.add_argument('--hidden_mlp', type=int, default=1024, help='mlp hidden width')
    p.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    p.add_argument('--epochs', type=int, default=30, help='Adam training epochs')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--split_ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   help='train/val/test image split ratios')
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or a specific cuda/mps/cpu')
    p.add_argument('--cache_ram', type=str, default='auto', choices=['auto', 'on', 'off'])
    p.add_argument('--cache_ram_gb', type=float, default=8.0)
    p.add_argument('--result_path', type=str, default='probe_classification_results.json')
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
    if len(folders_all) == 0:
        raise RuntimeError('no usable images after truncation')

    modules = pf.discover_modules(folders_all)
    shapes, d_by = pf.scan_shapes(folders_all, modules, args.input_layout)
    if args.modules:
        mods = list(dict.fromkeys(args.modules))
        for m in mods:
            if m not in modules:
                raise RuntimeError(f'module {m} not among candidates: {modules}')
    else:
        mods = modules
    D = sum(d_by[m] for m in mods)
    print(f'images: {len(folders_all)} | modules used: {mods} (GAP concat dim D={D})')

    # Labels and split (stratified by class)
    y_all, classes = build_labels(folders_all)
    print(f'n_classes C={len(classes)}: {classes}')
    splits, degenerate = pf.stratified_split(folders_all, args.split_ratios,
                                             args.split_seed)
    print(f"split: train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}  (class stratification"
          f"{'off, too little data' if degenerate else 'on'})")

    # Device and cache
    device = pf.pick_device(args.device)
    print(f'device: {device}')
    est_gb = pf.estimate_cache_bytes(folders_all, modules, shapes) / 1024 ** 3
    if args.cache_ram == 'on':
        use_cache = True
    elif args.cache_ram == 'off':
        use_cache = False
    else:
        use_cache = est_gb <= args.cache_ram_gb
    print(f'feature RAM cache: {"on" if use_cache else "off"} (estimated {est_gb:.2f} GB)')
    bank = pf.FeatureBank(use_cache, args.input_layout)

    # Extract GAP features once (n, D)
    t0 = time.time()
    X_all = gap_features(folders_all, bank, mods)
    name2row = {f.name: i for i, f in enumerate(folders_all)}
    row_of = {k: torch.tensor([name2row[f.name] for f in v], dtype=torch.long)
              for k, v in splits.items()}
    print(f'GAP feature extraction done: {tuple(X_all.shape)} ({time.time() - t0:.1f}s)')

    X_tr = X_all[row_of['train']]
    X_va = X_all[row_of['val']]
    X_te = X_all[row_of['test']]
    y_tr = y_all[row_of['train']]
    y_va = y_all[row_of['val']]
    y_te = y_all[row_of['test']]
    mu, inv_std = standardize_fit(X_tr)
    Xs_tr = (X_tr - mu) * inv_std
    Xs_va = (X_va - mu) * inv_std
    Xs_te = (X_te - mu) * inv_std
    C = len(classes)

    n_tr, D_ = Xs_tr.shape
    print(f'n_train={n_tr}, all-module concat dim D={D_}')

    # Validation: at least one probing task
    joint_probes = list(args.probe) if args.probe else []
    per_probes = ([] if 'none' in args.per_input_probes
                  else list(args.per_input_probes))
    if not joint_probes and not per_probes:
        raise RuntimeError('no probing task: --probe not given and '
                           '--per_input_probes=none')

    # All-modules concatenation (off by default, enabled via --probe)
    joint_res = {}
    if joint_probes:
        print('\n--- all-modules-concatenated probe ---', flush=True)
        for probe in joint_probes:
            joint_res[probe] = solve_and_eval(
                probe, Xs_tr, y_tr, Xs_va, y_va, Xs_te, y_te, C,
                args, device, seed=args.seed)

    # Per-input-module probing (GAP column slice); default linear only (closed-form, low cost)
    per_res = {}
    if per_probes:
        print('\n--- per-input-module probing (single-module GAP -> class) ---', flush=True)
        off = 0
        for m in mods:
            d_m = d_by[m]
            sl = slice(off, off + d_m)
            per_res[m] = {}
            print(f'  · module {m} (d={d_m})', flush=True)
            for q in per_probes:
                per_res[m][q] = solve_and_eval(
                    q, Xs_tr[:, sl], y_tr, Xs_va[:, sl], y_va,
                    Xs_te[:, sl], y_te, C, args, device, seed=args.seed,
                    label=m)
            off += d_m
        width_m = max(len(m) for m in mods) + 2
        print('\n--- per-input-module results (test top-1 / top-5) ---')
        print('  ' + 'module'.ljust(width_m)
              + ''.join(q.rjust(22) for q in per_probes))
        for m in mods:
            row = m.ljust(width_m)
            for q in per_probes:
                r = per_res[m][q]
                t5 = f"{r['top5']:.4f}" if 'top5' in r else '  -  '
                row += f"{r['top1']:.4f} / {t5}".rjust(22)
            print('  ' + row)

    if joint_probes:
        print('\n========== summary (all modules concatenated, test top-1 / top-5) ==========')
        for probe, r in joint_res.items():
            t5 = f"{r['top5']:.4f}" if 'top5' in r else '  -  '
            print(f"  {probe:<8} {r['top1']:.4f} / {t5}   ({r['solver']})")

    out = {
        'task': 'classification',
        'feature_root': str(root),
        'n_images': len(folders_all),
        'modules': mods,
        'module_dims': {m: d_by[m] for m in mods},
        'gap_dim': D_,
        'classes': classes,
        'n_classes': C,
        'splits': {k: [f.name for f in v] for k, v in splits.items()},
        'degenerate_split': degenerate,
        'args': {k: v for k, v in vars(args).items()},
        'results': {'joint': joint_res, 'per_input': per_res},
    }
    with open(args.result_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nresults saved to: {args.result_path}')


if __name__ == '__main__':
    main()
