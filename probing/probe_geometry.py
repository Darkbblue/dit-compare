"""Geometry probe: pair up input × reference representations one-to-one, compute
norm ratios and principal-direction cosine.

Usage:
  python probe_geometry.py --feature_root FEATS --inputs a b c --refs x y z

1. Pairing: --inputs and --refs must have the same count, paired by index
   (inputs[i] vs refs[i]); a module may appear on both sides (self-check
   should give ratio=1, cos=1).
2. Representation = all position vectors of a module over all images (each
   image's (h, w, d) flattened to (h*w, d) and concatenated per image; GAP
   1x1 grids also work). All quantities are defined in each module's own
   feature space, independent of the spatial grid, with no grid alignment.
3. Norm: norm_rms = sqrt(mean ||v||²) — per-position mean L2 norm, invariant
   to image count and grid size, the convention for norm ratios
   (norm_ratio = rms_in / rms_ref); the Frobenius norm
   (sqrt(sum ||v||²), depends on image and token counts) is also reported
   for reference.
4. Principal direction: top-eigenvalue eigenvector of the representation's
   centered covariance Gram (i.e. the first principal component, default;
   --no_center switches to the uncentered top singular direction, where the
   mean direction dominates).
   With equal channel counts (mode=principal_dir): cosine cos = |u_in · u_ref|
   — the sign of a principal direction is arbitrary, hence the absolute
   value; each side's variance explained ratio (evr) is also reported to
   judge representativeness.
   With mismatched channel counts the dot product is undefined; use the CCA
   canonical-correlation spectrum instead (mode=cca): the two sides are
   aligned token-by-token (interpolated to the input side's first-image grid
   when grids differ), covariances relatively regularized and whitened, and
   the SVD singular values = correlations of canonical direction pairs
   ρ_i ∈ [0,1] in descending order; the main cos = the first canonical
   correlation ρ1 — the standard method for cross-dim representation
   comparison (SVCCA idea), measuring the maximal linear alignment between
   the two spaces.
5. Compute: streamed per-image Gram accumulation (S += tokᵀtok; matmul on
   --device in fp32, S/sum always accumulated on CPU fp64, eigh always CPU
   fp64 — MPS lacks fp64), single pass over all images, each involved module
   read from disk once; memory O(d²) independent of image count; features
   themselves are not cached.
6. Results: per-pair console output + summary table + JSON (--result_path)
   for ranking comparisons.
   --selftest runs the built-in mathematical self-check (no feature_root
   needed).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import probe_features as pf


# =========================================================================== #
#                Statistics accumulation and geometry quantities              #
# =========================================================================== #

def token2d(x: torch.Tensor) -> torch.Tensor:
    """(d, h, w) -> (h*w, d): one spatial position's feature vector per row."""
    d, h, w = x.shape
    return x.permute(1, 2, 0).reshape(h * w, d)


class ModuleAcc:
    """Streamed accumulator for one module's representation statistics:
        S   = Σ tokᵀtok   (Gram, d×d, fp64 CPU; trace(S) = Σ||v||²)
        sum = Σ tok       (d, fp64 CPU; for centering)
        n   = total token count
    matmul on the given device in fp32 (GPU preferred), accumulation always
    CPU fp64 (MPS has no fp64)."""

    def __init__(self, d: int):
        self.d = d
        self.S = torch.zeros(d, d, dtype=torch.float64)
        self.sum = torch.zeros(d, dtype=torch.float64)
        self.n = 0

    def update(self, tok: torch.Tensor, device: torch.device):
        # tok: (n_i, d) fp32 CPU
        self.sum += tok.sum(0, dtype=torch.float64)
        t = tok.to(device)
        G = t.T @ t                              # fp32 on device
        self.S += G.detach().to('cpu').double()  # fp64 CPU accumulation
        self.n += tok.shape[0]

    def _trace(self) -> float:
        return float(self.S.diagonal().sum())

    def norm_rms(self) -> float:
        """Per-position mean L2 norm: sqrt(mean ||v||²) = sqrt(trace(S)/n)."""
        return math.sqrt(max(self._trace(), 0.0) / max(self.n, 1))

    def frobenius(self) -> float:
        """L2 norm of the whole flattened representation: sqrt(Σ||v||²)."""
        return math.sqrt(max(self._trace(), 0.0))


def principal_dir(acc: ModuleAcc, centered: bool):
    """The representation's principal direction (fp64 CPU eigh) and its
    variance-explained ratio evr.
    centered=True: first principal component (top eigenvector of the
    covariance Gram = S - n·μμᵀ);
    centered=False: top eigenvector of the uncentered Gram (the mean
    direction dominates).
    Returns (unit principal direction u, evr = top eigenvalue / nonneg sum
    of the spectrum)."""
    C = acc.S.clone()
    if centered:
        mu = acc.sum / max(acc.n, 1)
        C -= acc.n * torch.outer(mu, mu)
    C = (C + C.T) * 0.5                        # numerical symmetry safeguard
    evals, evecs = torch.linalg.eigh(C)        # fp64 CPU (MPS pitfall avoidance)
    ev_max = float(evals[-1])
    u = evecs[:, -1]
    denom = float(evals.clamp(min=0).sum())
    evr = ev_max / denom if denom > 0 else float('nan')
    return u, evr


def cos_principal(acc_a: ModuleAcc, acc_b: ModuleAcc, centered: bool):
    """Cosine between the two representations' principal directions (absolute
    value; None when channel counts differ)."""
    if acc_a.d != acc_b.d:
        return None
    ua, _ = principal_dir(acc_a, centered)
    ub, _ = principal_dir(acc_b, centered)
    return abs(float(ua @ ub))


CCA_REG = 1e-3      # relative covariance regularization (trace-scaled), whitening numerical stability
CCA_TOL = 1e-10     # relative eigenvalue truncation threshold (numerical rank)


class CcaAcc:
    """Streamed CCA accumulator for cross-dim pairs: after token-wise X/Y
    alignment, accumulate Sxx / Syy / Sxy Gram blocks + means per image.
    matmul on device fp32, accumulation CPU fp64 (same device/dtype
    convention as ModuleAcc)."""

    def __init__(self, d1: int, d2: int):
        self.d1, self.d2 = d1, d2
        self.Sxx = torch.zeros(d1, d1, dtype=torch.float64)
        self.Syy = torch.zeros(d2, d2, dtype=torch.float64)
        self.Sxy = torch.zeros(d1, d2, dtype=torch.float64)
        self.sum_x = torch.zeros(d1, dtype=torch.float64)
        self.sum_y = torch.zeros(d2, dtype=torch.float64)
        self.n = 0

    def update(self, tx: torch.Tensor, ty: torch.Tensor,
               device: torch.device):
        # tx (n_i, d1) / ty (n_i, d2) fp32 CPU, row-aligned (same image, same position)
        assert tx.shape[0] == ty.shape[0], 'CCA token counts must match on both sides'
        self.sum_x += tx.sum(0, dtype=torch.float64)
        self.sum_y += ty.sum(0, dtype=torch.float64)
        x, y = tx.to(device), ty.to(device)
        self.Sxx += (x.T @ x).detach().to('cpu').double()
        self.Syy += (y.T @ y).detach().to('cpu').double()
        self.Sxy += (x.T @ y).detach().to('cpu').double()
        self.n += tx.shape[0]


def cca_solve(acc: CcaAcc):
    """Canonical correlation analysis (CCA): returns (ρ spectrum fp64 CPU
    descending, numerical rank min(rx, ry)).
    Center the three Gram blocks → relative regularization → eigh whitening
    → SVD(Wxᵀ Cxy Wy); the singular values = Pearson correlation of each
    canonical direction pair ρ_i ∈ [0,1]; ρ1 = the maximal linear alignment
    between the two spaces (the cross-dim generalization of "principal
    direction cosine")."""
    n = max(acc.n, 1)
    mux, muy = acc.sum_x / n, acc.sum_y / n
    Cxx = (acc.Sxx - n * torch.outer(mux, mux)) / n
    Cyy = (acc.Syy - n * torch.outer(muy, muy)) / n
    Cxy = (acc.Sxy - torch.outer(acc.sum_x, acc.sum_y) / n) / n

    def whiten(C, d):
        reg = CCA_REG * float(C.diagonal().sum()) / max(d, 1)
        evals, evecs = torch.linalg.eigh(
            (C + C.T) * 0.5 + reg * torch.eye(d, dtype=torch.float64))
        keep = evals > CCA_TOL * float(evals[-1])   # eigh ascending; [-1] is max
        if not bool(keep.any()):
            return torch.zeros(0, d, dtype=torch.float64), 0
        return evecs[:, keep] / evals[keep].sqrt(), int(keep.sum())

    Wx, rx = whiten(Cxx, acc.d1)
    Wy, ry = whiten(Cyy, acc.d2)
    if rx == 0 or ry == 0:
        return torch.zeros(0, dtype=torch.float64), 0
    M = Wx.T @ Cxy @ Wy
    s = torch.linalg.svdvals(M)
    return s.clamp(0.0, 1.0), min(rx, ry)


# =========================================================================== #
#                                 Self-test                                    #
# =========================================================================== #

def _selftest():
    print('=== selftest: geometry quantity mathematical correctness ===')
    torch.manual_seed(0)
    dev = torch.device('cpu')
    n, chunk = 4000, 500

    # 1) Centered principal-direction recovery: strong single direction + weak isotropic noise
    d = 16
    u0 = F.normalize(torch.randn(d), dim=0)
    X = (3.0 * torch.randn(n)[:, None] * u0[None, :]
         + 0.1 * torch.randn(n, d))
    acc = ModuleAcc(d)
    for i in range(0, n, chunk):
        acc.update(X[i:i + chunk], dev)
    u, evr = principal_dir(acc, centered=True)
    dot = abs(float(u @ u0.double()))
    assert dot > 0.999, f'principal direction recovery failed: |u·u0|={dot:.4f}'
    assert evr > 0.9, f'abnormal evr: {evr:.4f}'
    print(f'  [1] centered principal direction recovery |u·u0|={dot:.6f}  evr={evr:.4f}  OK')

    # 2) Uncentered: the mean direction dominates the top singular direction
    mu = F.normalize(torch.randn(d), dim=0) * 5.0
    X = mu[None, :] + 0.01 * torch.randn(n, d)
    acc = ModuleAcc(d)
    for i in range(0, n, chunk):
        acc.update(X[i:i + chunk], dev)
    u, _ = principal_dir(acc, centered=False)
    dot = abs(float(u @ (mu / mu.norm()).double()))
    assert dot > 0.999, f'uncentered principal direction should ≈ mean direction: {dot:.4f}'
    print(f'  [2] uncentered top direction = mean direction |u·μ̂|={dot:.6f}  OK')

    # 3) Norm statistics vs direct computation
    d = 32
    X = torch.randn(n, d)
    acc = ModuleAcc(d)
    acc.update(X, dev)
    rms, frob = acc.norm_rms(), acc.frobenius()
    ref_rms = float(X.double().norm(dim=1).pow(2).mean().sqrt())
    ref_frob = float(X.double().norm())
    assert abs(rms / ref_rms - 1) < 1e-5, f'rms deviation: {rms} vs {ref_rms}'
    assert abs(frob / ref_frob - 1) < 1e-5, f'frobenius deviation: {frob} vs {ref_frob}'
    print(f'  [3] rms/frobenius match direct computation (rel<1e-5)  OK')

    # 4) Mirror invariance: global negation preserves the principal direction (basis for |cos|)
    X = torch.randn(n, d)
    a, b = ModuleAcc(d), ModuleAcc(d)
    for i in range(0, n, chunk):
        a.update(X[i:i + chunk], dev)
        b.update(-X[i:i + chunk], dev)
    c = cos_principal(a, b, centered=True)
    assert abs(c - 1.0) < 1e-6, f'cos after mirroring should be 1: {c}'
    print(f'  [4] mirror invariance cos={c:.6f}  OK')

    # 5) CCA cross-dim recovery: X = Z@A (d1=16), Y = Z@B (d2=3), sharing a 2-dim latent Z
    # (ρ2's finite-sample bias ~ √(d1·d2/n); large sample size, threshold within the bias)
    n5 = 12000
    Z = torch.randn(n5, 2)
    X = Z @ torch.randn(2, 16) + 0.05 * torch.randn(n5, 16)
    Y = Z @ torch.randn(2, 3) + 0.05 * torch.randn(n5, 3)
    acc = CcaAcc(16, 3)
    for i in range(0, n5, chunk):
        acc.update(X[i:i + chunk], Y[i:i + chunk], dev)
    spec, rank = cca_solve(acc)
    assert spec.numel() >= 3 and rank >= 3, f'abnormal CCA rank: {rank}'
    assert float(spec[0]) > 0.999, f'CCA ρ1 too low: {float(spec[0]):.4f}'
    assert float(spec[1]) > 0.9, f'CCA ρ2 too low: {float(spec[1]):.4f}'
    assert float(spec[2]) < 0.5, f'CCA ρ3 should be noise level: {float(spec[2]):.4f}'
    print(f'  [5] CCA cross-dim recovery ρ1/ρ2/ρ3='
          f'{float(spec[0]):.4f}/{float(spec[1]):.4f}/{float(spec[2]):.4f}  OK')

    print('SELFTEST ALL PASS')


# =========================================================================== #
#                                 Main flow                                    #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='geometry probe: pair input × reference representations '
                    'one-to-one; computes norm ratios (RMS convention) and '
                    'principal-direction cosines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument('--feature_root', type=str, default=None,
                   help='feature root directory containing {class}_{image_name}/ folders '
                        '(may be omitted with --selftest)')
    p.add_argument('--inputs', type=str, nargs='+', default=None,
                   help='input representation module names (= npy file names without extension)')
    p.add_argument('--refs', '--references', dest='refs',
                   type=str, nargs='+', default=None,
                   help='reference representation module names; same count as --inputs, paired by index')
    p.add_argument('--max_images', type=int, default=None,
                   help='use only features of the first N images sorted by folder name')
    p.add_argument('--input_layout', type=str, default='hwd',
                   choices=['hwd', 'chw'],
                   help='npy feature dim layout: hwd=(h, w, d) channels-last; '
                        'chw=(c, h, w) channels-first')
    # Geometry quantities
    p.add_argument('--no_center', action='store_true',
                   help='switch the principal direction to the uncentered top '
                        'singular direction (default: first principal component '
                        'of the centered covariance, i.e. the PCA direction)')
    # Misc
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or specify cuda/mps/cpu; '
                        'used only for Gram matmul, eigh is always CPU fp64')
    p.add_argument('--selftest', action='store_true',
                   help='run the built-in math self-check and exit (no features read)')
    p.add_argument('--result_path', type=str,
                   default='probe_geometry_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.feature_root:
        raise RuntimeError('--feature_root required (or use --selftest)')
    if not args.inputs or not args.refs:
        raise RuntimeError('--inputs and --refs required (same count, paired by index)')
    if len(args.inputs) != len(args.refs):
        raise RuntimeError(f'--inputs ({len(args.inputs)}) and --refs '
                           f'({len(args.refs)}) must have the same count, paired by index')

    root = Path(args.feature_root)
    if not root.is_dir():
        raise RuntimeError(f'feature root directory does not exist: {root}')

    folders_all = pf.list_image_folders(root)
    if args.max_images is not None:
        folders_all = folders_all[:max(args.max_images, 0)]
        print(f'--max_images={args.max_images}: kept the first '
              f'{len(folders_all)} images after sorting')
    if len(folders_all) == 0:
        raise RuntimeError('no usable images after truncation')

    modules = pf.discover_modules(folders_all)
    shapes, d_by_module = pf.scan_shapes(folders_all, modules,
                                         args.input_layout)
    f0 = folders_all[0].name
    print(f'images: {len(folders_all)} | modules: {len(modules)} | '
          f'input layout: {args.input_layout} -> internal (d, h, w)')
    print('module dims: ' + '  '.join(
        f'{m}(d={d_by_module[m]}, example shape {shapes[(f0, m)]})'
        for m in modules))

    # Pairing validation
    inputs, refs = list(args.inputs), list(args.refs)
    for m in inputs + refs:
        if m not in modules:
            raise RuntimeError(f'module {m} not among candidates: {modules}')
    pairs = [(inputs[i], refs[i]) for i in range(len(inputs))]
    involved = list(dict.fromkeys(inputs + refs))   # dedup, keep order; single-pass load
    dim_bad = [(mi, mr) for mi, mr in pairs
               if d_by_module[mi] != d_by_module[mr]]
    if dim_bad:
        detail = ', '.join(f'{a}({d_by_module[a]})~{b}({d_by_module[b]})'
                           for a, b in dim_bad[:4])
        if len(dim_bad) > 4:
            detail += '...'
        print(f'[note] {len(dim_bad)} pairs with mismatched channel counts ({detail}), '
              f'token-wise aligned CCA canonical-correlation spectrum used, '
              f'main cos = first canonical correlation ρ1')
    print(f'pairs: {len(pairs)} | involved modules {len(involved)} | '
          f'principal direction: {"centered PCA" if not args.no_center else "uncentered"}')

    device = pf.pick_device(args.device)
    bank = pf.FeatureBank(use_cache=False, layout=args.input_layout)
    print(f'device: {device} (matmul) | statistics and eigh: CPU fp64 | features not cached',
          flush=True)

    # Single pass: each involved module of each image loaded once. Same-dim pairs
    # accumulate into ModuleAcc (native grid; norm convention); cross-dim pairs
    # accumulate into CcaAcc (both sides interpolated/aligned to the input side's
    # first-image grid, token-wise pairing)
    accs = {m: ModuleAcc(d_by_module[m]) for m in involved}
    cca_accs, cca_hw = {}, {}
    for i, (mi, mr) in enumerate(pairs):
        if d_by_module[mi] != d_by_module[mr]:
            cca_accs[i] = CcaAcc(d_by_module[mi], d_by_module[mr])
            cca_hw[i] = shapes[(f0, mi)][:2]   # alignment grid = input side's first-image grid
    if cca_accs:
        print('CCA alignment grids: '
              + '; '.join(f'pair{i + 1}({mi})→{cca_hw[i]}'
                          for i, (mi, mr) in enumerate(pairs)
                          if i in cca_hw), flush=True)
    t0 = time.time()
    step = max(1, len(folders_all) // 10)
    for k, folder in enumerate(folders_all):
        raw = {m: bank.load(folder, m) for m in involved}   # (d, h, w)
        for m in involved:
            accs[m].update(token2d(raw[m]), device)
        for i in cca_accs:
            mi, mr = pairs[i]
            tx = token2d(pf.resize_spatial(raw[mi], cca_hw[i]))
            ty = token2d(pf.resize_spatial(raw[mr], cca_hw[i]))
            cca_accs[i].update(tx, ty, device)
        if (k + 1) % step == 0 or k + 1 == len(folders_all):
            print(f'  [accumulate] {k + 1}/{len(folders_all)} images '
                  f'({time.time() - t0:.0f}s)', flush=True)

    # Per-pair geometry quantities
    centered = not args.no_center
    print(f'\n=== per-pair geometry quantities (input vs reference, '
          f'{"centered" if centered else "uncentered"} principal direction) ===')
    pairs_res = []
    for i, (mi, mr) in enumerate(pairs):
        ai, ar = accs[mi], accs[mr]
        di, dr = d_by_module[mi], d_by_module[mr]
        rms_i, rms_r = ai.norm_rms(), ar.norm_rms()
        frob_i, frob_r = ai.frobenius(), ar.frobenius()
        entry = {
            'index': i, 'input': mi, 'reference': mr,
            'd_in': di, 'd_ref': dr,
            'n_tokens_in': int(ai.n), 'n_tokens_ref': int(ar.n),
            'norm_rms_in': round(rms_i, 6), 'norm_rms_ref': round(rms_r, 6),
            'norm_ratio': round(rms_i / rms_r, 6) if rms_r > 0 else None,
            'frobenius_in': round(frob_i, 6),
            'frobenius_ref': round(frob_r, 6),
            'frobenius_ratio': round(frob_i / frob_r, 6) if frob_r > 0
                               else None,
        }
        if di == dr:
            cos = cos_principal(ai, ar, centered)
            _, evr_i = principal_dir(ai, centered)
            _, evr_r = principal_dir(ar, centered)
            entry.update({'cos_principal': round(cos, 6),
                          'cos_available': True,
                          'mode': 'principal_dir',
                          'evr_in': round(evr_i, 6),
                          'evr_ref': round(evr_r, 6)})
            print(f'  [{i + 1}/{len(pairs)}] {mi} vs {mr}: '
                  f'norm_ratio={entry["norm_ratio"]:.4f} '
                  f'(rms {rms_i:.3f}/{rms_r:.3f})  '
                  f'cos={entry["cos_principal"]:.4f}  '
                  f'evr={entry["evr_in"]:.3f}/{entry["evr_ref"]:.3f}')
        else:
            spec, rank = cca_solve(cca_accs[i])
            has = spec.numel() > 0
            rho1 = float(spec[0]) if has else float('nan')
            entry.update({'cos_principal': round(rho1, 6) if has else None,
                          'cos_available': has,
                          'mode': 'cca',
                          'cca_spectrum': [round(float(v), 6)
                                           for v in spec.tolist()],
                          'cca_rank': rank,
                          'n_tokens_aligned': int(cca_accs[i].n),
                          'evr_in': None, 'evr_ref': None})
            if has:
                spec_s = '/'.join(f'{v:.3f}' for v in spec[:6].tolist())
                more = '...' if spec.numel() > 6 else ''
                print(f'  [{i + 1}/{len(pairs)}] {mi} vs {mr}: '
                      f'norm_ratio={entry["norm_ratio"]:.4f} '
                      f'(rms {rms_i:.3f}/{rms_r:.3f})  '
                      f'cos(cca ρ1)={rho1:.4f}  '
                      f'spectrum[{spec_s}{more}]  (rank {rank})')
            else:
                print(f'  [{i + 1}/{len(pairs)}] {mi} vs {mr}: '
                      f'norm_ratio={entry["norm_ratio"]:.4f}  '
                      f'cos(cca)=— (covariance numerically degenerate)')
        pairs_res.append(entry)

    # Summary table
    width = max(max(len(mi) + len(mr) for mi, mr in pairs) + 3, 8)
    print(f'\n========== summary (norm_ratio / cos_principal / evr) ==========')
    print('pair'.ljust(width) + 'norm_ratio'.rjust(12)
          + 'cos'.rjust(10) + '  evr(in/ref)')
    for e in pairs_res:
        name = f'{e["input"]}~{e["reference"]}'
        cos_s = f'{e["cos_principal"]:.4f}' if e['cos_available'] else '—'
        evr_s = (f"{e['evr_in']:.3f}/{e['evr_ref']:.3f}"
                 if e.get('evr_in') is not None else '—')
        print(name.ljust(width) + f"{e['norm_ratio']:.4f}".rjust(12)
              + cos_s.rjust(10) + '  ' + evr_s)

    out = {
        'feature_root': str(root),
        'n_images': len(folders_all),
        'input_layout': args.input_layout,
        'centered': centered,
        'device': str(device),
        'cos_method': 'principal_dir=same-dim centered PCA principal-direction cosine (|cos|); '
                      'cca=first canonical correlation ρ1 after cross-dim token-wise alignment',
        'pairs': pairs_res,
    }
    with open(args.result_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nresults saved to: {args.result_path}')


if __name__ == '__main__':
    main()
