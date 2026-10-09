"""Standardize-whiten on-disk representations (offline two-pass version).

Whitening definition (consistent with transform_network.solve_transform's
'standardize' branch):
  z' = (z − μ) / σ_d, per-dim standardization (each channel independent).
  μ/σ_d are computed from all tokens of the [to-be-whitened image set]
  (after N_IMAGES truncation) — float64 streamed accumulation (protocol
  pitfalls 2/3: centering/precision), per-layer
  independent (pitfall 4). standardize only needs the diagonal of the
  covariance, not the full Gram; the statistics are just Σz / Σz² (vs
  LayerStats' d×d Gram, saving memory and multiply-adds on wide layers).

IO/loop logic references delete_direction.py:
  Input  {feature_dir}/{image}/{feat_id}.npy — (c, h, w) channels-first
        (generic-diffusion-feature FeatureStore convention, same as the
        probe --input_layout chw; standardize acts per channel, independent
        of token grid flattening order)
  pass1 stream statistics over the selected image set (after N_IMAGES
        truncation) -> per-layer μ / σ (layer-independent);
  pass2 apply z' = (z − μ)⊙inv_std per image and write to disk (source
        files untouched);
  Output {feature_dir|output_dir}/{image}/{feat_id}-whitened.npy, with
        shape / layout / dtype (fp32) identical to the input.

Numerical protection: "dead channels" with variance no more than
  max(variance)·1e-12 (near-zero variance, common for inactive units)
  output 0 — reusing transform_network purge-x0's dead criterion.
  The standard branch's clamp_min(1e-30) would amplify constant channels
  ~1e15× into numerical garbage; in a persistent on-disk setting they must
  be zeroed, otherwise all downstream metrics are polluted.

Truncation (N_IMAGES, same as delete_direction.py's first 500): only the
  first N images in name order are processed (None = all) — pass1 statistics
  and pass2 writing [synchronously] cover exactly these N images; statistics
  range = whitening range (transductive), μ/σ do not skew due to truncation.

Differences from delete_direction.py:
  - pass2 includes a re-check: after whitening, per-dim mean should be ≈0
    and variance ≈1 (within fp32 rounding), end-to-end verification of
    statistics/solve/write in one go.

Usage: edit feature_dir / LAYERS / output_dir / N_IMAGES at the top and run
  directly; python whiten_features.py --selftest runs a synthetic end-to-end
  self-check (no external dependencies).
"""

import argparse
import os
import socket

import numpy as np
import torch

# ---- Config (same top-constant style as delete_direction.py; edit as needed) ----
feature_dir = ('/jindofs_temp/users/292869/intern/benyuan/outputs/probing/'
               'dit-compare-fm-a15-t300/')
output_dir = None    # None = same image subfolder as the source npy; else output_dir/{image}
LAYERS = [f'vit-block{l}-mlp' for l in range(9)]
N_IMAGES = 500       # only the first N images in name order (statistics and writing cover exactly this set); None = all

DEAD_REL = 1e-12     # dead-channel criterion: var <= var.max()·DEAD_REL -> output zeroed


def list_image_dirs(feature_dir, layers):
    """Image subfolders containing any target-layer npy (sorted; same
    directory scan as delete_direction.py, but with a more precise filter —
    only target-layer files count, unrelated npy excluded)."""
    return sorted(
        p for p in os.listdir(feature_dir)
        if os.path.isdir(os.path.join(feature_dir, p))
        and any(os.path.isfile(os.path.join(feature_dir, p, f'{lid}.npy'))
                for lid in layers))


def check_sources(feature_dir, imgs, layers):
    """Startup check: every (image, layer) source file exists (if missing,
    report all available modules at once, same style as delete_direction.py's
    missing check)."""
    missing = {}
    for i in imgs:
        root = os.path.join(feature_dir, i)
        for lid in layers:
            if not os.path.isfile(os.path.join(root, f'{lid}.npy')):
                missing.setdefault(i, []).append(lid)
    if missing:
        first = next(iter(missing))
        avail = sorted(f[:-4] for f in os.listdir(os.path.join(feature_dir, first))
                       if f.endswith('.npy'))
        n_bad = sum(len(v) for v in missing.values())
        raise RuntimeError(
            f'{n_bad} source files missing (across {len(missing)} images), e.g. '
            f'{first}: {missing[first]}; available modules in that image dir: {avail}')


class DiagStats:
    """Per-layer streamed diagonal statistics (float64 CPU):
    n / Σz / Σz² -> μ / per-dim variance.

    standardize only needs diag(C), not the full Gram (protocol pitfall 3:
    fp64 accumulation; the input data is already on CPU, no cross-device
    issue)."""

    def __init__(self):
        self.n = 0
        self.sum = None      # (d,) fp64, lazily initialized on the first image
        self.sq = None

    def update(self, x):
        """x: (c, h, w) torch tensor, channel dim = dim 0; flatten tokens and sum."""
        x = x.reshape(x.shape[0], -1).double()
        if self.sum is None:
            self.sum = torch.zeros(x.shape[0], dtype=torch.float64)
            self.sq = torch.zeros(x.shape[0], dtype=torch.float64)
        elif x.shape[0] != self.sum.shape[0]:
            raise RuntimeError(f'channel count mismatch: first image '
                               f'{self.sum.shape[0]} vs current {x.shape[0]} '
                               f'(statistics require same layer width)')
        self.n += x.shape[1]
        self.sum += x.sum(dim=1)
        self.sq += (x * x).sum(dim=1)

    def finalize(self):
        """-> (mu, var) fp64. Divisor n−1 matches transform_network.LayerStats
        (unbiased); variance undefined for n<2, raise directly."""
        if self.n < 2:
            raise RuntimeError(f'insufficient samples: only {self.n} tokens, variance undefined')
        mu = self.sum / self.n
        var = (self.sq - self.n * mu * mu) / (self.n - 1)
        return mu, var


def whiten_chw(feat, mu, inv_std):
    """(c, h, w) per-channel whitening: z' = (z − μ)⊙inv_std, shape/dtype unchanged.

    Equivalent to transform_network.apply_linear with M=diag(inv_std)
    (a diagonal transform is independent of token flattening order);
    mu/inv_std are first cast to feat's dtype (fp32), matching
    apply_linear's fp32 application convention."""
    mu32 = mu.to(feat.dtype)
    inv32 = inv_std.to(feat.dtype)
    return (feat - mu32[:, None, None]) * inv32[:, None, None]


def solve_standardize(var):
    """Per-dim variance -> (inv_std, dead): dead channels (var <=
    var.max()·DEAD_REL, including all-zero layers) get inv_std = 0 (their
    output is all 0); the rest get inv_std = 1/√var.
    run's solve and the selftest's recomputation share this implementation,
    guaranteeing bit-identical conventions."""
    dead = ~(var > var.max() * DEAD_REL)
    inv_std = torch.where(dead, torch.zeros_like(var),
                          var.clamp_min(1e-30).rsqrt())
    return inv_std, dead


def run(feature_dir, layers, output_dir=None, n_images=None):
    """Two-pass main flow: pass1 streamed statistics -> solve standardize ->
    pass2 whiten and write.
    n_images: only the first n_images images in name order (None = all);
    statistics and writing synchronously cover only the truncated set.
    Returns {layer: {'shape', 'tokens', 'n_dead', 'std_span'}} for self-test
    assertions."""
    imgs = list_image_dirs(feature_dir, layers)
    if not imgs:
        raise RuntimeError(f'no image subfolders with target-layer npy under '
                           f'{feature_dir} (layers: {layers})')
    n_all = len(imgs)
    if n_images is not None:
        if n_images < 1:
            raise RuntimeError(f'n_images={n_images} must be >= 1, or None for all')
        imgs = imgs[:n_images]
    check_sources(feature_dir, imgs, layers)
    cut = (f' ({n_all} images in the directory, truncated to the first '
           f'{len(imgs)} in name order — statistics and whitening '
           f'[synchronously] cover exactly this set, no skew)'
           if len(imgs) < n_all else '')
    print(f'{len(imgs)} images to process{cut}, first 3: {imgs[:3]}')
    print(f'layers ({len(layers)}): {layers}')
    print(f'output_dir: {output_dir or "(same dir as source, file name suffixed with -whitened)"}')
    print(f'running node: {socket.gethostname()} | '
          f'feature_dir realpath: {os.path.realpath(feature_dir)}')
    print(f'statistics range = the {len(imgs)} images to process (same as the whitening range)')

    # ---- pass1: streamed statistics μ/σ (per-layer independent, fp64) ----
    print(f'\n[pass1] streamed statistics ({len(imgs)} images × {len(layers)} layers):')
    stats = {lid: DiagStats() for lid in layers}
    shapes = {}
    for n, i in enumerate(imgs):
        for lid in layers:
            arr = np.load(os.path.join(feature_dir, i, f'{lid}.npy'))
            if arr.ndim != 3:
                raise RuntimeError(f'{i}/{lid}.npy shape {arr.shape}, expected 3 dims')
            if lid not in shapes:
                shapes[lid] = arr.shape
                c0, h0, w0 = arr.shape
                warn = ('  [!] last dim far larger than the first two; '
                        'suspected channels-last (h,w,c) layout — this script '
                        'assumes (c,h,w) channels-first, please confirm first!'
                        if w0 > 8 * max(c0, h0) else '')
                print(f'  {lid}: shape {arr.shape} (c, h, w), '
                      f'{h0 * w0} tokens per image{warn}')
            elif arr.shape != shapes[lid]:
                raise RuntimeError(f'{i}/{lid}.npy shape {arr.shape} mismatches '
                                   f'the first image {shapes[lid]} (same layer '
                                   f'requires the same grid)')
            stats[lid].update(torch.from_numpy(arr))
        if (n + 1) % 50 == 0 or n + 1 == len(imgs):
            print(f'  pass1 {n + 1}/{len(imgs)}', flush=True)

    # ---- Solve standardize: mu / inv_std (same formula as transform_network) ----
    mu_inv = {}
    summary = {}
    print('\n[solve] standardize (z−μ)/σ_d, per-layer diagnostics:')
    for lid in layers:
        mu, var = stats[lid].finalize()
        inv_std, dead = solve_standardize(var)
        mu_inv[lid] = (mu, inv_std)
        kept = var[~dead]
        span = float(kept.max() / kept.min()) if kept.numel() else 1.0
        summary[lid] = {'shape': shapes[lid], 'tokens': stats[lid].n,
                        'n_dead': int(dead.sum()), 'std_span': span}
        print(f'  {lid}: tokens {stats[lid].n}, std_span {span:.2e}'
              + (f', {int(dead.sum())} dead channels (output zeroed)' if dead.any()
                 else ''))

    # ---- pass2: whiten per image and write + post-whitening statistics re-check ----
    print(f'\n[pass2] whiten and write ({"same dir" if output_dir is None else output_dir}):')
    verify = {lid: DiagStats() for lid in layers}
    for n, i in enumerate(imgs):
        dst_root = os.path.join(feature_dir, i) if output_dir is None \
            else os.path.join(output_dir, i)
        os.makedirs(dst_root, exist_ok=True)
        for lid in layers:
            src = os.path.join(feature_dir, i, f'{lid}.npy')
            feat = torch.from_numpy(np.load(src)).float()
            out = whiten_chw(feat, *mu_inv[lid])
            dst = os.path.join(dst_root, f'{lid}-whitened.npy')
            np.save(dst, out.numpy())
            verify[lid].update(out)
            if n == 0 and lid == layers[0]:
                # Write self-check: read back immediately and compare element-wise
                # (rule out cross-node/view sync issues/corrupted writes)
                back = np.load(dst)
                assert back.shape == shapes[layers[0]] \
                    and back.dtype == np.float32, \
                    f'written shape/dtype abnormal: {back.shape} {back.dtype}'
                assert np.array_equal(back, out.numpy()), 'read-back differs from what was written'
                print(f'[write self-check] {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)'
                      f' — read-back element-wise identical', flush=True)
        print(f'[{n + 1}/{len(imgs)}] {i} done', flush=True)

    # ---- Re-check: post-whitening per-dim mean ≈0 / variance ≈1 ----
    print('\n[re-check] post-whitening empirical statistics (should be ≈0 / ≈1, within fp32 rounding):')
    for lid in layers:
        mu2, var2 = verify[lid].finalize()
        dead = mu_inv[lid][1] == 0
        print(f'  {lid}: |mean| max {float(mu2[~dead].abs().max()):.2e}, '
              f'std [{float(var2[~dead].min().sqrt()):.4f}, '
              f'{float(var2[~dead].max().sqrt()):.4f}]')
    print('\nall done')
    return summary


# --------------------------------------------------------------------------- #
#           selftest: synthetic end-to-end (no external dependencies)          #
# --------------------------------------------------------------------------- #

def _selftest():
    """Synthetic end-to-end: truncation/statistics/solve/write/re-check full flow, item-by-item assertions (covers the full execution path)."""
    import shutil
    import tempfile

    print('== whiten_features selftest: synthetic end-to-end ==')
    torch.manual_seed(0)
    c, h, w, n_img, n_trunc = 24, 8, 8, 6, 4
    layers = ['blk-a-mlp', 'blk-b-mlp']
    td = tempfile.mkdtemp(prefix='whiten_selftest_')
    try:
        # Synthetic data: σ spans 3 orders of magnitude; channel 0 constant
        # across the image = dead channel; out-of-truncation images
        # (img04/05) have active channels shifted by +100 — if statistics
        # mix them in, the recomputation must differ
        srcs = {}
        for lid in layers:
            mu_l = torch.randn(c) * 3
            sig_l = torch.logspace(1, -2, c) * torch.exp(0.3 * torch.randn(c))
            arrs = []
            for img_i in range(n_img):
                x = mu_l[:, None, None] \
                    + sig_l[:, None, None] * torch.randn(c, h, w)
                if img_i >= n_trunc:
                    x[1:] += 100.0
                x[0] = 7.0                       # constant channel -> variance 0 -> zeroed
                arrs.append(x.float().numpy())
            srcs[lid] = arrs
        for img_i in range(n_img):
            root = os.path.join(td, f'img{img_i:02d}')
            os.makedirs(root)
            for lid in layers:
                np.save(os.path.join(root, f'{lid}.npy'), srcs[lid][img_i])

        # ---- Assertion 1: truncation — only process/count the first n_trunc images ----
        summary = run(td, layers, n_images=n_trunc)
        for img_i in range(n_img):
            dst = os.path.join(td, f'img{img_i:02d}',
                               f'{layers[0]}-whitened.npy')
            assert os.path.isfile(dst) == (img_i < n_trunc), dst
        assert all(summary[lid]['tokens'] == n_trunc * h * w
                   for lid in layers)
        for lid in layers:
            # Recomputation comparison (direction-sensitive assertion):
            # whitening img00 with statistics of the first N images matches
            # the on-disk result bit-for-bit; statistics contaminated with
            # out-of-truncation images (+100 shift) must differ
            feat = torch.from_numpy(
                np.load(os.path.join(td, 'img00', f'{lid}.npy'))).float()

            def stats_whiten(feat, data_arrs):
                st = DiagStats()
                for a in data_arrs:
                    st.update(torch.from_numpy(a))
                mu, var = st.finalize()
                inv, _ = solve_standardize(var)
                return whiten_chw(feat, mu, inv)

            back = np.load(os.path.join(td, 'img00', f'{lid}-whitened.npy'))
            assert np.array_equal(
                back, stats_whiten(feat, srcs[lid][:n_trunc]).numpy()), \
                'statistics range != first N images'
            assert not np.array_equal(
                back, stats_whiten(feat, srcs[lid]).numpy()), \
                'statistics seem contaminated with out-of-truncation data'
        print('  [ok] truncation: only the first N images processed/written, '
              'statistics range = first N images '
              '(recomputation bit-identical; differs if contaminated)')

        # ---- Assertion 2: full run — outputs complete / naming / shape dtype / sources untouched ----
        summary = run(td, layers)
        for img_i in range(n_img):
            root = os.path.join(td, f'img{img_i:02d}')
            for lid in layers:
                dst = os.path.join(root, f'{lid}-whitened.npy')
                assert os.path.isfile(dst), dst
                back = np.load(dst)
                assert back.shape == (c, h, w) and back.dtype == np.float32, \
                    (back.shape, back.dtype)
                assert np.array_equal(
                    np.load(os.path.join(root, f'{lid}.npy')),
                    srcs[lid][img_i]), 'source file was modified'
        assert all(summary[lid]['shape'] == (c, h, w) for lid in layers)
        assert all(summary[lid]['tokens'] == n_img * h * w for lid in layers)
        print('  [ok] full run: outputs complete / -whitened naming / shape dtype / source files untouched')

        # ---- Assertion 3: post-whitening per-channel mean ≈0 / variance ≈1; dead channels all 0 ----
        # Note: outs is (n_img, c, h, w); the batch dim must first be permuted
        # behind the channel dim before reshape(c, -1) — reshaping directly
        # would mix multiple images and channels within each row, and the mean
        # would be dominated by inter-image sampling fluctuation ~σ/√n_img,
        # producing false positives (first-version pitfall)
        for lid in layers:
            outs = np.stack([np.load(os.path.join(
                td, f'img{i:02d}', f'{lid}-whitened.npy'))
                for i in range(n_img)])
            z = torch.from_numpy(outs).double().permute(1, 0, 2, 3) \
                .reshape(c, -1)
            mean, var = z.mean(dim=1), z.var(dim=1, unbiased=True)
            # Mean tolerance: |μ − fp32(μ)|/σ_min (~0.006) can reach ~1e-4
            # magnitude; 1e-3 is still far below real-bug signals (O(0.1+))
            assert float(mean[1:].abs().max()) < 1e-3, \
                float(mean[1:].abs().max())
            assert float((var[1:] - 1).abs().max()) < 1e-4, \
                float((var[1:] - 1).abs().max())
            assert float(z[0].abs().max()) == 0.0, 'dead channel should output all zeros'
            assert summary[lid]['n_dead'] == 1
        print('  [ok] per-channel mean≈0 variance≈1, dead channels zeroed (n_dead=1)')

        # ---- Assertion 4: output_dir branch (separate output directory; -whitened files not mistaken for sources) ----
        out2 = os.path.join(td, '_out')
        summary2 = run(td, layers, output_dir=out2)
        for img_i in range(n_img):
            for lid in layers:
                assert os.path.isfile(os.path.join(
                    out2, f'img{img_i:02d}', f'{lid}-whitened.npy'))
        assert summary2[layers[0]]['n_dead'] == summary[layers[0]]['n_dead']
        print('  [ok] output_dir branch: written to a separate directory, statistics consistent with the same-dir version')
        print('\nALL PASS')
    finally:
        shutil.rmtree(td)


def main():
    p = argparse.ArgumentParser(
        description='standardize-whiten on-disk representations (two passes: '
                    'statistics + writing); target/layers/output dir/truncation '
                    'configured as constants at the top of the file')
    p.add_argument('--selftest', action='store_true',
                   help='synthetic end-to-end self-check (does not touch the real feature_dir)')
    p.add_argument('--f', action='store_true')
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(feature_dir, LAYERS, output_dir, N_IMAGES)


if __name__ == '__main__':
    main()
