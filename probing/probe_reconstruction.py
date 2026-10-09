#!/usr/bin/env python3
"""
Pixel reconstruction probe (probe_reconstruction)
==================================================

Task: predict original-image pixels (RGB) from feature modules; report
R² / MSE / PSNR / per-position cosine similarity on a held-out test set.

Data organization::

    feature_root/
        {class}_{image_name}/
            {module}.npy          # 2D feature map (h, w, d) or channels-first
                                  # (c, h, w) (--input_layout chw, same as
                                  # probe_features)
    image_dir/                    # given by --image_dir
        {class}_{image_name}.JPEG # original image, file name matches the
                                  # feature folder (uppercase .JPEG)

Efficiency strategy
-------------------
1. Unified low-resolution reconstruction grid: both the original image and
   all input features are bilinearly resized to --out_size (default 64×64).
   Per-position regression samples = #images × out_size²; feature-map
   information at low resolution suffices for ranking comparisons; varying
   original image sizes are no problem (unified resizing).
2. The original image is decoded once and the low-res result cached
   (ImageBank); memory footprint is tiny (64×64×3×4B ≈ 48KB per image).
3. The linear probe reuses probe_features' closed-form standardized ridge
   regression (streamed Gram + one CPU float64 eigendecomposition + γ chosen
   on the validation set); other probes use Adam for a few epochs.
4. Image (folder) level train/val/test split, stratified by class (reuses
   probe_features).
5. By default each input module reconstructs the original image separately
   (--per_input_probes; the default linear closed-form solution is nearly
   free); the [all-inputs-concatenated] probe is off by default and must be
   enabled explicitly with --probe.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import probe_features as pf


# =========================================================================== #
#                 Original-image loading (low-res caching)                     #
# =========================================================================== #

class ImageBank:
    """Load the original image by folder name -> RGB float32 [0,1] -> resize to the reconstruction grid and cache."""

    def __init__(self, image_dir: Path, out_hw):
        self.image_dir = Path(image_dir)
        self.out_hw = tuple(out_hw)
        self._cache = {}

    def load(self, folder) -> torch.Tensor:
        key = folder.name
        if key not in self._cache:
            img = Image.open(self.image_dir / f'{key}.JPEG').convert('RGB')
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1).contiguous()          # (3, H, W)
            x = pf.resize_spatial(x, self.out_hw)
            self._cache[key] = x
        return self._cache[key]


class ReconDataset(pf.ProbeDataset):
    """X = input-module features (aligned to the reconstruction grid) concat
    along channels, y = low-res RGB of the original image.
    Layout semantics identical to probe_features.ProbeDataset:
        flat: X (H*W, D_in), y (H*W, 3);  map: X (D_in, H, W), y (3, H, W)
    """

    def __init__(self, folders, bank, input_modules, layout, ref_hw, img_bank):
        super().__init__(folders, bank, input_modules,
                         target_module='__pixels__', layout=layout, ref_hw=ref_hw)
        self.img_bank = img_bank

    def _load_pair(self, idx):
        folder = self.folders[idx]
        y = self.img_bank.load(folder)                   # (3, oh, ow)
        xs = [pf.resize_spatial(self.bank.load(folder, m), self.ref_hw)
              for m in self.input_modules]
        return torch.cat(xs, dim=0), y


# =========================================================================== #
#                                   Main                                       #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='pixel reconstruction probe: input-module features -> '
                    'low-res RGB of the original image; reports '
                    'R²/MSE/PSNR/cosine similarity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument('--feature_root', type=str, required=True,
                   help='feature root directory containing {class}_{image_name}/ folders')
    p.add_argument('--image_dir', type=str, required=True,
                   help='original-image directory; file names are {folder_name}.JPEG (uppercase extension)')
    p.add_argument('--inputs', type=str, nargs='*', default=None,
                   help='input module name list; defaults to all modules automatically')
    p.add_argument('--max_images', type=int, default=None,
                   help='use only the first N images after sorting by folder name')
    p.add_argument('--out_size', type=int, default=64,
                   help='side length of the unified reconstruction grid (image and features both resized to out_size×out_size)')
    p.add_argument('--input_layout', type=str, default='hwd', choices=['hwd', 'chw'],
                   help='npy feature dim layout: hwd=(h, w, d) channels-last; '
                        'chw=(c, h, w) channels-first (torch (C,H,W) feature '
                        "maps, ViT features saved as token sequences rearranged "
                        "to 'b c h w', etc.)")
    # Probes and training (semantics same as probe_features.py)
    p.add_argument('--probe', type=str, nargs='+', default=None,
                   choices=pf.PROBE_NAMES,
                   help='probes for the [all-inputs-concatenated] probe '
                        '(linear/conv are linear with closed-form solutions; '
                        'mlp/convmlp are nonlinear); by default no '
                        'concatenation is run, only per-input-module probes '
                        '(see --per_input_probes)')
    p.add_argument('--per_input_probes', type=str, nargs='+', default=['linear'],
                   choices=pf.PROBE_NAMES + ['none'],
                   help='probes used when each input module reconstructs '
                        'separately (default linear = closed-form solution, '
                        'nearly free); pass none to disable')
    p.add_argument('--hidden_mlp', type=int, default=1024)
    p.add_argument('--hidden_conv', type=int, default=256)
    p.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    p.add_argument('--epochs', type=int, default=20,
                   help='Adam training epochs (conv family may pair with --lr 3e-3)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=8,
                   help='images loaded per step (map layout: images per optimizer step)')
    p.add_argument('--batch_rows', type=int, default=8192,
                   help='positions per optimizer step in the flat layout')
    p.add_argument('--linear_solver', type=str, default='auto',
                   choices=['auto', 'closed', 'adam'],
                   help='linear solver: auto = closed-form when D_in<=max_dim')
    p.add_argument('--closed_form_max_dim', type=int, default=16384)
    p.add_argument('--gamma_grid', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0])
    p.add_argument('--gram_dtype', type=str, default='float32',
                   choices=['float32', 'float64'])
    # Splitting and running
    p.add_argument('--split_ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto',
                   help='auto=cuda>mps>cpu, or a specific cuda/mps/cpu')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--cache_ram', type=str, default='auto', choices=['auto', 'on', 'off'])
    p.add_argument('--cache_ram_gb', type=float, default=8.0)
    p.add_argument('--result_path', type=str, default='probe_reconstruction_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.feature_root)
    image_dir = Path(args.image_dir)
    if not root.is_dir():
        raise RuntimeError(f'feature root directory does not exist: {root}')
    if not image_dir.is_dir():
        raise RuntimeError(f'original-image directory does not exist: {image_dir}')

    folders_all = pf.list_image_folders(root)
    if args.max_images is not None:
        folders_all = folders_all[:max(args.max_images, 0)]
        print(f'--max_images={args.max_images}: kept the first {len(folders_all)} images after sorting')
    if len(folders_all) == 0:
        raise RuntimeError('no usable images after truncation')

    # Original-image existence check (file name = folder name + '.JPEG')
    missing = [f.name for f in folders_all
               if not (image_dir / f'{f.name}.JPEG').is_file()]
    if missing:
        show = missing[:3] + (['...'] if len(missing) > 3 else [])
        print(f'[warn] {len(missing)} images lack a matching original image {show}; removed')
        folders_all = [f for f in folders_all if f.name not in set(missing)]
    if not folders_all:
        raise RuntimeError('no usable images after removing missing originals')

    modules = pf.discover_modules(folders_all)
    shapes, d_by = pf.scan_shapes(folders_all, modules, args.input_layout)
    if args.inputs:
        inputs = list(dict.fromkeys(args.inputs))
        for m in inputs:
            if m not in modules:
                raise RuntimeError(f'input module {m} not among candidates: {modules}')
    else:
        inputs = modules
    d_in = sum(d_by[m] for m in inputs)
    ref_hw = (args.out_size, args.out_size)
    print(f'images: {len(folders_all)} | input modules: {len(inputs)}, D_in={d_in} | '
          f'reconstruction grid {ref_hw}')

    # Split (stratified by class)
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
    img_bank = ImageBank(image_dir, ref_hw)

    # Validation: at least one probing task
    joint_probes = list(args.probe) if args.probe else []
    per_probes = ([] if 'none' in args.per_input_probes
                  else list(args.per_input_probes))
    if not joint_probes and not per_probes:
        raise RuntimeError('no probing task: --probe not given and '
                           '--per_input_probes=none')

    # All-inputs concatenation (off by default, enabled via --probe; reuses the probe_features pipeline)
    joint_res = {}
    if joint_probes:
        for probe in joint_probes:
            def make_ds(fl, _layout=pf.PROBE_LAYOUTS[probe]):
                return ReconDataset(fl, bank, inputs, _layout, ref_hw, img_bank)
            t0 = time.time()
            res = pf.run_one(probe, 'pixels', inputs, d_in, 3, ref_hw, bank,
                             splits, args, device, seed=args.seed,
                             make_ds=make_ds)
            # PSNR (RGB mean MSE, pixel range [0,1])
            res['psnr'] = round(-10.0 * math.log10(max(res['mse'], 1e-12)), 2)
            res['seconds'] = round(time.time() - t0, 1)
            joint_res[probe] = res

    # Per-input-module probing (single module -> image pixels); default linear only (low cost)
    per_res = {}
    if per_probes:
        print('\n--- per-input-module probing (single module -> image pixels) ---', flush=True)
        for m in inputs:
            d_m = d_by[m]
            per_res[m] = {}
            for q in per_probes:
                def make_ds_m(fl, _m=m, _layout=pf.PROBE_LAYOUTS[q]):
                    return ReconDataset(fl, bank, [_m], _layout, ref_hw, img_bank)
                r = pf.run_one(q, 'pixels', [m], d_m, 3, ref_hw, bank, splits,
                               args, device, seed=args.seed, make_ds=make_ds_m,
                               label=m)
                r['psnr'] = round(-10.0 * math.log10(max(r['mse'], 1e-12)), 2)
                per_res[m][q] = r
        width_m = max(len(m) for m in inputs) + 2
        print('\n--- per-input-module results (test R2_avg / PSNR) ---')
        print('  ' + 'module'.ljust(width_m)
              + ''.join(q.rjust(26) for q in per_probes))
        for m in inputs:
            row = m.ljust(width_m)
            for q in per_probes:
                r = per_res[m][q]
                row += f"{r['r2_avg']:.4f} / {r['psnr']:.2f}dB".rjust(26)
            print('  ' + row)

    if joint_probes:
        print('\n========== summary (all inputs concatenated, test R2_avg / cos_sim / PSNR) ==========')
        for probe, r in joint_res.items():
            print(f"  {probe:<8} {r['r2_avg']:.4f} / {r['cos_sim']:.4f} / "
                  f"{r['psnr']:.2f} dB   ({r['solver']})")

    out = {
        'task': 'reconstruction',
        'feature_root': str(root),
        'image_dir': str(image_dir),
        'n_images': len(folders_all),
        'n_missing_images': len(missing),
        'modules': modules,
        'inputs': inputs,
        'module_dims': {m: d_by[m] for m in modules},
        'd_in': d_in,
        'out_size': args.out_size,
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
