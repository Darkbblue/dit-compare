#!/usr/bin/env python3
"""
DINOv2 representation extraction (compatible with probe_features.py input format)
=================================================================================

Extract DINOv2 features from an image directory into per-image folders; the
output organization matches probe_features.py's input format exactly::

    out_dir/
        {image_name}/             # = original file name without extension
                                 #   (corresponds to the same image as the
                                 #   diffusion folder name {class}_{image_name})
            {prefix}.npy         # final-layer representation, shape (d, h, w),
                                 # channels-first
                                 # (= probe_features.py's --input_layout chw)

Only DINOv2's final output (last_hidden_state) is extracted: ViT patch tokens
(cls / register tokens removed) are rearranged into an (h, w) grid for
saving, so like diffusion features it is a 2D feature map and can directly
participate in per-position probing as --inputs / --target (note that probe
scripts need --input_layout chw).

Two run modes
-------------

1. Extraction mode (loads DINOv2; --model_path must point to a local
   HuggingFace model directory):

     python dinov2.py --image_dir images/ --out_dir feats_vp_eps/ \
         --model_path /path/to/local/dinov2-base

   - No internet download: --model_path is a local directory,
     local_files_only throughout.
   - Input images are uniformly resized to --img_size x --img_size (must be a
     multiple of patch_size; default 518 = 37x14 patches; use 224 = 16x14 or
     448 = 32x14 if VRAM is insufficient).
   - Outputs a single module ({prefix}.npy), the representation of the
     model's final layer (l{L}).
   - Target npy files that already exist are skipped automatically (resumable);
     if all targets exist the model is not even loaded. --overwrite forces
     recomputation.

2. Copy mode (does not load DINOv2; copies already-extracted DINO
   representations from the source directory to out_dir):

     python dinov2.py --source feats_vp_eps/ --out_dir feats_vp_x0/ \
         [--max_images 100]

   - Copies {prefix}.npy inside each folder of source to the same-named
     folder in out_dir; only the first --max_images folders by name are
     processed (same semantics as probe_features' --max_images); existing
     target files are skipped automatically.

Then {prefix} can be used together with diffusion module names in the probe
scripts, e.g.:

     python probe_features.py --feature_root feats_vp_eps/ --target block_27 \
         --inputs block_0 ... dinov2 --input_layout chw
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTS = {'.jpeg', '.jpg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Common utilities
# --------------------------------------------------------------------------- #

def pick_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None \
            and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def list_images(image_dir: Path, max_images=None):
    """Image directory -> list of image paths sorted by name (extension
    stripped), first max_images kept.
    The sort key is the extension-stripped stem, consistent with the sort of
    diffusion representation folder names, so the image set selected by
    --max_images matches the folder set selected by probe_features."""
    files = [p for p in image_dir.iterdir()
             if p.is_file() and not p.name.startswith('.')
             and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.stem)
    seen, uniq = set(), []
    for p in files:                      # same stem, different extensions: keep the first in sort order
        if p.stem not in seen:
            seen.add(p.stem)
            uniq.append(p)
    if not uniq:
        raise RuntimeError(f'no image files found under {image_dir} '
                           f'(supported extensions: {sorted(IMAGE_EXTS)})')
    if max_images is not None:
        uniq = uniq[:max_images]
    return uniq


# --------------------------------------------------------------------------- #
# Extraction mode
# --------------------------------------------------------------------------- #

def preprocess(files, img_size: int, device) -> torch.Tensor:
    """(N images) -> standardized (N, 3, img_size, img_size), official DINOv2 preprocessing."""
    xs = []
    for p in files:
        img = Image.open(p).convert('RGB')
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None]                     # (1, 3, H, W)
        xs.append(F.interpolate(x, size=(img_size, img_size), mode='bicubic',
                                align_corners=False, antialias=True)[0])
    x = torch.stack(xs)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return ((x - mean) / std).to(device)


def extract(args):
    if not args.image_dir:
        raise RuntimeError('extraction mode requires --image_dir (original-image directory)')
    image_dir, out_dir = Path(args.image_dir), Path(args.out_dir)
    if not image_dir.is_dir():
        raise RuntimeError(f'image directory does not exist: {image_dir}')
    files = list_images(image_dir, args.max_images)

    from transformers import AutoConfig, AutoModel

    # Lightweight config load (no download, no weights): validate args + resumability check
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    L = int(config.num_hidden_layers)
    patch = int(config.patch_size)
    if args.img_size % patch != 0:
        raise RuntimeError(f'--img_size {args.img_size} is not a multiple of '
                           f'patch_size {patch}; use a multiple of {patch} '
                           f'(e.g. {patch * 16}, {patch * 24}, {patch * 37})')
    reg = int(getattr(config, 'num_register_tokens', 0) or 0)
    grid = args.img_size // patch
    d_model = int(config.hidden_size)

    def save_path(stem: str) -> Path:
        return out_dir / stem / f'{args.prefix}.npy'

    need = [p for p in files
            if args.overwrite or not save_path(p.stem).is_file()]
    print(f'images: {len(files)} (to extract {len(need)}) | model {args.model_path}')
    print(f'final layer (l{L}) output | patch {patch}, grid {grid}x{grid}, '
          f'd={d_model}, register tokens={reg} | output layout (d, h, w) = chw')
    if not need:
        print('all target representations already exist; model not loaded, done.')
        return

    model = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model.eval().to(pick_device(args.device))
    device = next(model.parameters()).device

    t0, n_written = time.time(), 0
    for bi in range(0, len(need), args.batch_size):
        batch = need[bi:bi + args.batch_size]
        x = preprocess(batch, args.img_size, device)
        with torch.no_grad():
            # patch tokens of the final-layer output (last_hidden_state)
            feats = model(x).last_hidden_state[:, 1 + reg:, :]
            B, P, _ = feats.shape
            assert P == grid * grid, (P, grid)
            fm = feats.transpose(1, 2).reshape(B, d_model, grid, grid)
            fm = fm.to(torch.float32).cpu().numpy()          # (B, d, h, w)
            for j, p in enumerate(batch):
                if args.overwrite or not save_path(p.stem).is_file():
                    save_path(p.stem).parent.mkdir(parents=True, exist_ok=True)
                    np.save(save_path(p.stem), fm[j])
                    n_written += 1
        print(f'  [{min(bi + len(batch), len(need))}/{len(need)}] image batch done, '
              f'{n_written} npy written so far', flush=True)

    dt = time.time() - t0
    print(f'extraction done: wrote {n_written} npy '
          f'({len(need)} images, fewer if some already existed), '
          f'took {dt:.1f}s ({dt / max(len(need), 1):.2f}s/image)')
    print(f'for probing use: --input_layout chw, module name {args.prefix}')


# --------------------------------------------------------------------------- #
# Copy mode
# --------------------------------------------------------------------------- #

def copy_feats(args):
    source, out_dir = Path(args.source), Path(args.out_dir)
    if not source.is_dir():
        raise RuntimeError(f'source directory does not exist: {source}')

    folders = sorted((p for p in source.iterdir() if p.is_dir()),
                     key=lambda p: p.name)
    if not folders:
        raise RuntimeError(f'no subfolders under {source}')
    if args.max_images is not None:
        folders = folders[:args.max_images]
    name = f'{args.prefix}.npy'
    n_copied = n_skipped = n_missing = 0
    t0 = time.time()
    for f in folders:
        src = f / name
        if not src.is_file():
            print(f'[warn] {f.name}: missing {name}; skipping this folder')
            n_missing += 1
            continue
        dst_dir = out_dir / f.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        if dst.exists():
            if args.overwrite and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                n_copied += 1
            else:
                n_skipped += 1
            continue
        shutil.copy2(src, dst)
        n_copied += 1
    dt = time.time() - t0
    print(f'copy done: {len(folders)} folders -> {out_dir} | '
          f'copied {n_copied}, skipped (exists) {n_skipped}, missing (warned) {n_missing} | '
          f'took {dt:.1f}s')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='DINOv2 representation extraction: output compatible with '
                    'probe_features.py input format ((c, h, w) layout, use '
                    '--input_layout chw when probing); supports extraction '
                    'mode (local HF model) and copy mode (--source, no model load)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--image_dir', type=str, default=None,
                   help='[extraction mode] original-image directory; image name without extension = output folder name')
    p.add_argument('--out_dir', type=str, required=True,
                   help='output feature root directory (may point directly at an '
                        'existing diffusion representation directory; DINO npy '
                        'files are written into the corresponding folders)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model_path', type=str, default=None,
                   help='[extraction mode] local HuggingFace DINOv2 model '
                        'directory (e.g. the downloaded facebook/dinov2-base), '
                        'no online download')
    g.add_argument('--source', type=str, default=None,
                   help='[copy mode] feature root of already-extracted DINO '
                        'representations; copies {prefix}.npy into --out_dir, '
                        'no model load')
    p.add_argument('--max_images', type=int, default=None,
                   help='only process the first N images/folders by name '
                        '(semantics aligned with probe_features --max_images)')
    p.add_argument('--prefix', type=str, default='dinov2',
                   help='output module name (DINOv2 final-layer output); npy file = {prefix}.npy')
    p.add_argument('--img_size', type=int, default=518,
                   help='[extraction mode] input images uniformly resized to '
                        'img_size x img_size (must be a multiple of patch_size; '
                        '518=37x14, 448=32x14, 224=16x14)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='[extraction mode] images per batch')
    p.add_argument('--device', type=str, default='auto',
                   help='[extraction mode] auto / cuda / mps / cpu')
    p.add_argument('--overwrite', action='store_true',
                   help='overwrite existing output files (default: skip, resumable)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.source is not None:
        copy_feats(args)                   # copy mode: no model load
    else:
        extract(args)                      # extraction mode: load DINOv2


if __name__ == '__main__':
    main()
