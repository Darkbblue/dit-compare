"""Modulate on-disk representations with deletion directions (offline pass2).

The direction npz is produced by probe_features.py --save_deletion_dirs; each
entry carries a space field deciding the deletion mode (branched here):
  fullspace (current default, SC1 iteration): directions are an orthonormal
      basis in the standardized space —
      z = (x−mu)⊙inv_std → z −= α·(z@Vᵀ)V → x' = z⊙std + mu
  r2_target / topk (baseline): directions are an orthonormal basis in the
      original representation space, deleted by direct projection.
Output: {feature_dir|output_dir}/{image}/{module}-del-{eps|x0}.npy, with
shape/layout identical to the input ((c, h, w) feature maps)."""
import json
import os
import socket

import numpy as np
import torch

feature_dir = '/jindofs_temp/users/292869/intern/benyuan/outputs/probing/dit-compare-fm-a15-t300/'
output_dir = None    # None = same image subfolder as the source npy; else output_dir/{image}

NPZ_PATH = '/jindofs_temp/users/292869/intern/benyuan/outputs/probing/dit-compare-fm-a15-t300-mlp-2.npz'


def load_deletion_table(npz_path, target, device):
    """Return {input: {'dirs': (k,d), 'space': str, 'mu': (d,), 'inv_std': (d,),
                       'protect': list, 'r2_del_before': float, ...}}"""
    data = np.load(npz_path)
    meta = json.loads(str(data['__meta__']))
    table = {}
    for e in meta['entries']:
        if e['target'] != target or e['k'] <= 0:
            continue
        item = {'dirs': torch.from_numpy(
                    np.asarray(data[e['keys']['dirs']])).float().to(device),
                'space': e.get('space', 'original'),
                'protect': e.get('protect'),
                'r2_del_before': e.get('r2_del_before'),
                'r2_pro_before': e.get('r2_pro_before'),
                'r2_pro_after': e.get('r2_pro_after')}
        if item['space'] == 'standardized':
            item['mu'] = torch.from_numpy(
                np.asarray(data[e['keys']['mu']])).float().to(device)
            item['inv_std'] = torch.from_numpy(
                np.asarray(data[e['keys']['inv_std']])).float().to(device)
        table[e['input']] = item
    return table


def subtract_flat(flat, V, alpha=1.0):
    V = V.to(device=flat.device, dtype=flat.dtype)
    return flat - alpha * ((flat @ V.T) @ V)


def subtract_dirs_chw(feat, item, alpha=1.0):
    """Delete directions from a (c, h, w) feature map per the entry's space semantics; shape unchanged."""
    V = item['dirs']
    c, h, w = feat.shape
    if item['space'] == 'standardized':
        # SC1 protocol: directions are defined in the standardized space —
        # standardize first, delete by projection, then de-standardize
        mu, inv = item['mu'], item['inv_std']
        z = (feat - mu[:, None, None]) * inv[:, None, None]
        flat = z.permute(1, 2, 0).reshape(-1, c)             # (h*w, c)
        flat = subtract_flat(flat, V, alpha)
        z = flat.reshape(h, w, c).permute(2, 0, 1)           # (c, h, w)
        return z / inv[:, None, None] + mu[:, None, None]
    # Original-space directions (r2_target/topk/fallback): direct projection
    flat = feat.permute(1, 2, 0).reshape(-1, c)
    flat = subtract_flat(flat, V, alpha)
    return flat.reshape(h, w, c).permute(2, 0, 1)


V_eps = load_deletion_table(NPZ_PATH, target='pipe-eps', device='cpu')
V_x0 = load_deletion_table(NPZ_PATH, target='pipe-x0', device='cpu')

LAYERS = [f'vit-block{l}-mlp' for l in range(9)]
ALPHA = 1.0    # deletion strength: 1 = full deletion, (0,1) = partial deletion (strength sweep)

# Startup check: direction entries for every layer exist in both components
# (if missing, report all available keys at once)
missing = [m for m in LAYERS if m not in V_eps or m not in V_x0]
if missing:
    raise RuntimeError(
        f'direction npz is missing entries for these input modules: {missing}\n'
        f'pipe-eps available: {sorted(V_eps)}\npipe-x0 available: {sorted(V_x0)}')
for name, tab in (('pipe-eps', V_eps), ('pipe-x0', V_x0)):
    it = tab[LAYERS[0]]
    if it['space'] == 'standardized':
        tag = 'standardized space (SC1 iteration)'
        if it.get('protect'):
            tag += (f" + protect {','.join(it['protect'])} (target R² "
                    f"{it['r2_del_before']:.3f}→residual, protected signal R² "
                    f"{it['r2_pro_before']:.3f}→{it['r2_pro_after']:.3f})")
    else:
        tag = 'original space'
    print(f'direction semantics {LAYERS[0]} -> {name}: space={it["space"]} ({tag}), '
          f'dirs {tuple(it["dirs"].shape)}')

imgs = sorted(
    p for p in os.listdir(feature_dir)
    if os.path.isdir(os.path.join(feature_dir, p))
    and any(f.endswith('.npy') for f in os.listdir(os.path.join(feature_dir, p)))
)[:500]
print(f'{len(imgs)} images to process, first 3: {imgs[:3]} '
      f'(cross-check against the splits.train prefix of the pass1 result json)')
print(f'running node: {socket.gethostname()} | '
      f'feature_dir realpath: {os.path.realpath(feature_dir)}')

for n, i in enumerate(imgs):
    # Output directory: default is the same image subfolder as the source npy,
    # feature_dir/{image}/ (was once mistakenly written to the feature_dir top
    # level, where the 500 images overwrote each other leaving only the last)
    dst_root = os.path.join(feature_dir, i) if output_dir is None \
        else os.path.join(output_dir, i)
    os.makedirs(dst_root, exist_ok=True)
    for li, feat_id in enumerate(LAYERS):
        src = os.path.join(feature_dir, i, f'{feat_id}.npy')
        feat = torch.from_numpy(np.load(src)).float()    # (c, h, w)

        feat_del_eps = subtract_dirs_chw(feat, V_eps[feat_id], ALPHA)
        feat_del_x0 = subtract_dirs_chw(feat, V_x0[feat_id], ALPHA)

        # np.save(file, arr): the first argument is the path (the original call had the argument order reversed)
        dst_eps = os.path.join(dst_root, f'{feat_id}-del-eps.npy')
        dst_x0 = os.path.join(dst_root, f'{feat_id}-del-x0.npy')
        np.save(dst_eps, feat_del_eps.numpy().astype(np.float32))
        np.save(dst_x0, feat_del_x0.numpy().astype(np.float32))
        if n == 0 and li == 0:
            # Write self-check: read back immediately to verify the write
            # (rule out cross-node/view sync issues)
            assert os.path.isfile(dst_eps), f'write failed: {dst_eps}'
            sz = os.path.getsize(dst_eps)
            print(f'[write self-check] {dst_eps} ({sz / 1e6:.1f} MB) — '
                  f'if visible on this machine but not on other nodes, the path '
                  f'is a node-local disk rather than shared storage; '
                  f'change output_dir to a shared path', flush=True)
    print(f'[{n + 1}/{len(imgs)}] {i} done', flush=True)

print('all done')
