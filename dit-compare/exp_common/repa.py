"""
REPA (representation alignment) integration for the unified framework.

Verbatim-faithful ports from ref/REPA-main:
  - build_mlp: the projector head (models/sit.py)
  - repa_proj_loss: per-token negative cosine similarity (loss.py; vectorized,
    mathematically identical to the per-sample loop for a single encoder)
  - dinov2_preprocess: the dinov2 branch of train.py's preprocess_raw_image,
    adapted for this framework's pipeline (images already in [-1, 1])

Two deliberate deviations from the reference:
  - DINOv2 loads from MANUAL local paths instead of auto-downloading (the
    cluster has no internet). Two weight formats are supported, auto-detected:
      * HuggingFace directory (config.json + model.safetensors) -> Dinov2Model
      * official pretrain .pth + local dinov2 repo checkout -> torch.hub
  - The projector is applied INSIDE the model forward (like REPA's own SiT, and
    unlike an external hook collector) so DDP gradient bucketing covers it —
    applying a DDP-wrapped submodule outside forward() desyncs the reducer.
    The model's feature_gatherer hooks stay protocol-compatible with
    diffusion_feature (evaluation via prepare_feature_extractor unchanged).

Enabled in train_exp.py via --repa-depth (None = off). The alignment layer
may also follow a progressive schedule (--repa-depth + --repa-depth-steps):
the aligned layer advances over training, reusing the same projector.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_mlp(hidden_size, projector_dim, z_dim):
    """REPA projector (ref/REPA-main/models/sit.py, verbatim)."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def attach_repa(model, repa_depths, z_dim, projector_dim=2048):
    """
    Attach the REPA projector as a model submodule, so it is part of
    state_dict / optimizer / EMA exactly like REPA's in-model projectors.
    Call BEFORE the EMA deepcopy / DDP wrap. The projector is consumed inside
    UnifiedDiT.forward(repa_depth=...) — never from outside the DDP forward.

    repa_depths: EVERY alignment layer the run will ever use (progressive
    schedules included). ONE shared projector is attached regardless of the
    schedule length — its shape depends only on (hidden_size, projector_dim,
    z_dim), not on the layer — so advancing the aligned layer reuses the same
    weights (curriculum-style continuation) and leaves optimizer / EMA / DDP
    completely untouched.
    """
    for repa_depth in repa_depths:
        assert 1 <= repa_depth <= len(model.blocks), \
            f"repa_depth {repa_depth} out of range [1, {len(model.blocks)}]"
    model.repa_projector = build_mlp(model.hidden_size, projector_dim, z_dim)
    # xavier-init the projector linears, matching REPA's _basic_init coverage
    for m in model.repa_projector.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


def parse_repa_schedule(repa_depth, repa_depth_steps):
    """
    Validate the --repa-depth / --repa-depth-steps pair and normalise it to
    (depths, steps), two aligned lists: the active alignment layer becomes
    depths[i] once train_steps >= steps[i]. Legacy behaviour (no
    --repa-depth-steps) requires a single depth and keeps it fixed forever;
    a multi-depth schedule WITHOUT steps is an error rather than a silent
    truncation to the first entry.
    """
    depths = list(repa_depth)
    assert depths, "--repa-depth requires at least one value"
    if repa_depth_steps is None:
        assert len(depths) == 1, \
            "a multi-depth REPA schedule needs --repa-depth-steps " \
            "(one switching step per depth, aligned 1:1 with --repa-depth)"
        return depths, None
    steps = list(repa_depth_steps)
    assert len(steps) == len(depths), \
        f"--repa-depth-steps ({len(steps)} values) must align 1:1 with " \
        f"--repa-depth ({len(depths)} values)"
    assert steps[0] == 0, \
        "the first --repa-depth-steps entry must be 0 (the initial depth is " \
        "active from the start of training)"
    assert all(s2 > s1 for s1, s2 in zip(steps, steps[1:])), \
        "--repa-depth-steps must increase strictly"
    return depths, steps


def repa_depth_at(depths, steps, train_steps):
    """
    The alignment layer active at the given global step (fixed depths[0]
    when steps is None). Every rank derives it from the same train_steps,
    so the parameter-free layer switch stays DDP-consistent without any
    synchronization.
    """
    if steps is None:
        return depths[0]
    active = depths[0]
    for depth, start in zip(depths, steps):
        if train_steps >= start:
            active = depth
    return active


class _HFDinov2Encoder(nn.Module):
    """
    Wrap a HuggingFace Dinov2Model with the hub-style forward_features() API
    used by REPA: 'x_norm_patchtokens' = patch tokens AFTER the final
    LayerNorm. HF's last_hidden_state is already layernormed; we slice off the
    prefix tokens (cls, plus register tokens for the with-registers variants —
    auto-detected from the length mismatch, no config field needed).
    Positional-embedding interpolation is delegated to the model
    (interpolate_pos_encoding=True), which is idempotent at 224px input.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.embed_dim = hf_model.config.hidden_size
        self.patch_size = hf_model.config.patch_size

    def forward_features(self, x):
        try:
            out = self.model(pixel_values=x, interpolate_pos_encoding=True)
        except TypeError:  # older transformers without the kwarg (224px only)
            out = self.model(pixel_values=x)
        h = out.last_hidden_state  # (B, 1[+R]+T, D)
        n_patches = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        return {'x_norm_patchtokens': h[:, h.shape[1] - n_patches:]}


@torch.no_grad()
def load_dinov2(ckpt_path, repo_dir=None, arch='dinov2_vitb14', image_size=256, device='cuda'):
    """
    Load a frozen DINOv2 encoder from local paths (no auto-download).
    ckpt_path decides the format:
      - a DIRECTORY (HF format: config.json + model.safetensors) -> loads via
        transformers.Dinov2Model; repo_dir is unused and `arch` is ignored
        (config.json decides the architecture, embed_dim comes from it too)
      - an official pretrain .pth FILE -> torch.hub source='local'; requires
        repo_dir = local checkout of facebookresearch/dinov2 (hubconf.py at
        root); 'teacher'-wrapped and 'backbone.'-prefixed checkpoints are
        also handled
    """
    if os.path.isdir(ckpt_path):
        from transformers import Dinov2Model
        encoder = _HFDinov2Encoder(Dinov2Model.from_pretrained(ckpt_path))
        encoder.eval().requires_grad_(False).to(device)
        return encoder
    assert repo_dir is not None, \
        "official .pth weights need --dinov2-repo (local dinov2 checkout)"
    import timm
    encoder = torch.hub.load(repo_dir, arch, source='local', pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'teacher' in state:
        state = state['teacher']
    if any(k.startswith('backbone.') for k in state):
        state = {k[len('backbone.'):]: v for k, v in state.items()
                 if k.startswith('backbone.')}
    encoder.load_state_dict(state, strict=True)
    del encoder.head
    encoder.head = nn.Identity()
    patch_resolution = 16 * (image_size // 256)
    if patch_resolution != 16:
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
    encoder.eval().requires_grad_(False).to(device)
    return encoder


def dinov2_preprocess(x, image_size):
    """
    REPA's dinov2 preprocessing (train.py preprocess_raw_image), adapted:
    this framework's pipeline yields images in [-1, 1], so (x+1)/2 is exactly
    REPA's x/255. Order matches the reference: normalize, then resize.
    """
    x = (x + 1.0) / 2.0
    x = TF.normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    x = F.interpolate(x, 224 * (image_size // 256), mode='bicubic')
    return x


def repa_proj_loss(z, z_tilde):
    """
    REPA projection loss (loss.py): per-token negative cosine similarity,
    averaged over tokens and batch. Computed in fp32 for stable normalization
    under bf16 autocast.
    """
    z = F.normalize(z.float(), dim=-1)
    z_tilde = F.normalize(z_tilde.float(), dim=-1)
    return -(z * z_tilde).sum(dim=-1).mean()
