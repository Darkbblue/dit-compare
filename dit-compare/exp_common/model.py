"""
Shared backbone for all 2x3+1 cells (chat record section 2-G: identical backbone,
identical parameter count). Adapted from DiT-v/models.py with two deliberate
changes:

  - learn_sigma is removed entirely: every cell is a pure regression with C
    output channels (no sigma head, no VB term; SiT's dead half-head is gone too).
  - attention defaults to timm's Attention (SiT's exact implementation, no
    qk_norm). qk_norm=True keeps the custom implementation WITH the bf16 cast
    fix — note timm's own qk_norm does not cast back after LayerNorm, so under
    autocast it silently falls off flash kernels just like the original bug.
    With qk_norm=False the timm and custom implementations have IDENTICAL
    state_dict keys (qkv/proj), so checkpoints are interchangeable.

The conditioning clock is aligned externally: both frameworks feed the same
standardized logSNR value into `t_embedder` (see formulas.cond_input).
"""

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention as TimmAttention
import torch.nn.functional as F

from .skip_table import break_points as make_break_points


class Attention(nn.Module):
    """
    Multi-head attention with optional QK-normalization (LayerNorm on queries
    and keys, per head). Identical to the DiT-v modification.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Under autocast, LayerNorm runs in fp32 and returns fp32; feeding mixed
        # dtypes (fp32 q/k, bf16 v) silently drops SDPA to the slow math kernel.
        # Cast back to v's dtype so flash/mem-efficient kernels stay eligible.
        # (The normalization statistics are still computed in fp32.)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # (B, N, H, D) -> (B, H, N, D); head_dim stride stays 1, so SDPA accepts
        # these non-contiguous views directly (same as timm's Attention) — no
        # need for the three full-tensor .contiguous() copies.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations. Receives the
    standardized logSNR (a float) from both frameworks.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout
    for classifier-free guidance (shared dropout_prob=0.1 across all cells).
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################

def subtract_dirs(u, item, alpha=1.0):
    """u: (B, T, d) token sequence -- each token is a per-position d-dim vector
    of probe semantics. Standardize -> project-delete -> de-standardize, strictly
    isomorphic to delete_direction.py; mu/inv broadcast along the last dim.
    alpha=1 fully removes, (0,1) sweeps the removal strength."""
    V   = item['V'].to(device=u.device, dtype=u.dtype)
    mu  = item['mu'].to(device=u.device, dtype=u.dtype)
    inv = item['inv'].to(device=u.device, dtype=u.dtype)
    z = (u - mu) * inv                       # standardize
    z = z - alpha * ((z @ V.T) @ V)          # project-delete
    return z / inv + mu                      # de-standardize

@torch.no_grad()
def replace_dirs_with_noise(u, item, alpha=1.0):
    """Replace the components of u along the deletion directions with iid random
    noise matched in mean/variance.

    RNG alignment: noise generation is wrapped in fork_rng -- on entry it saves
    the global generator states (CPU + the CUDA device hosting u), and restores
    them as-is on exit (exception-safe). This operator is fully transparent to
    the global random stream and consumes no random numbers, so it does not
    affect the seed alignment of the leading shuffle and per-sample gt noise
    sampling at the top of the script; and the intervention run itself is
    reproducible when re-run (same global state at that point => same noise).
    """
    V   = item['V'].to(device=u.device, dtype=u.dtype)     # (k,d) row-orthonormal
    mu  = item['mu'].to(device=u.device, dtype=u.dtype)    # (d,)
    inv = item['inv'].to(device=u.device, dtype=u.dtype)   # (d,)

    z = (u - mu) * inv                                     # standardize
    coef = (z @ V.T).float()                               # (...,k) coefficients being removed
    stat_dims = tuple(range(coef.dim() - 1))               # across batch+token
    mu_c = coef.mean(dim=stat_dims)                        # (k,)
    sd_c = coef.std(dim=stat_dims)                         # (k,)

    _devs = [u.device] if u.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=_devs):
        noise = torch.randn_like(coef) * sd_c + mu_c       # (...,k) leaves the global stream untouched

    z = z + alpha * (noise.to(z.dtype) - (z @ V.T)) @ V    # in-subspace replacement
    return z / inv + mu                                    # de-standardize

def set_to_zero(u):
    return 0

class Block(nn.Module):
    """adaLN-Zero transformer block (identical in DiT and SiT)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if qk_norm:
            # custom attention WITH the autocast dtype fix (flash-eligible)
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=True, **block_kwargs)
        else:
            # SiT's exact implementation
            self.attn = TimmAttention(hidden_size, num_heads=num_heads,
                                      qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, index=None, V=None, intervention_layers=[]):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        update = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + update
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(update, f'attn', shift=shift_msa, scale=scale_msa)
        update = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if intervention_layers and index in intervention_layers:
            # update = replace_dirs_with_noise(update, V[f'vit-block{index}-mlp'])
            update = 0
        x = x + update
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BreakingLayer(nn.Module):
    """Residual-stream breaker (the "breaking-residual" cut): FinalLayer's
    non-residual replacement applied in TOKEN space (hidden -> hidden instead
    of hidden -> p*p*C). x = f(x) OVERWRITES the stream rather than adding to
    it, so information from earlier segments can only pass through this
    adaLN-modulated linear readout. Inserted between evenly-split block
    segments (see skip_table.break_points); the last segment still ends at the
    model's own final_layer.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class UnifiedDiT(nn.Module):
    """
    One backbone for every cell of the grid. C output channels always
    (pure regression; the learn_sigma head does not exist here).
    break_segments: None = plain backbone. An int S splits the blocks into S
    even segments and inserts a BreakingLayer after every segment boundary
    ("breaking-residual", e.g. 'XL/2-brk'); the loss is still computed on the
    final unpatchified output only.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        qk_norm=False,
        break_segments=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # breaking-residual: overwrite the residual stream after each segment.
        # break_points stays EMPTY for the plain backbone, so its forward loop
        # is byte-identical (zero overhead, zero state_dict keys).
        self.break_segments = break_segments
        self.break_points = {}  # 0-based block index -> breaking layer index
        if break_segments is not None:
            pts = make_break_points(depth, break_segments)
            self.break_layers = nn.ModuleList([
                BreakingLayer(hidden_size) for _ in pts
            ])
            self.break_points = {b: i for i, b in enumerate(pts)}
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # breaking layers: same adaLN-Zero zero-init philosophy as final_layer
        # (the stream restarts from 0 at every segment boundary)
        if self.break_segments is not None:
            for bl in self.break_layers:
                nn.init.constant_(bl.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(bl.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(bl.linear.weight, 0)
                nn.init.constant_(bl.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, external_skip=None, repa_depth=None, V=None, intervention_layers=[]):
        """
        x: (N, C, H, W) latent
        t: (N,) standardized logSNR (conditioning clock, shared across frameworks)
        y: (N,) class labels
        external_skip: optional external skip features; only used by LSC addons
            outside the basic/v-pred scope (kept for hook compatibility with
            the reference repos; basic/v-pred ignore it).
        repa_depth: None = pure denoising forward (sampling/evaluation, single
            tensor return). An int enables the REPA branch: the forward returns
            (output, z_tilde) where z_tilde is self.repa_projector applied to
            the output of block (repa_depth - 1) — 1-based, matching REPA's
            encoder_depth semantics. The projector runs INSIDE the DDP forward
            so its gradients all-reduce like every other parameter.
        """
        x = self.x_embedder(x) + self.pos_embed
        if external_skip is not None:
            external_skip = self.external_embedder(external_skip)  # (N, T, D), no pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        # LSC hook: verbatim copy of the code shared by DiT-v/models.py and
        # SiT/models.py (fuse skip BEFORE the up block, store AFTER the down block)
        skips = {}
        for layer, block in enumerate(self.blocks):
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['up']:
                    skip = skips[self.lsc.skip_table['up'][layer]]
                    x = self.lsc(x, skip, layer, external_skip)
            x = block(x, c, layer, V, intervention_layers)
            if hasattr(self, 'feature_gatherer'):
                self.feature_gatherer.gather(x, f'block{layer}-out')
            if repa_depth is not None and (layer + 1) == repa_depth:
                z_tilde = self.repa_projector(x)  # (N, T, D) -> (N, T, z_dim)
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['down']:
                    skips[layer] = x
            if layer in self.break_points:
                # breaking-residual cut: x = f(x) OVERWRITES the stream (the
                # only non-residual transition besides the final_layer itself)
                x = self.break_layers[self.break_points[layer]](x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if repa_depth is not None:
            return x, z_tilde
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Batched conditional/unconditional forward for classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=1)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   Configs                                     #
#################################################################################

def _mk(depth, hidden_size, patch_size, num_heads, **kwargs):
    return UnifiedDiT(depth=depth, hidden_size=hidden_size, patch_size=patch_size,
                      num_heads=num_heads, **kwargs)


Exp_models = {
    'XL/2': lambda **kw: _mk(28, 1152, 2, 16, **kw),
    # breaking-residual: XL/2 with the residual stream OVERWRITTEN by an adaLN
    # breaking layer after every 7 blocks (4 segments, cuts at blocks 6/13/20);
    # each segment must re-derive its representation — no residual bypass.
    'XL/2-brk': lambda **kw: _mk(28, 1152, 2, 16, break_segments=4, **kw),
    'XL/4': lambda **kw: _mk(28, 1152, 4, 16, **kw),
    'XL/8': lambda **kw: _mk(28, 1152, 8, 16, **kw),
    'L/2':  lambda **kw: _mk(24, 1024, 2, 16, **kw),
    'L/4':  lambda **kw: _mk(24, 1024, 4, 16, **kw),
    'L/8':  lambda **kw: _mk(24, 1024, 8, 16, **kw),
    'B/2':  lambda **kw: _mk(12, 768, 2, 12, **kw),
    'B/4':  lambda **kw: _mk(12, 768, 4, 12, **kw),
    'B/8':  lambda **kw: _mk(12, 768, 8, 12, **kw),
    'S/2':  lambda **kw: _mk(12, 384, 2, 6, **kw),
    'S/4':  lambda **kw: _mk(12, 384, 4, 6, **kw),
    'S/8':  lambda **kw: _mk(12, 384, 8, 6, **kw),
    # /16 variants: intended for --data-space pixel (JiT-alignment ablation),
    # where 256px images give 256 tokens — JiT's exact sequence length. On
    # latents (32px) they leave a degenerate 2x2 grid; not recommended there.
    'XL/16': lambda **kw: _mk(28, 1152, 16, 16, **kw),
    'L/16':  lambda **kw: _mk(24, 1024, 16, 16, **kw),
    'B/16':  lambda **kw: _mk(12, 768, 16, 12, **kw),
    'S/16':  lambda **kw: _mk(12, 384, 16, 6, **kw),
}
