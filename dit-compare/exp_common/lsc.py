"""
LSC (Long Skip Connections) for the unified framework, for the follow-up
experiments after the 2x3+1 grid.

Faithfully replicates the adapter implementations shared by the two reference
repos (DiT-v/train_nas.py and SiT/train_nas.py both build on the same
`diffusion_feature.add_lsc`; the adapter math below is copied from
generic-diffusion-feature/feature/components/lsc_adapter.py). Only the two
addon types requested for the follow-up experiments are kept:

  lsc_addon = "basic"  : x -> Linear(LayerNorm(cat([x, skip])))   (U-ViT style)
  lsc_addon = "v-pred" : x -> x + gamma * LayerNorm(skip), gamma init 0
                         (gated residual; at init it is EXACTLY the w/o model,
                         the skip contribution grows from zero like a learned
                         correction term)

args.lsc_type selects the skip topology (see make_skip_table):
  "w/o"    : no LSC (default; the 2x3+1 grid runs in this mode)
  "w/"     : dense mirrored skips (0<->n-1, 1<->n-2, ...), identical to the
             explicit 'w/ LSC dense' table in both reference train_nas.py
  "skewed" : shifted skips, down segment starts at layer 6 (6<->n-1, ...),
             identical to SiT/train_nas.py's 'skewed LSC' table

The model-side hook (UnifiedDiT.forward) is copied verbatim from the code
shared by DiT-v/models.py and SiT/models.py.
"""

import torch
import torch.nn as nn

from .skip_table import make_skip_table  # re-exported for callers of this module

__all__ = ["make_skip_table", "LSCAdapter", "LSCAdapterVPred",
           "LSCAdapterGroup", "add_lsc"]


# ---------------------------------------------------------------------------
# adapters (replicated from the reference lsc_adapter.py)
# ---------------------------------------------------------------------------

class LSCAdapter(nn.Module):
    """lsc_addon='basic': fuse by concatenation + LayerNorm + Linear."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x, skip, external_skip=None):
        if external_skip is not None:
            skip = external_skip
        x = torch.cat([x, skip], dim=-1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class LSCAdapterVPred(nn.Module):
    """lsc_addon='v-pred': gated residual skip, gamma initialized to zero
    (training starts exactly at the w/o-LSC model)."""

    def __init__(self, dim):
        super().__init__()
        self.skip_norm = nn.LayerNorm(dim, 1e-6, True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, skip, external_skip=None):
        x = x + self.gamma * self.skip_norm(skip)
        return x


_LSC_TYPES = {
    "basic": LSCAdapter,
    "v-pred": LSCAdapterVPred,
}


class LSCAdapterGroup(nn.Module):
    """Per-up-layer adapters + skip table; attached to the model as `.lsc`."""

    def __init__(self, dim, layer_count, lsc_type, skip_table=None):
        super().__init__()
        if lsc_type not in _LSC_TYPES:
            raise ValueError(f"unknown lsc addon: {lsc_type} (choices: {list(_LSC_TYPES)})")
        self.lsc_type = lsc_type
        self.layer_count = layer_count
        self.skip_table = skip_table if skip_table is not None \
            else make_skip_table(layer_count, "w/")
        self.all_lsc = nn.ModuleDict()
        for layer in self.skip_table["up"]:
            self.all_lsc[str(layer)] = _LSC_TYPES[lsc_type](dim)

    def forward(self, x, skip, layer_idx, external_skip=None):
        adapter = self.all_lsc[str(layer_idx)]
        return adapter(x, skip, external_skip=external_skip)

    def get_layer(self, layer_idx):
        return self.all_lsc[str(layer_idx)]


def add_lsc(transformer, dim, layer_count, lsc_type="basic", skip_table=None):
    """Attach the LSC group to the transformer (same contract as the reference
    `diffusion_feature.add_lsc`): accessible as `transformer.lsc`; the model's
    forward hook picks it up via hasattr(self, 'lsc')."""
    transformer.lsc = LSCAdapterGroup(dim, layer_count, lsc_type, skip_table)
    if hasattr(transformer, "device"):
        transformer.lsc.to(transformer.device)
    if hasattr(transformer, "dtype"):
        transformer.lsc.to(transformer.dtype)
    return transformer
