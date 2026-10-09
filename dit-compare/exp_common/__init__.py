"""
exp_common: unified experiment framework for the 2x3(+1) DiT-vs-SiT comparison.

Design source: the experiment plan discussed in "OpenRouter Chat Sun Aug 16 2026.md".
The effective training loss of every cell is written as

    L = p(lam) * w(lam) * c_tgt(lam) * R_theta(lam)

and the framework flattens every factor except trajectory geometry (framework
axis) and output semantics (target axis):

  factor 1  p(lam)  : uniform sampling in logSNR space, shared [lam_min, lam_max]
  factor 2  w(lam)  : explicit min-SNR weight, shared gamma
  factor 3  c_tgt   : exact target-space conversion, subsumed into the per-target
                      min-SNR forms (all become pure functions of lam, path-free)

Additionally flattened: conditioning clock (standardized lam fed to the same
TimestepEmbedder), backbone (one shared class), optimizer hyperparameters,
EMA, label dropout, common random pairing (single script + shared seed).
"""

from . import formulas
from .paths import get_path, VPPath, LinearPath
from .objectives import get_objective, Objective
