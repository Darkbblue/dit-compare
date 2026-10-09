"""
Per-lambda binned diagnostics (chat record section 2-J): the direct numerical
evidence that flattening succeeded. Accumulates, per lambda bin:

  - count            : effective sample count (should be uniform across bins)
  - native_mse       : mean native-target MSE
  - eps_mse          : mean epsilon-space MSE = native_mse / c_eps(lam)
                       (after conversion, all cells regress the same quantity)
  - weight           : mean applied min-SNR weight w(lam)
  - weighted_mse     : mean w(lam) * native_mse  (the effective objective)

Usage: call `update(...)` every training step (cheap, stays on GPU), then
`all_reduce()` + `snapshot()` at log time, and `reset()`.
"""

import torch


class LambdaBinLogger:
    def __init__(self, lam_min, lam_max, num_bins=12, device=None):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.num_bins = num_bins
        self.edges = torch.linspace(lam_min, lam_max, num_bins + 1, device=device)
        self.reset(device)

    def reset(self, device=None):
        device = device if device is not None else self.edges.device
        self.count = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.native_sum = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.eps_sum = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.weight_sum = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.weighted_sum = torch.zeros(self.num_bins, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, lam, native_mse, eps_mse, weight):
        """lam/native_mse/eps_mse/weight: 1-D tensors of shape [B].

        Fully vectorized with index_add_: the previous per-bin python loop
        evaluated `if n > 0` on a 0-dim tensor, which forces a GPU->CPU sync
        num_bins times EVERY training step and stalls the async pipeline.
        """
        idx = torch.bucketize(lam, self.edges[1:-1].to(lam.device))
        self.count.index_add_(0, idx, torch.ones_like(lam, dtype=torch.float64))
        self.native_sum.index_add_(0, idx, native_mse.to(torch.float64))
        self.eps_sum.index_add_(0, idx, eps_mse.to(torch.float64))
        self.weight_sum.index_add_(0, idx, weight.to(torch.float64))
        self.weighted_sum.index_add_(0, idx, (weight * native_mse).to(torch.float64))

    def all_reduce(self):
        """Synchronize bins across DDP ranks (call before snapshot)."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for buf in (self.count, self.native_sum, self.eps_sum,
                        self.weight_sum, self.weighted_sum):
                torch.distributed.all_reduce(buf)

    def snapshot(self):
        """Return a JSON-serializable dict of per-bin means."""
        centers = (0.5 * (self.edges[:-1] + self.edges[1:])).tolist()
        count = self.count.clamp(min=1)
        return {
            "bin_center": centers,
            "count": self.count.tolist(),
            "native_mse": (self.native_sum / count).tolist(),
            "eps_mse": (self.eps_sum / count).tolist(),
            "weight": (self.weight_sum / count).tolist(),
            "weighted_mse": (self.weighted_sum / count).tolist(),
        }
