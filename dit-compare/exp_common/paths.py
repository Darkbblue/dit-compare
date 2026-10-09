"""Path objects: thin wrappers over exp_common.formulas for the framework axis."""

from . import formulas


class ContinuousPath:
    """A path x_t = alpha(lam)*data + sigma(lam)*noise parameterized by logSNR.

    Both paths are continuous in lam: there is no timestep quantization anywhere
    (this removes the staircase artifact of lambda->integer-t inversion noted in
    section 2-D of the chat record).
    """

    path_type = None

    def alpha_sigma(self, lam):
        return formulas.path_alpha_sigma(self.path_type, lam)

    def d_alpha_sigma(self, lam):
        return formulas.path_d_alpha_sigma(self.path_type, lam)

    def snr(self, lam):
        return formulas.snr(lam)

    def sample_lambda(self, batch_size, lam_min, lam_max, device=None, generator=None,
                      mode="uniform", p_mean=-0.8, p_std=0.8):
        """Factor 1: sampling density over logSNR.

        uniform      : U[lam_min, lam_max] on the shared support (the 2x3+1 default).
        logit-normal : JiT's t = sigmoid(z) with z ~ N(p_mean, p_std^2); since
                       lam = 2*logit(t) = 2z, this is lam ~ N(2*p_mean, (2*p_std)^2).
                       Unclamped (matching JiT); lam_min/lam_max still normalize
                       the conditioning clock and the diagnostics bins.
        """
        import torch
        if mode == "logit-normal":
            z = torch.randn(batch_size, device=device, generator=generator)
            return (2.0 * p_std) * z + 2.0 * p_mean
        u = torch.rand(batch_size, device=device, generator=generator)
        return lam_min + (lam_max - lam_min) * u


class VPPath(ContinuousPath):
    """Continuous VP (DDPM) path: alpha^2 + sigma^2 = 1."""
    path_type = "vp"


class LinearPath(ContinuousPath):
    """Linear (flow matching) path: alpha + sigma = 1."""
    path_type = "fm"


def get_path(path_type):
    if path_type == "vp":
        return VPPath()
    if path_type == "fm":
        return LinearPath()
    raise ValueError(f"unknown path_type: {path_type}")
