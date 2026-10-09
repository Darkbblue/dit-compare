"""Objective objects: thin wrappers over exp_common.formulas for the target axis."""

from . import formulas


class Objective:
    """Prediction target y = a(lam)*noise + c(lam)*data with its exact min-SNR weight.

    `rotation` (the +1 extra experiment) is only meaningful on the fm path per
    the experiment plan; this is enforced by the launcher / train script, not here.
    """

    def __init__(self, target, lam_min, lam_max):
        if target == "rotation" and not (lam_min < lam_max):
            raise ValueError("rotation target requires lam_min < lam_max")
        self.target = target
        self.lam_min = lam_min
        self.lam_max = lam_max

    def coeffs(self, lam):
        return formulas.objective_coeffs(self.target, lam, self.lam_min, self.lam_max)

    def make_target(self, lam, noise, data):
        """y = a*noise + c*data. Shapes broadcast: lam [B], noise/data [B,C,H,W]."""
        a, c = self.coeffs(lam)
        if not hasattr(a, "shape") or a.shape != lam.shape:
            a = lam * 0.0 + a
        if not hasattr(c, "shape") or c.shape != lam.shape:
            c = lam * 0.0 + c
        view = (-1,) + (1,) * (noise.dim() - 1)
        return a.view(view) * noise + c.view(view) * data

    def weight(self, lam, gamma):
        """Exact per-target min-SNR weight multiplying the native MSE."""
        return formulas.min_snr_weight(self.target, lam, gamma, self.lam_min, self.lam_max)

    def u_error_factor(self, lam):
        """Native-error -> velocity (u) space error conversion factor (ablation)."""
        return formulas.u_error_factor(self.target, lam, self.lam_min, self.lam_max)

    def c_eps(self, lam):
        a, c = self.coeffs(lam)
        return formulas.c_eps_factor(a, c, lam)

    def c_x0(self, lam):
        a, c = self.coeffs(lam)
        return formulas.c_x0_factor(a, c, lam)

    def eps_from_pred(self, pred, x_t, lam, alpha, sigma):
        return formulas.eps_from_pred(
            self.target, pred, x_t, lam, alpha, sigma, self.lam_min, self.lam_max
        )

    def cond_input(self, cond_mode, lam):
        return formulas.cond_input(cond_mode, lam, self.lam_min, self.lam_max)


def get_objective(target, lam_min, lam_max):
    return Objective(target, lam_min, lam_max)
