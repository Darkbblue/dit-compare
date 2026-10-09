"""
Single source of truth for all flattening math in the 2x3+1 framework.

Every function is duck-typed: it accepts either python floats (verified locally
by check_alignment.py with the stdlib only) or torch tensors of any broadcastable
shape (used by train_exp.py / sample_exp.py on GPU). Do NOT add a second copy of
these formulas anywhere else; the whole point is that training, sampling and the
alignment check share one implementation.

Conventions (chat record notation):
  - `data`  = clean latent (a.k.a. x0 in DDPM papers, x1 in SiT code)
  - `noise` = standard Gaussian (a.k.a. epsilon in DDPM papers, x0 in SiT code)
  - x_t = alpha(lam) * data + sigma(lam) * noise,  SNR = (alpha/sigma)^2 = exp(lam)
  - prediction target y = a(lam) * noise + c(lam) * data
"""

import math

# ---------------------------------------------------------------------------
# duck-typed elementary ops
# ---------------------------------------------------------------------------

def _exp(x):
    return x.exp() if hasattr(x, "exp") else math.exp(x)


def _sqrt(x):
    return x.sqrt() if hasattr(x, "sqrt") else math.sqrt(x)


def _cos(x):
    return x.cos() if hasattr(x, "cos") else math.cos(x)


def _sin(x):
    return x.sin() if hasattr(x, "sin") else math.sin(x)


def _minimum(x, y):
    # elementwise min(x, y); y may be a python scalar
    if hasattr(x, "clamp"):
        return x.clamp(max=y) if isinstance(y, float) else x.minimum(y)
    return min(x, y)


def sigmoid(x):
    return 1.0 / (1.0 + _exp(-x))


# ---------------------------------------------------------------------------
# factor 1: logSNR-uniform sampling density p(lam), shared support
# ---------------------------------------------------------------------------

def snr(lam):
    """SNR = exp(lam)."""
    return _exp(lam)


def default_lambda_range(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Compute the logSNR support that covers the FULL generative path, from the
    original DDPM chain (T=1000, linear beta schedule, as used by DiT/ADM).

    The previous fixed [-6, 6] support was designed for representation-learning
    probes (it emphasizes the semantically strong SNR region). For generation
    the sampler must traverse the whole path, so the support endpoints are
    COMPUTED, not hard-coded:

        lam_max = logSNR(abar_1) = log((1-beta_1)/beta_1)   (~ +9.21)
                = the most data-like point of the DDPM chain
        lam_min = logSNR(abar_T)                             (~ -10.12)
                = the most noise-like point of the DDPM chain

    Both frameworks share the same SNR at a given lam on either path, so this
    single computed range covers the full sampling trajectory of the VP column
    (it IS the DDPM chain, continuously interpolated) and of the FM column
    (identical SNR coverage end to end).
    """
    log_abar = 0.0
    lam_min = lam_max = None
    for i in range(num_timesteps):
        beta = beta_start + (beta_end - beta_start) * i / (num_timesteps - 1)
        log_abar += math.log(1.0 - beta)
        abar = math.exp(log_abar)
        lam = log_abar - math.log1p(-abar)  # log(abar) - log(1 - abar) = logSNR
        if i == 0:
            lam_max = lam  # t = 1: closest to data
    lam_min = lam  # t = T: closest to pure noise
    return lam_min, lam_max


# ---------------------------------------------------------------------------
# paths (framework axis). Coefficients are functions of lam only; the discrete
# DDPM chain is replaced by the continuous VP path (chat record section 2-D),
# which removes the lambda->integer-t quantization staircase entirely.
# ---------------------------------------------------------------------------

def path_alpha_sigma(path_type, lam):
    """Return (alpha, sigma) = (data coefficient, noise coefficient)."""
    if path_type == "vp":
        # VP / DDPM: alpha^2 + sigma^2 = 1
        alpha = _sqrt(sigmoid(lam))
        sigma = _sqrt(sigmoid(-lam))
    elif path_type == "fm":
        # Linear / flow matching: alpha + sigma = 1 (SiT's native t equals alpha)
        alpha = sigmoid(0.5 * lam)
        sigma = sigmoid(-0.5 * lam)
    else:
        raise ValueError(f"unknown path_type: {path_type}")
    return alpha, sigma


def path_d_alpha_sigma(path_type, lam):
    """Derivatives d(alpha)/d(lam) and d(sigma)/d(lam) (for the PF-ODE drift)."""
    alpha, sigma = path_alpha_sigma(path_type, lam)
    if path_type == "vp":
        d_alpha = 0.5 * alpha * sigma * sigma
        d_sigma = -0.5 * sigma * alpha * alpha
    elif path_type == "fm":
        d_alpha = 0.5 * alpha * sigma
        d_sigma = -0.5 * alpha * sigma
    else:
        raise ValueError(f"unknown path_type: {path_type}")
    return d_alpha, d_sigma


# ---------------------------------------------------------------------------
# objectives (target axis). y = a(lam) * noise + c(lam) * data
# ---------------------------------------------------------------------------

def theta_schedule(lam, lam_min, lam_max):
    """Uniform angular-velocity sweep: theta goes pi/2 -> 0 linearly in lam,
    so the noise:data proportion rotates at constant angular velocity along lam.
    theta(lam_min) = pi/2 (pure data target), theta(lam_max) = 0 (pure noise)."""
    return 0.5 * math.pi * (lam_max - lam) / (lam_max - lam_min)


# Angle-interpolated targets (finer target axis), theta from the noise axis.
# a0 is deliberately ABSENT: (cos0, -sin0) = (1, 0) is byte-identical to eps.
# a80 densifies the grid near the x0 end (a75->a80->a90, 5-degree step); its
# mirror partner a10 is not implemented, so it does not join the mirror pairs.
ANGLE_TARGETS_DEG = {"a15": 15.0, "a30": 30.0, "a45": 45.0,
                     "a60": 60.0, "a75": 75.0, "a80": 80.0, "a90": 90.0}


def objective_coeffs(target, lam, lam_min=None, lam_max=None):
    """Return (a, c) such that the prediction target is y = a*noise + c*data.

    eps      : y = noise              (fixed 1:0 proportion)
    x0       : y = data               (fixed 0:1 proportion)
    u        : y = noise - data       (fixed 1:1 proportion, chat's final choice)
    rotation : y = cos(theta)*noise - sin(theta)*data, theta = theta_schedule(lam)
               (proportion sweeps with lam; the +1 extra experiment, FM only)
    a15/a30/a45/a60/a75/a80/a90 : angle-interpolated fixed targets on the
               noise/data unit circle, y = cos(theta)*noise - sin(theta)*data
               with theta measured FROM THE NOISE AXIS in degrees. Together
               with eps (0), u (45) and x0 (90, up to an irrelevant global
               sign) they form a uniform 15-degree grid; theta and 90-theta
               are mirror images (noise/data swapped), so a15 mirrors a75 and
               a30 mirrors a60 about u; a80 densifies the x0 end at a 5-degree
               step (its partner a10 is not implemented). Being unit-norm,
               they differ from u by an overall 1/sqrt(2) scale — immaterial:
               every downstream factor (min-SNR weight, u_error_factor, c_eps)
               is homogeneous in (a, c) and absorbs any constant target scale
               exactly. a45 == u/||u|| (same ray, unit norm); a90 == -x0
               (same ray, flipped). a0 is not offered: it equals eps exactly
               (identical coefficients, hence an identical training run).
    """
    if target == "eps":
        return 1.0, 0.0
    if target == "x0":
        return 0.0, 1.0
    if target == "u":
        return 1.0, -1.0
    if target == "rotation":
        if lam_min is None or lam_max is None:
            raise ValueError("rotation target requires lam_min/lam_max")
        theta = theta_schedule(lam, lam_min, lam_max)
        return _cos(theta), -_sin(theta)
    if target in ANGLE_TARGETS_DEG:
        theta = math.radians(ANGLE_TARGETS_DEG[target])
        return _cos(theta), -_sin(theta)
    raise ValueError(f"unknown target: {target}")


def c_eps_factor(a, c, lam):
    """Target-space conversion factor (factor 3), epsilon space:
    ||d y||^2 = c_eps * ||d noise||^2  =>  c_eps = (a - c * exp(-lam/2))^2.
    Depends on lam only, i.e. identical on both paths (chat record, final table)."""
    return (a - c * _exp(-0.5 * lam)) ** 2


def c_x0_factor(a, c, lam):
    """Target-space conversion factor, x0(data) space:
    ||d y||^2 = c_x0 * ||d data||^2  =>  c_x0 = (a*sqrt(s) - c)^2, s = exp(lam)."""
    return (a * _sqrt(snr(lam)) - c) ** 2


def min_snr_weight(target, lam, gamma, lam_min=None, lam_max=None):
    """Per-target min-SNR weight w(lam) multiplying the NATIVE MSE of `target`.

    min-SNR (Hang et al., 2023) reads: convert every target's loss to the common
    x0 space, then cap the weight at gamma. With c_x0 above, this gives one
    closed form covering all cells:

        w(lam) = min(s, gamma) / (a*sqrt(s) - c)^2 ,  s = exp(lam)

      eps      : min(s, gamma) / s
      x0       : min(s, gamma)
      u        : min(s, gamma) / (1 + sqrt(s))^2        (chat record, final reply)
      rotation : min(s, gamma) / (sqrt(s)*cos(theta) + sin(theta))^2

    It is a pure function of lam -> the effective weight profile W(lam)
    coincides pointwise across all 2x3+1 cells.
    """
    if gamma is None:
        # weighting disabled: return multiplicative identity with lam's type
        return lam * 0.0 + 1.0 if hasattr(lam, "__sub__") else 1.0
    a, c = objective_coeffs(target, lam, lam_min, lam_max)
    return _minimum(snr(lam), gamma) / c_x0_factor(a, c, lam)


# ---------------------------------------------------------------------------
# weighting bump (ablation): extra multiplicative Gaussian weight centred at
# mu, on top of the min-SNR (or v-loss) weight. Explicit weighting and sampling
# weighting are equivalent, so only the explicit form is implemented.
# ---------------------------------------------------------------------------

def sampling_bump_weight(lam, mu, sigma, lam_min, lam_max):
    """Mean-one Gaussian bump in logSNR: bump(lam) = exp(-(lam-mu)^2/(2*sigma^2))
    divided by its expectation under the training prior lam ~ U[lam_min, lam_max].

    Multiplying the per-sample loss by this weight IS training with lam drawn
    from the tilted density p(lam) proportional to U[lam_min,lam_max] * bump(lam)
    (importance-weight identity), while the normalisation keeps the total loss
    scale identical to the un-bumped run, so loss curves stay comparable.
    mu/sigma and the support are python floats, so the normaliser is a scalar
    computed with math.erf; only the numerator is duck-typed.
    """
    z = (lam - mu) / sigma
    bump = _exp(-0.5 * z * z)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    mean = (sigma / (lam_max - lam_min)) * math.sqrt(0.5 * math.pi) * (
        math.erf((lam_max - mu) / sigma * inv_sqrt2)
        - math.erf((lam_min - mu) / sigma * inv_sqrt2))
    return bump / mean


def flat_uniform_weight(lam, gamma, lam_min, lam_max):
    """Mean-one normalised RECIPROCAL of the aligned min-SNR profile min(s, gamma).

    The 'wn-flat' ablation: multiplying the min-SNR weight by this factor
    cancels the aligned weight profile exactly. With
    w_total = min_snr_weight * flat_uniform_weight,

        w_total * c_x0 = min(s, gamma) / E_U[1/min(s, gamma)] = const

    i.e. the effective training weight per unit of logSNR becomes CONSTANT --
    the run is exactly the plain, unweighted x0-space MSE (in eps space the
    profile is min(s, gamma)/s instead; the two differ by the deterministic
    Jacobian, as everywhere).

    Closed-form normaliser (piecewise integral of 1/min(e^lam, gamma), kink
    at lam_star = ln gamma):
        E_U[1/min(s,g)] = [ (e^-lam_min - 1/g) + (lam_max - ln g)/g ] / R
    All of gamma/lam_min/lam_max are python floats so the normaliser is a
    scalar; only the numerator is duck-typed (tensor lam allowed).
    """
    lam_star = math.log(gamma)
    integral = (math.exp(-lam_min) - 1.0 / gamma) + (lam_max - lam_star) / gamma
    mean = integral / (lam_max - lam_min)
    return 1.0 / (_minimum(_exp(lam), gamma) * mean)


def mirror_minsnr_weight(lam, gamma, lam_min, lam_max):
    """Mean-one normalised compensation turning the aligned min-SNR profile
    min(s, gamma) into its LEFT-RIGHT MIRROR about the support midpoint.

    The 'wn-mirror' ablation: multiplying the min-SNR weight by this factor
    replaces the effective aligned profile (in x0 space) by

        W_mirror(lam) = W(Lam - lam) = min(e^(Lam - lam), gamma),
        Lam = lam_min + lam_max   (mirror axis = support midpoint 2*mid)

    up to the global 1/E scale fixed by mean-one normalisation: wherever the
    original profile put high weight the mirrored run puts low weight, and
    vice versa (monotone decreasing instead of increasing).

    Closed-form normaliser over the three regions cut by q = Lam - ln(gamma)
    (mirrored kink) and p = ln(gamma) (original kink); for the project's
    gamma with exp(mid) <= gamma <= exp(lam_max) the order is
    lam_min <= q <= p <= lam_max and

        E_U[comp] = [ gamma * (e^-lam_min - e^-q)                (lam < q: gamma*e^-lam)
                    + (e^(2p-Lam) - e^(Lam-2p)) / 2              (q < lam < p: e^(Lam-2lam))
                    + (e^(Lam-p) - e^lam_min) / gamma ] / R      (lam > p: e^(Lam-lam)/gamma)

    gamma/lam_min/lam_max are python floats so the normaliser is a scalar;
    only the numerator/denominator are duck-typed (tensor lam allowed).
    """
    mid = 0.5 * (lam_min + lam_max)
    if not (math.exp(mid) <= gamma <= math.exp(lam_max)):
        raise ValueError("wn-mirror requires exp(midpoint) <= gamma <= exp(lam_max) "
                         "so that both kinks stay inside the support")
    R = lam_max - lam_min
    p = math.log(gamma)
    Lam = lam_min + lam_max
    q = Lam - p
    integral = (gamma * (math.exp(-lam_min) - math.exp(-q))
                + 0.5 * (math.exp(2.0 * p - Lam) - math.exp(Lam - 2.0 * p))
                + (math.exp(Lam - p) - math.exp(Lam - lam_max)) / gamma)
    mean = integral / R
    num = _minimum(_exp(Lam - lam), gamma)   # mirrored profile
    den = _minimum(_exp(lam), gamma)         # original min-SNR profile
    return num / (den * mean)


# ---------------------------------------------------------------------------
# u-space profile ablations. The aligned min-SNR effective profile read in U
# space is target-independent (c_x0 * c_u = s*(1+r)^2 = (1+sqrt(s))^2 for
# every target, the (a - c*r)^2 factors cancel exactly):
#
#     W_u(lam) = min(s, gamma) / (1 + sqrt(s))^2 ,  s = exp(lam)
#
# so a compensation c(lam) multiplying the min-SNR weight reshapes W_u to
# W_u(lam)*c(lam) identically for all cells. Two shapes are provided:
# umirror (the midpoint mirror of W_u) and ucenter (the mean-one
# normalised geometric mean of W_u and its mirror, i.e. the effective
# profile is EXACTLY symmetric about the support midpoint).
# ---------------------------------------------------------------------------

def _u_profile_ratio(lam, gamma, lam_min, lam_max, exponent):
    """[W_u(Lam-lam)/W_u(lam)]**exponent, elementwise (duck-typed).

    u = e^(lam/2) = sqrt(s) maps the mirror point Lam-lam to c/u with
    c = e^(Lam/2); the ratio is piecewise smooth with kinks where the
    min() of either profile switches (u = c/sqrt(gamma) and u = sqrt(gamma)).
    """
    Lam = lam_min + lam_max
    num = _minimum(_exp(Lam - lam), gamma) / (1.0 + _exp(0.5 * (Lam - lam)))**2
    den = _minimum(_exp(lam), gamma) / (1.0 + _exp(0.5 * lam))**2
    return (num / den)**exponent


def _u_ratio_mean(gamma, lam_min, lam_max, exponent):
    """E_U{[W_u(Lam-lam)/W_u(lam)]**exponent} under lam ~ U[lam_min, lam_max],
    closed form (python floats only; called once per weight evaluation).

    Substitute u = e^(lam/2), dlam = 2 du/u, c = e^(Lam/2); with
    R(lam) = W_u(Lam-lam)/W_u(lam) the integrand splits at u1 = c/sqrt(gamma)
    and u2 = sqrt(gamma) into three regions:

      exponent=1   R = gamma*(1+u)^2/(u+c)^2            (u < u1)
                       = c^2*(1+u)^2/(u^2*(u+c)^2)      (u1 < u < u2)
                       = c^2*(1+u)^2/(gamma*(u+c)^2)    (u > u2)
      exponent=0.5 sqrt(R) = sqrt(gamma)*(1+u)/(u+c)
                       = c*(1+u)/(u*(u+c))
                       = (c/sqrt(gamma))*(1+u)/(u+c)

    each integrated by partial fractions (antiderivatives P/P0/Q/S for the
    exponent-1 pieces -- (1+u)^2 splits as u^2+2u+1 over u^3, hence P0 --
    G/H for the sqrt pieces).
    """
    if not (math.exp(0.5 * (lam_min + lam_max)) <= gamma <= math.exp(lam_max)):
        raise ValueError("wn-umirror/ucenter require exp(midpoint) <= gamma <= "
                         "exp(lam_max) so that both u-space kinks stay inside "
                        "the support")
    R = lam_max - lam_min
    c = math.exp(0.5 * (lam_min + lam_max))
    u_min = math.exp(0.5 * lam_min)
    u_max = math.exp(0.5 * lam_max)
    sg = math.sqrt(gamma)
    u1 = c / sg
    u2 = sg

    if exponent == 1.0:
        def P(u):   # int (1+u)^2 / (u*(u+c)^2) du
            return (math.log(u) / (c * c)
                    + (1.0 - 1.0 / (c * c)) * math.log(u + c)
                    + (1.0 - c)**2 / (c * (u + c)))
        def Q(u):   # int 1 / (u^2*(u+c)^2) du
            return (-2.0 * math.log(u) / c**3 - 1.0 / (c * c * u)
                    + 2.0 * math.log(u + c) / c**3 - 1.0 / (c * c * (u + c)))
        def S(u):   # int 1 / (u^3*(u+c)^2) du
            return (3.0 * math.log(u) / c**4 + 2.0 / (c**3 * u)
                    - 1.0 / (2.0 * c * c * u * u)
                    - 3.0 * math.log(u + c) / c**4 + 1.0 / (c**3 * (u + c)))
        def P0(u):  # int 1 / (u*(u+c)^2) du
            return (math.log(u) / (c * c)
                    - math.log(u + c) / (c * c)
                    + 1.0 / (c * (u + c)))
        def FB(u):  # int (1+u)^2 / (u^3*(u+c)^2) du == S + 2Q + P0
            return S(u) + 2.0 * Q(u) + P0(u)
        integral = (2.0 * gamma * (P(u1) - P(u_min))
                    + 2.0 * c * c * (FB(u2) - FB(u1))
                    + 2.0 * (c * c / gamma) * (P(u_max) - P(u2)))
    elif exponent == 0.5:
        def G(u):   # int (1+u) / (u*(u+c)) du
            return (math.log(u) / c
                    + (1.0 - 1.0 / c) * math.log(u + c))
        def H(u):   # int (1+u) / (u^2*(u+c)) du
            return ((c - 1.0) * math.log(u) / (c * c) - 1.0 / (c * u)
                    + (1.0 - c) * math.log(u + c) / (c * c))
        integral = (2.0 * sg * (G(u1) - G(u_min))
                    + 2.0 * c * (H(u2) - H(u1))
                    + 2.0 * (c / sg) * (G(u_max) - G(u2)))
    else:
        raise ValueError("exponent must be 1.0 (umirror) or 0.5 (ucenter)")
    return integral / R


def umirror_weight(lam, gamma, lam_min, lam_max):
    """Mean-one normalised compensation turning the U-SPACE effective profile
    W_u(lam) = min(s, gamma)/(1+sqrt(s))^2 into its LEFT-RIGHT MIRROR about
    the support midpoint.

    The 'wn-umirror' ablation: multiplying the min-SNR weight by this factor
    replaces the effective u-space profile (identical for every target) by

        W_u_mirror(lam) = W_u(Lam - lam),   Lam = lam_min + lam_max

    up to the global 1/E scale from mean-one normalisation. Where the base
    profile put weight the mirrored run puts the opposite: base carries
    18% of its mass in the noise half / 82% in the clean half (peak at
    lam = ln gamma, steep clean-side decay), the mirror carries 82/18
    (peak at Lam - ln gamma, steep noise-side decay).

    Closed-form normaliser: see _u_ratio_mean (exponent=1 piecewise partial
    fractions in u = e^(lam/2)). gamma/lam_min/lam_max are python floats so
    the normaliser is a scalar; only the ratio itself is duck-typed.
    """
    mean = _u_ratio_mean(gamma, lam_min, lam_max, 1.0)
    return _u_profile_ratio(lam, gamma, lam_min, lam_max, 1.0) / mean


def ucenter_weight(lam, gamma, lam_min, lam_max):
    """Mean-one normalised compensation making the U-SPACE effective profile
    EXACTLY symmetric about the support midpoint.

    The 'wn-ucenter' ablation: multiplying the min-SNR weight by this factor
    replaces the effective u-space profile (identical for every target) by
    the geometric mean of the base profile and its midpoint mirror,

        W_u_center(lam) = sqrt( W_u(lam) * W_u(Lam - lam) ) / E,

    which satisfies W_u_center(lam) == W_u_center(Lam - lam) pointwise:
    the peak sits exactly at the support midpoint and the mass splits
    exactly 50/50 between the noise and clean halves (base: 18/82;
    wn-umirror: 82/18). The log-domain reading is the symmetric part of
    log W_u -- the symmetric profile closest to base in log space.

    Closed-form normaliser: see _u_ratio_mean (exponent=0.5). Same scalar
    normaliser / duck-typed ratio split as umirror_weight.
    """
    mean = _u_ratio_mean(gamma, lam_min, lam_max, 0.5)
    return _u_profile_ratio(lam, gamma, lam_min, lam_max, 0.5) / mean


# ---------------------------------------------------------------------------
# loss-space conversion (ablation): express any target's native error in the
# velocity (u) space, at fixed x_t. JiT's v-loss is the same object (v = -u,
# identical squared loss).
# ---------------------------------------------------------------------------

def u_error_factor(target, lam, lam_min=None, lam_max=None):
    """e_u = factor * e_y, converting the native target-space error to u space.

    Derivation (per pixel, fixed x_t = alpha*data + sigma*noise):
    the target bijection inverts to d(noise) = d(y)/(a - c*sigma/alpha) and
    d(data) = -(sigma/alpha)*d(noise), so with u = noise - data

        d(u) = d(y) * (1 + sigma/alpha)/(a - c*sigma/alpha),  sigma/alpha = exp(-lam/2)

    which depends on lam only (path-independent, like c_eps/c_x0). Squared and
    multiplied by the native MSE it IS the velocity-space loss; for target='x0'
    the squared factor is (1+sqrt(s))^2 — exactly JiT's v-space loss applied to
    an x0-output network.
    """
    a, c = objective_coeffs(target, lam, lam_min, lam_max)
    r = _exp(-0.5 * lam)
    return (1.0 + r) / (a - c * r)


def eps_from_pred(target, pred, x_t, lam, alpha, sigma, lam_min=None, lam_max=None):
    """Invert the target bijection: recover the noise estimate from model output.

    pred = a*noise + c*data and data = (x_t - sigma*noise)/alpha give
    pred = (a - c*sigma/alpha) * noise + (c/alpha) * x_t, hence

        noise_hat = (pred - (c/alpha) * x_t) / (a - c*sigma/alpha)

    The denominator a - c*exp(-lam/2) is strictly positive on the training
    support for eps / u / rotation (theta < pi/2); for x0 it equals
    -exp(-lam/2) (the usual 1/sigma blow-up at the data endpoint, harmless on
    lam <= lam_max). Any cell can therefore feed any sampler.
    """
    a, c = objective_coeffs(target, lam, lam_min, lam_max)
    return (pred - (c / alpha) * x_t) / (a - c * sigma / alpha)


# ---------------------------------------------------------------------------
# conditioning clock (chat record section 2-C): both frameworks feed the SAME
# scalar to the same TimestepEmbedder.
# ---------------------------------------------------------------------------

def cond_input(cond_mode, lam, lam_min, lam_max):
    """Map lam to the value fed into the model's t_embedder.

    "lambda-std" (default): standardized lam (zero mean, unit variance under
        lam ~ U[lam_min, lam_max]); identical distribution on both frameworks.
    "lambda": raw lam.
    """
    if cond_mode == "lambda":
        return lam
    if cond_mode == "lambda-std":
        mean = 0.5 * (lam_min + lam_max)
        std = (lam_max - lam_min) / math.sqrt(12.0)
        return (lam - mean) / std
    raise ValueError(f"unknown cond_mode: {cond_mode}")
