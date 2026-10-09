"""
Numerical self-check for the 2x3+1 alignment framework (pure stdlib).

This script calls the SAME implementation the GPU training/sampling code uses
(exp_common/formulas.py, duck-typed) with python floats, and verifies every
flattening claim of the experiment plan:

  1. path identities        : VP alpha^2+sigma^2=1, FM alpha+sigma=1, and both
                              paths see the SAME SNR=exp(lam) at the same lam
  2. path derivatives       : d_alpha/d_lam, d_sigma/d_lam vs finite differences
                              (these drive the unified PF-ODE sampler)
  3. theta schedule         : rotation target has truly uniform angular velocity
  4. objective coefficients : eps/x0/u/rotation (a,c) values
  5. conversion factors     : c_eps = c_x0 / s (epsilon vs x0 space relation)
  6. min-SNR closed forms   : w(lam) matches the per-target closed forms from
                              the chat record's final reply:
                                eps      min(s,g)/s
                                x0       min(s,g)
                                u        min(s,g)/(1+sqrt(s))^2
                                rotation min(s,g)/(sqrt(s)*cos(t)+sin(t))^2
  7. W(lam) coincidence     : effective weight profile W(lam) = w(lam)*c_x0(lam)
                              is EXACTLY min(s,g) for all 7 cells, pointwise
  8. eps_from_pred          : exact inversion of every target's bijection on
                              both paths (this is what lets one sampler serve
                              all cells)
  9. conditioning clock     : standardized lam has mean 0 and |value|=sqrt(3)
                              at both support endpoints, identical for both
                              frameworks by construction
 10. weighting disabled     : gamma=None returns the multiplicative identity
 11. lambda support         : default_lambda_range() reproduces the DDPM chain
                              endpoints and covers the full generative path
                              (the old fixed [-6, 6] was a representation-
                              learning probe range, not a generation range)
 12. LSC skip tables        : make_skip_table reproduces the reference repos'
                              explicit dense / skewed tables and stays sane
                              on smaller backbones
 13. u_error_factor         : target->velocity error conversion equals the exact
                              Jacobian du/dy at fixed x_t (finite differences),
                              and for x0 reproduces JiT's v-loss weight (1+sqrt(s))^2
 14. angle targets          : a15..a90 == (cos, -sin) at theta degrees from
                              the noise axis: unit-norm, mirror-symmetric about u
                              (a15<->a75, a30<->a60), a uniform 15-degree grid
                              with eps (0 deg), u (45 deg), x0 (90 deg, up to sign);
                              a45 == u/||u|| and a90 == -x0 close the unit-norm grid;
                              a80 densifies the x0 end at a 5-degree step (a10 absent)
 15. sampling bump          : the wn ablation weight is mean-one under lam~U (so
                              the overall loss scale is preserved), peaks at mu,
                              and its clean/noisy presets are mirror images about
                              the support midpoint
 16. breaking-residual      : 'XL/2-brk' cuts the 28-block stream into 4 even
                              7-block segments (breaking layers after blocks
                              6/13/20); indivisible depths raise
 17. wn-flat compensation  : the flat wn mode divides out the aligned min-SNR
                              profile min(s, gamma) (mean-one normalised), so
                              the effective weight profile w*c_x0 is exactly
                              CONSTANT in lam == the plain x0-space MSE
 18. wn-mirror compensation: the mirror wn mode swaps high/low weights: it
                              divides by min(s, gamma) and multiplies by its
                              left-right mirror W(Lam-lam) about the support
                              midpoint (mean-one normalised), so the effective
                              weight profile is EXACTLY the mirrored profile
 19. wn-umirror           : the umirror wn mode mirrors the U-SPACE effective
                              profile W_u = min(s, gamma)/(1+sqrt(s))^2 (which
                              is identical for every target) about the support
                              midpoint: closed-form normaliser exact, effective
                              profile == W_u(Lam-lam)/E, mass split 82/18
 20. wn-ucenter           : the ucenter wn mode makes that profile EXACTLY
                              midpoint-symmetric (geometric mean of the profile
                              and its mirror): closed-form normaliser exact,
                              pointwise symmetry, 50/50 mass split
Run:  python check_alignment.py
Exit code is 0 iff every check passes (suitable as a pre-flight gate).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_common import formulas as F
from exp_common.skip_table import make_skip_table, break_points

LAM_MIN, LAM_MAX = F.default_lambda_range()  # computed, not hard-coded (check 11)
GAMMA = 5.0
EXACT_TOL = 1e-12
FD_TOL = 1e-6

# the 2x3+1 grid: (cell name, framework, target)
CELLS = [
    ("vp-eps", "vp", "eps"),
    ("vp-x0", "vp", "x0"),
    ("vp-u", "vp", "u"),
    ("fm-eps", "fm", "eps"),
    ("fm-x0", "fm", "x0"),
    ("fm-u", "fm", "u"),
    ("fm-rotsweep", "fm", "rotation"),
]

# refined target axis (check 14): angle-interpolated targets (aNN, N degrees
# from the noise axis; a80 included, its mirror partner a10 is not)
ANGLE_TARGETS = tuple(F.ANGLE_TARGETS_DEG)


def lam_grid(n=241):
    h = (LAM_MAX - LAM_MIN) / (n - 1)
    return [LAM_MIN + i * h for i in range(n)]


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, name, max_err, tol):
        ok = max_err <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:58s} max_err={max_err:.3e} (tol={tol:.0e})")
        if not ok:
            self.failures += 1
        return ok


def main():
    c = Checker()
    grid = lam_grid()

    print("1. path identities (shared SNR, VP sum-of-squares, FM sum-to-one)")
    err_vp = err_fm = err_snr = 0.0
    for lam in grid:
        a_vp, s_vp = F.path_alpha_sigma("vp", lam)
        a_fm, s_fm = F.path_alpha_sigma("fm", lam)
        err_vp = max(err_vp, abs(a_vp**2 + s_vp**2 - 1.0))
        err_fm = max(err_fm, abs(a_fm + s_fm - 1.0))
        # relative error: exp(lam) reaches ~1e4 at the data end of the computed
        # support, absolute tolerances no longer apply there
        err_snr = max(err_snr, abs((a_vp / s_vp) ** 2 - math.exp(lam)) / math.exp(lam))
        err_snr = max(err_snr, abs((a_fm / s_fm) ** 2 - math.exp(lam)) / math.exp(lam))
    c.check("VP: alpha^2 + sigma^2 == 1", err_vp, EXACT_TOL)
    c.check("FM: alpha + sigma == 1", err_fm, EXACT_TOL)
    c.check("both paths: (alpha/sigma)^2 == exp(lam)", err_snr, EXACT_TOL)

    print("2. path derivatives vs central finite differences")
    h = 1e-6
    for path_type in ("vp", "fm"):
        err = 0.0
        for lam in grid:
            da, ds = F.path_d_alpha_sigma(path_type, lam)
            a_p, s_p = F.path_alpha_sigma(path_type, lam + h)
            a_m, s_m = F.path_alpha_sigma(path_type, lam - h)
            err = max(err, abs(da - (a_p - a_m) / (2 * h)))
            err = max(err, abs(ds - (s_p - s_m) / (2 * h)))
        c.check(f"{path_type}: d(alpha)/dlam, d(sigma)/dlam analytic", err, FD_TOL)

    print("3. rotation theta schedule: uniform angular velocity")
    t_lo = F.theta_schedule(LAM_MIN, LAM_MIN, LAM_MAX)
    t_hi = F.theta_schedule(LAM_MAX, LAM_MIN, LAM_MAX)
    t_mid = F.theta_schedule(0.5 * (LAM_MIN + LAM_MAX), LAM_MIN, LAM_MAX)
    c.check("theta(lam_min) == pi/2", abs(t_lo - math.pi / 2), EXACT_TOL)
    c.check("theta(lam_max) == 0", abs(t_hi), EXACT_TOL)
    c.check("theta linear in lam (midpoint)", abs(t_mid - math.pi / 4), EXACT_TOL)
    # constant angular velocity: equal lam increments -> equal theta increments
    err = 0.0
    step = (LAM_MAX - LAM_MIN) / 50.0
    for i in range(50):
        t0 = F.theta_schedule(LAM_MIN + i * step, LAM_MIN, LAM_MAX)
        t1 = F.theta_schedule(LAM_MIN + (i + 1) * step, LAM_MIN, LAM_MAX)
        err = max(err, abs((t0 - t1) - (math.pi / 2) / 50.0))
    c.check("dtheta/dlam constant (uniform angular velocity)", err, EXACT_TOL)

    print("4. objective coefficients (a, c) for y = a*noise + c*data")
    c.check("eps == (1, 0)", max(abs(F.objective_coeffs("eps", 0.0)[0] - 1.0),
                               abs(F.objective_coeffs("eps", 0.0)[1])), EXACT_TOL)
    c.check("x0 == (0, 1)", max(abs(F.objective_coeffs("x0", 0.0)[0]),
                                abs(F.objective_coeffs("x0", 0.0)[1] - 1.0)), EXACT_TOL)
    c.check("u == (1, -1)", max(abs(F.objective_coeffs("u", 0.0)[0] - 1.0),
                                abs(F.objective_coeffs("u", 0.0)[1] + 1.0)), EXACT_TOL)
    lam_mid = 0.5 * (LAM_MIN + LAM_MAX)  # theta(lam_mid) = pi/4 by construction
    a_r, c_r = F.objective_coeffs("rotation", lam_mid, LAM_MIN, LAM_MAX)
    c.check("rotation @ lam_mid == (sqrt2/2, -sqrt2/2)",
            max(abs(a_r - math.sqrt(0.5)), abs(c_r + math.sqrt(0.5))), EXACT_TOL)

    print("5. conversion-factor relation c_eps == c_x0 / s")
    err = 0.0
    for lam in grid:
        for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
            a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
            ref = F.c_x0_factor(a, cc, lam) / math.exp(lam)
            err = max(err, abs(F.c_eps_factor(a, cc, lam) - ref) / ref)
    c.check("c_eps * s == c_x0 for every target (relative)", err, EXACT_TOL)

    print("6. min-SNR weight: implementation matches the per-target closed forms")
    def closed_form(target, lam):
        s = math.exp(lam)
        m = min(s, GAMMA)
        if target == "eps":
            return m / s
        if target == "x0":
            return m
        if target == "u":
            return m / (1.0 + math.sqrt(s)) ** 2
        if target == "rotation":
            th = F.theta_schedule(lam, LAM_MIN, LAM_MAX)
            return m / (math.sqrt(s) * math.cos(th) + math.sin(th)) ** 2
        if target in F.ANGLE_TARGETS_DEG:  # same closed form, constant theta
            th = math.radians(F.ANGLE_TARGETS_DEG[target])
            return m / (math.sqrt(s) * math.cos(th) + math.sin(th)) ** 2
        raise ValueError(target)

    for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
        err = max(abs(F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
                      - closed_form(target, lam)) for lam in grid)
        c.check(f"w_{target}(lam) == closed form", err, EXACT_TOL)

    print("7. W(lam) = w(lam) * c_x0(lam) coincides at min(s,g) for ALL 7 cells")
    for name, framework, target in CELLS:
        err = 0.0
        for lam in grid:
            a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
            w = F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
            w_eff = w * F.c_x0_factor(a, cc, lam)
            err = max(err, abs(w_eff - min(math.exp(lam), GAMMA)))
        c.check(f"cell {name}: W(lam) == min(s, {GAMMA:g})", err, EXACT_TOL)

    print("8. eps_from_pred inverts every target's bijection on both paths")
    data, noise = 0.7315, -1.2047  # arbitrary fixed scalars
    for framework in ("vp", "fm"):
        for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
            err = 0.0
            for lam in grid:
                alpha, sigma = F.path_alpha_sigma(framework, lam)
                x_t = alpha * data + sigma * noise
                a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
                pred = a * noise + cc * data
                noise_hat = F.eps_from_pred(target, pred, x_t, lam, alpha, sigma,
                                            LAM_MIN, LAM_MAX)
                err = max(err, abs(noise_hat - noise))
            c.check(f"{framework}/{target}: noise recovery exact", err, 1e-9)

    print("9. conditioning clock: standardized lam, identical on both frameworks")
    conds = [F.cond_input("lambda-std", lam, LAM_MIN, LAM_MAX) for lam in lam_grid(2401)]
    mean = sum(conds) / len(conds)
    c.check("lambda-std: mean == 0", abs(mean), EXACT_TOL)
    c.check("lambda-std: value at endpoints == +/-sqrt(3)",
            max(abs(F.cond_input("lambda-std", LAM_MIN, LAM_MIN, LAM_MAX) + math.sqrt(3)),
                abs(F.cond_input("lambda-std", LAM_MAX, LAM_MIN, LAM_MAX) - math.sqrt(3))),
            EXACT_TOL)
    var = sum((v - mean) ** 2 for v in conds) / len(conds)
    c.check("lambda-std: variance ~= 1 under lam~U", abs(var - 1.0), 2e-3)

    print("10. weighting disabled (gamma=None) is the identity")
    err = max(abs(F.min_snr_weight(t, lam, None, LAM_MIN, LAM_MAX) - 1.0)
              for t in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS for lam in grid)
    c.check("w(lam) == 1 when gamma=None", err, EXACT_TOL)

    print("11. lambda support: computed range covers the full generative path")
    # data endpoint: t=1 of the DDPM chain, abar_1 = 1 - beta_1 exactly
    c.check("lam_max == log((1-beta_1)/beta_1) (DDPM t=1)",
            abs(LAM_MAX - math.log((1.0 - 1e-4) / 1e-4)), EXACT_TOL)
    # noise endpoint: t=T; independently rebuild abar_T from the beta schedule
    log_abar_T = sum(math.log(1.0 - (1e-4 + (0.02 - 1e-4) * i / 999.0))
                     for i in range(1000))
    abar_T = math.exp(log_abar_T)
    c.check("sigmoid(lam_min) == abar_T (VP alpha^2 lands on DDPM t=T)",
            abs(F.sigmoid(LAM_MIN) - abar_T), EXACT_TOL)
    c.check("abar_T ~= 4e-5 (sanity: literature value of the DDPM endpoint)",
            max(0.0, abar_T - 1e-4) + max(0.0, 1e-5 - abar_T), 0.0)
    # coverage: at lam_min the VP interpolant is (1-abar_T) pure noise,
    # at lam_max it carries exactly 1-beta_1 signal — the whole path is trained
    a_lo, s_lo = F.path_alpha_sigma("vp", LAM_MIN)
    a_hi, s_hi = F.path_alpha_sigma("vp", LAM_MAX)
    c.check("VP at lam_min: sigma^2 == 1 - abar_T (pure-noise end covered)",
            abs(s_lo**2 - (1.0 - abar_T)), EXACT_TOL)
    c.check("VP at lam_max: alpha^2 == 1 - beta_1 (data end covered)",
            abs(a_hi**2 - (1.0 - 1e-4)), EXACT_TOL)
    # FM sees the identical SNR coverage at the same lam
    a_fm, s_fm = F.path_alpha_sigma("fm", LAM_MIN)
    c.check("FM at lam_min: same SNR as VP (shared coverage)",
            abs((a_fm / s_fm) ** 2 - math.exp(LAM_MIN)), EXACT_TOL)

    print("12. LSC skip tables match the reference repos' explicit tables")
    dense_ref = {i: 27 - i for i in range(13)}  # both repos' 'w/ LSC dense' table
    dense_gen = make_skip_table(28, "w/")
    c.check("dense@28 == explicit table in DiT-v/SiT train_nas.py",
            0.0 if dense_gen["down"] == dense_ref and
            dense_gen["up"] == {u: d for d, u in dense_ref.items()} else 1.0, 0.0)
    skew_ref = {i: 33 - i for i in range(6, 16)}  # SiT's 'skewed LSC' table
    skew_gen = make_skip_table(28, "skewed")
    c.check("skewed@28 == SiT train_nas.py explicit table",
            0.0 if skew_gen["down"] == skew_ref and
            skew_gen["up"] == {u: d for d, u in skew_ref.items()} else 1.0, 0.0)
    # smaller backbones stay well-formed (up strictly after down, no overlap)
    def table_sane(n, mode):
        t = make_skip_table(n, mode)
        if set(t["down"]) & set(t["up"]):
            return False
        if len(t["down"]) != len(t["up"]):
            return False
        return all(d < u for d, u in t["down"].items())
    c.check("dense@24 (L/2) has 11 pairs, sane",
            0.0 if len(make_skip_table(24, "w/")["down"]) == 11 and
            table_sane(24, "w/") else 1.0, 0.0)
    c.check("dense@12 (B/2, S/2) has 5 pairs, sane",
            0.0 if len(make_skip_table(12, "w/")["down"]) == 5 and
            table_sane(12, "w/") else 1.0, 0.0)
    c.check("skewed@12 clamped to 3 pairs, sane",
            0.0 if len(make_skip_table(12, "skewed")["down"]) == 3 and
            table_sane(12, "skewed") else 1.0, 0.0)

    print("13. u_error_factor: exact target->velocity Jacobian (JiT v-loss ablation)")
    data, noise = 0.7315, -1.2047  # arbitrary fixed scalars
    err = 0.0
    h = 1e-7
    for framework in ("vp", "fm"):
        for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
            for lam in grid:
                alpha, sigma = F.path_alpha_sigma(framework, lam)
                a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
                # perturb at fixed x_t: d_data = +h, d_noise = -(alpha/sigma)*h
                y0 = a * noise + cc * data
                y1 = a * (noise - (alpha / sigma) * h) + cc * (data + h)
                u0 = noise - data
                u1 = (noise - (alpha / sigma) * h) - (data + h)
                jac = (u1 - u0) / (y1 - y0)
                ref = F.u_error_factor(target, lam, LAM_MIN, LAM_MAX)
                err = max(err, abs(jac - ref) / abs(ref))
    c.check("u_error_factor == du/dy at fixed x_t (all targets, both paths)", err, 1e-5)
    err = max(abs(F.u_error_factor("x0", lam) ** 2 - (1.0 + math.sqrt(math.exp(lam))) ** 2)
              / (1.0 + math.sqrt(math.exp(lam))) ** 2 for lam in grid)
    c.check("x0: u-space weight == (1+sqrt(s))^2 (JiT v-loss on an x0 head)", err, EXACT_TOL)
    c.check("u target: u_error_factor == 1 (native == velocity space)",
            max(abs(F.u_error_factor("u", lam, LAM_MIN, LAM_MAX) - 1.0) for lam in grid),
            EXACT_TOL)

    print("14. angle-interpolated targets a15..a90 (refined target axis)")
    # coefficients: y = cos(theta)*noise - sin(theta)*data, theta measured FROM
    # THE NOISE AXIS in degrees
    err = 0.0
    for name, deg in F.ANGLE_TARGETS_DEG.items():
        a, cc = F.objective_coeffs(name, 0.0)
        err = max(err, abs(a - math.cos(math.radians(deg))),
                  abs(cc + math.sin(math.radians(deg))))
    c.check("aN == (cos(N deg), -sin(N deg))", err, EXACT_TOL)
    # unit norm: the 1/sqrt(2) scale vs u is absorbed exactly by every
    # downstream factor (min-SNR, u_error_factor, c_eps are all (a,c)-homogeneous)
    err = max(abs(a * a + cc * cc - 1.0)
              for name in ANGLE_TARGETS
              for a, cc in [F.objective_coeffs(name, 0.0)])
    c.check("unit norm: a^2 + c^2 == 1", err, EXACT_TOL)
    # grid anchors: eps = 0 deg exactly, u/||u|| = 45 deg, x0 = 90 deg up to a
    # global sign (flipping the target's sign is the same regression task)
    a_u, c_u = F.objective_coeffs("u", 0.0)
    n_u = math.hypot(a_u, c_u)
    a_x, c_x = F.objective_coeffs("x0", 0.0)
    err = max(abs(a_u / n_u - math.cos(math.radians(45.0))),
              abs(c_u / n_u + math.sin(math.radians(45.0))),
              abs(a_x + math.cos(math.radians(90.0))),
              abs(c_x - math.sin(math.radians(90.0))))
    c.check("eps/u/x0 anchor the 0/45/90-degree grid points", err, EXACT_TOL)
    # the new members close the unit-norm grid at the legacy anchors:
    # a45 is exactly the normalized u, a90 is exactly -x0
    a45, c45 = F.objective_coeffs("a45", 0.0)
    a90, c90 = F.objective_coeffs("a90", 0.0)
    err = max(abs(a45 - a_u / n_u), abs(c45 - c_u / n_u),
              abs(a90 + a_x), abs(c90 + c_x))
    c.check("a45 == u/||u|| and a90 == -x0 (unit-norm grid closure)", err, EXACT_TOL)
    # mirror symmetry about u: theta and 90-theta swap the noise/data content,
    # i.e. (a, c)_{90-theta} == (-c, -a)_theta; a80 has no partner (a10 absent)
    err = 0.0
    for lo, hi in (("a15", "a75"), ("a30", "a60")):
        a_lo, c_lo = F.objective_coeffs(lo, 0.0)
        a_hi, c_hi = F.objective_coeffs(hi, 0.0)
        err = max(err, abs(a_hi + c_lo), abs(c_hi + a_lo))
    c.check("mirror about u: a15<->a75 and a30<->a60 (noise/data swapped)", err, EXACT_TOL)
    # the refined cells flatten identically: W(lam) = w(lam)*c_x0 == min(s, gamma)
    err = 0.0
    for lam in grid:
        for name in ANGLE_TARGETS:
            a, cc = F.objective_coeffs(name, lam, LAM_MIN, LAM_MAX)
            w = F.min_snr_weight(name, lam, GAMMA, LAM_MIN, LAM_MAX)
            err = max(err, abs(w * F.c_x0_factor(a, cc, lam) - min(math.exp(lam), GAMMA)))
    c.check("W(lam) == min(s, gamma) for the 7 angle cells as well", err, EXACT_TOL)

    print("15. sampling-bump weight: mean-one Gaussian on top of min-SNR (wn ablation)")
    bump_sigma = 1.6  # train_exp.py --wn-sigma default
    mid = 0.5 * (LAM_MIN + LAM_MAX)
    quarter = 0.25 * (LAM_MAX - LAM_MIN)
    presets = {"clean": mid + quarter, "midclean": mid + 0.5 * quarter, "mid": mid,
               "midnoisy": mid - 0.5 * quarter, "noisy": mid - quarter}  # train_exp.py
    fine = lam_grid(40001)
    h_fine = (LAM_MAX - LAM_MIN) / (len(fine) - 1)

    def bump_mean_and_vals(mu, sig):
        vals = [F.sampling_bump_weight(lam, mu, sig, LAM_MIN, LAM_MAX) for lam in fine]
        trapz = (0.5 * vals[0] + sum(vals[1:-1]) + 0.5 * vals[-1]) * h_fine
        return trapz / (LAM_MAX - LAM_MIN), vals

    for mode, mu in presets.items():
        mean_u, vals = bump_mean_and_vals(mu, bump_sigma)
        c.check(f"wn-{mode}: E_U[bump] == 1 (overall loss scale preserved)",
                abs(mean_u - 1.0), 1e-6)
        peak_lam = fine[max(range(len(fine)), key=lambda i: vals[i])]
        c.check(f"wn-{mode}: the bump peaks at mu", abs(peak_lam - mu), 2.0 * h_fine)
    for sig in (0.8, 3.2):  # mean-one holds for any bandwidth, not just the default
        mean_u, _ = bump_mean_and_vals(presets["mid"], sig)
        c.check(f"wn-mid (sigma={sig}): E_U[bump] == 1", abs(mean_u - 1.0), 1e-6)
    # wn-mid is symmetric about mu; clean/noisy are mirror images about the
    # support midpoint (the truncated-normaliser is symmetric there as well)
    err = max(abs(F.sampling_bump_weight(mid + d, presets["mid"], bump_sigma, LAM_MIN, LAM_MAX)
                  - F.sampling_bump_weight(mid - d, presets["mid"], bump_sigma, LAM_MIN, LAM_MAX))
              for d in (0.5, 1.7, 3.3, 6.1))
    c.check("wn-mid: bump(mu + d) == bump(mu - d)", err, EXACT_TOL)
    err = max(abs(F.sampling_bump_weight(mid + d, presets["clean"], bump_sigma, LAM_MIN, LAM_MAX)
                  - F.sampling_bump_weight(mid - d, presets["noisy"], bump_sigma, LAM_MIN, LAM_MAX))
              for d in (0.0, 0.8, 2.4, 4.0))
    c.check("wn-clean/wn-noisy are mirror images about the support midpoint", err, EXACT_TOL)
    # the mid* presets are the exact midpoints between mid and the clean/noisy
    # presets (the full-noisy bump proved too strong in practice)
    err = max(abs(presets["midclean"] - 0.5 * (presets["mid"] + presets["clean"])),
              abs(presets["midnoisy"] - 0.5 * (presets["mid"] + presets["noisy"])))
    c.check("wn-midclean/wn-midnoisy sit halfway between mid and clean/noisy", err, EXACT_TOL)
    # midclean and midnoisy are mirror images about the support midpoint, just
    # like clean/noisy (the truncated normaliser is symmetric there as well)
    err = max(abs(F.sampling_bump_weight(mid + d, presets["midclean"], bump_sigma, LAM_MIN, LAM_MAX)
                  - F.sampling_bump_weight(mid - d, presets["midnoisy"], bump_sigma, LAM_MIN, LAM_MAX))
              for d in (0.0, 0.8, 2.4, 4.0))
    c.check("wn-midclean/wn-midnoisy are mirror images about the support midpoint", err, EXACT_TOL)

    print("16. breaking-residual split points (the 'XL/2-brk' architecture)")
    pts = break_points(28, 4)
    c.check("XL/2-brk: cuts after blocks 6/13/20",
            0.0 if pts == [6, 13, 20] else 1.0, 0.0)
    # every segment has exactly 7 blocks: [0..6], [7..13], [14..20], [21..27]
    segs, prev = [], -1
    for p in pts + [27]:
        segs.append(p - prev)
        prev = p
    c.check("the 4 segments are even (7 blocks each)",
            0.0 if segs == [7, 7, 7, 7] else 1.0, 0.0)
    c.check("segments None/<=1 mean no cut",
            0.0 if break_points(28, None) == []
            and break_points(28, 1) == [] else 1.0, 0.0)
    try:
        break_points(28, 3)
        raised = 1.0
    except AssertionError:
        raised = 0.0
    c.check("an indivisible depth raises", raised, 0.0)

    print("17. wn-flat compensation: cancels the aligned min-SNR profile min(s, gamma)")
    # the wn-flat weight is 1/min(s, gamma) normalised to mean one under lam~U
    flat_vals = [F.flat_uniform_weight(lam, GAMMA, LAM_MIN, LAM_MAX) for lam in fine]
    trapz = (0.5 * flat_vals[0] + sum(flat_vals[1:-1]) + 0.5 * flat_vals[-1]) * h_fine
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    # the closed-form normaliser inside flat_uniform_weight must agree with
    # the brute-force numerical integral of the returned weight
    c.check("wn-flat: E_U[compensation] == 1 (closed-form normaliser exact)",
            abs(mean_u - 1.0), 1e-6)
    # it cancels the profile exactly: min(s, gamma) * compensation == const
    lam_star = math.log(GAMMA)
    closed_mean = ((math.exp(-LAM_MIN) - 1.0 / GAMMA)
                   + (LAM_MAX - lam_star) / GAMMA) / (LAM_MAX - LAM_MIN)
    err = max(abs(min(math.exp(lam), GAMMA) * v - 1.0 / closed_mean)
              for lam, v in zip(fine, flat_vals))
    c.check("wn-flat: min(s, gamma) * compensation is CONSTANT in lam", err, EXACT_TOL)
    # end-to-end: w_total * c_x0 == the same constant for every target, i.e.
    # the wn-flat run trains the plain, unweighted x0-space MSE
    err = 0.0
    for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
        for lam in grid:
            a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
            w_tot = (F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
                     * F.flat_uniform_weight(lam, GAMMA, LAM_MIN, LAM_MAX))
            err = max(err, abs(w_tot * F.c_x0_factor(a, cc, lam) - 1.0 / closed_mean))
    c.check("wn-flat: w_total * c_x0 == const for all targets (plain x0 MSE)",
            err, EXACT_TOL)

    print("18. wn-mirror compensation: mirrors min(s, gamma) about the support midpoint")
    # the wn-mirror weight is W(Lam-lam)/W(lam) normalised to mean one under lam~U
    Lam = LAM_MIN + LAM_MAX
    p = math.log(GAMMA)
    q = Lam - p
    closed_E = ((GAMMA * (math.exp(-LAM_MIN) - math.exp(-q))
                 + 0.5 * (math.exp(2.0 * p - Lam) - math.exp(Lam - 2.0 * p))
                 + (math.exp(Lam - p) - math.exp(LAM_MIN)) / GAMMA)
                / (LAM_MAX - LAM_MIN))
    mirror_vals = [F.mirror_minsnr_weight(lam, GAMMA, LAM_MIN, LAM_MAX) for lam in fine]
    trapz = (0.5 * mirror_vals[0] + sum(mirror_vals[1:-1]) + 0.5 * mirror_vals[-1]) * h_fine
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    # the closed-form normaliser must agree with the brute-force integral
    c.check("wn-mirror: E_U[compensation] == 1 (closed-form normaliser exact)",
            abs(mean_u - 1.0), 1e-6)
    # it replaces the profile by its mirror: W(lam)*comp == W(Lam-lam)/E
    err = max(abs(min(math.exp(lam), GAMMA) * v
                  - min(math.exp(Lam - lam), GAMMA) / closed_E)
              for lam, v in zip(fine, mirror_vals))
    c.check("wn-mirror: W(lam)*comp == W(Lam-lam)/E (mirrored profile)", err, EXACT_TOL)
    # end-to-end: w_total * c_x0 == the mirrored profile for every target
    err = 0.0
    for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
        for lam in grid:
            a, cc = F.objective_coeffs(target, lam, LAM_MIN, LAM_MAX)
            w_tot = (F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
                     * F.mirror_minsnr_weight(lam, GAMMA, LAM_MIN, LAM_MAX))
            err = max(err, abs(w_tot * F.c_x0_factor(a, cc, lam)
                               - min(math.exp(Lam - lam), GAMMA) / closed_E))
    c.check("wn-mirror: w_total * c_x0 == mirrored profile for all targets",
            err, EXACT_TOL)

    print("19. wn-umirror compensation: mirrors the u-space profile about the midpoint")
    # W_u(lam) = min(s, gamma)/(1+sqrt(s))^2 is the target-independent u-space
    # effective profile; wn-umirror multiplies min-SNR by [W_u(Lam-lam)/W_u(lam)]
    # normalised to mean one under lam~U. The closed-form normaliser integrates
    # piecewise partial fractions in u = e^(lam/2) (three segments split at the
    # two min() kinks) -- the brute-force integral catches any antiderivative slip.
    Lam = LAM_MIN + LAM_MAX

    def wu(lam):
        s = math.exp(lam)
        return min(s, GAMMA) / (1.0 + math.sqrt(s)) ** 2
    ratio_vals = [wu(Lam - lam) / wu(lam) for lam in fine]
    trapz = (0.5 * ratio_vals[0] + sum(ratio_vals[1:-1]) + 0.5 * ratio_vals[-1]) * h_fine
    closed_E = F._u_ratio_mean(GAMMA, LAM_MIN, LAM_MAX, 1.0)
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    c.check("wn-umirror: E_U[W_u(Lam-lam)/W_u(lam)] == closed form",
            abs(mean_u - closed_E) / closed_E, 1e-6)
    umirror_vals = [F.umirror_weight(lam, GAMMA, LAM_MIN, LAM_MAX) for lam in fine]
    trapz = (0.5 * umirror_vals[0] + sum(umirror_vals[1:-1]) + 0.5 * umirror_vals[-1]) * h_fine
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    c.check("wn-umirror: E_U[compensation] == 1 (closed-form normaliser exact)",
            abs(mean_u - 1.0), 1e-6)
    # it replaces the u-space profile by its mirror: W_u(lam)*comp == W_u(Lam-lam)/E
    err = max(abs(wu(lam) * v - wu(Lam - lam) / closed_E)
              for lam, v in zip(fine, umirror_vals))
    c.check("wn-umirror: W_u(lam)*comp == W_u(Lam-lam)/E (mirrored profile)",
            err, EXACT_TOL)
    # end-to-end: w_total / c_u == the mirrored u-profile for every target
    # (w_total/c_u = min(s,gamma)*comp/(c_x0*c_u) by the check-17 identity)
    err = 0.0
    for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
        for lam in grid:
            w_tot = (F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
                     * F.umirror_weight(lam, GAMMA, LAM_MIN, LAM_MAX))
            err = max(err, abs(w_tot / F.u_error_factor(target, lam, LAM_MIN, LAM_MAX) ** 2
                               - wu(Lam - lam) / closed_E))
    c.check("wn-umirror: w_total / c_u == mirrored u-profile for all targets",
            err, EXACT_TOL)

    print("20. wn-ucenter compensation: makes the u-space profile midpoint-symmetric")
    # wn-ucenter multiplies min-SNR by sqrt[W_u(Lam-lam)/W_u(lam)] (mean-one
    # normalised): the effective profile becomes the geometric mean of the base
    # profile and its mirror, pointwise symmetric about the support midpoint.
    ratio_vals = [math.sqrt(wu(Lam - lam) / wu(lam)) for lam in fine]
    trapz = (0.5 * ratio_vals[0] + sum(ratio_vals[1:-1]) + 0.5 * ratio_vals[-1]) * h_fine
    closed_E = F._u_ratio_mean(GAMMA, LAM_MIN, LAM_MAX, 0.5)
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    c.check("wn-ucenter: E_U[sqrt(W_u(Lam-lam)/W_u(lam))] == closed form",
            abs(mean_u - closed_E) / closed_E, 1e-6)
    ucenter_vals = [F.ucenter_weight(lam, GAMMA, LAM_MIN, LAM_MAX) for lam in fine]
    trapz = (0.5 * ucenter_vals[0] + sum(ucenter_vals[1:-1]) + 0.5 * ucenter_vals[-1]) * h_fine
    mean_u = trapz / (LAM_MAX - LAM_MIN)
    c.check("wn-ucenter: E_U[compensation] == 1 (closed-form normaliser exact)",
            abs(mean_u - 1.0), 1e-6)
    # effective profile == geometric mean of the profile and its mirror / E
    err = max(abs(wu(lam) * v - math.sqrt(wu(lam) * wu(Lam - lam)) / closed_E)
              for lam, v in zip(fine, ucenter_vals))
    c.check("wn-ucenter: W_u(lam)*comp == sqrt(W_u(lam)*W_u(Lam-lam))/E",
            err, EXACT_TOL)
    # pointwise midpoint symmetry of the effective profile (fine is a uniform
    # symmetric grid, so the mirror of fine[i] is fine[::-1][i])
    err = max(abs(wu(lam) * v - wu(Lam - lam) * v_m)
              for lam, v, v_m in zip(fine, ucenter_vals, ucenter_vals[::-1]))
    c.check("wn-ucenter: effective profile is pointwise midpoint-symmetric",
            err, EXACT_TOL)
    # end-to-end: w_total / c_u == the symmetric u-profile for every target
    err = 0.0
    for target in ("eps", "x0", "u", "rotation") + ANGLE_TARGETS:
        for lam in grid:
            w_tot = (F.min_snr_weight(target, lam, GAMMA, LAM_MIN, LAM_MAX)
                     * F.ucenter_weight(lam, GAMMA, LAM_MIN, LAM_MAX))
            err = max(err, abs(w_tot / F.u_error_factor(target, lam, LAM_MIN, LAM_MAX) ** 2
                               - math.sqrt(wu(lam) * wu(Lam - lam)) / closed_E))
    c.check("wn-ucenter: w_total / c_u == symmetric u-profile for all targets",
            err, EXACT_TOL)

    print("=" * 78)
    if c.failures == 0:
        print("ALL CHECKS PASSED: W(lam) coincides across the 2x3+1 grid, "
              "min-SNR closed forms are exact, sampler inversion is exact.")
        return 0
    print(f"{c.failures} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
