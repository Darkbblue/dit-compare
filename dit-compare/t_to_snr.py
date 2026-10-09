#!/usr/bin/env python3
"""t -> SNR conversion for the dit_compare.py feature-extraction protocol.

Pure stdlib (no torch): mirrors the mapping inside DiTFeaturePipeline.__call__
(for_feature/dit_compare.py), where a DDPM-style extraction timestep t
(0 = data, 1000 = pure noise) is converted to the protocol's implied logSNR:

    u   = (1000 - t) / 1000 + 1e-6
    u_c = clip(u, 1e-6, 1 - 1e-6)
    lam = 2 * log(u_c / (1 - u_c))          # logSNR
    SNR = exp(lam) = (u_c / (1 - u_c))^2

Both vp and fm pipelines share this t -> lam mapping; the paths differ only in
the noising curve applied AT that lam, so one table serves both. Keep this in
sync with dit_compare.py if the protocol ever changes.

Usage:
    python t_to_snr.py                      # 11-point table over [0, 1000]
    python t_to_snr.py 100 500 900          # query specific timesteps
    python t_to_snr.py --range 0 1000 --steps 21
    python t_to_snr.py --plot t_to_snr.png  # also save the t->SNR curve
"""

import argparse
import math


def t_to_logsnr(t):
    """Exact t -> logSNR mapping of DiTFeaturePipeline.__call__."""
    u = (1000.0 - t) / 1000.0 + 1e-6
    u_c = min(max(u, 1e-6), 1.0 - 1e-6)
    return 2.0 * math.log(u_c / (1.0 - u_c))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("t", type=float, nargs="*",
                        help="DDPM-style timesteps in [0, 1000] (0=data, 1000=noise)")
    parser.add_argument("--range", type=float, nargs=2, default=(0.0, 1000.0),
                        metavar=("T0", "T1"), help="scan range when no t is given")
    parser.add_argument("--steps", type=int, default=11,
                        help="number of scan points (with --range)")
    parser.add_argument("--plot", metavar="PNG", default=None,
                        help="save a matplotlib t->SNR curve to this path")
    args = parser.parse_args()

    if args.t:
        ts = args.t
    else:
        lo, hi = args.range
        n = max(args.steps - 1, 1)
        ts = [lo + i * (hi - lo) / n for i in range(args.steps)]

    print(f"{'t':>10} {'u':>12} {'logSNR':>10} {'SNR':>14} {'SNR(dB)':>10}")
    print("-" * 60)
    for t in ts:
        u = (1000.0 - t) / 1000.0 + 1e-6
        lam = t_to_logsnr(t)
        snr = math.exp(lam)
        snr_db = 10.0 * math.log10(snr)
        print(f"{t:>10.2f} {u:>12.6f} {lam:>10.4f} {snr:>14.6g} {snr_db:>10.2f}")

    if args.plot:
        plot_curve(args.plot)


def training_lambda_range(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """DDPM-chain logSNR endpoints (mirrors formulas.default_lambda_range):
    the training support [lam_min, lam_max] ~ [-10.12, +9.21], NOT symmetric.
    """
    log_abar = 0.0
    lam_max = None
    for i in range(num_timesteps):
        beta = beta_start + (beta_end - beta_start) * i / (num_timesteps - 1)
        log_abar += math.log(1.0 - beta)
        if i == 0:
            lam_max = log_abar - math.log1p(-math.exp(log_abar))
    lam_min = log_abar - math.log1p(-math.exp(log_abar))
    return lam_min, lam_max


def plot_curve(png, lo=0.0, hi=1000.0, n=2001):
    """Save the t -> SNR curve (log-scale y axis) over the full t range,
    with the asymmetric training-support interval overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
    snrs = [math.exp(t_to_logsnr(t)) for t in ts]

    # training support: lam in [lam_min, lam_max] (asymmetric, midpoint -0.454)
    lam_min, lam_max = training_lambda_range()

    def t_of(lam):  # inverse of lam = 2*logit(u)
        u = 1.0 / (1.0 + math.exp(-0.5 * lam))
        return 1000.0 * (1.0 - u + 1e-6)

    t_data, t_noise = t_of(lam_max), t_of(lam_min)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(ts, snrs, lw=2.0, color="C0",
            label="protocol map  $\\mathrm{SNR}=e^{\\lambda}$,  $\\lambda=2\\cdot$logit$(u)$")
    ax.set_yscale("log")
    ax.axvspan(t_data, t_noise, color="C1", alpha=0.15,
               label="training support  $\\lambda\\in$[%.2f, +%.2f]"
                     "  (midpoint %.2f)" % (lam_min, lam_max, 0.5 * (lam_min + lam_max)))
    for lam, t_end in ((lam_max, t_data), (lam_min, t_noise)):
        y = math.exp(lam)
        ax.scatter([t_end], [y], s=28, color="C1", zorder=5)
        dx = 40 if t_end < 500 else -235
        ax.annotate("t=%.1f: $\\lambda$=%+.2f" % (t_end, lam),
                    xy=(t_end, y), xytext=(t_end + dx, y * 6),
                    fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.6))
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(500.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.annotate("t=500: SNR=1",
                xy=(500.0, 1.0), xytext=(520.0, 3e-2), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.6))
    ax.set_xlabel("DDPM-style timestep t   (0 = data, 1000 = pure noise)")
    ax.set_ylabel("SNR = exp(lam)   (log scale)")
    ax.set_title("dit-compare feature-extraction protocol: t -> SNR\n"
                 "the map is symmetric about t=500 ($\\lambda$=0), but the\n"
                 "training support visited by the model is NOT symmetric")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"saved curve plot -> {png}")


if __name__ == "__main__":
    main()
