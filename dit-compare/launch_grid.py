"""
Single entry point that launches the whole 2x3+1 grid, one cell at a time.

The seven cells (chat record, final experiment plan):

    framework axis (path geometry)   target axis (output semantics)
    ------------------------------   ------------------------------
    vp  (continuous DDPM path)       eps, x0, u
    fm  (linear / flow-matching)     eps, x0, u
    fm  (+1 extra experiment)        rotation  (uniform angular-velocity sweep)

Every cell goes through the SAME train_exp.py with the SAME flattening config
(lam support, min-SNR gamma, conditioning clock, optimizer hyperparameters,
global seed), so all cells share: init weights, batch order (imagefolder
backend), lambda/noise/dropout draws -> common random pairing (section 2-H).

Examples
--------
Dry-run (print the 7 commands, launch nothing):
    python launch_grid.py --data-path /data/imagenet/train --dry-run

Launch the full grid on 8 GPUs:
    python launch_grid.py --data-path /data/imagenet/train --nproc-per-node 8

Launch only the velocity cells:
    python launch_grid.py --data-path /data/imagenet/train --only vp-u,fm-u

Launch the angle-refined target cells (finer target axis, 15-degree grid):
    python launch_grid.py --data-path /data/imagenet/train --only vp-a15,fm-a30,...

Any extra argument after `--` is forwarded verbatim to every train_exp.py call,
e.g.:
    python launch_grid.py --data-path /data/imagenet/train -- --train-steps 100000 --ckpt-every 20000

When forwarding ablation knobs (--wn-mode / --data-space / ...), also pass
--tag so the run folders get a suffix and do not overwrite the base cells:
    python launch_grid.py --data-path /data/imagenet/train --tag -wn-mid -- --wn-mode mid
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# the 2x3+1 grid
# ---------------------------------------------------------------------------

CELLS_BASE = [
    # name          framework  target
    ("vp-eps",       "vp",      "eps"),
    ("vp-x0",        "vp",      "x0"),
    ("vp-u",         "vp",      "u"),
    ("fm-eps",       "fm",      "eps"),
    ("fm-x0",        "fm",      "x0"),
    ("fm-u",         "fm",      "u"),
    ("fm-rotsweep",  "fm",      "rotation"),  # the +1 extra experiment
]

# angle-interpolated targets (the refined target axis): a15/a30 sit between
# eps and u, a60/a75 between u and x0, mirror-symmetric about u
CELLS_REFINED = [
    ("vp-a15",       "vp",      "a15"),
    ("vp-a30",       "vp",      "a30"),
    ("vp-a60",       "vp",      "a60"),
    ("vp-a75",       "vp",      "a75"),
    ("fm-a15",       "fm",      "a15"),
    ("fm-a30",       "fm",      "a30"),
    ("fm-a60",       "fm",      "a60"),
    ("fm-a75",       "fm",      "a75"),
]

CELLS = CELLS_BASE + CELLS_REFINED

CELL_NAMES = [c[0] for c in CELLS]


def build_command(args, name, framework, target):
    """Assemble the torchrun command for one cell. All flattening knobs are
    taken from the shared CLI namespace, so every cell sees the same values."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "train_exp.py",
        "--framework", framework,
        "--target", target,
        # --tag appends a suffix (e.g. "-wn-mid") so ablation reruns land in
        # their own folders instead of overwriting the base cell's; non-default
        # architectures likewise get their own run names ('/' stripped, same
        # rule as train_exp.py's automatic cell_name suffix). A non-default
        # --vae-scale gets the same treatment here (train_exp.py skips its
        # automatic suffixes whenever --run-name is given, so the launcher
        # must extend the name itself)
        "--run-name", name
        + ("-" + args.model.replace("/", "") if args.model != "XL/2" else "")
        + (f"-vs{args.vae_scale:g}" if args.vae_scale is not None else "")
        + args.tag,
        # flattening knobs (shared across cells)
        "--min-snr-gamma", str(args.min_snr_gamma),
        "--cond-mode", args.cond_mode,
        # aligned optimization hyperparameters
        "--model", args.model,
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--global-batch-size", str(args.global_batch_size),
        "--label-dropout", str(args.label_dropout),
        "--grad-clip", str(args.grad_clip),
        "--epochs", str(args.epochs),
        # data
        "--data-path", args.data_path,
        "--data-backend", args.data_backend,
        "--image-size", str(args.image_size),
        "--num-classes", str(args.num_classes),
        # logging / checkpoints / pairing
        "--results-dir", args.results_dir,
        "--global-seed", str(args.global_seed),
        "--log-every", str(args.log_every),
        "--ckpt-every", str(args.ckpt_every),
    ]
    # lam support: omitted -> train_exp.py computes it from the DDPM chain
    # endpoints (full generative path); pass explicitly only to override.
    if args.lam_min is not None:
        cmd.extend(["--lam-min", str(args.lam_min)])
    if args.lam_max is not None:
        cmd.extend(["--lam-max", str(args.lam_max)])
    if args.qk_norm:
        cmd.append("--qk-norm")
    if args.no_bf16:
        cmd.append("--no-bf16")
    if args.vae_path:
        cmd.extend(["--vae-path", args.vae_path])
    if args.vae_scale is not None:
        cmd.extend(["--vae-scale", str(args.vae_scale)])
    if args.train_steps:
        cmd.extend(["--train-steps", str(args.train_steps)])
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def main(args):
    # default (no --only) runs the base 2x3+1 grid, unchanged from before;
    # the refined angle cells are reached explicitly via --only
    selected = [c for c in CELLS if c[0] in args.only] if args.only else CELLS_BASE
    unknown = set(args.only or []) - set(CELL_NAMES)
    assert not unknown, f"unknown cell names in --only: {sorted(unknown)} (valid: {CELL_NAMES})"
    assert selected, "no cells selected"

    print("=" * 78)
    print(f"2x3+1 grid launcher: {len(selected)} cell(s) queued")
    lam_disp = (f"[{args.lam_min}, {args.lam_max}]" if args.lam_min is not None
                else "auto (DDPM-chain endpoints, full generative path)")
    print(f"flattening config: lam={lam_disp} "
          f"min_snr_gamma={args.min_snr_gamma} cond_mode={args.cond_mode} "
          f"model={args.model} qk_norm={args.qk_norm} "
          f"vae_scale={args.vae_scale if args.vae_scale is not None else 0.18215}")
    print(f"aligned hyper: lr={args.lr} (constant) warmup={args.warmup_steps} "
          f"global_batch={args.global_batch_size} label_dropout={args.label_dropout} "
          f"grad_clip={args.grad_clip} bf16={not args.no_bf16} "
          f"global_seed={args.global_seed}")
    print("=" * 78)

    results = {}
    for i, (name, framework, target) in enumerate(selected):
        cmd = build_command(args, name, framework, target)
        cmd_str = shlex.join(cmd)
        print(f"\n[{i + 1}/{len(selected)}] cell {name}  (framework={framework}, target={target})")
        print(f"  $ {cmd_str}")
        if args.dry_run:
            results[name] = "dry-run"
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        hours = (time.time() - t0) / 3600.0
        if proc.returncode == 0:
            results[name] = f"OK ({hours:.2f}h)"
            print(f"[{i + 1}/{len(selected)}] cell {name} finished in {hours:.2f}h")
        else:
            results[name] = f"FAILED (exit {proc.returncode})"
            print(f"[{i + 1}/{len(selected)}] cell {name} FAILED with exit code {proc.returncode}")
            if not args.continue_on_error:
                break

    print("\n" + "=" * 78)
    print("grid summary:")
    for name, status in results.items():
        print(f"  {name:14s} {status}")
    print("=" * 78)
    if any(str(s).startswith("FAILED") for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the 2x3+1 DiT-vs-SiT comparison grid from one entry point.",
        epilog="arguments after `--` are forwarded verbatim to train_exp.py",
    )
    # cell selection / execution
    parser.add_argument("--only", type=str, default=None,
                        help=f"comma-separated subset of cells (default: the base 2x3+1 "
                             f"grid): {','.join(CELL_NAMES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without launching anything")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep launching remaining cells even if one fails")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix appended to every --run-name (e.g. '-wn-mid'); use it "
                             "when forwarding ablation knobs so reruns do not overwrite the "
                             "base cells' folders")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29500)
    # shared flattening knobs (must stay identical across cells; mirror train_exp.py defaults)
    parser.add_argument("--lam-min", type=float, default=None,
                        help="default: computed from the DDPM chain endpoints (full "
                             "generative path); set explicitly only for ablations")
    parser.add_argument("--lam-max", type=float, default=None,
                        help="default: computed from the DDPM chain endpoints (full "
                             "generative path); set explicitly only for ablations")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--cond-mode", type=str, default="lambda-std",
                        choices=["lambda-std", "lambda"])
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true")
    # shared aligned hyperparameters
    parser.add_argument("--model", type=str, default="XL/2")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--label-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--train-steps", type=int, default=None,
                        help="optional step cap forwarded to every cell")
    parser.add_argument("--no-bf16", dest="no_bf16", action="store_true")
    # shared data config
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-backend", type=str, default="imagefolder",
                        choices=["imagefolder", "nas"])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--vae-scale", type=float, default=None,
                        help="latent scaling factor forwarded to every train_exp.py call "
                             "(default: omit, train_exp.py uses its 0.18215 SD default; "
                             "pass 0.2204 for the custom-dataset VAE); non-default values "
                             "also append a -vs suffix to every run-name")
    # shared logging / pairing
    parser.add_argument("--results-dir", type=str, default="results_grid")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="KEEP IDENTICAL across cells: it is what pairs init "
                             "weights, batch order and noise draws (section 2-H)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="everything after `--` is forwarded to train_exp.py")
    args = parser.parse_args()

    if args.only:
        args.only = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    main(args)
