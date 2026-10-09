"""
Unified sampling entry for the 2x3+1 grid (aligned evaluation protocol, chat
record section 2-I).

One deterministic probability-flow ODE integrator (Heun, fixed NFE) in logSNR
space is used for BOTH frameworks; on the VP path this PF-ODE is exactly what
DDIM with eta=0 discretizes, so the framework axis is not confounded by the
solver family. Any cell's prediction target is first converted to epsilon via
the exact bijection inverse (formulas.eps_from_pred), so every target feeds
the same sampler. The model's conditioning input is the standardized lambda of
the current step (same clock as training).

Usage mirrors the base repos' sample_ddp.py: DDP, per-rank png dump, then npz.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from tqdm import tqdm
import os
import shutil
from PIL import Image
import numpy as np
import math
import argparse

from diffusers.models import AutoencoderKL

from exp_common import formulas
from exp_common.model import Exp_models
from exp_common.paths import get_path
from exp_common.objectives import get_objective
from exp_common.lsc import add_lsc, make_skip_table


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    # np.savez writes a zip archive and needs seekable random-access I/O, which
    # some FS mounts reject. Build the archive in the local working directory
    # first, then move it (a plain sequential stream copy) to the target path.
    local_path = os.path.join(os.getcwd(), f".npz_build_{os.getpid()}.npz")
    np.savez(local_path, arr_0=samples)
    shutil.move(local_path, npz_path)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@torch.no_grad()
def pfode_drift(x, lam, model_fn, path, objective, cond_mode, model_kwargs):
    """dx/dlam = d_alpha * x0_hat + d_sigma * eps_hat (probability flow ODE)."""
    alpha, sigma = path.alpha_sigma(lam)
    d_alpha, d_sigma = path.d_alpha_sigma(lam)
    t_cond = objective.cond_input(cond_mode, lam)
    t_batch = torch.full((x.shape[0],), t_cond, device=x.device) if not hasattr(t_cond, "shape") \
        else t_cond.expand(x.shape[0])
    pred = model_fn(x, t_batch, **model_kwargs)
    view = (-1,) + (1,) * (x.dim() - 1)
    eps_hat = objective.eps_from_pred(pred, x, lam, alpha, sigma)
    x0_hat = (x - sigma.view(view) * eps_hat) / alpha.view(view)
    return d_alpha.view(view) * x0_hat + d_sigma.view(view) * eps_hat


@torch.no_grad()
def sample_pfode(model_fn, shape, path, objective, cond_mode, model_kwargs,
                 lam_min, lam_max, num_steps, device, solver="heun", init=None):
    """Heun (or Euler) integration of the PF-ODE from lam_min (noise) to
    lam_max (data). NFE = 2*num_steps for heun, num_steps for euler — identical
    schedule and budget for every cell of the grid. If `init` is given it is
    used as the starting noise (keeps the same RNG-stream behavior as the base
    repos' sample_ddp.py); otherwise it is drawn here."""
    x = torch.randn(*shape, device=device) if init is None else init
    lam_grid = torch.linspace(lam_min, lam_max, num_steps + 1, device=device)
    for i in range(num_steps):
        lam0 = lam_grid[i]
        lam1 = lam_grid[i + 1]
        h = lam1 - lam0
        f0 = pfode_drift(x, lam0, model_fn, path, objective, cond_mode, model_kwargs)
        if solver == "euler":
            x = x + h * f0
        else:
            x_euler = x + h * f0
            f1 = pfode_drift(x_euler, lam1, model_fn, path, objective, cond_mode, model_kwargs)
            x = x + 0.5 * h * (f0 + f1)
    return x


def main(args):
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load checkpoint and recover the cell configuration stored at train time:
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    train_args = checkpoint.get("args")
    get = lambda k, default: getattr(train_args, k, default) if train_args is not None else default

    framework = args.framework or get("framework", None)
    target = args.target or get("target", None)
    auto_min, auto_max = formulas.default_lambda_range()
    lam_min = get("lam_min", None)
    lam_max = get("lam_max", None)
    lam_min = auto_min if lam_min is None else lam_min
    lam_max = auto_max if lam_max is None else lam_max
    # manual overrides win; default: read from checkpoint args. NOTE the LSC
    # adapters are rebuilt BEFORE the strict state_dict load, so an override
    # that disagrees with the training config fails the load (intentionally).
    lsc_type = args.lsc_type or get("lsc_type", "w/o")
    lsc_addon = args.lsc_addon or get("lsc_addon", "v-pred")
    cond_mode = get("cond_mode", "lambda-std")
    # JiT-alignment ablations; loss_space/lam_sampling only affect TRAINING,
    # mirrored here solely so the sample folder name matches the cell name
    data_space = get("data_space", "latent")
    loss_space = get("loss_space", "native")
    lam_sampling = get("lam_sampling", "uniform")
    # vae_scale: the latent scaling factor used at train time (SD's default
    # 0.18215; a custom-dataset VAE may store 0.2204). Explicit --vae-scale
    # wins, then the checkpoint args, then the SD default
    vae_scale = args.vae_scale if args.vae_scale is not None else get("vae_scale", 0.18215)
    # wn (sampling-bump reweighting) likewise only affects TRAINING; resolved
    # here solely so the sample folder name matches the cell name
    wn_mode = get("wn_mode", "off")
    model_name = args.model or get("model", "XL/2")
    # qk_norm: new checkpoints store the flag directly; fall back to the old
    # inverted field (no_qk_norm, default-on era) for earlier checkpoints
    qk_norm = get("qk_norm", None)
    if qk_norm is None:
        qk_norm = not get("no_qk_norm", False)
    # num_classes / label_dropout: the label embedding table has
    # num_classes + (label_dropout > 0) rows, so BOTH must match the training
    # values for the strict state_dict load to succeed. Resolve them from the
    # checkpoint args (manual --num-classes wins when given). At eval time the
    # dropout never fires (model.eval()), only the table size matters.
    num_classes = args.num_classes if args.num_classes is not None \
        else get("num_classes", 1000)
    label_dropout = get("label_dropout", 0.1)
    assert framework and target, "framework/target not found in checkpoint args; pass --framework/--target"

    path = get_path(framework)
    objective = get_objective(target, lam_min, lam_max)

    if data_space == "pixel":
        # JiT-alignment ablation: the model operated on the [-1,1] image at
        # full resolution (in_channels=3, no VAE)
        input_size, in_channels = args.image_size, 3
    else:
        input_size, in_channels = args.image_size // 8, 4
    model = Exp_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=label_dropout,
        qk_norm=qk_norm,
    )
    # rebuild the LSC structure recorded in the checkpoint BEFORE loading
    # weights (adapter shapes must exist for load_state_dict to succeed)
    if lsc_type != 'w/o':
        skip_table = make_skip_table(len(model.blocks), mode=lsc_type)
        add_lsc(model, model.hidden_size, layer_count=len(model.blocks),
                lsc_type=lsc_addon, skip_table=skip_table)
    state_dict = checkpoint["ema"] if args.use_ema and "ema" in checkpoint else checkpoint["model"]
    # the REPA projector only exists for the training loss; sampling never
    # consumes it, so drop those keys before the strict load
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("repa_projector.")}
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    vae = None
    if data_space == "latent":
        vae = AutoencoderKL.from_pretrained(args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0
    using_cfg = args.cfg_scale > 1.0

    # cell naming mirrors train_exp.py exactly (ablation suffixes included)
    cell_name = f"{framework}-{target}"
    if model_name != "XL/2":
        # architecture variants get their own folders ('/' stripped)
        cell_name += "-" + model_name.replace("/", "")
    if data_space != "latent":
        cell_name += f"-{data_space}"
    if vae_scale != 0.18215:
        # mirror train_exp.py's suffix so the sample folder matches the cell
        cell_name += f"-vs{vae_scale:g}"
    if loss_space != "native":
        cell_name += "-vloss"
    if lam_sampling != "uniform":
        cell_name += "-lnt"
    if wn_mode != "off":
        cell_name += f"-wn-{wn_mode}"
    if lsc_type != 'w/o':
        cell_name += f"-lsc{'' if lsc_type == 'w/' else '-skewed'}-{lsc_addon}"
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"{cell_name}-{ckpt_string_name}-cfg-{args.cfg_scale}-" \
                  f"{args.solver}-{args.num_sampling_steps}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    for _ in pbar:
        z = torch.randn(n, model.in_channels, input_size, input_size, device=device)
        y = torch.randint(0, num_classes, (n,), device=device)

        if using_cfg:
            z_in = torch.cat([z, z], 0)
            y_null = torch.tensor([num_classes] * n, device=device)
            y_in = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            z_in = z
            model_kwargs = dict(y=y)
            model_fn = model.forward

        # pass the pre-drawn z directly as the initial noise (exact same
        # RNG-stream behavior as the base repos' sample_ddp.py)
        samples = sample_pfode(
            model_fn, z_in.shape, path, objective, cond_mode, model_kwargs,
            lam_min, lam_max, args.num_sampling_steps, device, solver=args.solver,
            init=z_in,
        )
        del z_in

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        if vae is not None:
            samples = vae.decode(samples / vae_scale).sample
        # pixel mode: samples already live in [-1,1]; the [0,255] clamp below
        # performs the same [-1,1] truncation the latent path implicitly gets
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--framework", type=str, default=None, choices=[None, "vp", "fm"],
                        help="default: read from checkpoint args")
    parser.add_argument("--target", type=str, default=None,
                        choices=[None, "eps", "x0", "u", "rotation",
                                 "a15", "a30", "a45", "a60", "a75", "a80", "a90"])
    parser.add_argument("--model", type=str, default=None, choices=[None] + list(Exp_models.keys()))
    parser.add_argument("--lsc-type", type=str, default=None,
                        choices=[None, "w/o", "w/", "skewed"],
                        help="default: read from checkpoint args; must match the training "
                             "value (the LSC structure is rebuilt before the state_dict load)")
    parser.add_argument("--lsc-addon", type=str, default=None, choices=[None, "basic", "v-pred"],
                        help="default: read from checkpoint args; must match the training value")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--vae-scale", type=float, default=None,
                        help="latent scaling factor for vae.decode; default: read from "
                             "checkpoint args (fallback 0.18215 = SD's default; pass "
                             "0.2204 for the custom-dataset VAE); must match training")
    parser.add_argument("--sample-dir", type=str, default="samples_exp")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=None,
                        help="default: read from checkpoint args (fallback 1000); must "
                             "match the training value, the CFG null token sits AT index "
                             "num_classes so passing num_classes+1 breaks the embedding load")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"])
    parser.add_argument("--num-sampling-steps", type=int, default=50,
                        help="PF-ODE steps in lambda space; NFE = 2x for heun")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    args = parser.parse_args()
    main(args)
