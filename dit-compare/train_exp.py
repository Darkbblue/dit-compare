"""
Unified training entry for the 2x3+1 DiT-vs-SiT comparison.

One script trains every cell of the grid; the cell is selected by
`--framework {vp,fm}` and `--target {eps,x0,u,rotation}`. All flattening
decisions from the experiment plan live in exp_common/ and are shared:

  factor 1  uniform logSNR sampling on the shared support [lam_min, lam_max],
            continuous paths (no t-quantization on the VP side);
  factor 3  exact per-target min-SNR weight w(lam) = min(s,g)/(a*sqrt(s)-c)^2,
            a pure function of lam, identical for both frameworks;
  velocity  u = noise - data in BOTH frameworks (DDPM and FM columns);
            rotation target (uniform angular-velocity sweep) only on FM (+1);
  clock     both frameworks feed standardized lam to the same TimestepEmbedder;
  backbone  one shared UnifiedDiT (no learn_sigma head anywhere);
  hyper     lr 1e-4 constant, Adam(0.9, 0.999), wd 0, shared warmup,
            EMA 0.9999, batch 256, label dropout 0.1, grad clip;
  pairing   single script + shared seed -> identical init weights, batch order
            (imagefolder backend), lambda/noise/dropout draws across cells.

Diagnostics (chat record section 2-J): per-lambda binned native / eps-space /
weighted MSE and effective weight are logged every --log-every steps to
log.txt and diagnostics.jsonl in the experiment directory.

Optional REPA mode (--repa-depth N): adds the reference DINOv2 representation-
alignment loss on the block-N output (projector in-model, frozen DINOv2 from a
local HF directory or official .pth via --dinov2-ckpt); off by default. The
aligned layer may also follow a PROGRESSIVE SCHEDULE: --repa-depth N1 N2 ...
with --repa-depth-steps 0 S2 ... activates depth Ni once train_steps >= Si
(the shared projector weights simply carry over at each switch); without
--repa-depth-steps a single depth is fixed for the whole run (legacy mode).

Ablation knobs (JiT-alignment; off by default, each one breaks exactly ONE of
the flattening factors that separate the grid from JiT's training regime):
  --data-space pixel   : skip the VAE entirely; the model operates on the [-1,1]
                         image directly (in_channels=3, input_size=image_size).
  --loss-space u       : UNWEIGHTED MSE in velocity space (u = noise - data;
                         JiT's v-loss is the identical loss since v = -u).
                         Implemented as the exact target->u Jacobian squared
                         multiplying the native MSE; min-SNR is ignored.
  --lam-sampling logit-normal --p-mean -0.8 --p-std 0.8
                     : JiT's t = sigmoid(z), z ~ N(p_mean, p_std^2), i.e.
                         lam = 2z, replacing uniform-in-logSNR (factor 1).
                         lam_min/lam_max still normalize the conditioning clock.
  --wn-mode {clean,midclean,mid,midnoisy,noisy,flat,mirror,umirror,ucenter}
                     : extra Gaussian bump w(lam) on top of the min-SNR (or
                         v-loss) weight, centred at the clean / mid-clean /
                         middle / mid-noisy / noisy region of the lambda
                         support (positions mid+R/4, mid+R/8, mid, mid-R/8,
                         mid-R/4; the mid* presets sit halfway between mid and
                         the two quartile presets, mirror images about the
                         support midpoint — the full-noisy bump proved too
                         extreme). Explicit weighting IS the equivalent
                         sampling weighting (importance identity); the bump is
                         normalised to mean 1 under the uniform prior so the
                         total loss scale is unchanged. 'flat' is no bump but
                         the exact INVERSE of the aligned min-SNR profile
                         min(s, gamma) (mean-one normalised): the effective
                         weight profile W(lam) = p*w*c_x0 becomes CONSTANT,
                         i.e. plain x0-space MSE training (requires
                         --loss-space native and --min-snr-gamma > 0;
                         --wn-mu/--wn-sigma do not apply). 'mirror' instead
                         swaps high and low weights: it multiplies by the
                         mean-one normalised ratio of the mirrored profile
                         W(Lam-lam) to the original one, so the effective
                         profile becomes the LEFT-RIGHT MIRROR of min(s, gamma)
                         about the support midpoint (monotone decreasing).
                         'umirror'/'ucenter' act on the U-SPACE effective
                         profile W_u(lam) = min(s, gamma)/(1+sqrt(s))^2
                         (identical for every target by the c_x0*c_u
                         identity): umirror multiplies by the mean-one
                         normalised ratio W_u(Lam-lam)/W_u(lam), turning the
                         base mass split 18/82 (noise/clean halves) into
                         82/18; ucenter multiplies by its square root,
                         replacing the profile by the geometric mean with the
                         mirror, EXACTLY symmetric about the midpoint (peak
                         at the midpoint, 50/50 split). All three require
                         --loss-space native and --min-snr-gamma > 0;
                         --wn-mu/--wn-sigma do not apply to them).

Finer target axis (angle interpolation): --target also accepts a15/a30/a45/a60/
a75/a80/a90, fixed unit-norm directions y = cos(theta)*noise - sin(theta)*data
with theta in degrees from the noise axis; eps=0, u=45, x0=90 complete a
uniform 15-degree grid, mirror-symmetric about u (a15<->a75, a30<->a60); a80
densifies the x0 end at a 5-degree step (its partner a10 is not implemented).
a45 == u/||u|| and a90 == -x0 (unit-norm closure); a0 is omitted, being
identical to eps. All conversion factors and the sampler inversion apply
unchanged (they are homogeneous in (a, c)).
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import json
import logging
import os

from diffusers.models import AutoencoderKL

from exp_common import formulas
from exp_common.model import Exp_models
from exp_common.paths import get_path
from exp_common.objectives import get_objective
from exp_common.diagnostics import LambdaBinLogger
from exp_common.lsc import add_lsc, make_skip_table
from exp_common.repa import (attach_repa, load_dinov2, dinov2_preprocess,
                             repa_proj_loss, parse_repa_schedule, repa_depth_at)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    # fused foreach ops: 2 kernel launches total instead of 2 per parameter
    # (ema is a deepcopy, so parameters() iterate in the identical order)
    ema_params = list(ema_model.parameters())
    model_params = list(model.parameters())
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, model_params, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_state_maybe_repa(module, state, label, repa_enabled, logger):
    """Load a (possibly REPA-augmented) state_dict, tolerating a projector
    mismatch: resuming a REPA run from a plain 2x3+1 checkpoint keeps the
    freshly initialized projector; loading a REPA checkpoint with REPA off
    drops the projector keys. Any other key mismatch is an error."""
    has_repa_keys = any(k.startswith("repa_projector.") for k in state)
    if has_repa_keys and not repa_enabled:
        state = {k: v for k, v in state.items() if not k.startswith("repa_projector.")}
        logger.info(f"{label}: checkpoint has a REPA projector but REPA is off; dropped those keys")
    incompatible = module.load_state_dict(state, strict=False)
    missing_ok = all(k.startswith("repa_projector.") for k in incompatible.missing_keys)
    assert missing_ok and len(incompatible.unexpected_keys) == 0, \
        f"{label} state mismatch: missing={incompatible.missing_keys[:4]} " \
        f"unexpected={incompatible.unexpected_keys[:4]}"
    if incompatible.missing_keys:
        logger.info(f"{label}: REPA projector not in checkpoint; keeping the fresh init")


def get_lr_scheduler(optimizer, warmup_steps):
    """Linear warmup to base lr, then constant (shared across all cells)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


#################################################################################
#                                  Data backends                                #
#################################################################################

def build_loader(args, rank, transform):
    if args.data_backend == "imagefolder":
        # Exact cross-cell pairing: DistributedSampler with a fixed seed yields
        # the identical batch order in every cell (same world size).
        dataset = ImageFolder(args.data_path, transform=transform)
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=rank,
            shuffle=True, seed=args.global_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=8,
        )
        return loader, sampler, len(dataset)

    raise ValueError(f"unknown data backend: {args.data_backend}")


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if args.target == "rotation":
        assert args.framework == "fm", \
            "the rotation (angular-sweep) target is the +1 extra experiment and runs on the FM framework only"
    if args.min_snr_gamma == 0:
        args.min_snr_gamma = None
    # velocity-space loss (JiT's v-loss) is by definition UNWEIGHTED; min-SNR is
    # a native-target-space weight and does not apply there
    minsnr_ignored = args.loss_space != "native" and args.min_snr_gamma is not None
    if minsnr_ignored:
        args.min_snr_gamma = None

    # REPA is switched ON by --repa-depth (the alignment layer(s), 1-based) and
    # needs manually specified local DINOv2 weights (no auto-download):
    # an HF-format directory (config.json + model.safetensors) or the official
    # .pth plus a local dinov2 repo checkout. A multi-value --repa-depth is a
    # progressive schedule: --repa-depth-steps aligns 1:1 and gives each
    # depth's activation step (first entry must be 0); without it the run
    # falls back to the legacy fixed single layer.
    repa_enabled = args.repa_depth is not None
    repa_depths, repa_steps = None, None
    if repa_enabled:
        assert args.dinov2_ckpt, \
            "--repa-depth requires --dinov2-ckpt (HF directory or official .pth)"
        assert os.path.isdir(args.dinov2_ckpt) or args.dinov2_repo, \
            "official .pth weights also require --dinov2-repo (local dinov2 checkout)"
        repa_depths, repa_steps = parse_repa_schedule(args.repa_depth,
                                                      args.repa_depth_steps)

    # logSNR support: default is COMPUTED from the DDPM chain endpoints so the
    # training distribution covers the full generative sampling path (the old
    # fixed [-6, 6] support only targets representation-learning probes).
    if args.lam_min is None or args.lam_max is None:
        auto_min, auto_max = formulas.default_lambda_range()
        if args.lam_min is None:
            args.lam_min = auto_min
        if args.lam_max is None:
            args.lam_max = auto_max

    # weighting-bump ablation: resolve the bump centre (quartile presets are
    # mirror-symmetric about the support midpoint; midclean/midnoisy are the
    # midpoints between mid and the clean/noisy presets, i.e. mid +/- R/8)
    wn_mu = None
    if args.wn_mode in ("flat", "mirror", "umirror", "ucenter"):
        # all divide out the aligned min-SNR profile min(s, gamma) (flat: 1/min;
        # mirror: mirrored/min; umirror/ucenter: mirrored/centered u-space
        # profile W_u), so they are only defined when it is active
        if args.loss_space != "native" or args.min_snr_gamma is None:
            raise ValueError(f"--wn-mode {args.wn_mode} modifies the aligned min-SNR "
                             "profile min(s, gamma) and therefore requires "
                             "--loss-space native with a positive --min-snr-gamma")
    elif args.wn_mode != "off":
        mid = 0.5 * (args.lam_min + args.lam_max)
        quarter = 0.25 * (args.lam_max - args.lam_min)
        preset = {"clean": mid + quarter, "midclean": mid + 0.5 * quarter,
                  "mid": mid, "midnoisy": mid - 0.5 * quarter,
                  "noisy": mid - quarter}
        wn_mu = args.wn_mu if args.wn_mu is not None else preset[args.wn_mode]

    # Setup DDP. Same seed formula as both base repos; combined with the single
    # shared script this gives common random pairing across all cells.
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    cell_name = args.run_name or f"{args.framework}-{args.target}"
    if args.run_name is None:
        # ablation suffixes keep follow-up runs in separate experiment folders
        if args.model != "XL/2":
            # architecture variants get their own folders ('/' stripped so the
            # suffix never becomes a nested directory)
            cell_name += "-" + args.model.replace("/", "")
        if args.data_space != "latent":
            cell_name += f"-{args.data_space}"
        if args.vae_scale != 0.18215:
            # non-default latent scaling factor (e.g. a custom-dataset VAE)
            cell_name += f"-vs{args.vae_scale:g}"
        if args.loss_space != "native":
            cell_name += "-vloss"
        if args.lam_sampling != "uniform":
            cell_name += "-lnt"
        if args.wn_mode != "off":
            cell_name += f"-wn-{args.wn_mode}"
        if args.lsc_type != 'w/o':
            cell_name += f"-lsc{'' if args.lsc_type == 'w/' else '-skewed'}-{args.lsc_addon}"
        if repa_enabled:
            # single depth keeps the old "-repa8" suffix; a progressive
            # schedule concatenates: "-repa8-16"
            cell_name += "-repa" + "-".join(str(d) for d in repa_depths)

    # Setup an experiment folder (only rank 0 writes logs/checkpoints/diagnostics):
    checkpoint_path = None
    checkpoint_dir = None
    diag_path = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.resume:
            if os.path.isdir(args.resume):
                checkpoint_dir = args.resume if args.resume.endswith("checkpoints") \
                    else os.path.join(args.resume, "checkpoints")
                experiment_dir = os.path.dirname(checkpoint_dir)
                ckpts = sorted(glob(os.path.join(checkpoint_dir, "*.pt")))
                checkpoint_path = ckpts[-1] if ckpts else None
            else:
                checkpoint_path = args.resume
                checkpoint_dir = os.path.dirname(checkpoint_path)
                experiment_dir = os.path.dirname(checkpoint_dir)
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Could not find checkpoint to resume from: {args.resume}")
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming training from {checkpoint_path}")
        else:
            experiment_dir = os.path.join(args.results_dir, cell_name)
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
            with open(f"{experiment_dir}/config.json", "w") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)
        diag_path = os.path.join(experiment_dir, "diagnostics.jsonl")
    else:
        logger = create_logger(None)

    paths_to_bcast = [checkpoint_path]
    dist.broadcast_object_list(paths_to_bcast, src=0)
    checkpoint_path = paths_to_bcast[0]

    # Flattening objects:
    path = get_path(args.framework)
    objective = get_objective(args.target, args.lam_min, args.lam_max)

    # Create the shared backbone (identical architecture & parameter count everywhere):
    if args.data_space == "pixel":
        # JiT-style ablation: no VAE, the model operates on the [-1,1] image
        input_size, in_channels = args.image_size, 3
    else:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        input_size, in_channels = args.image_size // 8, 4
    model = Exp_models[args.model](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.label_dropout,
        qk_norm=args.qk_norm,
    )
    # LSC (follow-up experiments; w/o for the 2x3+1 grid). Attached BEFORE the
    # EMA deepcopy so EMA tracks the adapter parameters too
    if args.lsc_type != 'w/o':
        skip_table = make_skip_table(len(model.blocks), mode=args.lsc_type)
        add_lsc(model, model.hidden_size, layer_count=len(model.blocks),
                lsc_type=args.lsc_addon, skip_table=skip_table)
    # REPA (optional; --repa-depth is the on/off switch). The frozen DINOv2
    # encoder loads from local paths; the projector attaches BEFORE the EMA
    # deepcopy so EMA tracks it too (REPA's projectors are in-model as well).
    repa_encoder = None
    if repa_enabled:
        repa_encoder = load_dinov2(args.dinov2_ckpt, repo_dir=args.dinov2_repo,
                                   arch=args.dinov2_arch,
                                   image_size=args.image_size, device=device)
        args.repa_z_dim = repa_encoder.embed_dim  # persisted for ckpt consumers
        attach_repa(model, repa_depths, z_dim=args.repa_z_dim,
                    projector_dim=args.repa_projector_dim)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    vae = None
    if args.data_space == "latent":
        vae = AutoencoderKL.from_pretrained(args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Cell={cell_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Aligned optimizer hyperparameters (chat record section 2-G):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
    )
    scheduler = get_lr_scheduler(opt, args.warmup_steps)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    loader, sampler, dataset_len = build_loader(args, rank, transform)
    logger.info(f"Dataset contains {dataset_len:,} images ({args.data_path}, backend={args.data_backend})")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    prev_repa_depth = None  # last logged active alignment layer (schedule switch log)
    running_loss = 0.0
    running_grad_norm = 0.0
    running_proj = torch.zeros((), device=device)  # tensor accumulation: no per-step GPU sync

    use_bf16 = args.bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if not use_bf16 else None
    autocast_kwargs = dict(device_type='cuda', dtype=amp_dtype)
    logger.info(f"Using AMP dtype: {amp_dtype}")

    bin_logger = LambdaBinLogger(args.lam_min, args.lam_max, num_bins=args.num_lam_bins, device=device)

    # Resume from checkpoint if specified:
    resume_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_state_maybe_repa(model.module, checkpoint["model"], "model", repa_enabled, logger)
        load_state_maybe_repa(ema, checkpoint["ema"], "ema", repa_enabled, logger)
        train_steps = checkpoint.get("train_steps", 0)
        resume_epoch = checkpoint.get("epoch", 0)
        cur_n = sum(len(g["params"]) for g in opt.param_groups)
        ckpt_n = sum(len(g["params"]) for g in checkpoint["opt"]["param_groups"])
        if ckpt_n == cur_n:
            opt.load_state_dict(checkpoint["opt"])
            if "scaler" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                scheduler.last_epoch = train_steps
        else:
            # e.g. resuming a plain 2x3+1 checkpoint with REPA enabled: the
            # optimizer state cannot absorb the new projector, so keep the
            # fresh optimizer and skip warmup (lr enters the constant phase).
            scheduler.last_epoch = train_steps
            logger.warning(
                f"optimizer state has {ckpt_n} params vs current {cur_n} "
                "(REPA projector added/removed); starting with FRESH optimizer state")
        del checkpoint
        torch.cuda.empty_cache()
        logger.info(f"Resumed from epoch {resume_epoch}, train step {train_steps}")

    if wn_mu is not None:
        wn_desc = f"mu={wn_mu},sigma={args.wn_sigma}"
    elif args.wn_mode == "flat":
        wn_desc = "compensation=1/min(s,gamma),mean-one"
    elif args.wn_mode == "mirror":
        wn_desc = "compensation=W(Lam-lam)/W(lam),mean-one"
    elif args.wn_mode == "umirror":
        wn_desc = "compensation=W_u(Lam-lam)/W_u(lam),mean-one"
    elif args.wn_mode == "ucenter":
        wn_desc = "compensation=sqrt(W_u(Lam-lam)/W_u(lam)),mean-one"
    else:
        wn_desc = "off"
    logger.info(
        f"Flattening config: framework={args.framework} target={args.target} "
        f"lam=[{args.lam_min:.4f}, {args.lam_max:.4f}] min_snr_gamma={args.min_snr_gamma} "
        f"cond_mode={args.cond_mode} lr={args.lr} warmup={args.warmup_steps} "
        f"ema=0.9999 batch={args.global_batch_size} label_dropout={args.label_dropout} "
        f"grad_clip={args.grad_clip} qk_norm={args.qk_norm} "
        f"data_space={args.data_space} loss_space={args.loss_space} "
        f"vae_scale={args.vae_scale:g} "
        f"lam_sampling={args.lam_sampling}(p_mean={args.p_mean},p_std={args.p_std}) "
        f"wn_mode={args.wn_mode}({wn_desc}) "
        f"lsc_type={args.lsc_type} lsc_addon={args.lsc_addon} "
        f"repa_depth={args.repa_depth} repa_depth_steps={args.repa_depth_steps} "
        f"repa_coeff={args.repa_coeff}"
    )
    if minsnr_ignored:
        logger.warning("--loss-space u trains the UNWEIGHTED velocity-space MSE "
                       "(JiT's v-loss); --min-snr-gamma was ignored")
    logger.info(f"Training for {args.epochs} epochs (max {args.train_steps or -1} steps)...")

    start_time = time()
    done = False
    for epoch in range(resume_epoch, args.epochs):
        if done:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            with torch.autocast(**autocast_kwargs):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.no_grad():
                    # REPA target features come from the RAW image (before the
                    # VAE latent overwrites x); frozen encoder, no_grad.
                    if repa_encoder is not None:
                        z = repa_encoder.forward_features(
                            dinov2_preprocess(x, args.image_size))['x_norm_patchtokens']
                    if vae is not None:
                        x = vae.encode(x).latent_dist.sample().mul_(args.vae_scale)
                    # pixel mode: x stays the [-1, 1] image

                B = x.shape[0]
                # factor 1: uniform-in-logSNR sampling on the shared support
                # (JiT-style logit-normal when --lam-sampling=logit-normal)
                lam = path.sample_lambda(B, args.lam_min, args.lam_max, device=device,
                                         mode=args.lam_sampling,
                                         p_mean=args.p_mean, p_std=args.p_std)
                noise = torch.randn_like(x)
                alpha, sigma = path.alpha_sigma(lam)
                view = (-1,) + (1,) * (x.dim() - 1)
                x_t = alpha.view(view) * x + sigma.view(view) * noise
                # target axis + conditioning clock (standardized lam, both frameworks)
                target = objective.make_target(lam, noise, x)
                t_cond = objective.cond_input(args.cond_mode, lam)

                if repa_encoder is not None:
                    # progressive schedule: resolve the active alignment layer
                    # for this step. The switch is parameter-free (the shared
                    # projector just moves to a different block), and all
                    # ranks derive it from the same train_steps, so no
                    # synchronization is needed.
                    cur_repa_depth = repa_depth_at(repa_depths, repa_steps, train_steps)
                    if cur_repa_depth != prev_repa_depth:
                        logger.info(f"REPA alignment layer set to {cur_repa_depth} "
                                    f"(step {train_steps})")
                        prev_repa_depth = cur_repa_depth
                    model_output, z_tilde = model(x_t, t_cond, y, repa_depth=cur_repa_depth)
                else:
                    model_output = model(x_t, t_cond, y)
                native_mse = mean_flat((target - model_output) ** 2)
                if args.loss_space == "u":
                    # JiT's v-loss: unweighted MSE in velocity space. The squared
                    # target->u Jacobian times the native MSE IS the u-space MSE,
                    # so diagnostics' weighted_mse stays the true training loss.
                    w = objective.u_error_factor(lam) ** 2
                else:
                    # factor 3: exact per-target min-SNR weight
                    w = objective.weight(lam, args.min_snr_gamma)
                if wn_mu is not None:
                    # wn ablation: mean-1 Gaussian bump (explicit weighting ==
                    # sampling from the tilted density; identical loss scale)
                    w = w * formulas.sampling_bump_weight(lam, wn_mu, args.wn_sigma,
                                                          args.lam_min, args.lam_max)
                elif args.wn_mode == "flat":
                    # wn-flat ablation: divide out the aligned min-SNR profile
                    # min(s, gamma) (mean-one normalised) so the effective
                    # weight profile W(lam) = p*w*c_x0 is CONSTANT in lam ==
                    # exactly the plain x0-space MSE training
                    w = w * formulas.flat_uniform_weight(lam, args.min_snr_gamma,
                                                         args.lam_min, args.lam_max)
                elif args.wn_mode == "mirror":
                    # wn-mirror ablation: swap high and low weights -- divide by
                    # min(s, gamma) and multiply by its left-right mirror about
                    # the support midpoint, so the effective weight profile
                    # W(lam) = p*w*c_x0 becomes the MIRRORED min-SNR profile
                    w = w * formulas.mirror_minsnr_weight(lam, args.min_snr_gamma,
                                                          args.lam_min, args.lam_max)
                elif args.wn_mode == "umirror":
                    # wn-umirror ablation: mirror the U-SPACE effective profile
                    # W_u = min(s, gamma)/(1+sqrt(s))^2 (identical for every
                    # target) about the support midpoint: the base mass split
                    # 18/82 (noise/clean halves) becomes 82/18
                    w = w * formulas.umirror_weight(lam, args.min_snr_gamma,
                                                    args.lam_min, args.lam_max)
                elif args.wn_mode == "ucenter":
                    # wn-ucenter ablation: replace the U-SPACE effective profile
                    # by the geometric mean of itself and its mirror, EXACTLY
                    # symmetric about the midpoint (peak at the midpoint, 50/50
                    # mass split)
                    w = w * formulas.ucenter_weight(lam, args.min_snr_gamma,
                                                    args.lam_min, args.lam_max)
                loss = (w * native_mse).mean()
                if repa_encoder is not None:
                    proj_loss = repa_proj_loss(z, z_tilde)
                    loss = loss + args.repa_coeff * proj_loss

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) \
                    if args.grad_clip > 0 else torch.tensor(0.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) \
                    if args.grad_clip > 0 else torch.tensor(0.0)
                opt.step()
            scheduler.step()
            update_ema(ema, model.module)

            # diagnostics: native / eps-space / weighted MSE per lambda bin
            with torch.no_grad():
                eps_mse = native_mse / objective.c_eps(lam)
                bin_logger.update(lam, native_mse, eps_mse, w)

            running_loss += loss.item()
            running_grad_norm += float(grad_norm)
            if repa_encoder is not None:
                running_proj += proj_loss.detach()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_grad = torch.tensor(running_grad_norm / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_grad, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_grad = avg_grad.item() / dist.get_world_size()
                bin_logger.all_reduce()
                snap = bin_logger.snapshot()
                proj_msg = ""
                proj_json = {}
                if repa_encoder is not None:
                    avg_proj = running_proj / log_steps
                    dist.all_reduce(avg_proj, op=dist.ReduceOp.SUM)
                    avg_proj = avg_proj.item() / dist.get_world_size()
                    proj_msg = f", Proj Loss: {avg_proj:.4f}"
                    proj_json = {"proj_loss": avg_proj}
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Grad Norm: {avg_grad:.3f}{proj_msg}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if diag_path is not None:
                    with open(diag_path, "a") as f:
                        f.write(json.dumps({
                            "step": train_steps, "loss": avg_loss,
                            "grad_norm": avg_grad, **proj_json, **snap,
                        }) + "\n")
                running_loss = 0.0
                running_grad_norm = 0.0
                running_proj = torch.zeros((), device=device)
                log_steps = 0
                start_time = time()
                bin_logger.reset()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "scheduler": scheduler.state_dict(),
                    }
                    if scaler is not None:
                        checkpoint["scaler"] = scaler.state_dict()
                    ckpt_file = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, ckpt_file)
                    logger.info(f"Saved checkpoint to {ckpt_file}")
                dist.barrier()

            if args.train_steps and train_steps >= args.train_steps:
                done = True
                break

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # cell definition
    parser.add_argument("--framework", type=str, required=True, choices=["vp", "fm"],
                        help="vp = continuous DDPM path (DiT column); fm = linear path (SiT column)")
    parser.add_argument("--target", type=str, required=True,
                        choices=["eps", "x0", "u", "rotation",
                                 "a15", "a30", "a45", "a60", "a75", "a80", "a90"],
                        help="prediction target; rotation = uniform angular-velocity sweep "
                             "(FM only, the +1); a15..a90 = fixed angle-interpolated "
                             "targets on the noise/data unit circle (theta in degrees from "
                             "the noise axis; eps=0, u=45, x0=90)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="experiment subfolder name under results-dir (default: {framework}-{target})")
    # flattening knobs (shared defaults from the experiment plan)
    parser.add_argument("--lam-min", type=float, default=None,
                        help="logSNR support lower end; default: computed from the DDPM "
                             "chain endpoints (full generative path, ~= -10.12)")
    parser.add_argument("--lam-max", type=float, default=None,
                        help="logSNR support upper end; default: computed from the DDPM "
                             "chain endpoints (full generative path, ~= +9.21)")
    parser.add_argument("--lsc-type", type=str, default="w/o", choices=["w/o", "w/", "skewed"],
                        help="LSC skip topology for follow-up experiments; w/o = no LSC "
                             "(the 2x3+1 grid runs in this mode)")
    parser.add_argument("--lsc-addon", type=str, default="v-pred", choices=["basic", "v-pred"],
                        help="LSC fusion adapter: basic = Linear(LN(cat[x,skip])); "
                             "v-pred = x + gamma*LN(skip) with gamma init 0")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0,
                        help="min-SNR cap gamma; 0 disables weighting")
    parser.add_argument("--cond-mode", type=str, default="lambda-std", choices=["lambda-std", "lambda"],
                        help="conditioning clock fed to t_embedder on both frameworks")
    # JiT-alignment ablation knobs (each breaks exactly one flattening factor)
    parser.add_argument("--data-space", type=str, default="latent", choices=["latent", "pixel"],
                        help="latent = SD-VAE latent x the --vae-scale factor (0.18215 = "
                             "SD's default, the 2x3+1 grid); "
                             "pixel = operate directly on the [-1,1] image (JiT-style; "
                             "no VAE, in_channels=3 at full resolution)")
    parser.add_argument("--loss-space", type=str, default="native", choices=["native", "u"],
                        help="native = MSE on the cell's own target (min-SNR weight applies); "
                             "u = unweighted MSE in velocity space (JiT's v-loss; v = -u "
                             "gives the identical squared loss; min-SNR is ignored)")
    parser.add_argument("--lam-sampling", type=str, default="uniform",
                        choices=["uniform", "logit-normal"],
                        help="uniform = U[lam_min, lam_max] (factor 1); logit-normal = "
                             "JiT's t = sigmoid(z), z ~ N(p_mean, p_std^2), i.e. lam = 2z")
    parser.add_argument("--p-mean", type=float, default=-0.8,
                        help="JiT logit-normal mean in z space (only with --lam-sampling logit-normal)")
    parser.add_argument("--p-std", type=float, default=0.8,
                        help="JiT logit-normal std in z space (only with --lam-sampling logit-normal)")
    # weighting-bump ablation (extra weight on top of min-SNR/v-loss)
    parser.add_argument("--wn-mode", type=str, default="off",
                        choices=["off", "clean", "midclean", "mid", "midnoisy", "noisy",
                                 "flat", "mirror", "umirror", "ucenter"],
                        help="Gaussian weight bump centred at the clean (high-lam) / "
                             "mid-clean / middle / mid-noisy / noisy (low-lam) region of "
                             "the lambda support; off = the 2x3+1 default flat-ish "
                             "profile. Presets are the support midpoint, midpoint +/- "
                             "quarter range, and midpoint +/- eighth range (each pair "
                             "mirror-symmetric about the midpoint). flat = no bump "
                             "but the exact inverse of the aligned min-SNR profile "
                             "min(s, gamma) (mean-one normalised): the effective "
                             "weight profile becomes CONSTANT, i.e. plain x0-space "
                             "MSE training. mirror = swap high/low weights: the "
                             "effective profile becomes the left-right MIRROR of "
                             "min(s, gamma) about the support midpoint. umirror/"
                             "ucenter act on the U-SPACE effective profile W_u = "
                             "min(s, gamma)/(1+sqrt(s))^2 (identical for every "
                             "target): umirror mirrors it about the midpoint "
                             "(base 18/82 noise/clean mass split becomes 82/18); "
                             "ucenter replaces it by its geometric mean with the "
                             "mirror, EXACTLY midpoint-symmetric (peak at the "
                             "midpoint, 50/50 split). All require --loss-space "
                             "native and --min-snr-gamma > 0; --wn-mu/--wn-sigma "
                             "do not apply to them)")
    parser.add_argument("--wn-mu", type=float, default=None,
                        help="explicit bump centre in logSNR; default: the --wn-mode preset")
    parser.add_argument("--wn-sigma", type=float, default=1.6,
                        help="bump std in logSNR (default 1.6, JiT lnt's lambda-domain std)")
    parser.add_argument("--num-lam-bins", type=int, default=12)
    # aligned optimization hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--label-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=False,
                        help="enable QK-LayerNorm in attention (custom flash-eligible "
                             "implementation); default off = timm Attention, SiT's exact "
                             "implementation")
    # REPA (representation alignment with a frozen DINOv2 teacher; off by default)
    parser.add_argument("--repa-depth", type=int, nargs='+', default=None, metavar="N",
                        help="alignment layer, 1-based block index (REPA's encoder_depth, "
                             "reference default 8); None disables REPA entirely. A "
                             "multi-value list defines a PROGRESSIVE schedule (e.g. "
                             "--repa-depth 8 16 aligns block 8 first, then block 16) "
                             "and requires --repa-depth-steps")
    parser.add_argument("--repa-depth-steps", type=int, nargs='+', default=None, metavar="S",
                        help="activation step of each --repa-depth value, aligned 1:1 "
                             "(--repa-depth 8 16 --repa-depth-steps 0 400000: block 8 "
                             "from step 0, block 16 once train_steps >= 400000). The "
                             "first entry must be 0 and the steps increase strictly; "
                             "the shared projector weights carry over at each switch. "
                             "Omitting it keeps the legacy behaviour: the single "
                             "--repa-depth value is fixed for the whole run")
    parser.add_argument("--repa-coeff", type=float, default=0.5,
                        help="weight of the REPA projection loss (reference default 0.5)")
    parser.add_argument("--repa-projector-dim", type=int, default=2048,
                        help="hidden width of the REPA MLP projector (reference default 2048)")
    parser.add_argument("--dinov2-repo", type=str, default=None,
                        help="local checkout of facebookresearch/dinov2 (hubconf.py at root); "
                             "only needed when --dinov2-ckpt points to an official .pth")
    parser.add_argument("--dinov2-ckpt", type=str, default=None,
                        help="DINOv2 weights: an HF-format directory (config.json + "
                             "model.safetensors) or an official pretrain .pth file "
                             "('teacher'-wrapped / 'backbone.'-prefixed also accepted)")
    parser.add_argument("--dinov2-arch", type=str, default="dinov2_vitb14",
                        help="dinov2 hubconf entrypoint, only used with official .pth "
                             "weights (HF directories read their config.json)")
    # data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-backend", type=str, default="imagefolder", choices=["imagefolder"])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    # model / vae
    parser.add_argument("--model", type=str, choices=list(Exp_models.keys()), default="XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="full HF id or local path overriding --vae")
    parser.add_argument("--vae-scale", type=float, default=0.18215,
                        help="latent scaling factor multiplied after vae.encode (SD's "
                             "0.18215 default; pass 0.2204 for the custom-dataset VAE). "
                             "Persisted in the checkpoint args, so sample_exp.py / "
                             "eval_exp.py / build_feature_pipeline pick it up")
    # schedule / logging
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--train-steps", type=int, default=0,
                        help="if >0, stop after this many steps (aligned training length)")
    parser.add_argument("--results-dir", type=str, default="results_exp")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
