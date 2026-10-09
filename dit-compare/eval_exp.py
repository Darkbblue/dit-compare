"""
Unified fixed-timestep loss evaluation for the 2x3+1 grid (causal-intervention
probes; companion to train_exp.py / sample_exp.py).

Instead of full generation, this entry loads a trained checkpoint and measures
the cell's TRAINING loss at ONE FIXED noise level on the training data itself:

  - no test split: for causal interventions the phenomenon must show on the
    training distribution, so the training dataset backend is reused as-is;
  - deterministic data side: the dataset is walked in its DEFAULT order (no
    shuffle, and the random horizontal flip is dropped), and only the first
    --num-images images are evaluated before the aggregate is printed and the
    script exits;
  - aligned randomness: ALL stochastic draws (the VAE reparameterization eps
    and the diffusion noise) are pulled ONCE, up front, from a single
    --global-seed stream in dataset order, independent of the batch size —
    the i-th image receives bit-identical randomness on every run and on
    every checkpoint, so loss differences across models at the same seed are
    causal, not sampling noise.

The fixed noise level comes either directly in the framework-native clock
(--lam, logSNR) or as a DDPM-style t in [0, 1000] (--t) converted with the
project's evaluation protocol (u = (1000-t)/1000 + 1e-6 clamped to
[1e-6, 1-1e-6], lam = 2*log(u/(1-u)); verbatim from for_feature/t_to_snr.py —
keep the two in sync). The same lam drives the per-path noising curve
(alpha/sigma) and the conditioning clock (standardized lam), exactly as in
training; the reported loss is the cell's training loss (native MSE x min-SNR
/ v-loss weight x wn bump), all recovered from the checkpoint args. The REPA
alignment term is NOT reproduced here (it needs the DINOv2 teacher and only
shapes training); REPA checkpoints load fine with the projector keys dropped.

Single GPU, no DDP, fp32 throughout (evaluation precision, fully
deterministic per batch composition); the aggregate is printed and the
script exits.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import math

from diffusers.models import AutoencoderKL

from exp_common import formulas
from exp_common.model import Exp_models
from exp_common.paths import get_path
from exp_common.objectives import get_objective
from exp_common.lsc import add_lsc, make_skip_table


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def t_to_logsnr(t):
    """DDPM-style t in [0, 1000] -> logSNR, the project's evaluation protocol
    (verbatim from for_feature/t_to_snr.py — keep the two in sync)."""
    u = (1000.0 - t) / 1000.0 + 1e-6
    u_c = min(max(u, 1e-6), 1.0 - 1e-6)
    return 2.0 * math.log(u_c / (1.0 - u_c))


def center_crop_arr(pil_image, image_size):
    # verbatim from train_exp.py: deterministic preprocessing (the RANDOM flip
    # of the training transform is deliberately NOT applied at eval time)
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


@torch.no_grad()
def main(args):
    assert args.num_images > 0, "--num-images must be positive"
    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU."
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # checkpoint + recovered cell configuration (sample_exp.py pattern:
    # every knob defaults to the value stored at train time; explicit CLI
    # flags win)
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    train_args = checkpoint.get("args")
    get = lambda k, default: getattr(train_args, k, default) if train_args is not None else default

    # image size: explicit CLI flag wins; default to the training value
    args.image_size = args.image_size if args.image_size is not None else get("image_size", 256)

    framework = args.framework or get("framework", None)
    target = args.target or get("target", None)
    assert framework and target, "framework/target not found in checkpoint args; pass --framework/--target"

    auto_min, auto_max = formulas.default_lambda_range()
    lam_min = get("lam_min", None)
    lam_max = get("lam_max", None)
    lam_min = auto_min if lam_min is None else lam_min
    lam_max = auto_max if lam_max is None else lam_max
    cond_mode = get("cond_mode", "lambda-std")

    # fixed noise level: exactly one of --t / --lam
    if (args.t is None) == (args.lam is None):
        raise SystemExit("pass exactly one of --t (DDPM-style, 0..1000) or --lam (logSNR)")
    lam_fixed = args.lam if args.lam is not None else t_to_logsnr(args.t)

    path = get_path(framework)
    objective = get_objective(target, lam_min, lam_max)

    # training-loss weights, resolved exactly as train_exp.py did at train time
    min_snr_gamma = get("min_snr_gamma", 5.0)
    if min_snr_gamma == 0:
        min_snr_gamma = None
    loss_space = get("loss_space", "native")
    # v-loss (u-space) is unweighted by definition; min-SNR is a native-space
    # weight and does not apply there
    if loss_space != "native" and min_snr_gamma is not None:
        min_snr_gamma = None
    wn_mode = get("wn_mode", "off")
    wn_mu = None
    wn_sigma = None
    if wn_mode in ("flat", "mirror", "umirror", "ucenter"):
        # all divide out the aligned min-SNR profile min(s, gamma), so they are
        # only defined when that profile was actually active at train time
        if loss_space != "native" or min_snr_gamma is None:
            raise SystemExit(f"--wn-mode {wn_mode} requires --loss-space native with a "
                             "positive min_snr_gamma in the checkpoint args")
    elif wn_mode != "off":
        mid = 0.5 * (lam_min + lam_max)
        quarter = 0.25 * (lam_max - lam_min)
        preset = {"clean": mid + quarter, "midclean": mid + 0.5 * quarter,
                  "mid": mid, "midnoisy": mid - 0.5 * quarter,
                  "noisy": mid - quarter}
        wn_mu = get("wn_mu", None)
        if wn_mu is None:
            wn_mu = preset[wn_mode]
        wn_sigma = get("wn_sigma", 1.6)

    # ------------------------------------------------------------------
    # model construction (mirrors sample_exp.py: rebuild the recorded LSC
    # structure BEFORE the strict state_dict load; drop REPA projector keys)
    # ------------------------------------------------------------------
    data_space = get("data_space", "latent")
    # vae_scale: the latent scaling factor used at train time (SD's default
    # 0.18215; a custom-dataset VAE may store 0.2204). Explicit --vae-scale
    # wins, then the checkpoint args, then the SD default
    vae_scale = args.vae_scale if args.vae_scale is not None else get("vae_scale", 0.18215)
    model_name = args.model or get("model", "XL/2")
    lsc_type = args.lsc_type or get("lsc_type", "w/o")
    lsc_addon = args.lsc_addon or get("lsc_addon", "v-pred")
    qk_norm = get("qk_norm", None)
    if qk_norm is None:
        qk_norm = not get("no_qk_norm", False)
    num_classes = args.num_classes if args.num_classes is not None else get("num_classes", 1000)
    label_dropout = get("label_dropout", 0.1)

    if data_space == "pixel":
        # JiT-style ablation: the model operates on the [-1,1] image (no VAE)
        input_size, in_channels = args.image_size, 3
    else:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        input_size, in_channels = args.image_size // 8, 4
    model = Exp_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=label_dropout,
        qk_norm=qk_norm,
    )
    if lsc_type != 'w/o':
        skip_table = make_skip_table(len(model.blocks), mode=lsc_type)
        add_lsc(model, model.hidden_size, layer_count=len(model.blocks),
                lsc_type=lsc_addon, skip_table=skip_table)
    state_dict = checkpoint["ema"] if args.use_ema and "ema" in checkpoint else checkpoint["model"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("repa_projector.")}
    model.load_state_dict(state_dict)
    model.eval()  # disables the in-model label dropout (train=False in y_embedder)
    model = model.to(device)
    vae = None
    if data_space == "latent":
        vae = AutoencoderKL.from_pretrained(args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # ------------------------------------------------------------------
    # data: the TRAINING dataset backend, default order (no shuffle), the
    # training preprocessing minus the random flip -> the data side is fully
    # deterministic
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    if args.data_backend == "imagefolder":
        dataset = ImageFolder(args.data_path, transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        dataset_len = len(dataset)
    elif args.data_backend == "nas":
        # NAS table backend (mirrors train_exp.py's build_loader, but with
        # shuffle=False so the default table order is preserved)
        from qtrain.train.dataset import NasTableImageDataset
        import cv2

        def img_loader(img):
            data = np.asarray(bytearray(img), dtype='uint8')
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            return Image.fromarray(data.astype('uint8'))

        # single-process eval: read the WHOLE table in default order (slice 0
        # of 1) — unlike DDP training, where each rank falls back to slice
        # `rank` of `world_size`; explicit CLI overrides still win
        slice_id = args.nas_slice_id if args.nas_slice_id is not None else 0
        slice_count = args.nas_slice_count if args.nas_slice_count is not None else 1
        dataset = NasTableImageDataset(
            args.data_path,
            slice_id=slice_id, slice_count=slice_count,
            shuffle=True, thread_num=args.num_workers,
            target_col_idx=args.nas_target_col_idx, image_col_idx=args.nas_image_col_idx,
            transform=transform, loader=img_loader,
            queue_size=args.nas_queue_size,
            sampling_by_weight=False, sampling_repeat=1,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False,
        )
        dataset_len = len(dataset)
    else:
        raise ValueError(f"unknown data backend: {args.data_backend}")

    # ------------------------------------------------------------------
    # aligned randomness: pre-draw EVERYTHING from one seed stream in
    # dataset order, independent of the batch size. The i-th image thus gets
    # the i-th (vae_eps, noise) pair on any run / any checkpoint / any
    # --batch-size / --num-images value.
    # ------------------------------------------------------------------
    torch.manual_seed(args.global_seed)
    n_eval = min(args.num_images, dataset_len)
    if n_eval < args.num_images:
        print(f"WARNING: dataset has only {dataset_len} images; evaluating the first {n_eval}.")
    # kept on CPU: only per-batch slices are moved to the GPU
    vae_eps_all = torch.randn(n_eval, 4, input_size, input_size) \
        if vae is not None else None
    noise_all = torch.randn(n_eval, in_channels, input_size, input_size)

    # fixed-time scalars (identical for every sample, as in training)
    alpha0, sigma0 = path.alpha_sigma(lam_fixed)
    t_cond = objective.cond_input(cond_mode, lam_fixed)

    print("=" * 76)
    print(f"ckpt           : {args.ckpt} (EMA: {bool(args.use_ema and 'ema' in checkpoint)})")
    print(f"cell           : {framework}-{target} | model {model_name} | data_space {data_space}"
          f" | qk_norm {qk_norm} | lsc {lsc_type}/{lsc_addon} | vae_scale {vae_scale:g}")
    if args.t is not None:
        print(f"noise level    : t = {args.t:g} -> lam = {lam_fixed:.4f}"
              f" (SNR = {math.exp(lam_fixed):.4g})  [project t->logSNR protocol]")
    else:
        print(f"noise level    : lam = {lam_fixed:.4f} (SNR = {math.exp(lam_fixed):.4g})")
    print(f"                 alpha = {float(alpha0):.4f}, sigma = {float(sigma0):.4f}"
          f" ({framework} path); cond clock '{cond_mode}' -> t_cond = {t_cond:.4f}")
    if not (lam_min <= lam_fixed <= lam_max):
        print(f"                 WARNING: lam is OUTSIDE the training support "
              f"[{lam_min:.4f}, {lam_max:.4f}] (out-of-distribution intervention)")
    weight_desc = f"min_snr_gamma={min_snr_gamma}" if loss_space == "native" \
        else f"loss_space={loss_space} (unweighted v-loss)"
    if wn_mu is not None:
        weight_desc += f", wn bump mu={wn_mu:.4f} sigma={wn_sigma:.4f}"
    elif wn_mode == "flat":
        weight_desc += ", wn flat (x min(s,gamma)^-1, mean-one)"
    elif wn_mode == "mirror":
        weight_desc += ", wn mirror (x W(Lam-lam)/W(lam), mean-one)"
    elif wn_mode == "umirror":
        weight_desc += ", wn umirror (x W_u(Lam-lam)/W_u(lam), mean-one)"
    elif wn_mode == "ucenter":
        weight_desc += ", wn ucenter (x sqrt(W_u(Lam-lam)/W_u(lam)), mean-one)"
    print(f"loss           : native MSE in '{target}' space x weight ({weight_desc})")
    print(f"data           : {args.data_path} ({args.data_backend}), first {n_eval} of"
          f" {dataset_len} images in default order (no shuffle/flip),"
          f" batch {args.batch_size}, seed {args.global_seed}")
    print("-" * 76)

    import json
    def load_deletion_dirs(npz_path, target, device='cuda'):
        """-> {module name: {'V': (k,d), 'mu': (d,), 'inv': (d,)}}
        Only entries with space='standardized' (fullspace/SC1 products)"""
        data = np.load(npz_path)
        meta = json.loads(str(data['__meta__']))
        table = {}
        for e in meta['entries']:
            if e['target'] != target or e.get('space') != 'standardized':
                continue
            table[e['input']] = {
                'V':   torch.from_numpy(np.asarray(data[e['keys']['dirs']])).float().to(device),
                'mu':  torch.from_numpy(np.asarray(data[e['keys']['mu']])).float().to(device),
                'inv': torch.from_numpy(np.asarray(data[e['keys']['inv_std']])).float().to(device),
            }
        return table
    
    if args.V:
        V = load_deletion_dirs(args.V, target=args.V_target,  device='cuda')
    else:
        V = None
        intervention_layers = []

    # ------------------------------------------------------------------
    # evaluation loop: native MSE x weight == the exact training loss of the
    # cell at this lam (all weights are constant in lam here)
    # ------------------------------------------------------------------
    total_native = 0.0
    total_eps = 0.0
    total_weighted = 0.0
    seen = 0
    idx = 0
    for x, y in loader:
        b = min(x.shape[0], n_eval - idx)
        if b <= 0:
            break
        x = x[:b].to(device, non_blocking=True)
        y = y[:b].to(device, non_blocking=True)

        # VAE reparameterization with the pre-drawn eps (statistically the
        # same draw as training's latent_dist.sample(), but from the aligned
        # stream); pixel mode: x stays the [-1,1] image
        if vae is not None:
            latent_dist = vae.encode(x).latent_dist
            x = (latent_dist.mean + latent_dist.std * vae_eps_all[idx: idx + b].to(device)) \
                .mul_(vae_scale)

        noise = noise_all[idx: idx + b].to(device)
        idx += b

        lam = torch.full((b,), lam_fixed, device=device)
        alpha, sigma = path.alpha_sigma(lam)
        view = (-1,) + (1,) * (x.dim() - 1)
        x_t = alpha.view(view) * x + sigma.view(view) * noise
        tgt = objective.make_target(lam, noise, x)
        t_batch = torch.full((b,), float(t_cond), device=device)

        pred = model(x_t, t_batch, y, V=V, intervention_layers=args.intervention_layers)
        native_mse = mean_flat((tgt - pred) ** 2)
        if loss_space == "u":
            w = objective.u_error_factor(lam) ** 2
        else:
            w = objective.weight(lam, min_snr_gamma)
        if wn_mu is not None:
            w = w * formulas.sampling_bump_weight(lam, wn_mu, wn_sigma, lam_min, lam_max)
        elif wn_mode == "flat":
            w = w * formulas.flat_uniform_weight(lam, min_snr_gamma, lam_min, lam_max)
        elif wn_mode == "mirror":
            w = w * formulas.mirror_minsnr_weight(lam, min_snr_gamma, lam_min, lam_max)
        elif wn_mode == "umirror":
            w = w * formulas.umirror_weight(lam, min_snr_gamma, lam_min, lam_max)
        elif wn_mode == "ucenter":
            w = w * formulas.ucenter_weight(lam, min_snr_gamma, lam_min, lam_max)
        eps_mse = native_mse / objective.c_eps(lam)  # common epsilon space (diagnostics)

        total_native += native_mse.sum().item()
        total_eps += eps_mse.sum().item()
        total_weighted += (w * native_mse).sum().item()
        seen += b

    avg_native = total_native / seen
    avg_eps = total_eps / seen
    avg_weighted = total_weighted / seen
    print(f"n_eval                        : {seen}")
    print(f"native MSE ('{target}' space) : {avg_native:.6f}")
    print(f"eps-space MSE                 : {avg_eps:.6f}")
    print(f"train-equivalent weighted loss: {avg_weighted:.6f}")
    print("=" * 76)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # checkpoint / cell definition
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--framework", type=str, default=None, choices=[None, "vp", "fm"],
                        help="default: read from checkpoint args")
    parser.add_argument("--target", type=str, default=None,
                        choices=[None, "eps", "x0", "u", "rotation",
                                 "a15", "a30", "a45", "a60", "a75", "a80", "a90"],
                        help="default: read from checkpoint args")
    parser.add_argument("--model", type=str, default=None, choices=[None] + list(Exp_models.keys()))
    parser.add_argument("--lsc-type", type=str, default=None,
                        choices=[None, "w/o", "w/", "skewed"],
                        help="default: read from checkpoint args; must match the training "
                             "value (the LSC structure is rebuilt before the state_dict load)")
    parser.add_argument("--lsc-addon", type=str, default=None, choices=[None, "basic", "v-pred"],
                        help="default: read from checkpoint args; must match the training value")
    # fixed noise level (exactly one of the two)
    parser.add_argument("--t", type=float, default=None,
                        help="fixed DDPM-style timestep in [0, 1000] (0 = data, 1000 = pure "
                             "noise); converted to logSNR with the project's evaluation "
                             "protocol (for_feature/t_to_snr.py)")
    parser.add_argument("--lam", type=float, default=None,
                        help="fixed logSNR directly (the framework-native clock); "
                             "mutually exclusive with --t")
    # evaluation size / batching
    parser.add_argument("--num-images", type=int, default=1000,
                        help="evaluate only the FIRST n images of the dataset in default "
                             "order, print the aggregate and exit")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="inference batch size; does NOT affect which noise each image "
                             "receives (randomness is pre-drawn per image)")
    # data (the training dataset, reused without a test split)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-backend", type=str, default="nas", choices=["nas", "imagefolder"])
    parser.add_argument("--image-size", type=int, default=None,
                        help="default: read from checkpoint args (fallback 256); must "
                             "match the training value")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="default: read from checkpoint args (fallback 1000); must "
                             "match the training value so the label-embedding load succeeds")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--nas-slice-id", type=int, default=None,
                        help="default 0 (single process reads the whole table; "
                             "training instead falls back to the DDP rank)")
    parser.add_argument("--nas-slice-count", type=int, default=None,
                        help="default 1 (single process reads the whole table; "
                             "training instead falls back to the DDP world size)")
    parser.add_argument("--nas-target-col-idx", type=int, default=1)
    parser.add_argument("--nas-image-col-idx", type=int, default=0)
    parser.add_argument("--nas-queue-size", type=int, default=1024)
    # model / vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="full HF id or local path overriding --vae")
    parser.add_argument("--vae-scale", type=float, default=None,
                        help="latent scaling factor applied after vae.encode; default: "
                             "read from checkpoint args (fallback 0.18215 = SD's default; "
                             "pass 0.2204 for the custom-dataset VAE); must match training")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    # randomness
    parser.add_argument("--global-seed", type=int, default=0,
                        help="single seed governing the pre-drawn VAE eps and diffusion "
                             "noise streams; same seed -> bit-identical inputs across "
                             "checkpoints")
    parser.add_argument('--V', type=str)
    parser.add_argument('--V_target', type=str)
    parser.add_argument('--intervention_layers', type=int, nargs='+')
    args = parser.parse_args()
    main(args)
