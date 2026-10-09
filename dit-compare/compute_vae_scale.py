"""
Compute the SD-VAE latent scale factor for a dataset (the ImageNet+SD-VAE
replacement for the hardcoded 0.18215).

PROTOCOL (the official 0.18215 recipe, cf. LDM paper Sec. 4.3 and the
diffusers#437 reproductions):
  1. load the VAE ONLY (no DiT, no diffusion);
  2. encode EVERY image of the training dataset with the TRAINING
     preprocessing (center-crop resize to --image-size, [-1,1] normalize);
  3. pool the raw latents (pre-scaling) of the whole dataset and compute the
     GLOBAL variance / standard deviation over all latent elements;
  4. scale factor = 1 / std(latents).

  The factor aligns the latent variance with the standard Gaussian noise:
      Var(c * z) = c^2 * Var(z) = 1   =>   c = 1/sqrt(Var) = 1/std.
  NOTE it is the reciprocal of the STANDARD DEVIATION, not of the variance
  (multiplying by 1/var would leave variance at 1/var, not 1). The official
  0.18215 = 1/5.4902 confirms this: the SD-VAE latent std on ImageNet is
  ~5.49 (variance ~30.1, so 1/var ~ 0.0332). The mean is NOT subtracted
  (SD protocol scales only); it is reported for diagnostics.
  A single global scalar is the official choice; per-channel moments are
  printed as diagnostics only.

The statistic is computed with EXACT float64 streaming accumulators
(sum / sum-of-squares; nothing is ever materialized), in fp32 VAE-encode
precision (no autocast — this is a reference statistic, not training).
The reparameterization draw (latent_dist.sample) is what training actually
consumes and is the default; --latent-mode mode gives the deterministic
posterior-mean variant (differs by the ~unit posterior noise, negligible
at dataset scale).

Usage:
  single GPU : python compute_vae_scale.py --data-path <ImageNet or table> \
                   --data-backend {imagefolder,nas} [--image-size 256]
  multi GPU  : torchrun --nproc_per_node=8 compute_vae_scale.py ... \
                   (the nas backend is sharded by slices like train_exp.py;
                    imagefolder by a DistributedSampler; float64 sums are
                    all-reduced, rank 0 writes the JSON)

Plug the result into training as-is:
  python train_exp.py ... --vae-scale <scale>
(the -vs<scale> run-name suffix is appended automatically by train_exp.py,
and sample_exp.py / eval_exp.py / build_feature_pipeline inherit it from the
checkpoint args).

--self-check verifies the estimator on synthetic N(0, 5.49^2) latents: the
recovered scale must reproduce ~1/5.49 ~= 0.18215 (no GPU / VAE / data needed).
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import json
import math
import os
from time import time
from datetime import datetime


#################################################################################
#                              Training preprocessing                           #
#################################################################################

def center_crop_arr(pil_image, image_size):
    # verbatim from train_exp.py (the RandomHorizontalFlip of the training
    # transform is dropped: deterministic, and a flip leaves the value
    # distribution — hence the variance statistic — unchanged anyway)
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


#################################################################################
#                          float64 streaming statistics                         #
#################################################################################

class LatentStats:
    """Exact streaming moments (global + per-channel) of the raw latents.

    Everything accumulates in float64 on `device`; with ~5e9 elements for
    full ImageNet@256 the float64 sum-of-squares error is ~1e-7 relative —
    far below any meaningful digit of the scale factor."""

    def __init__(self, channels, device):
        self.channels = channels
        self.device = device
        f64 = dict(dtype=torch.float64, device=device)
        self.n = torch.zeros((), **f64)            # latent elements
        self.n_img = torch.zeros((), **f64)        # images
        self.sum = torch.zeros((), **f64)
        self.sumsq = torch.zeros((), **f64)
        self.ch_sum = torch.zeros(channels, **f64)
        self.ch_sumsq = torch.zeros(channels, **f64)

    @torch.no_grad()
    def update(self, z):
        # z: (B, C, h, w) raw latents, pre-scaling, any float dtype
        zd = z.to(torch.float64)
        self.n += zd.numel()
        self.n_img += z.shape[0]
        self.sum += zd.sum()
        self.sumsq += zd.pow(2).sum()
        self.ch_sum += zd.sum(dim=(0, 2, 3))
        self.ch_sumsq += zd.pow(2).sum(dim=(0, 2, 3))

    def all_reduce(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in (self.n, self.n_img, self.sum, self.sumsq, self.ch_sum, self.ch_sumsq):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def finalize(self):
        """Biased moments (divide by N): at dataset scale the unbiased
        correction is O(1/N), i.e. far below the reported digits."""
        mean = (self.sum / self.n).item()
        var = (self.sumsq / self.n - (self.sum / self.n).pow(2)).item()
        n_ch = (self.n / self.channels).item()
        ch_mean = (self.ch_sum / n_ch).tolist()
        ch_var = (self.ch_sumsq / n_ch - (self.ch_sum / n_ch).pow(2)).tolist()
        return {
            "num_images": int(self.n_img.item()),
            "num_elements": int(self.n.item()),
            "global_mean": mean,
            "global_var": var,
            "global_std": math.sqrt(max(var, 0.0)),
            "per_channel_mean": ch_mean,
            "per_channel_var": ch_var,
            "per_channel_std": [math.sqrt(max(v, 0.0)) for v in ch_var],
        }


#################################################################################
#                              dataset construction                             #
#################################################################################

def build_loader(args, rank, world_size):
    """The train_exp.py data backends, walked in default order (statistics are
    order-invariant sums; no shuffle, no flip -> fully deterministic data side).
    Returns (loader, dataset_len)."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    if args.data_backend == "imagefolder":
        dataset = ImageFolder(args.data_path, transform=transform)
        dataset_len = len(dataset)
        if world_size > 1:
            # drop_last: never duplicate-padding samples into the statistic
            # (drops len % world_size images, ~0.0006% of ImageNet at most)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=True)
            if rank == 0 and dataset_len % world_size:
                print(f"note: DistributedSampler drop_last skips {dataset_len % world_size} images")
        else:
            sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        return loader, dataset_len

    elif args.data_backend == "nas":
        # NAS table backend (mirrors train_exp.py's build_loader; the table is
        # sharded by slices across ranks exactly as in DDP training)
        from qtrain.train.dataset import NasTableImageDataset
        import cv2

        def img_loader(img):
            data = np.asarray(bytearray(img), dtype='uint8')
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            return Image.fromarray(data.astype('uint8'))

        # DDP: each rank takes slice `rank` of `world_size` (train_exp default);
        # single process: the WHOLE table, slice 0 of 1 (an explicit None passed
        # straight through would hit the NasTable slice_count>0 assert)
        slice_id = args.nas_slice_id if args.nas_slice_id is not None else rank
        slice_count = args.nas_slice_count if args.nas_slice_count is not None else world_size
        dataset = NasTableImageDataset(
            args.data_path,
            slice_id=slice_id, slice_count=slice_count,
            shuffle=True, thread_num=args.num_workers,
            target_col_idx=args.nas_target_col_idx, image_col_idx=args.nas_image_col_idx,
            transform=transform, loader=img_loader,
            queue_size=args.nas_queue_size,
            sampling_by_weight=False, sampling_repeat=1,
        )
        dataset_len = len(dataset)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False,
        )
        return loader, dataset_len

    raise ValueError(f"unknown data backend: {args.data_backend}")


#################################################################################
#                          self-check (no GPU / VAE / data)                     #
#################################################################################

def run_self_check(args):
    """Synthetic latents ~ N(0.01, 5.49^2) must recover scale ~ 1/5.49 ~ 0.18215
    (the official ImageNet+SD-VAE value), var ~ 30.14, and post-scale variance 1."""
    torch.manual_seed(args.global_seed)
    device = torch.device("cpu")
    stats = LatentStats(channels=4, device=device)
    TRUE_STD, TRUE_MEAN = 5.49, 0.01
    n_batches, batch = 800, 64  # 800*64*4*32*32 = 2.097e8 elements
    for _ in range(n_batches):
        z = torch.randn(batch, 4, 32, 32) * TRUE_STD + TRUE_MEAN
        stats.update(z)
    s = stats.finalize()
    scale = 1.0 / s["global_std"]
    ok = (abs(scale - 1 / TRUE_STD) / (1 / TRUE_STD) < 0.01
          and abs(s["global_mean"] - TRUE_MEAN) < 0.01)
    print("=" * 76)
    print(f"self-check: synthetic latents ~ N({TRUE_MEAN}, {TRUE_STD}^2), "
          f"{s['num_elements']:,} elements")
    print(f"  mean = {s['global_mean']:+.6f} (expect ~{TRUE_MEAN})")
    print(f"  var  = {s['global_var']:.6f} (expect ~{TRUE_STD ** 2:.4f})")
    print(f"  std  = {s['global_std']:.6f} (expect ~{TRUE_STD})")
    print(f"  scale factor = 1/std = {scale:.6f} (expect ~{1 / TRUE_STD:.6f} "
          "= the official 0.18215 protocol)")
    print(f"  sanity: scale^2 * var = {scale ** 2 * s['global_var']:.6f} (expect 1)")
    print(f"  1/var  = {1 / s['global_var']:.6f} (for reference only — NOT unit-variance)")
    print(f"  per-channel std: {[f'{v:.4f}' for v in s['per_channel_std']]}")
    print(f"SELF-CHECK {'PASSED' if ok else 'FAILED'}")
    print("=" * 76)
    return 0 if ok else 1


#################################################################################
#                                      main                                     #
#################################################################################

@torch.no_grad()
def main(args):
    if args.self_check:
        raise SystemExit(run_self_check(args))
    assert torch.cuda.is_available(), "VAE encoding currently requires at least one GPU."

    # DDP only when actually launched under torchrun (world size > 1); a plain
    # single-GPU `python compute_vae_scale.py` needs no process group
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        assert world_size == dist.get_world_size()
    else:
        rank = 0
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    # per-rank RNG stream for the latent_dist.sample() reparameterization draws
    # (same formula as train_exp.py, so the pass is reproducible)
    torch.manual_seed(args.global_seed * world_size + rank)

    # VAE ONLY — no DiT, no diffusion machinery anywhere in this script
    from diffusers.models import AutoencoderKL
    vae_id = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval()
    vae.requires_grad_(False)

    latent_hw = args.image_size // 8
    stats = LatentStats(channels=4, device=device)
    loader, dataset_len = build_loader(args, rank, world_size)
    # --num-images is a per-dataset cap, split evenly across ranks (approximate
    # under DDP: only used for quick-subset estimates, never the full protocol)
    limit = None if args.num_images is None else math.ceil(args.num_images / world_size)

    if rank == 0:
        print("=" * 76)
        print(f"vae            : {vae_id} (fp32, no autocast) | image {args.image_size} "
              f"-> latent [4, {latent_hw}, {latent_hw}] | draw mode: {args.latent_mode}")
        print(f"dataset        : {args.data_path} (backend={args.data_backend}, "
              f"{dataset_len:,} images, world_size={world_size}"
              + (f", capped at {args.num_images}" if args.num_images is not None else "") + ")")
        print(f"encoding all latents and accumulating float64 moments ...")

    t0 = time()
    processed = 0
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        latent_dist = vae.encode(x).latent_dist
        # sample = the reparameterized latent training actually consumes (the
        # official recipe); mode = deterministic posterior mean (variant)
        z = latent_dist.sample() if args.latent_mode == "sample" else latent_dist.mode()
        stats.update(z)
        processed += x.shape[0]
        if limit is not None and processed >= limit:
            break
        if rank == 0 and (i + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time() - t0
            done_frac = min(processed * world_size / max(dataset_len, 1), 1.0)
            eta = elapsed / done_frac - elapsed if done_frac > 0 else float("nan")
            print(f"  [{(i + 1):6d} batches] {processed * world_size:>9,d}/{dataset_len:,} images "
                  f"({100 * done_frac:5.2f}%) | {processed * world_size / elapsed:7.1f} img/s | "
                  f"ETA {eta / 60:6.1f} min")

    stats.all_reduce()
    s = stats.finalize()
    scale = 1.0 / s["global_std"]

    if rank == 0:
        n_used = s["num_images"]
        summary = {
            "protocol": "scale = 1/std(raw latents) over the full dataset "
                        "(the SD 0.18215 recipe; unit-variance alignment: "
                        "Var(c*z)=c^2*Var(z)=1 => c=1/sqrt(Var)=1/std)",
            "vae": vae_id,
            "image_size": args.image_size,
            "latent_shape": [4, latent_hw, latent_hw],
            "latent_mode": args.latent_mode,
            "data_path": args.data_path,
            "data_backend": args.data_backend,
            "num_images_expected": dataset_len,
            "fp64_streaming": True,
            "encode_precision": "fp32 (tf32 matmul allowed, no autocast)",
            "global_seed": args.global_seed,
            "world_size": world_size,
            **s,
            "scale_factor": scale,                       # <-- pass as --vae-scale
            "scale_if_reciprocal_variance": 1.0 / s["global_var"],  # reference only
            "post_scale_variance": scale ** 2 * s["global_var"],    # sanity ~= 1
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if args.output is not None:
            out_path = args.output
        else:
            stem = vae_id.rstrip("/").split("/")[-1] or "vae"
            out_dir = "results_exp/vae_scale_stats"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}_s{args.image_size}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("-" * 76)
        print(f"latents        : {n_used:,} images, {s['num_elements']:,} elements")
        print(f"statistics     : mean = {s['global_mean']:+.6f} | var = {s['global_var']:.6f} "
              f"| std = {s['global_std']:.6f}")
        print(f"per-channel std: {[f'{v:.4f}' for v in s['per_channel_std']]} "
              f"(diagnostics; the official protocol uses ONE global scalar)")
        print(f"SCALE FACTOR   : 1/std = {scale:.6f}")
        print(f"                 -> python train_exp.py ... --vae-scale {scale:g}"
              f"   (run-name auto-suffix: -vs{scale:g})")
        print(f"                 Var(c*z)=c^2*Var(z)=1 needs c=1/sqrt(Var)=1/std;")
        print(f"                 official 0.18215 = 1/5.4902 (SD-VAE on ImageNet)")
        print(f"1/var (ref)    : {1 / s['global_var']:.6f}  (NOT unit-variance: "
              f"post-scale var would be {s['global_var'] / s['global_var'] ** 2:.4f})")
        print(f"sanity         : scale^2 * var = {scale ** 2 * s['global_var']:.6f} (expect 1)")
        if args.num_images is not None and n_used < dataset_len:
            print(f"NOTE           : subset estimate (--num-images {args.num_images}); "
                  f"re-run without the cap for the final factor")
        print(f"saved          : {out_path}")
        print("=" * 76)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true",
                        help="verify the estimator on synthetic N(0, 5.49^2) latents "
                             "(no GPU / VAE / data needed) and exit")
    # data (train_exp.py backends, walked in default order)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-backend", type=str, default="nas", choices=["nas", "imagefolder"])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-images", type=int, default=None,
                        help="cap the pass at this many images (quick estimate; "
                             "omit for the full-dataset protocol)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    # nas backend (defaults follow train_exp.py; single process reads slice 0 of 1)
    parser.add_argument("--nas-slice-id", type=int, default=None)
    parser.add_argument("--nas-slice-count", type=int, default=None)
    parser.add_argument("--nas-target-col-idx", type=int, default=1)
    parser.add_argument("--nas-image-col-idx", type=int, default=0)
    parser.add_argument("--nas-queue-size", type=int, default=1024)
    # vae (train_exp.py loading; VAE only — nothing else is ever loaded)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="full HF id or local path overriding --vae")
    parser.add_argument("--latent-mode", type=str, default="sample", choices=["sample", "mode"],
                        help="sample = reparameterized latent (what training consumes, "
                             "the official recipe); mode = deterministic posterior mean")
    # output
    parser.add_argument("--output", type=str, default=None,
                        help="JSON path (default: results_exp/vae_scale_stats/"
                             "{vae}_s{image_size}.json)")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not args.self_check:
        assert args.data_path, "--data-path is required (or use --self-check)"
    main(args)
