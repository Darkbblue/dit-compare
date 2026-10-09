"""
inspect_ckpt.py — read diagnostic weights out of a dit-compare checkpoint.

Two reports:

  1. LSC v-pred gammas: the scalar gate of every LSCAdapterVPred, per up layer
     (with the paired down layer from the skip table). gamma is zero-init, so
     its trained value measures how much each long skip contributes.

  2. t(lambda) -> layer influence: the adaLN-Zero modulation weights. The
     conditioning clock reaches each block ONLY through
     blocks.<i>.adaLN_modulation[-1] (SiLU -> Linear(hidden, 6*hidden)), which
     is zero-initialized; the trained magnitude of this matrix IS how much
     conditioning dependence the layer has learned. Since c = t_emb + y_emb
     shares one matrix, we additionally report the composite gain through the
     t-embedder output matrix, ||W_adaLN @ W_t2||_F, isolating the t pathway.

Loading mirrors sample_exp.py / for_feature/dit_compare.py exactly:
    explicit CLI arg  >  checkpoint['args']  >  state-dict sniffing  >  default.
Runs on CPU; no DDP, no VAE needed.

Usage:
    python inspect_ckpt.py --ckpt path/to/ckpt.pt
    python inspect_ckpt.py --ckpt ... --no-ema
    python inspect_ckpt.py --ckpt ... --grep adaLN_modulation
"""

import argparse
import torch

from exp_common.model import Exp_models
from exp_common.lsc import add_lsc, make_skip_table


# ---------------------------------------------------------------------------
# checkpoint loading (same resolution chain as sample_exp.py / dit_compare.py)
# ---------------------------------------------------------------------------

def load_model(args):
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if train_args is not None and not isinstance(train_args, dict):
        train_args = vars(train_args)  # argparse.Namespace
    train_args = train_args or {}
    get = lambda k, default: train_args.get(k, default)

    # pick the weight dict (ema by default), falling back like dit_compare.py
    key = "ema" if args.use_ema else "model"
    if not isinstance(checkpoint, dict) or key not in checkpoint:
        key = "model" if isinstance(checkpoint, dict) and "model" in checkpoint else None
    state_dict = checkpoint[key] if key is not None else checkpoint
    # the REPA projector only exists for the training loss; drop before strict load
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("repa_projector.")}

    # qk_norm: new ckpts store the flag directly; older ones store the inverted
    # no_qk_norm; oldest have neither -> sniff for q_norm weights
    qk_norm = get("qk_norm", None)
    if qk_norm is None:
        if "no_qk_norm" in train_args:
            qk_norm = not get("no_qk_norm", False)
        else:
            qk_norm = any(k.endswith("attn.q_norm.weight") for k in state_dict)

    model_name = args.model or get("model", "XL/2")
    lsc_type = get("lsc_type", "w/o")
    lsc_addon = get("lsc_addon", "v-pred")
    data_space = get("data_space", "latent")
    image_size = get("image_size", 256)
    num_classes = args.num_classes if args.num_classes is not None \
        else get("num_classes", 1000)
    # label_dropout affects the embedding table size (num_classes + (p > 0)),
    # so it must match the checkpoint even though eval never drops
    label_dropout = get("label_dropout", 0.1)

    if data_space == "pixel":
        input_size, in_channels = image_size, 3   # JiT ablation: no VAE
    else:
        input_size, in_channels = image_size // 8, 4

    model = Exp_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=label_dropout,
        qk_norm=qk_norm,
    )
    # rebuild the LSC structure recorded in the checkpoint BEFORE loading
    # weights (adapter shapes must exist for load_state_dict to succeed)
    if lsc_type != "w/o":
        skip_table = make_skip_table(len(model.blocks), mode=lsc_type)
        add_lsc(model, model.hidden_size, layer_count=len(model.blocks),
                lsc_type=lsc_addon, skip_table=skip_table)

    model.load_state_dict(state_dict)
    model.eval()

    cfg = dict(model=model_name, weights=key, lsc_type=lsc_type, lsc_addon=lsc_addon,
               data_space=data_space, image_size=image_size, num_classes=num_classes,
               label_dropout=label_dropout, qk_norm=qk_norm,
               framework=get("framework", None), target=get("target", None),
               lam_min=get("lam_min", None), lam_max=get("lam_max", None),
               cond_mode=get("cond_mode", None))
    return model, state_dict, cfg


# ---------------------------------------------------------------------------
# report 1: LSC v-pred gammas
# ---------------------------------------------------------------------------

def report_lsc_gammas(model):
    print("\n=== LSC gammas (v-pred addon) ===")
    if not hasattr(model, "lsc"):
        print("checkpoint has no LSC (lsc_type=w/o); nothing to report.")
        return
    lsc = model.lsc
    if lsc.lsc_type != "v-pred":
        print(f"lsc_addon='{lsc.lsc_type}' has no gamma parameter "
              f"(only v-pred adapters do); nothing to report.")
        return
    down_of = lsc.skip_table["up"]  # {up_layer: down_layer}
    rows = []
    for name, adapter in lsc.all_lsc.items():
        up = int(name)
        rows.append((up, down_of[up], adapter.gamma.detach().float().item()))
    rows.sort()
    print(f"{'up layer':>8} {'<- down':>8} {'gamma':>12}")
    for up, down, g in rows:
        print(f"{up:>8} {down:>8} {g:>+12.6f}")
    vals = torch.tensor([g for _, _, g in rows])
    print(f"summary: mean={vals.mean():+.6f}  std={vals.std():.6f}  "
          f"min={vals.min():+.6f}  max={vals.max():+.6f}  "
          f"mean|gamma|={vals.abs().mean():.6f}")


# ---------------------------------------------------------------------------
# report 2: t(lambda) -> layer influence (adaLN-Zero modulation weights)
# ---------------------------------------------------------------------------

_CHUNKS = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")


def _chunk_norms(weight, n_chunks):
    return [c.norm().item() for c in weight.detach().float().chunk(n_chunks, dim=0)]


def report_t_influence(model):
    print("\n=== t(lambda) influence per layer (adaLN-Zero modulation weights) ===")
    print("columns: per-chunk Frobenius norms of blocks.<i>.adaLN_modulation[-1].weight;")
    print("  ||W||_F      : full 6D x D matrix norm (zero at init -> learned t-dependence)")
    print("  ||W@W_t2||_F : composite through t_embedder.mlp[2], isolating the t pathway")
    W_t2 = model.t_embedder.mlp[2].weight.detach().float()

    header = f"{'block':>5} " + " ".join(f"{c:>10}" for c in _CHUNKS) + \
             f" {'||W||_F':>10} {'||W@W_t2||_F':>13}"
    print(header)
    for i, block in enumerate(model.blocks):
        W = block.adaLN_modulation[-1].weight.detach().float()
        norms = _chunk_norms(W, 6)
        total = W.norm().item()
        t_gain = (W @ W_t2).norm().item()
        print(f"{i:>5} " + " ".join(f"{n:>10.4f}" for n in norms) +
              f" {total:>10.4f} {t_gain:>13.4f}")

    W = model.final_layer.adaLN_modulation[-1].weight.detach().float()
    s, sc = _chunk_norms(W, 2)
    print(f"final_layer: shift={s:.4f}  scale={sc:.4f}  ||W||_F={W.norm().item():.4f}  "
          f"||W@W_t2||_F={(W @ W_t2).norm().item():.4f}")


# ---------------------------------------------------------------------------

def grep_state_dict(state_dict, pattern):
    print(f"\n=== state_dict entries matching '{pattern}' ===")
    hits = 0
    for k, v in sorted(state_dict.items()):
        if pattern in k:
            hits += 1
            if v.numel() <= 8:
                flat = ", ".join(f"{x:+.6f}" for x in v.detach().float().flatten())
                print(f"{k:60s} shape={tuple(v.shape)}  values=[{flat}]")
            else:
                vf = v.detach().float()
                print(f"{k:60s} shape={tuple(v.shape)}  mean={vf.mean():+.6f}  "
                      f"absmean={vf.abs().mean():.6f}  norm={vf.norm():.6f}")
    if hits == 0:
        print("(no matches)")


def main(args):
    model, state_dict, cfg = load_model(args)
    print("resolved checkpoint config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    report_lsc_gammas(model)
    report_t_influence(model)
    if args.grep:
        grep_state_dict(state_dict, args.grep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, choices=[None] + list(Exp_models.keys()),
                        help="default: read from checkpoint args")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="default: read from checkpoint args (fallback 1000); must "
                             "match the training value (CFG null token sits AT index "
                             "num_classes)")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--grep", type=str, default=None,
                        help="also dump stats of raw state_dict entries containing "
                             "this substring (e.g. 'adaLN_modulation', 'lsc.all_lsc')")
    args = parser.parse_args()
    main(args)
