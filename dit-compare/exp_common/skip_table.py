"""
Architecture-topology generation (pure python, no torch): LSC skip tables and
breaking-residual split points.

Kept in a torch-free module so check_alignment.py can verify the generated
tables against the explicit tables hard-coded in the two reference repos.
Used by exp_common.lsc (which attaches the adapters), exp_common.model (which
builds the breaking layers), and by train/sample.
"""


def make_skip_table(layer_count, mode, skew_offset=6, skew_pairs=10):
    """Generate the down/up skip table instead of hard-coding one.

    Returns {'down': {d: u, ...}, 'up': {u: d, ...}} where down layer d stores
    its block output and up layer u fuses it BEFORE its block (the hook
    convention shared by DiT-v/models.py and SiT/models.py).

    mode "w/"     : dense mirror, pairs = layer_count//2 - 1 starting at 0.
                    For 28 layers: {0:27, 1:26, ..., 12:15} — matches the
                    explicit 'w/ LSC dense' table in both reference repos.
    mode "skewed" : down segment shifted by `skew_offset` (default 6), with
                    `skew_pairs` (default 10) pairs, clamped to what fits.
                    For 28 layers: {6:27, 7:26, ..., 15:18} — matches SiT's
                    explicit 'skewed LSC' table.
    """
    if mode == "w/":
        offset, pairs = 0, layer_count // 2 - 1
    elif mode == "skewed":
        offset = skew_offset
        # every up layer must sit strictly after every paired down layer:
        # n - pairs > offset + pairs - 1  =>  pairs <= (n - offset) // 2
        pairs = min(skew_pairs, max(0, (layer_count - offset) // 2))
    else:
        raise ValueError(f"unknown lsc mode: {mode}")

    down, up = {}, {}
    for i in range(offset, offset + pairs):
        u = layer_count - 1 - (i - offset)
        down[i] = u
        up[u] = i
    return {"down": down, "up": up}


def break_points(depth, segments):
    """Evenly split `depth` blocks into `segments` residual segments; return the
    0-based block indices AFTER WHICH a breaking layer applies (segment ends,
    except the last segment — it ends at the model's existing final_layer).

    For depth=28, segments=4: [6, 13, 20], i.e. three breaking layers cut the
    28-block stream into four 7-block segments. segments None/<=1 means no cut.
    """
    if segments is None or segments <= 1:
        return []
    assert depth % segments == 0, \
        f"depth {depth} is not evenly divisible into {segments} segments"
    seg = depth // segments
    return [seg * (i + 1) - 1 for i in range(segments - 1)]
