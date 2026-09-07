"""Quantize the ARC4-Encoder (Llama-3.2-3B backbone) to int8 on CPU.

The released checkpoint is fp32 (3.03B params, 12.1 GB). This applies the same
bitsandbytes vectorwise int8 pass MIBURI uses for Moshi to every nn.Linear in
the encoder and the bridge module.

The `ArcEncoderConditioner` wrapper is deliberately bypassed: constructing it
pulls the Llama-3.2-3B-Instruct tokenizer from a gated HF repo, which is not
needed to touch weights. The two weight-bearing submodules -- `embedder` and
`bridge_module` -- are built directly, matching the checkpoint's key prefixes.

  python "MIBURI User/scripts/quantize_arc_q8.py"
"""

import argparse
import sys
import time
import os
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT  # noqa: E402

MOSHI_RAG = PROJECT / "moshi-rag" / "moshi"
# A locally built xformers, if there is one. Point XFORMERS_PATH at its
# directory; without it the SDPA fallback in arc_encoder.py is used instead.
XF = Path(os.environ["XFORMERS_PATH"]) if os.environ.get("XFORMERS_PATH") else None
SRC = PROJECT / "weights" / "ARC4_Encoder_Llama" / "model.safetensors"
DST = SRC.with_name("model.q8.safetensors")

# From loaders._lm_kwargs["conditioners"]["reference_with_time"].
COMPRESS_RATE = -4
BRIDGE = {"in_dim": 3072, "out_dim": 4096, "hidden_dim": 2048}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.dst.exists() and not args.force:
        print(f"exists, refusing to overwrite: {args.dst}  (use --force)")
        return 0

    sys.path.insert(0, str(MOSHI_RAG))
    if XF is not None and XF.exists():
        sys.path.insert(0, str(XF))

    from moshi.conditioners.arc_encoder import ArcEncoderTransformer, EmbProjector
    from moshi.utils.quantize import replace_linear_with_qlinear

    t0 = time.time()

    class Arc(nn.Module):
        """Container reproducing the checkpoint's `embedder.*` / `bridge_module.*` layout."""

        def __init__(self):
            super().__init__()
            self.embedder = ArcEncoderTransformer(compression_rate=COMPRESS_RATE)
            self.bridge_module = EmbProjector(**BRIDGE)

    print("building ARC encoder on cpu")
    model = Arc()

    print(f"reading {args.src}")
    with safe_open(args.src, "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    src_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"  {len(state)} tensors, {src_bytes / 1e9:.2f} GB")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  MISSING {len(missing)}: {missing[:5]}")
        print(f"  UNEXPECTED {len(unexpected)}: {unexpected[:5]}")
        print("refusing to quantize an incompletely loaded model")
        return 1
    del state
    print(f"  loaded clean in {time.time() - t0:.0f}s")

    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"quantizing {n_lin} nn.Linear modules to int8")
    replace_linear_with_qlinear(model)

    out = {k: v.contiguous() for k, v in model.state_dict().items()}
    dst_bytes = sum(t.numel() * t.element_size() for t in out.values())
    print(f"  {len(out)} tensors, {dst_bytes / 1e9:.2f} GB "
          f"({100 * dst_bytes / src_bytes:.0f}% of source)")

    print(f"writing {args.dst}")
    save_file(out, str(args.dst))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())