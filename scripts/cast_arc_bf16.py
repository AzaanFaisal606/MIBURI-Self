"""Cast the ARC4-Encoder checkpoint from fp32 to bf16.

The released checkpoint is fp32: 3.03B parameters, 12.1 GB. It has to shrink,
and on this machine it has to shrink *without* int8, because the encoder runs on
CPU -- bitsandbytes' QLinear needs CUDA, and the card's ~0.7 GB of headroom
alongside Moshi and the gesture stack is nowhere near 6 GB. bf16 halves it to
~6.1 GB, keeps the exponent range fp32 had (so no rescaling), and CPU matmul
handles it fine at the ~730 ms/retrieval this path costs.

This is a pure dtype cast over the safetensors file: no model is built, no
module layout is assumed, and integer/bool tensors are passed through untouched.
That makes it independent of the encoder's architecture, unlike
`quantize_arc_q8.py`, which has to instantiate the modules before it can swap
their Linears.

  python "MIBURI User/scripts/cast_arc_bf16.py"

`run_rag_demo.sh` points ARC_ENCODER_WEIGHTS at the result.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PROJECT  # noqa: E402

SRC = PROJECT / "weights" / "ARC4_Encoder_Llama" / "model.safetensors"
DST = SRC.with_name("model.bf16.safetensors")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                   help="bf16 keeps fp32's exponent range; fp16 does not")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.src.is_file():
        print(f"source not found: {args.src}")
        return 1
    if args.dst.exists() and not args.force:
        print(f"exists, refusing to overwrite: {args.dst}  (use --force)")
        return 0

    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    print(f"reading {args.src}")
    out: dict[str, torch.Tensor] = {}
    src_bytes = dst_bytes = 0
    n_cast = 0
    # Tensor by tensor: holding the fp32 and the bf16 copies of a 12 GB
    # checkpoint at once would need 18 GB of the 31 GB in this box.
    with safe_open(str(args.src), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            src_bytes += t.numel() * t.element_size()
            if t.dtype.is_floating_point and t.dtype != dtype:
                t = t.to(dtype)
                n_cast += 1
            out[k] = t.contiguous()
            dst_bytes += t.numel() * t.element_size()

    print(f"  {len(keys)} tensors, {src_bytes / 1e9:.2f} GB -> "
          f"{dst_bytes / 1e9:.2f} GB ({n_cast} cast to {args.dtype})")

    print(f"writing {args.dst}")
    save_file(out, str(args.dst))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
