"""Quantize the MoshiRAG (Moshika) LM to int8 on CPU.

Mirrors what `LMModel(quantize=True)` does at load time -- bitsandbytes
vectorwise int8 over every nn.Linear -- but does it once, offline, and writes
the result so the demo can load a q8 checkpoint directly.

Only nn.Linear is quantized. Embeddings, norms and the conditioner weights stay
bf16, which is why the output is ~9-10 GB rather than a clean half of 15.4 GB.

The conditioner tensors are copied through untouched: the model is built
without conditioners (so instantiating it does not pull the 12 GB ARC encoder),
then those keys are merged back verbatim from the source checkpoint.

  python "MIBURI User/scripts/quantize_moshi_q8.py"
"""

import argparse
import copy
import sys
import time
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT  # noqa: E402

MOSHI_RAG = PROJECT / "moshi-rag" / "moshi"
SRC = PROJECT / "weights" / "moshika-rag-pytorch-bf16" / "model.safetensors"
DST = SRC.with_name("model.q8.safetensors")

# `arc_encoder` imports xformers at module scope; the conditioners are stripped
# below so it is never reached, but the path is added for parity with runtime.
# A locally built xformers, if there is one. Point XFORMERS_PATH at its
# directory; without it the SDPA fallback in arc_encoder.py is used instead.
XF = Path(os.environ["XFORMERS_PATH"]) if os.environ.get("XFORMERS_PATH") else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = p.parse_args()

    if args.dst.exists() and not args.force:
        print(f"exists, refusing to overwrite: {args.dst}  (use --force)")
        return 0

    sys.path.insert(0, str(MOSHI_RAG))
    if XF is not None and XF.exists():
        sys.path.insert(0, str(XF))

    from moshi.models import loaders
    from moshi.models.lm import LMModel
    from moshi.utils.quantize import replace_linear_with_qlinear

    t0 = time.time()

    kwargs = copy.deepcopy(loaders._lm_kwargs)
    cond_keys_expected = kwargs.pop("conditioners", None) is not None
    kwargs.pop("fuser", None)

    print("building LMModel on cpu (no conditioners)")
    model = LMModel(device="cpu", dtype=torch.bfloat16, **kwargs)

    print(f"reading {args.src}")
    with safe_open(args.src, "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    src_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"  {len(state)} tensors, {src_bytes / 1e9:.2f} GB")

    # Conditioner weights are not part of the stripped model; hold them aside and
    # merge them back after quantization so the output stays a complete
    # checkpoint.
    cond = {k: v for k, v in state.items() if k.startswith("condition_provider.")}
    lm_state = {k: v for k, v in state.items() if not k.startswith("condition_provider.")}
    if cond_keys_expected:
        print(f"  holding aside {len(cond)} conditioner tensors")

    # `in_proj_weight` -> `in_projs.{i}.weight` is handled by a load hook on
    # StreamingMultiheadAttention (transformer.py:398), so load, don't assign.
    missing, unexpected = model.load_state_dict(lm_state, strict=False)
    missing = [k for k in missing if not k.startswith("condition_provider.")]
    if missing or unexpected:
        print(f"  MISSING {len(missing)}: {missing[:5]}")
        print(f"  UNEXPECTED {len(unexpected)}: {unexpected[:5]}")
        if missing or unexpected:
            print("refusing to quantize an incompletely loaded model")
            return 1
    del state, lm_state
    print(f"  loaded clean in {time.time() - t0:.0f}s")

    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"quantizing {n_lin} nn.Linear modules to int8")
    replace_linear_with_qlinear(model)

    out = dict(model.state_dict())
    out.update(cond)
    out = {k: (v.contiguous() if v.is_floating_point() or v.dtype == torch.int8 else v)
           for k, v in out.items()}
    dst_bytes = sum(t.numel() * t.element_size() for t in out.values())
    print(f"  {len(out)} tensors, {dst_bytes / 1e9:.2f} GB "
          f"({100 * dst_bytes / src_bytes:.0f}% of source)")

    print(f"writing {args.dst}")
    save_file(out, str(args.dst))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())