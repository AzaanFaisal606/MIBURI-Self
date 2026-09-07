"""End-to-end check of the MoshiRAG conditioner port inside MIBURI RAG.

Builds MIBURI's own `LMModel` with MoshiRAG's conditioners and fuser, loads the
q8 checkpoint plus the separate ARC encoder, then drives a few generation steps
with a retrieved reference injected mid-stream.

  python "MIBURI User/scripts/test_rag_conditioners.py" --device cpu
"""

import argparse
import sys
import time
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT  # noqa: E402

ROOT = PROJECT
RAG = ROOT / "MIBURI RAG"
# A locally built xformers, if there is one. Point XFORMERS_PATH at its
# directory; without it the SDPA fallback in arc_encoder.py is used instead.
XF = Path(os.environ["XFORMERS_PATH"]) if os.environ.get("XFORMERS_PATH") else None
MOSHI_Q8 = ROOT / "weights/moshika-rag-pytorch-bf16/model.q8.safetensors"
ARC_BF16 = ROOT / "weights/ARC4_Encoder_Llama/model.bf16.safetensors"

REFERENCE = ("Priya owns the rollback plan. The storage migration is postponed "
             "until after the audit closes, and the shard layout stays as it is.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--skip-weights", action="store_true",
                    help="build and wire everything without loading checkpoints")
    args = ap.parse_args()

    sys.path.insert(0, str(RAG))
    if XF is not None and XF.exists():
        sys.path.insert(0, str(XF))
    import os
    os.environ.setdefault("ARC_ENCODER_WEIGHTS", str(ARC_BF16))

    from miburi.models import loaders
    from miburi.models.lm import LMModel, LMGen

    dev = args.device
    t0 = time.time()

    kwargs = loaders.get_rag_lm_kwargs()
    print(f"rag_token_id = {kwargs['rag_token_id']}")
    print(f"fuser        = { {k: v for k, v in kwargs['fuser'].items() if v} }")

    # Same wiring get_moshi_lm does, kept explicit here so the test exercises
    # the conditioner path rather than the checkpoint-download path.
    kwargs["condition_provider"] = loaders.get_conditioner_provider(
        kwargs["dim"], dev, kwargs)
    kwargs.pop("conditioners")
    kwargs["fuser"] = loaders.get_condition_fuser(kwargs)

    print(f"\nbuilding LMModel on {dev}")
    model = LMModel(device=dev, dtype=torch.bfloat16, quantize=not args.skip_weights,
                    **kwargs)
    print(f"  built in {time.time() - t0:.0f}s")
    print(f"  conditioners: {list(model.condition_provider.conditioners)}")
    print(f"  fuser.streaming_sum -> {model.fuser.fuse2cond['streaming_sum']}")
    print(f"  fuser.prepend       -> {model.fuser.fuse2cond['prepend']}")

    if not args.skip_weights:
        from safetensors.torch import load_file
        print(f"\nloading {MOSHI_Q8.name}")
        state = load_file(str(MOSHI_Q8), device=dev)
        missing, unexpected = model.load_state_dict(state, assign=True, strict=False)
        missing = [k for k in missing if not k.startswith("condition_provider.conditioners.reference_with_time.embedder")
                   and not k.startswith("condition_provider.conditioners.reference_with_time.bridge_module")]
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5] or unexpected[:5]:
            print(f"  missing[:5]={missing[:5]}")
            print(f"  unexpected[:5]={unexpected[:5]}")
        del state

        print("loading ARC encoder weights")
        arc = model.condition_provider.conditioners["reference_with_time"]
        arc.load_weights()
        arc.to(dtype=torch.bfloat16)
        print("  ok")

    model.eval()

    # Build the conditions the fuser expects: a speaker tag, and the retrieved
    # reference the ARC encoder compresses.
    from miburi.conditioners.base import ConditionAttributes
    attrs = [ConditionAttributes(
        text={"first_speaker": "SPEAKER_MAIN", "reference_with_time": REFERENCE},
        wav={})]
    prepared = model.condition_provider.prepare(attrs)
    condition_tensors = model.condition_provider(prepared)
    for name, ct in condition_tensors.items():
        print(f"  condition {name}: {tuple(ct.condition.shape)}")

    lm_gen = LMGen(model, condition_tensors=condition_tensors)
    needed = model.num_codebooks - model.dep_q - 1
    print(f"\nrunning {args.steps} steps (user stream = {needed} codebooks)")
    with lm_gen.streaming(1):
        ref = condition_tensors["reference_with_time"].condition[0]
        lm_gen.update_streaming_sum(ref)
        print(f"  injected reference: {tuple(ref.shape)} -> "
              f"{ref.shape[0]} frames of streaming_sum")
        for i in range(args.steps):
            tokens = torch.zeros(1, needed, 1, dtype=torch.long, device=dev)
            out = lm_gen.step(tokens)
            st = lm_gen._streaming_state
            left = 0 if st.pending_streaming_sum is None else st.pending_streaming_sum.shape[0]
            live = st.condition_streaming_sum.abs().sum().item()
            print(f"  step {i}: out={None if out is None else tuple(out.shape)}  "
                  f"pending={left}  active_offset={'yes' if live > 0 else 'no'}")

    print(f"\nOK in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())