"""Stage 1 -- engine reproduction.

Drives the Moshi-free gesture engine with BEATX audio and BEATX's own
word-aligned transcript.  No ASR, no renderer.  Answers exactly one question:
is our token plumbing correct?

Ground truth for the same clip sits in ``smplxflame_25/<id>.npz``.  Output is
generative, so it will not match frame-for-frame -- we are checking that the
motion is well-formed, plausible, and beat-aligned, not identical.

    python "MIBURI User/run_stage1.py" \
        --wav      datasets/beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav \
        --transcript datasets/beat_english_v2.0.0/whisper_transcription/1_wayne_0_1_1.json \
        --out      out/stage1_wayne.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition import build_condition, describe_condition  # noqa: E402
from engine import REPO, GestureEngine  # noqa: E402
from paths import default_out, resolve_out  # noqa: E402

DEFAULT_GLM_CONFIG = REPO / "experiments" / "demoexp_release_gtdm3_goodspk" / "config.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--out", default=None,
                   help="bare filename -> output/npz/; a path is used verbatim. "
                        "Defaults to the wav's stem.")
    p.add_argument("--glm-config", default=str(DEFAULT_GLM_CONFIG))
    p.add_argument("--character-id", type=int, default=15, help="15=wayne, 3=lawrence, 4=solomon, 2=stewart")
    p.add_argument("--glm-cfg-coef", type=float, default=1.3)
    p.add_argument("--max-seconds", type=float, default=None, help="truncate input, for quick checks")
    p.add_argument("--seed", type=int, default=2342,
                   help="generation seed; the same seed reproduces the same motion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump-text-stream", action="store_true", help="print the aligned text row")
    args = p.parse_args()

    out = resolve_out(args.out, ".npz") if args.out else default_out(args.wav, ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    engine = GestureEngine(
        glm_config=args.glm_config,
        device=args.device,
        character_id=args.character_id,
        glm_cfg_coef=args.glm_cfg_coef,
        seed=args.seed,
    )
    print(f"[stage1] models loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"[stage1] VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    condition, duration = build_condition(
        wav_path=args.wav,
        transcript_path=args.transcript,
        mimi=engine.mimi,
        text_tokenizer=engine.text_tokenizer,
        device=args.device,
        max_seconds=args.max_seconds,
    )
    print(f"[stage1] condition {tuple(condition.shape)} over {duration:.2f}s "
          f"({condition.shape[2]} ticks @ {engine.mimi.frame_rate} Hz)")

    text_row = condition[0, 0]
    n_words = int((text_row > 3).sum().item())
    print(f"[stage1] text row: {n_words} non-padding tokens")
    if args.dump_text_stream:
        print(describe_condition(condition, engine.text_tokenizer))

    t0 = time.time()
    motion = engine.generate(condition)
    dt = time.time() - t0
    print(f"[stage1] generated {motion.num_frames} frames in {dt:.1f}s "
          f"({motion.num_frames / dt:.1f} fps, realtime = 25)")
    if torch.cuda.is_available():
        print(f"[stage1] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    motion.save_npz(out)
    print(f"[stage1] wrote {out}")

    # Sanity checks -- cheap, and they catch the failure modes that matter.
    print(f"[stage1] poses       {motion.poses.shape}  "
          f"range [{motion.poses.min():+.2f}, {motion.poses.max():+.2f}]")
    print(f"[stage1] expressions {motion.expressions.shape}  "
          f"range [{motion.expressions.min():+.2f}, {motion.expressions.max():+.2f}]")
    print(f"[stage1] trans       {motion.trans.shape}  "
          f"drift {np.linalg.norm(motion.trans[-1] - motion.trans[0]):.3f} m")
    if not np.isfinite(motion.poses).all():
        print("[stage1] WARNING: non-finite values in poses")
    motion_per_frame = np.abs(np.diff(motion.poses, axis=0)).mean()
    print(f"[stage1] mean abs pose delta/frame: {motion_per_frame:.4f} "
          f"(near-zero would mean a frozen avatar)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
