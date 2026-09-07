"""Diff our ASR output against BEATX's ground-truth transcript for the same wav.

Stage 2's risk decomposes into three parts (findings §10), and this measures all
three against a reference rather than guessing:

1. **Timing** -- a tick is 80 ms, so errors below ~40 ms quantise away. The metric
   that actually matters is not "seconds off" but whether ``floor(start * 12.5)``
   lands on the same tick, because that is what the Interleaver consumes.
2. **Text style** -- casing, punctuation, digits-vs-words. Different strings mean
   different SentencePiece tokens, i.e. off-distribution conditioning.
3. **Accuracy** -- wrong words, dropped words, hallucinated words.

Usage:
    python "MIBURI User/compare_transcripts.py" \
        --ref datasets/beat_english_v2.0.0/whisper_transcription/1_wayne_0_1_1.json \
        --hyp out/asr_wayne.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import iter_words  # noqa: E402

FRAME_RATE = 12.5  # Mimi ticks per second


def load(path: str | Path) -> list[tuple[str, float, float]]:
    with open(path) as f:
        return [(w, s, e) for w, s, e in iter_words(json.load(f)) if s is not None]


def norm(word: str) -> str:
    """Casing/punctuation-insensitive key, for matching words across transcripts."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True, help="BEATX ground-truth transcript")
    p.add_argument("--hyp", required=True, help="our ASR output")
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()

    ref, hyp = load(args.ref), load(args.hyp)
    if args.max_seconds:
        ref = [w for w in ref if w[1] < args.max_seconds]
        hyp = [w for w in hyp if w[1] < args.max_seconds]

    print(f"words: ref={len(ref)}  hyp={len(hyp)}")

    # --- 3. accuracy: align the two word sequences on normalised forms ---
    rn, hn = [norm(w) for w, _, _ in ref], [norm(w) for w, _, _ in hyp]
    sm = difflib.SequenceMatcher(a=rn, b=hn, autojunk=False)
    pairs: list[tuple[int, int]] = []
    n_eq = n_sub = n_del = n_ins = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            n_eq += i2 - i1
            pairs += [(i1 + k, j1 + k) for k in range(i2 - i1)]
        elif tag == "replace":
            n_sub += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            n_del += i2 - i1
        elif tag == "insert":
            n_ins += j2 - j1
    wer = (n_sub + n_del + n_ins) / max(len(ref), 1)
    print(f"\n--- accuracy ---")
    print(f"matched {n_eq}/{len(ref)}   sub={n_sub} del={n_del} ins={n_ins}   WER={wer:.1%}")

    if not pairs:
        print("no matched words -- transcripts disagree completely")
        return

    # --- 1. timing, on matched words only ---
    d_start = np.array([hyp[j][1] - ref[i][1] for i, j in pairs])
    ref_tick = np.array([int(ref[i][1] * FRAME_RATE) for i, _ in pairs])
    hyp_tick = np.array([int(hyp[j][1] * FRAME_RATE) for _, j in pairs])
    tick_err = hyp_tick - ref_tick

    print(f"\n--- timing (matched words, n={len(pairs)}) ---")
    print(f"start delta   mean {d_start.mean()*1000:+7.1f} ms   "
          f"median {np.median(d_start)*1000:+7.1f} ms   "
          f"|p90| {np.percentile(np.abs(d_start),90)*1000:6.1f} ms")
    print(f"tick error    exact {(tick_err==0).mean():.1%}   "
          f"within±1 {(np.abs(tick_err)<=1).mean():.1%}   "
          f"within±2 {(np.abs(tick_err)<=2).mean():.1%}")
    print(f"              worst {tick_err.min():+d} .. {tick_err.max():+d} ticks "
          f"({tick_err.min()*80:+d} .. {tick_err.max()*80:+d} ms)")

    # --- 2. text style, on matched words ---
    exact = sum(ref[i][0] == hyp[j][0] for i, j in pairs)
    case_only = sum(
        ref[i][0] != hyp[j][0] and ref[i][0].lower() == hyp[j][0].lower() for i, j in pairs
    )
    punct_only = sum(
        ref[i][0].lower() != hyp[j][0].lower()
        and norm(ref[i][0]) == norm(hyp[j][0])
        for i, j in pairs
    )
    print(f"\n--- text style (matched words) ---")
    print(f"byte-identical {exact}/{len(pairs)} ({exact/len(pairs):.1%})   "
          f"casing-only diff {case_only}   punctuation-only diff {punct_only}")
    shown = 0
    for i, j in pairs:
        if ref[i][0] != hyp[j][0] and shown < 8:
            print(f"    ref {ref[i][0]!r:20s} hyp {hyp[j][0]!r}")
            shown += 1

    print(f"\n--- verdict ---")
    if wer < 0.05 and (np.abs(tick_err) <= 1).mean() > 0.9:
        print("ASR is close enough to BEATX -- proceed to motion A/B")
    else:
        print("gap is material -- inspect before trusting on our own audio")


if __name__ == "__main__":
    main()
