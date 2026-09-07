"""Stage 2 -- audio in, word-aligned transcript out, in BEATX's exact schema.

The Gesture LM needs one text token per 80 ms tick, placed on the tick where its
word starts (see `condition.py`).  BEATX supplied that for Stage 1; for our own
recordings we have to produce it.

We use **openai-whisper** with `word_timestamps=True` rather than faster-whisper
or WhisperX, deliberately:

* BEATX's transcripts are openai-whisper's own segment dicts -- `seek`,
  `avg_logprob`, `compression_ratio`, `no_speech_prob`, and a per-word
  `probability` field.  WhisperX emits `score` instead and re-aligns with
  wav2vec2, so its word strings and timings come from a different distribution.
  Matching the training pipeline matters here (findings §5).
* it runs on plain torch, inheriting the verified cu130/sm_120 build.
  faster-whisper goes through CTranslate2 with its own bundled CUDA.
* we are offline, so its speed disadvantage costs nothing.

Output is written as the same JSON the BEATX loader reads, so
`_extract_alignments` consumes it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "large-v3"


def transcribe(
    wav_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    language: str | None = "en",
    _cache: dict[str, Any] = {},
) -> dict[str, Any]:
    """Transcribe with word-level timestamps. Returns whisper's raw result dict.

    Whisper keeps casing and punctuation, and prefixes each word with a space
    (" The"). `_extract_alignments` strips that, and the cased/punctuated form is
    what goes into SentencePiece -- do not normalise it here.
    """
    import whisper

    if model_name not in _cache:
        _cache[model_name] = whisper.load_model(model_name, device=device)
    model = _cache[model_name]

    return model.transcribe(
        str(wav_path),
        word_timestamps=True,
        language=language,
        # Greedy + whisper's own fallback ladder, matching stock defaults.
        verbose=False,
    )


def save_transcript(result: dict[str, Any], out_path: str | Path) -> Path:
    """Write whisper's result as BEATX-schema JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": result.get("text", ""),
        "segments": result["segments"],
        "language": result.get("language", "en"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    return out_path


def transcribe_to_file(
    wav_path: str | Path,
    out_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    language: str | None = "en",
) -> Path:
    return save_transcript(transcribe(wav_path, model_name, device, language), out_path)


def iter_words(transcript: dict[str, Any]):
    """Yield ``(word, start, end)`` across all segments, whitespace stripped."""
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if not text:
                continue
            yield text, w.get("start"), w.get("end")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import default_out, resolve_out  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True)
    p.add_argument("--out", default=None,
                   help="bare filename -> output/json/; a path is used verbatim. "
                        "Defaults to the wav's stem.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = resolve_out(args.out, ".json") if args.out else default_out(args.wav, ".json")
    path = transcribe_to_file(args.wav, out, args.model, args.device)
    with open(path) as f:
        n = sum(len(s.get("words", [])) for s in json.load(f)["segments"])
    print(f"[align] wrote {path} ({n} words)")
