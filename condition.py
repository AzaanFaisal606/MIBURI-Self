"""Build the ``[B, 9, T]`` conditioning stream MIBURI's Gesture LM consumes.

Row 0 is the Moshi text token, rows 1-8 are Mimi audio codes -- one column per
80 ms frame (12.5 Hz).  In the stock demo Moshi produces this stream for free as
a side effect of its inner monologue.  We have to manufacture it, which is
exactly the problem the authors already solved to build their training HDF5s.

So this module is a thin wrapper: the real work is done by ``Interleaver`` and
``InterleavedTokenizer``, imported unchanged from the upstream clone.  Params
are copied verbatim from ``build_hdf5_beatx.py:758`` -- they must match training
or the conditioning silently drifts off-distribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent / "miburi"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trainers.dataloaders.beatx.build_hdf5_beatx import (  # noqa: E402
    _extract_alignments,
    _load_audio,
    _resample_audio,
)
from scripts.trainers.dataloaders.utils.interleaver import (  # noqa: E402
    Interleaver,
    InterleavedTokenizer,
)

MIMI_AUDIO_FPS = 24000  # build_hdf5_beatx.py:1424 default


def load_alignments(json_path: str | Path) -> list[tuple[str, tuple[float, float], str]]:
    """Read a BEATX-style transcript into ``(word, (start, end), speaker)`` tuples.

    Schema (OpenAI Whisper with ``word_timestamps=True``)::

        {"segments": [{"words": [{"word": " The", "start": 0.96, "end": 1.48}, ...]}]}

    Word strings keep their original casing and punctuation -- they go straight
    into SentencePiece, where "Hello", "hello" and "Hello," are different tokens.
    """
    with open(json_path) as f:
        data = json.load(f)
    return _extract_alignments(data["segments"])


def build_tokenizer(mimi, text_tokenizer, device: str | torch.device) -> InterleavedTokenizer:
    """Interleaver with the exact training-time parameters."""
    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        text_padding=3,
        end_of_text_padding=0,
        zero_padding=-1,
        keep_main_only=True,
        device=device,
    )
    return InterleavedTokenizer(mimi, interleaver)


def build_condition(
    wav_path: str | Path,
    transcript_path: str | Path,
    mimi,
    text_tokenizer,
    device: str | torch.device,
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, float]:
    """``(wav, transcript)`` -> ``([1, 9, T]`` conditioning tensor, duration_sec)``.

    Duration is floored to a whole number of Mimi frames so the motion decoder
    gets a clean frame count.
    """
    audio, sr = _load_audio(str(wav_path))
    audio = _resample_audio(audio, sr, MIMI_AUDIO_FPS)
    if audio.shape[0] > 1:  # downmix to mono; Mimi is single-channel
        audio = audio.mean(axis=0, keepdims=True)

    duration = audio.shape[1] / MIMI_AUDIO_FPS
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    # Floor to whole 80 ms frames.
    n_frames = int(duration * mimi.frame_rate)
    duration = n_frames / mimi.frame_rate
    audio = audio[:, : int(duration * MIMI_AUDIO_FPS)]

    alignments = load_alignments(transcript_path)
    tokenizer = build_tokenizer(mimi, text_tokenizer, device)
    text_tokens, audio_tokens = tokenizer(audio, 0.0, alignments, duration)

    # Row 0 text, rows 1-8 audio -- the order GestureLMGen.step slices back apart
    # (gesture_lm.py:846).  Never reorder.
    condition = torch.cat([text_tokens, audio_tokens], dim=1)
    assert condition.shape[1] == 9, f"expected 9 rows, got {condition.shape[1]}"
    return condition.to(device), duration


def describe_condition(condition: torch.Tensor, text_tokenizer) -> str:
    """Human-readable dump of the text row, for eyeballing alignment."""
    text_row = condition[0, 0].tolist()
    special = {3: "<pad>", 0: "<eot>", -1: "<zero>"}
    pieces = []
    for tick, tok in enumerate(text_row):
        if tok in special:
            continue
        piece = text_tokenizer.id_to_piece(int(tok)).replace("▁", " ")
        pieces.append(f"{tick * 0.08:6.2f}s {piece!r}")
    return "\n".join(pieces)
