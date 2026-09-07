# MIBURI Self

Gesture synthesis driven by **your own recorded speech**.

Audio of someone talking goes in; an mp4 comes out of a humanoid avatar,
chest-up and frontal like a Zoom call, gesturing in time with the speech, with
the input audio muxed on top.

> Clones to a directory named `MIBURI User` — that's the name `paths.py` and the
> rest of the code expect.

## What's different from MIBURI

Upstream [MIBURI](https://github.com/m-hamza-mughal/miburi) is a live demo: Moshi
7B generates a spoken *reply* to you, and the avatar gestures to **Moshi's
reply**. This drives the same gesture engine from **your input audio** instead.

- **No Moshi at runtime.** The gesture LM checkpoint already carries the
  embedding tables it needs, so Moshi 7B is never loaded. That's what makes it
  fit on a 12 GB card — the stock demo needs 24 GB+.
- **Three commands, not a server.** Transcribe, generate, render.
- **Own transcripts**, from whisper large-v3, in the schema upstream's loader
  already reads.
- **Chest-up framing** with the audio muxed on, rather than a live viewer.
- **Nothing is written into the clone.** `paths.py` owns the layout and sorts
  outputs by type; the clone stays pristine and importable.
- Root translation is **zeroed by default** — the generated drift is 0.65 m over
  a minute, which walks the avatar out of frame.

## The pipeline

Every tool resolves its paths from `paths.py`, so these run from anywhere.

```bash
U="MIBURI User"
python "$U/align.py"      --wav clip.wav                        # -> output/json/clip.json
python "$U/run_stage1.py" --wav clip.wav --transcript output/json/clip.json
                                                                # -> output/npz/clip.npz
python "$U/render.py"     --npz output/npz/clip.npz --wav clip.wav --translation zero
                                                                # -> output/mp4/clip_smplx.mp4
```

`--out` is optional everywhere: a bare filename lands in the folder for its type,
a real path is used as-is.

A 60 s clip on an RTX 5070: ASR 11 s, generate 14 s (2.97 GB peak VRAM, 110 fps),
render 15 s (107 fps). Same `--seed` gives the same motion, byte for byte.

## Setup

Needs Python 3.12, an NVIDIA card and ffmpeg. Sits next to a read-only clone of
upstream MIBURI:

```bash
mkdir -p ~/MIBURI && cd ~/MIBURI
git clone https://github.com/m-hamza-mughal/miburi.git miburi
git -C miburi checkout 2a32e03                # the commit this was built against
git clone https://github.com/AzaanFaisal606/MIBURI-Self.git "MIBURI User"

bash "MIBURI User/setup_env.sh"                # conda env, torch cu130, editable install
pip install openai-whisper                     # align.py only

cd miburi && miburi-download-assets && miburi-download-checkpoints
```

SMPL-X needs its own registration; the release assets bundle covers it.

The tree the code expects:

```
MIBURI/
├── miburi/          upstream clone — READ-ONLY
├── tools/           headless Blender, only for FBX fitting
└── MIBURI User/     ← this repo
    └── output/          npz/  mp4/  json/  characters/
```

## Tools

| | |
| --- | --- |
| `align.py` | audio → word-aligned transcript (whisper large-v3) |
| `run_stage1.py` | wav + transcript → SMPL-X motion `.npz` |
| `render.py` | `.npz` → chest-up mp4; `smplx` and `mixamo` backends |
| `fit_character.py` | Mixamo FBX → character bundle |
| `textures.py` | per-submesh UV textures as a sidecar beside a bundle |
| `compare_transcripts.py` | ASR vs reference diff |
| `scripts/` | quantization + download helpers for the sibling RAG demo |

## Hardware

RTX 5070, 12 GB, Blackwell sm_120 — needs the CUDA 13 torch build. 31 GB RAM.
Resolved package versions in `docs/env-pinned-versions.txt`.

## Docs

`docs/PROJECT.md` is the real reference: current phase, the non-negotiable
technical invariants, and the empirical findings worth not re-deriving. Read it
before changing anything.

## Related

[MIBURI-RAG](https://github.com/AzaanFaisal606/MIBURI-RAG) — the realtime,
dialogue-driven half, with MoshiRAG and retrieval.

## Licensing

MIBURI weights are CC-BY-NC 4.0; SMPL-X is non-commercial. Research and demo
only.
