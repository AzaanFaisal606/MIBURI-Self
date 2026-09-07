# MIBURI User

**Gesture synthesis driven by your own recorded speech.**

Audio of a person talking goes in; an mp4 comes out of a humanoid avatar,
chest-up and frontal — Zoom-call framing — performing speech-synchronised
gestures with the input audio muxed on top.

This is the offline half of the project. It takes
[MIBURI](https://github.com/m-hamza-mughal/miburi)'s gesture synthesis engine
and drives it with **our own speech** instead of Moshi's generated replies.
Stock upstream does something different: Moshi 7B generates a spoken *reply* to
your audio and MIBURI gestures to **Moshi's reply**, not to your input. The
whole point here is redirecting that.

Realtime is the eventual goal; the realtime, dialogue-driven half lives in the
sibling **MIBURI RAG** repo. It is not the goal here.

## The pipeline

Three commands. Every tool resolves its paths from `paths.py`, not from the
working directory, so they run from anywhere.

```bash
U="MIBURI User"
python "$U/align.py"      --wav clip.wav                        # -> output/json/clip.json
python "$U/run_stage1.py" --wav clip.wav --transcript output/json/clip.json
                                                                # -> output/npz/clip.npz
python "$U/render.py"     --npz output/npz/clip.npz --wav clip.wav --translation zero
                                                                # -> output/mp4/clip_smplx.mp4
```

`--out` is optional everywhere: a bare filename lands in the folder for its
type, an explicit path is used verbatim.

On an RTX 5070, a 60 s clip end to end: ASR 11 s, generate 21 s (2.9 GB peak
VRAM, 113 fps), render 14 s (108 fps). Deterministic given `--seed`.

| file | role |
| --- | --- |
| `paths.py` | project layout, output routing |
| `align.py` | audio → BEATX-schema JSON (whisper large-v3) |
| `condition.py` | wav + transcript → `[1, 9, T]` conditioning tensor |
| `engine.py` | Moshi-free GLM + 3 codecs → SMPL-X `.npz` |
| `run_stage1.py` | generation CLI |
| `render.py` | `.npz` → chest-up mp4; `smplx` and `mixamo` backends |
| `fit_character.py` | Mixamo FBX → character bundle (wraps upstream's fitter) |
| `textures.py` | per-submesh UV textures as a sidecar next to a bundle |
| `_blender_extract_textures.py` | Blender-side helper for the above |
| `compare_transcripts.py` | ASR vs reference diff |
| `beat_align.py` | beat metrics — insufficient, see `docs/findings.md` |
| `setup_env.sh` | env build, idempotent |
| `scripts/quantize_moshi_q8.py` | MoshiRAG bf16 -> int8, for the sibling RAG demo |
| `scripts/cast_arc_bf16.py` | ARC encoder fp32 -> bf16, ditto |
| `scripts/test_rag_conditioners.py` | checks the RAG conditioner wiring, no weights needed |
| `scripts/dl_stt.sh` | resumable pull of the streaming ASR model |

## Layout

This repo is **one directory inside a working tree**, not the whole thing.
`paths.py` resolves everything relative to its own location, and expects:

```
MIBURI/
├── miburi/          upstream clone — READ-ONLY, never patch
│   ├── experiments/     released checkpoints (1.7 GB)
│   ├── assets_dep/      SMPL-X + y_bot + demo static + kyutai cache
│   └── datasets/        BEATX
├── tools/           headless Blender 4.5.12 LTS, for FBX fitting
├── logs/            setup + download logs
└── MIBURI User/     ← this repo
    └── output/          npz/  mp4/  json/  characters/
```

So clone it as `MIBURI User` next to a clone of upstream MIBURI:

```bash
mkdir -p ~/Desktop/MIBURI && cd ~/Desktop/MIBURI
git clone https://github.com/m-hamza-mughal/miburi.git miburi
git -C miburi checkout 2a32e03            # the commit this was built against
git clone <this repo> "MIBURI User"
bash "MIBURI User/setup_env.sh"           # conda env + torch cu130 + editable install
```

**The clone stays pristine.** Install it `pip install -e` and import from it;
every behaviour change lives here. Nothing we generate goes inside it — outputs
are sorted by type under `MIBURI User/output/`. The read-only exceptions are all
upstream artifacts: `assets_dep/` (including the Mimi/tokenizer download cache)
and `experiments/`.

Then fetch the assets:

```bash
miburi-download-assets          # ~200 MB -> assets_dep/
miburi-download-checkpoints     # ~1.7 GB -> experiments/
```

SMPL-X needs its own registration.

## Hardware

RTX 5070, **12 GB VRAM**, Blackwell sm_120 → needs the CUDA 13 torch build
(`torch 2.9.0+cu130`; resolved versions in `docs/env-pinned-versions.txt`).
31 GB RAM.

**The stock demo does not run on this card.** `gest-server` and
`run_gestinference` load Moshi 7B and need 24 GB+. They are reference source,
never a working baseline — which is exactly why this tree exists.

## Docs

- [`docs/PROJECT.md`](docs/PROJECT.md) — project instructions, current phase, and
  the non-negotiable technical invariants. **Read this first.**
- [`docs/findings.md`](docs/findings.md) — early research notes. Partial and
  predates the working pipeline; background only.
- `docs/env-pinned-versions.txt` — the resolved package set of the build that
  produced the timings above.

## Licensing

MIBURI weights are CC-BY-NC 4.0; SMPL-X is non-commercial. **Research and demo
only.** Not a commercial product.
