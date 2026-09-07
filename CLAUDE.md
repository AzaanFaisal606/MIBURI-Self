# MIBURI Self — working notes

Take MIBURI's gesture synthesis engine and drive it with **our own recorded
speech** instead of Moshi's generated replies. Audio in, chest-up avatar mp4 out,
input audio muxed on top.

Realtime is the eventual goal and lives in the sibling **MIBURI RAG** repo. It is
not the goal here.

## Layout

This repo clones as `MIBURI User` and resolves everything from `paths.py`, not
from the working directory:

```
MIBURI/
├── miburi/          upstream clone — READ-ONLY, never patch  (pinned at 2a32e03)
│   ├── experiments/     released checkpoints
│   ├── assets_dep/      SMPL-X + y_bot + demo static + kyutai cache
│   └── datasets/        BEATX
├── tools/           headless Blender 4.5.12 LTS, for FBX fitting
└── MIBURI User/     ← this repo
    └── output/          npz/  mp4/  json/  characters/
```

**The clone stays pristine.** Install it `pip install -e` and import from it;
every behaviour change lives here. Nothing we generate goes inside it. The only
read-only exceptions are upstream artifacts: `assets_dep/` (including the
Mimi/tokenizer download cache) and `experiments/`.

## What differs from upstream MIBURI

Upstream gestures to *Moshi's reply*. This gestures to *your input*, offline.

- Moshi 7B is never loaded — the GLM checkpoint carries its own copies of the
  embedding tables, which is what makes this fit in 12 GB.
- Own ASR (`align.py`, whisper large-v3) producing BEATX's exact transcript
  schema, so the upstream loader consumes it unchanged.
- A three-command CLI instead of `gest-server`.
- Chest-up frontal rendering with audio muxed, instead of a live viser scene.
- `paths.py` output routing, so the clone never accumulates our artifacts.

## Non-negotiable invariants

1. **Never call `CheckpointInfo.from_hf_repo()`** — it unconditionally pulls
   Moshi's 16 GB `model.safetensors` even when Moshi is never used. Fetch
   `tokenizer_spm_32k_3.model` and the Mimi checkpoint directly with
   `hf_hub_download`.
2. **Moshi 7B is not needed at runtime.** Construct `GTemporalDepthModel3` with
   zero-filled tensors of the right shape; strict `load_state_dict` overwrites
   them.
3. **The conditioning tensor is `[B, 9, T]`** — row 0 text token, rows 1-8 Mimi
   codes. Never reorder.
4. **Interleaver params must match training exactly:** `text_padding=3`,
   `end_of_text_padding=0`, `zero_padding=-1`, `keep_main_only=True`,
   `audio_frame_rate=mimi.frame_rate`. Wrong values fail silently.
5. **Reuse, don't reimplement.** `Interleaver` / `InterleavedTokenizer` already
   do what's needed.
6. **Motion is 25 fps.** The `.npz` says `mocap_frame_rate=30`; it's a
   Blender-addon artifact and it's wrong.

## Settled — don't redo these

- The 2x motion-energy number is real but not a visible defect at this framing.
  Default `character_id` stays 15. `glm_cfg_coef` moves nothing.
- ASR timing is fine; the ~140 ms bias is below threshold.
- Eyes never move — the face codec drives jaw + expression coefficients only.
  Architectural, matches upstream.
- `y_bot` has no face rig. Raw SMPL-X is the only backend with a working face.
- Never judge conditioning by pairwise motion distance; it saturates.

Full detail, with the measured numbers, in `docs/PROJECT.md`.

## Current state

Stages 1 and 2 are closed and the pipeline is verified end to end. Stage 3 —
a textured Mixamo character — is blocked on an FBX a human has to download from
Mixamo behind an Adobe login. `fit_character.py` and `textures.py` are written
but have **never run against a real FBX**; expect to debug them.

## Working style

- Verify against the source before repeating a claim; cite `file:line`.
- Use `conda`/`mamba`, not `venv` + `pip`.
- `docs/findings.md` is an early research snapshot and partly lost. Background
  only — where it disagrees with `docs/PROJECT.md`, PROJECT.md wins.
