# MIBURI User — project instructions

## What this project is

Extract MIBURI's gesture synthesis engine and drive it with **our own recorded speech**
instead of Moshi's generated replies.

Input: an audio file of a person talking.
Output: an mp4 of a humanoid avatar, chest-up and frontal (Zoom-call framing), performing
speech-synchronised gestures, with the input audio muxed on top.

Realtime is the eventual goal. It is **not** the current goal.

## Current phase: STAGE 3 — textured Mixamo character

Stages 1 and 2 are closed. The renderer works and settled both carried
questions by eye: motion energy is fine at this framing, and our ASR is as good as
BEATX's own transcript. What remains is a textured Mixamo character.

**The blocker is an FBX.** It has to be downloaded from Mixamo by a human behind
an Adobe login — it cannot be scripted and it is not on HuggingFace. Once there
is one: `fit_character.py` builds the bundle, `textures.py` writes the UV
sidecar beside it, and `render.py --backend mixamo --character <slug> --texture
uv` draws it. Both of those tools were rebuilt from their call sites and have
never been run against a real FBX; expect to debug them.

### What exists

Env `miburi` (torch 2.9.0+cu130, sm_120 verified, ffmpeg installed); resolved
package set in `docs/env-pinned-versions.txt`. Run with
`~/miniforge3/envs/miburi/bin/python` **from anywhere** — every tool resolves
its paths from `paths.py`, not from the working directory.

| file | role | status |
| --- | --- | --- |
| `paths.py` | project layout, output routing | works |
| `condition.py` | wav + transcript -> `[1,9,T]` | works |
| `engine.py` | Moshi-free GLM + 3 codecs -> SMPL-X `.npz` | works |
| `run_stage1.py` | CLI (`--seed`, `--glm-cfg-coef`, `--character-id`, `--max-seconds`) | works |
| `align.py` | audio -> BEATX-schema JSON (whisper large-v3) | works |
| `compare_transcripts.py` | ASR vs reference diff | works |
| `beat_align.py` | beat metrics | **insufficient, see §15** |
| `render.py` | `.npz` -> chest-up mp4, smplx + mixamo backends | works |
| `textures.py` | per-submesh UV textures as a sidecar beside a bundle | rebuilt; sidecar round-trips, the no-sidecar fallback is exercised by the mixamo render |
| `fit_character.py` | Mixamo FBX -> character bundle (wraps upstream) | rebuilt, **untested** — needs an FBX |
| `setup_env.sh` | env build, idempotent | done |

```bash
U="Desktop/MIBURI/MIBURI User"
python "$U/align.py"      --wav clip.wav                       # -> output/json/clip.json
python "$U/run_stage1.py" --wav clip.wav --transcript output/json/clip.json
                                                               # -> output/npz/clip.npz
python "$U/render.py"     --npz output/npz/clip.npz --wav clip.wav --translation zero
                                                               # -> output/mp4/clip_smplx.mp4
```
`--out` is optional everywhere; a bare filename lands in the folder for its type, an
explicit path is used verbatim. 60 s clip end to end: ASR 11 s, generate 21 s
(2.9 GB peak VRAM, 113 fps), render 14 s (108 fps). Deterministic given `--seed`.

Re-measured 2026-09-07 on a rebuilt env, 62.6 s clip: generate 14.2 s (110 fps,
**2.97 GB** peak VRAM), render 14.6 s (107 fps). Same `--seed` reproduces the
`.npz` byte for byte; a different one does not. The numbers above stand.

### Carry into Stage 3

- **The 2x energy number is real but not a visible defect** (§16). Head travel is ~1.9x GT
  and `character_id` moves it (speaker 2 = 0.114 m vs 15 = 0.183 m, GT 0.096 m), but on
  screen at chest-up framing 15 and 2 both look fine. Default stays 15. `glm_cfg_coef`
  moves nothing at all. Don't tune against these tables.
- **ASR timing is settled** (§16). Our whisper transcript renders as well-synced as
  BEATX's own; the ~140 ms bias is below threshold. No correction needed.
- **Root translation is zeroed by default** (`--translation zero`). Generated drift is
  0.65 m lateral over 60 s, which walks the avatar out of a chest-up frame.
- **Eyes never move.** Face codec drives jaw (joint 22) + 100 expression coeffs only.
  Architectural, matches upstream. No blinks — very visible at chest-up framing.
- **`y_bot` has no face rig** (no `face_vert_mask`/`expr_dirs_face`/`anatomical_jaw_pivot`).
  Body gestures only. Raw SMPL-X is the only backend with working face today; a textured
  Mixamo character needs `fit_mixamo_character.py --with_face` + headless Blender + an FBX
  a human must download (Adobe login).
- **Never judge conditioning by pairwise motion distance** — it saturates (§15). Watch it.

### Renderer starting points

- `render_smplx_debug_video()` — `scripts/trainers/dataloaders/utils/visualize.py:454`.
  pyrender/EGL -> cv2 -> ffmpeg mux. Takes `camera_pose` (4x4, overrides framing) and
  `only_face`. Builds a per-frame `trimesh` and sets `mesh.visual.vertex_colors`
  (`visualize.py:552`), so the Mixamo swap is a drop-in.
- `pose_mixamo_character()` — `miburi/utils/mixamo_character.py:362`.
- Texture is baked per-vertex RGBA, not UV mapping (`motion_vis_server.py:33`).

## Layout

```
Desktop/MIBURI/
├── miburi/          upstream clone — READ-ONLY, never patch  (pinned at 2a32e03)
│   ├── experiments/     released checkpoints (1.7 GB)
│   ├── assets_dep/      SMPL-X + y_bot + demo static + kyutai cache
│   └── datasets/        BEATX
├── moshi-rag/       clone of kyutai-labs/moshi-rag + our two patches (ASR only)
├── weights/         MoshiRAG + ARC checkpoints (39 GB), for the RAG demo
├── tools/           headless Blender 4.5.12 LTS (1.2 GB), for FBX fitting
├── logs/            setup + download logs
├── MIBURI User/     ← repo: the offline pipeline (this document lives here)
│   ├── docs/
│   └── output/          npz/  mp4/  json/  characters/
└── MIBURI RAG/      ← repo: the realtime MoshiRAG demo, a fork of the clone
```

The two `MIBURI *` directories are **separate git repositories**; everything
else in the tree is downloaded or cloned and is not version-controlled here.

The clone stays pristine. Install it `pip install -e` and import from it. Every
behaviour change lives in `MIBURI User/`.

**Nothing we generate goes inside the clone.** `paths.py` owns the layout; outputs are
sorted by type under `MIBURI User/output/`. Read-only exceptions, all upstream artifacts:
`assets_dep/` (incl. the Mimi/tokenizer download cache) and `experiments/`.

## Hardware constraints — these drive the architecture

- RTX 5070, **12 GB VRAM**. Blackwell, sm_120 → needs the CUDA 13 torch build.
- 31 GB RAM. ~33 GB free disk — watch this, model pulls are large.
- **The stock demo cannot run here.** `gest-server` / `run_gestinference` load Moshi 7B
  and need 24 GB+. Do not try to use them as a working baseline; they are reference
  source only.

## Non-negotiable technical invariants

1. **Never call `CheckpointInfo.from_hf_repo()`.** It calls `hf_get(moshi_name, ...)`
   unconditionally and downloads Moshi's 16 GB `model.safetensors` even when Moshi is
   never loaded. Fetch the two files we need directly with `hf_hub_download`:
   `tokenizer_spm_32k_3.model` and `tokenizer-e351c8d8-checkpoint125.safetensors` (Mimi).

2. **Moshi 7B is not needed at runtime.** The GLM checkpoint carries its own copies of
   Moshi's embedding tables. Construct `GTemporalDepthModel3` with zero-filled tensors of
   the right shape; strict `load_state_dict` overwrites them with the real weights.

3. **The conditioning tensor is `[B, 9, T]`** — row 0 text token, rows 1-8 Mimi codes.
   Never reorder. `glm_gen.step()` slices it as `condition[:, :1]` / `condition[:, 1:]`.

4. **Interleaver params must match training exactly:**
   `text_padding=3, end_of_text_padding=0, zero_padding=-1, keep_main_only=True`,
   `audio_frame_rate=mimi.frame_rate`. Wrong values silently produce wrong gestures.

5. **Reuse, don't reimplement.** `Interleaver` and `InterleavedTokenizer` already do what
   we need. The handoff's "port it verbatim" advice is obsolete.

6. **Motion is 25 fps.** The `.npz` field `mocap_frame_rate=30` is a Blender-addon
   artifact and is wrong. Ignore it.

## Licensing

MIBURI weights CC-BY-NC 4.0; SMPL-X non-commercial. **Research and demo only.** Not a
commercial product.

## Working style for this project

- Verify claims against the source before repeating them. The original handoff
  doc (`docs/handoff(ECA).md`) is gone and is not worth chasing: it was already
  known to carry at least one stale claim, and its "port it verbatim" advice is
  obsolete (invariant 5). `docs/findings.md` survives only in part — it is an
  early-phase snapshot that predates this pipeline, so treat it as background.
- Cite `file:line` for anything load-bearing.
- Use `mamba`/`conda` for environments, not `venv`+`pip`.
