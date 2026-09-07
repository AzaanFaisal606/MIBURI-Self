# Findings — Extracting MIBURI's Gesture Synthesis Engine

> **Historical.** This is the research record from before any code existed, and it is
> also **incomplete** — roughly the first half of the original survives. Where it
> disagrees with `PROJECT.md`, `PROJECT.md` wins: the pipeline it plans for was
> subsequently built, measured, and in places contradicted. Kept for the reasoning,
> not as a current record.

Research record. Everything here was verified against the cloned source and the downloaded
checkpoints unless explicitly marked as an assumption. Status: **research complete enough
to plan; no code written.**

---

## 1. Goal

Feed MIBURI a recorded audio file of a person speaking; get back an mp4 of a humanoid
avatar gesturing in sync, framed chest-up and frontal like a Zoom call, with the input
audio muxed on.

The stock repo does something different: Moshi 7B *generates a spoken reply* to your
audio, and MIBURI gestures to **Moshi's reply**, not to your input. Everything below is
about redirecting that.

---

## 2. How MIBURI actually works

Four models in a chain:

| Component | Role | Size |
| --- | --- | ---: |
| **Mimi** (Kyutai) | Audio codec. 24 kHz wav → 8 discrete codes per 80 ms frame (12.5 Hz) | ~300 MB |
| **Moshi 7B** (Kyutai) | Conversational speech LM. Generates a spoken reply | ~16 GB |
| **Gesture LM** (`GTemporalDepthModel3`, "GTDM3") | Consumes text+audio tokens, autoregressively emits 20 gesture codebook tokens per frame | 590 MB |
| **3 gesture codecs** | RVQ decoders: upper+hands (8 codebooks), lower+trans (8), face+expression (4) | ~270 MB |

Output is SMPL-X motion at 25 fps: 55 joint rotations, 100 facial expression
coefficients, root translation. Not video, not a mesh — parameters.

Frame-rate arithmetic, which matters later:
- Mimi frame rate = 12.5 Hz (`loaders.py:32`)
- Gesture codec frame rate = `motion_fps / frame_chunk_size` = 25 / 2 = 12.5 Hz
  (`gesture_codec.py:86`, config `frame_chunk_size: 2`, `motion_fps: 25`)
- Therefore `query2mem_scale = int(12.5 / 12.5) = 1` — **one GLM step per Mimi frame**,
  each emitting 2 motion frames.

---

## 3. Key finding — Moshi 7B is not needed. Confirmed at the tensor level.

The handoff asserted this; it is now proven rather than inferred.

The Gesture LM consumes **discrete tokens**, not Moshi hidden states
(`gest-server.py:375`, `run_gestinference.py:355`):

```python
gmoshi_tokens = self.glm_gen.step(moshi_tokens, ca_query_padding_mask=...)
```

where `moshi_tokens` is `[B, 9, T]` — row 0 a text token, rows 1-8 Mimi audio codes.
`GestureLMGen.step` (`gesture_lm.py:846`) splits it right back apart:

```python
audio_emb = condition[:, 1:]   # [B, 8, S]
text_emb  = condition[:, :1]   # [B, 1, S]
```

The GLM embeds these with **its own** tables. At construction those tables are seeded from
Moshi's (`run_gestinference.py:557`), but the released checkpoint carries its own copies —
read directly from the safetensors header of
`experiments/demoexp_release_gtdm3_goodspk/last_6500.safetensors`:

```
module.text_procemb.weight       BF16 [32001, 4096]
module.audio_procemb.{0..7}.weight  BF16 [2049, 4096]   ×8
```

≈ 400 MB of the 590 MB checkpoint. `load_checkpoints` uses a strict
`model.load_state_dict(states)` (`motion_utils.py:1896`), so these **overwrite** whatever
was passed at construction.

**Consequence:** Moshi 7B is needed only to supply correctly-shaped tensors at
construction time. Zero-filled tensors of shape `[32001, 4096]` and 8× `[2049, 4096]` work
identically. This drops ~16 GB of VRAM and brings the pipeline to roughly **1.5 GB peak**
— Mimi + GLM + three codecs. Comfortable on a 12 GB card.

`textaudio_emb_freeze: true` in the config confirms these tables were frozen during GLM
training, i.e. they are literally Moshi's weights, stored in our checkpoint.

### 3a. Download trap

`CheckpointInfo.from_hf_repo()` (`loaders.py:412`) calls `hf_get(moshi_name, hf_repo)`
unconditionally — it pulls Moshi's 16 GB `model.safetensors` even if `get_moshi()` is never
called. Bypass it. Fetch only:

- `tokenizer_spm_32k_3.model` (~800 KB) — SentencePiece text tokenizer
- `tokenizer-e351c8d8-checkpoint125.safetensors` (~300 MB) — Mimi

both from `kyutai/moshiko-pytorch-bf16`.

---

## 4. There is already a reference implementation of the path we want

`scripts/trainers/uflgtdm3_trainer.py:1423` — the trainer's own `test()` loop **already
bypasses Moshi**:

```python
audio_text_codes = torch.cat((text_codes, audio_codes), dim=1)   # B x K=9 x T
...
gmoshi_tokens = glmgen.step(
    condition=audiotext_chunk,
    ca_query_padding_mask=lower_cross_attn_mask if self.args.drop_lower_crossattn else None,
)
```

It feeds ground-truth text + audio tokens straight into the GLM, chunked by
`query2mem_scale`, then decodes through the three codecs. This is our target architecture,
written by the authors, sitting in the repo. Read `uflgtdm3_trainer.py:1440-1580` before
writing anything.

---

## 5. The conditioning stream — the actual hard part

The GLM was trained on Moshi's "inner monologue" format: **one text token per 80 ms
frame**, aligned to the audio, with four special padding tokens.

The spec lives in `scripts/trainers/dataloaders/utils/interleaver.py:176-215`
(`Interleaver.build_token_stream`). Roughly:

- default fill is `text_padding`
- on the frame where a word starts, emit that word's first SentencePiece token
- subsequent tokens of a multi-token word go on subsequent frames
- frames inside a word's span, after its tokens are exhausted, get `in_word_padding`
- the frame *before* a word's first token gets `end_of_text_padding`
- `zero_padding` (-1) means "use a zero embedding instead of a real one"

**Correction to the handoff:** it says "port it verbatim." That is unnecessary.
`Interleaver` and `InterleavedTokenizer` are self-contained, importable classes with no
heavy dependencies (json, math, numpy, sentencepiece, torch). `InterleavedTokenizer.__call__`
already does exactly the job — wav + alignments → `(text_tokens, audio_tokens)`,
including `mimi.encode()`. Reuse it.

### Exact instantiation params (from `build_hdf5_beatx.py:758`)

```python
Interleaver(
    text_tokenizer,
    mimi.frame_rate,          # 12.5
    text_padding=3,
    end_of_text_padding=0,
    zero_padding=-1,
    keep_main_only=True,
    device=device,
)
```

Mimi is set to 8 codebooks (`build_hdf5_beatx.py:754`, default `--mimi_codebooks 8`).

### Alignment input schema

`_extract_alignments` (`build_hdf5_beatx.py:428`) reads WhisperX-style JSON:

```json
{"segments": [{"words": [{"word": "Hello", "start": 0.42, "end": 0.68}, ...]}]}
```

and emits `(word_text, (start, end), "SPEAKER_MAIN")`. Whatever ASR we use must produce
this shape. Missing `start`/`end` fall back to the previous word's values.

**Casing and punctuation matter.** BEATX's transcripts are described as carrying "casing +
punctuation". The word text goes straight into SentencePiece (`interleaver.py:41`), and
`"Hello,"`, `"Hello"`, and `"hello"` produce different token IDs. Any normalisation we do
for alignment purposes must not leak into the text handed to the tokenizer.

---

## 6. BEATX — what it is and why it matters to us

BEAT2-English: 23 speakers of motion-captured **standing monologue**, retargeted to
SMPL-X, with audio and transcripts. MIBURI's fork adds 25 fps motion and the Whisper
transcriptions.

We do **not** need the 16.6 GB dataset. Individual files are fetchable from
`m-hamza-mughal/beat2-additional-annotations` (dataset repo, 17,978 files). Verified
present:

```
beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav
beat_english_v2.0.0/whisper_transcription/1_wayne_0_1_1.json      ← word-level start/end
beat_english_v2.0.0/whisper_transcription/1_wayne_0_1_1.TextGrid
beat_english_v2.0.0/smplxflame_25/1_wayne_0_1_1.npz               ← ground-truth motion
```

Two uses for us:

1. **A correctness anchor.** Their transcripts are already word-aligned, so we can drive
   the engine at full fidelity with zero alignment code, and compare against ground-truth
   motion. This is our *only* baseline — the stock demo needs 24 GB and cannot run here.
2. **Evidence about their ASR pipeline.** Transcripts ship as both `.json` and
   `.TextGrid`. TextGrid is Montreal Forced Aligner's native format, which strongly
   suggests transcribe-then-force-align rather than raw Whisper timestamps.

`wayne` is a good first sample: `character_id 15`, the default speaker in
`run_gestinference.py:515`, and the sentinel file the repo's own downloader checks
(`download_beatx_dataset.py:51`).

---

## 7. Rendering — what exists and what doesn't

### The offline path (what we want)

`render_smplx_debug_video()` — `scripts/trainers/dataloaders/utils/visualize.py:454`.
pyrender/EGL headless → cv2 VideoWriter → ffmpeg audio mux. 640×480 @ 25 fps, checkerboard
floor, single flat mesh colour `(36, 73, 156, 255)`.

Two knobs already relevant to us:
- `camera_pose: Optional[np.ndarray]` — caller-supplied 4×4 overrides framing entirely.
  Chest-up frontal is a matrix change, not new code.
- `only_face: bool` — tight head framing, built for face-codec debugging.

Per frame it builds a `trimesh.Trimesh` and assigns `mesh.visual.vertex_colors`
(`visualize.py:552-556`) — which makes the Mixamo swap a drop-in.

### The avatar question

**What the demo actually ships:** `scripts/miburi-demo` launches with no
`--mixamo-character` flag; `gest-server.py:764` defaults it to `None`. So the demo is
**raw SMPL-X**, painted pastel blue (`motion_vis_server.py:76`,
`[80/255, 150/255, 250/255, 1.0]`), in the Viser browser viewer. The textured woman in
`assets/MIBURI_TEASER.png` is a paper figure — no code in this repo produces it. The photo
of the live demo in the same image shows the flat blue figure.

**Face animation works on SMPL-X and only on SMPL-X.** The face codec decodes to 6
jaw/eye rotation values + 100 expression coefficients, fed straight into the SMPL-X
forward pass. Fully wired in both render paths.

**`y_bot` cannot move its face.** Bundle contents, read from
`assets_dep/mixamo_characters_release/y_bot.npz`:

```
verts_tpose    (27850, 3)     lbs_weights  (27850, 55)
uv_coords      (27850, 2)     vertex_colors (27850, 4)  ← present
face_vert_mask ✗   expr_dirs_face ✗   anatomical_jaw_pivot ✗
```

It was fit **without** `--with_face`. `load_mixamo_character` (`mixamo_character.py:105`)
treats those as optional, so it loads fine and silently renders a frozen face.

**Why they didn't care:** their demo is a full-body standing figure viewed from across a
room. A face is a few pixels there. Our chest-up framing inverts that priority completely.

### Texture mechanism

Not UV texture mapping — **baked per-vertex RGBA**. The Blender exporter samples each
submesh's texture at each vertex's UV and stores colours in the bundle
(`_blender_fbx_export.py:744`). There is a load-time fallback that bakes from
`uv_coords` + `texture_png` (`motion_vis_server.py:33`, `_sample_texture_at_uvs`).

Trade-off: colour resolution equals vertex density, not texture resolution. Fine at
Zoom-window size, soft on a close-up. Real UV texturing is possible in pyrender via
`trimesh.visual.TextureVisuals`; the repo just doesn't do it.

### Making a textured character

`scripts/fit_mixamo_character.py` — one-time, offline, per character:

1. Headless Blender imports the FBX, extracts the rest-pose mesh
2. Two-phase chamfer Adam fits SMPL-X (betas, global R/t, per-axis scale, A-pose) to it
3. Transfers SMPL-X per-vertex LBS weights onto the Mixamo verts by barycentric projection
4. `--with_face`: K-NN-transfers FLAME expression blendshapes onto the Mixamo face verts,
   plus jaw skinning and an anatomical jaw pivot (`fit_mixamo_character.py:699+`)
5. Writes the runtime `.npz`

`--with_face` is **fully implemented** — the module docstring calling it a "v2
placeholder" is stale. But it has never been run in this repo's released output, so it is
the least-proven code path we depend on.

At runtime `pose_mixamo_character()` (`mixamo_character.py:362`) takes the same SMPL-X
joint rotations and skins the Mixamo mesh instead.

**Requires from us:** a headless Blender install and a Mixamo FBX (Adobe login needed —
must be downloaded by a human).

---

## 8. What is on disk

Downloaded and verified.

`experiments/` (1.7 GB):

| Directory | Role | Latest ckpt |
| --- | --- | --- |
| `demoexp_release_gtdm3_goodspk/` | Gesture LM, 4-speaker subset — **demo default** | `last_6500.safetensors` |
| `demoexp_release_uppercodec/` | Upper + hands codec | `last_180.safetensors` |
| `demoexp_release_lowercodec/` | Lower + trans-velocity codec | `last_440.safetensors` |
| `demoexp_release_facecodec/` | Face / expression codec | `last_100.safetensors` |
| `allspk_release_gtdm3_exp/` | Gesture LM, 23 speakers | `last_1720.safetensors` |
| `allspk_release_{upper,lowertrans,face}codec/` | All-speaker codec variants | — |

`assets_dep/` (169 MB): `smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz`,
`mixamo_characters_release/y_bot.npz`, `demo-static/`.

Notable values from `demoexp_release_gtdm3_goodspk/config.yaml`:

```yaml
frame_chunk_size: 2        motion_fps: 25       num_frames: 250
drop_lower_crossattn: true vad_guidance: true   textaudio_emb_freeze: true
dataset_ratio: goodspk_beatx_lowervalid
deps_path: assets_dep/
```

Speaker IDs for the goodspk experiment: `lawrence: 3, solomon: 4, wayne: 15, stewart: 2`
(`run_gestinference.py:515`). Default `--glm-cfg-coef 1.3`.

---

## 9. Corrections to `handoff(ECA).md`

| Handoff claim | Reality |
| --- | --- |
| "Port `interleaver.py` verbatim" | Unnecessary. `Interleaver` / `InterleavedTokenizer` are importable and reusable as-is. |
| "Rendering — already exists" | True for flat SMPL-X. **No offline textured renderer exists** — the textured path lives only in the Viser browser viewer. |
| "optional Mixamo `y_bot`" (implied as a look upgrade) | `y_bot` has no face rig. Body gestures only. |
| Domain mismatch listed as a risk | **Confirmed, not hypothetical.** `dataset_ratio: goodspk_beatx_lowervalid` filters BEATX only; the seated dyadic `embody3d` data never enters the released checkpoints despite the config referencing its cache path. |
| "Moshi 7B is not needed" | Correct, and now verified at the tensor level (§3). |

Everything else in the handoff held up.

---

## 10. Proposed implementation plan

Four modules in `MIBURI User/`, each independently testable. The clone stays read-only.

**`align.py`** — (wav, transcript-or-nothing) → `[{word, start, end}]` in BEATX's schema, so
`_extract_alignments` consumes it unchanged. Stage 2 only.

**`condition.py`** — (wav, alignments) → `[B, 9, T]`. Thin wrapper over the repo's
`Interleaver` + `InterleavedTokenizer` with the §5 params.

**`engine.py`** — the Moshi-free core. Builds `GTemporalDepthModel3` with zero-filled
embedding tables, loads GLM + 3 codec checkpoints, steps one Mimi frame at a time
(`query2mem_scale = 1`), decodes, emits SMPL-X `.npz`. Modelled on
`uflgtdm3_trainer.py:1440-1580`.

**`render.py`** — `.npz` → mp4. Chest-up frontal camera, audio muxed. Two backends behind
one interface: SMPL-X (full face, flat colour) and Mixamo (textured, swappable character).

### Staging

**Stage 1 — engine only, borrowed speech.** BEATX audio + BEATX's own word-aligned
transcript, straight through `condition.py` → `engine.py`. No ASR code, no renderer.
Answers exactly one question: *is our token plumbing correct?* Ground-truth `.npz` is the
reference.

**Stage 2 — our speech.** ASR in-program: audio → transcript → word timestamps → same
path. Both real risks land here — ASR timing precision, and whether a standing-podium
model does anything sane with seated conversational audio.

Candidates: **WhisperX** (faster-whisper + wav2vec2 forced alignment; closest match to how
BEATX's own transcripts were made) or **easytranscriber** (same idea, GPU Viterbi, 35-100%
faster). Default to WhisperX for distribution match.

**Stage 3 — renderer.** Chest-up frontal framing, both avatar backends, character
swappable.

Property worth preserving: stage 1 touches neither ASR nor renderer, so a failure there is
unambiguously our plumbing.

---

## 11. Risks, ordered by how much they would hurt

1. **Domain mismatch — confirmed.** Released checkpoints are BEATX-only: standing speakers
   making large podium gestures. We want seated, chest-up, conversational. Validate on
   real conversational audio at stage 1, before building anything on top of it.
2. **Face quality under close framing — unmeasured.** The face codec drives SMPL-X
   expression coefficients. "Expressive" is not "viseme-accurate", and their demo never
   showed a face at our scale. Judge it before committing to the framing.
3. **`--with_face` Mixamo fit — never run.** The most load-bearing untested code path if
   we want a textured avatar that can talk.
4. **Idle / silence behaviour — untested.** May go degenerate. Real recordings have pauses.
5. **Casing/punctuation drift** between our ASR output and BEATX's transcript style,
   shifting token IDs off the training distribution.

---

## 12. Open items

- Mixamo FBX must be downloaded by a human (Adobe login).
- Headless Blender install needed before any textured-character work.
- Environment not yet built: conda/mamba env, Python 3.12, torch 2.9.0 **cu130** (RTX 5070
  is Blackwell/sm_120), then `pip install -e` the clone.
- Choice between `demoexp_release_gtdm3_goodspk` and `allspk_release_gtdm3_exp` is
  empirical. Start with goodspk — it matches the demo defaults and is the known-good path.
- Whether `character_id` (speaker style) meaningfully damps podium-scale gesturing is
  worth a sweep at stage 1.
