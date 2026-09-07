"""Moshi-free MIBURI gesture engine.

The stock pipeline runs Moshi 7B to generate a spoken reply, then gestures to
*that*.  We want gestures for our own audio, so Moshi is cut out entirely.

That is possible because the Gesture LM consumes discrete tokens, not Moshi
hidden states, and because the released GLM checkpoint carries its own copies of
Moshi's embedding tables::

    module.text_procemb.weight          BF16 [32001, 4096]
    module.audio_procemb.{0..7}.weight  BF16 [ 2049, 4096]

``load_checkpoints`` uses a strict ``load_state_dict``, so tables passed at
construction are overwritten by the real weights.  We therefore pass zeros of
the right shape and never touch Moshi -- dropping ~16 GB of VRAM.

Structure follows ``run_gestinference.py`` (model wiring) and
``uflgtdm3_trainer.py:1440-1580`` (the generation loop, which already bypasses
Moshi for evaluation).
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent / "miburi"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import smplx  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from miburi.models import GestureLMGen  # noqa: E402
from miburi.models.loaders import CheckpointInfo  # noqa: E402
from miburi.utils import rotation_conversions as rc  # noqa: E402
from miburi.utils.motion_utils import (  # noqa: E402
    get_gesture_condition_tensors,
    get_smplx_bodypart_masks,
    inverse_selection_smplx,
    velocity2position_mixeddiff,
)

MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
KYUTAI_CACHE = str(REPO / "assets_dep" / "kyutai_cache")

# Shapes of Moshi's embedding tables (loaders._lm_kwargs: dim=4096,
# text_card=32000, card=2048; each table carries one extra row for the
# null/pad token).  Values are irrelevant -- the checkpoint overwrites them.
TEXT_EMB_SHAPE = (32001, 4096)
AUDIO_EMB_SHAPE = (2049, 4096)
N_AUDIO_CODEBOOKS = 8

# SMPL-X joints per codec, in the order the GLM emits them.
MOTION_JOINTS_PER_CODEC = (43, 9, 1)  # upper+hands, lower, face


def _checkpoint_info_without_moshi() -> CheckpointInfo:
    """Build a CheckpointInfo that never pulls Moshi's 16 GB weights.

    ``CheckpointInfo.from_hf_repo`` calls ``hf_get(moshi_name, ...)``
    unconditionally (loaders.py:412), downloading ``model.safetensors`` even when
    ``get_moshi()`` is never called.  It is a plain dataclass, so we construct it
    directly with only the two files we actually need.

    ``lm_config=None`` makes ``get_mimi`` use 8 codebooks (loaders.py:456),
    matching ``--mimi_codebooks 8`` in the dataset builders.
    """
    tokenizer = hf_hub_download(MOSHI_REPO, TEXT_TOKENIZER_NAME, cache_dir=KYUTAI_CACHE)
    mimi = hf_hub_download(MOSHI_REPO, MIMI_NAME, cache_dir=KYUTAI_CACHE)
    return CheckpointInfo(
        moshi_weights=Path("/nonexistent/moshi-is-not-loaded.safetensors"),
        mimi_weights=Path(mimi),
        tokenizer=Path(tokenizer),
        lm_config=None,
    )


def _dummy_moshi_embeddings() -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Zero-filled stand-ins for Moshi's embedding tables.

    Only their shapes matter: ``GTemporalDepthModel3`` reads
    ``num_textemb, dim_textemb = text_procemb.shape`` (gesture_lm.py:94) to size
    its own ``ScaledEmbedding`` layers.  The strict ``load_state_dict`` that
    follows replaces every value.
    """
    text = torch.zeros(TEXT_EMB_SHAPE, dtype=torch.bfloat16)
    audio = [torch.zeros(AUDIO_EMB_SHAPE, dtype=torch.bfloat16) for _ in range(N_AUDIO_CODEBOOKS)]
    return text, audio


def _resolve(path: str) -> str:
    """Config checkpoint paths are relative to the clone root."""
    p = Path(path)
    return str(p if p.is_absolute() else (REPO / p))


@dataclass
class Motion:
    """SMPL-X motion at 25 fps, in the layout run_gestinference.py saves."""

    poses: np.ndarray        # (T, 165) flat axis-angle, 55 joints
    expressions: np.ndarray  # (T, 100)
    trans: np.ndarray        # (T, 3)
    betas: np.ndarray        # (T, 300) zeros

    @property
    def num_frames(self) -> int:
        return self.poses.shape[0]

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            str(path),
            betas=self.betas[0],
            poses=self.poses,
            expressions=self.expressions,
            trans=self.trans,
            model="smplx",
            gender="neutral",
            # 30 is wrong but expected by the SMPL-X Blender addon; real rate
            # is 25.  Kept for compatibility with run_gestinference.py.
            mocap_frame_rate=30,
        )


class GestureEngine:
    """Gesture LM + three codecs, driven by a ``[1, 9, T]`` conditioning stream."""

    def __init__(
        self,
        glm_config: str | Path,
        device: str = "cuda",
        character_id: int = 15,
        glm_cfg_coef: float = 1.3,
        seed: int | None = 2342,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        with open(glm_config) as f:
            args = SimpleNamespace(**yaml.safe_load(f))
        self.args = args
        self.device = device
        self.character_id = character_id

        info = _checkpoint_info_without_moshi()
        self.mimi = info.get_mimi(device=device)
        self.text_tokenizer = info.get_text_tokenizer()

        self.smplx_model = (
            smplx.create(
                _resolve(args.deps_path) + "/smplx_2020/",
                model_type="smplx",
                gender="NEUTRAL_2020",
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                use_pca=False,
            )
            .to(device)
            .eval()
        )
        for p in self.smplx_model.parameters():
            p.requires_grad = False

        # --- body-part masks, same ordering as run_gestinference.py:610 ---
        upper = torch.from_numpy(get_smplx_bodypart_masks("upper"))
        hands = torch.from_numpy(get_smplx_bodypart_masks("hands"))
        lower = torch.from_numpy(get_smplx_bodypart_masks("lower"))
        face = torch.from_numpy(get_smplx_bodypart_masks("face"))
        self.motion_mask = [upper + hands, lower, face]

        # --- gesture codecs ---
        codec_weights = [
            _resolve(args.upperbodycodec_ckpt),
            _resolve(args.lowerbodycodec_ckpt),
            _resolve(args.facecodec_ckpt),
        ]
        glm_gen_config = {
            "use_sampling": True,
            "temp_gtemporal": 0.9,
            "temp_gdepth": 0.9,
            "top_p_gtemporal": 0.8,
            "top_p_gdepth": 0.95,
            "check": True,
        }
        info.load_gesture_weights(codec_weights, _resolve(args.test_ckpt), glm_gen_config)

        self.codecs = info.get_gesture_codecs(
            device=device,
            codec_kwargs=dict(
                num_frames=args.num_frames,
                frame_chunk_size=args.frame_chunk_size,
                upperlower_nfeats=args.upperlower_nfeats,
                face_nfeats=args.face_nfeats,
                lowertrans_nfeats=args.lowertrans_nfeats,
                motion_fps=args.motion_fps,
                transformer_heads=args.transformer_heads,
                transformer_layers=args.transformer_layers,
                convblock_layers=args.convblock_layers,
            ),
        )

        # The GLM is conditioned on frozen copies of the codecs' RVQ codebooks.
        codec_layers = (
            copy.deepcopy(self.codecs[0].quantizer.vq.layers)
            + copy.deepcopy(self.codecs[1].quantizer.vq.layers)
            + copy.deepcopy(self.codecs[2].quantizer.vq.layers)
        )
        for layer in codec_layers:
            for p in layer.parameters():
                p.requires_grad = False
        codec_layers.eval()

        # --- gesture LM, built with zeroed Moshi tables ---
        text_procemb, audio_procemb = _dummy_moshi_embeddings()
        self.query2mem_scale = int(self.mimi.frame_rate / self.codecs[0].frame_rate)
        self.glm = info.get_gesture_lm(
            device=device,
            dtype=None,
            gesture_lm_kwargs=dict(
                num_heads=args.gestureformer_heads,
                num_layers=args.gestureformer_layers,
                depformer_heads=args.gestureformer_depformer_heads,
                depformer_layers=args.gestureformer_depformer_layers,
                query2mem_scale=self.query2mem_scale,
                num_temp_classifiers=args.num_temp_classifiers,
                text_procemb=text_procemb,
                audio_procemb=audio_procemb,
                gesture_codec_layers=codec_layers,
                vad_guidance=args.vad_guidance,
                body_parts=3,
                bp_dist=None,
                textaudio_emb_freeze=getattr(args, "textaudio_emb_freeze", False),
            ),
        )

        self.glm_gen = GestureLMGen(
            self.glm,
            condition_tensors=get_gesture_condition_tensors(character_id),
            cfg_coef=glm_cfg_coef,
            **glm_gen_config,
        )

        self.n_upper = self.codecs[0].num_codebooks
        self.n_lower = self.codecs[1].num_codebooks
        self.n_face = self.codecs[2].num_codebooks
        self.frame_chunk = self.codecs[0].frame_chunk_size

        # Cross-attention is applied to upper and face tokens only
        # (run_gestinference.py:118, config drop_lower_crossattn: true).
        self.lower_cross_attn_mask = torch.zeros(
            1, self.n_upper + self.n_lower + self.n_face, 1, device=device, dtype=torch.bool
        )
        self.lower_cross_attn_mask[:, self.n_upper : self.n_upper + self.n_lower, :] = True

    @torch.no_grad()
    def generate(self, condition: torch.Tensor) -> Motion:
        """``[1, 9, T]`` conditioning -> SMPL-X motion at 25 fps.

        One GLM step per Mimi frame (``query2mem_scale == 1`` for the released
        checkpoints), each emitting ``frame_chunk_size`` motion frames.
        """
        assert condition.shape[0] == 1, "batch size 1 only"
        assert condition.shape[1] == 9, f"expected 9 conditioning rows, got {condition.shape[1]}"

        # Root starts at hip height, matching run_gestinference.py:142.
        final_pos = (torch.zeros(1) + torch.tensor([0.0, 1.3, 0.0])).to(self.device)

        uhj, lj, fj = MOTION_JOINTS_PER_CODEC
        nframes = self.frame_chunk
        masks = [m.cpu().numpy() for m in self.motion_mask]

        chunks: list[np.ndarray] = []
        total = condition.shape[2]

        with self.codecs[0].streaming(1), self.codecs[1].streaming(1), self.codecs[
            2
        ].streaming(1), self.glm_gen.streaming(1):
            # Inside the contexts, not before them: `reset_streaming` asserts
            # there is a state to reset. Entering already allocates a fresh
            # one, so this only guarantees a clean slate per call.
            self.glm_gen.reset_streaming()
            for codec in self.codecs:
                codec.reset_streaming()

            for start in range(0, total, self.query2mem_scale):
                end = start + self.query2mem_scale
                if end > total:
                    break  # drop a trailing partial chunk; the GLM needs a full one
                tokens = self.glm_gen.step(
                    condition[:, :, start:end],
                    ca_query_padding_mask=self.lower_cross_attn_mask,
                )

                upper_tok = tokens[:, : self.n_upper, :]
                lower_tok = tokens[:, self.n_upper : self.n_upper + self.n_lower, :]
                face_tok = tokens[:, self.n_upper + self.n_lower :, :]

                upper = self.codecs[0].decode(upper_tok)
                lowertrans = self.codecs[1].decode(lower_tok)
                faceexp = self.codecs[2].decode(face_tok)

                bs = lowertrans.shape[0]

                upper = rc.matrix_to_axis_angle(
                    rc.rotation_6d_to_matrix(upper.reshape(bs, nframes, uhj, 6))
                ).reshape(bs, nframes, uhj * 3)

                lower = rc.matrix_to_axis_angle(
                    rc.rotation_6d_to_matrix(lowertrans[:, :, : lj * 6].reshape(bs, nframes, lj, 6))
                ).reshape(bs, nframes, lj * 3)
                trans_vel = lowertrans[:, :, lj * 6 : lj * 6 + 3]
                trans, final_pos = velocity2position_mixeddiff(
                    trans_vel, 1 / self.codecs[1].motion_fps, init_pos=final_pos
                )

                face = rc.matrix_to_axis_angle(
                    rc.rotation_6d_to_matrix(faceexp[:, :, : fj * 6].reshape(bs, nframes, fj, 6))
                ).reshape(bs, nframes, fj * 3)
                exps = faceexp[:, :, fj * 6 :]

                # Scatter each part back into the full 55-joint SMPL-X vector.
                combined = (
                    inverse_selection_smplx(upper[0].cpu().numpy(), masks[0], nframes)
                    + inverse_selection_smplx(lower[0].cpu().numpy(), masks[1], nframes)
                    + inverse_selection_smplx(face[0].cpu().numpy(), masks[2], nframes)
                )
                chunks.append(
                    np.concatenate(
                        [combined, exps[0].cpu().numpy(), trans[0].cpu().numpy()], axis=-1
                    )
                )

        motion = np.concatenate(chunks, axis=0)
        poses = motion[:, :-103]
        expressions = motion[:, -103:-3]
        trans = motion[:, -3:]
        return Motion(
            poses=poses,
            expressions=expressions,
            trans=trans,
            betas=np.zeros((poses.shape[0], 300), dtype=np.float32),
        )
