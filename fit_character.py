#!/usr/bin/env python
"""Mixamo FBX -> character bundle, routed into our output tree.

The real work is the clone's `scripts/fit_mixamo_character.py`: import the FBX
through headless Blender, chamfer-fit SMPL-X to the body submesh, transfer
skinning weights, write `<slug>.npz` plus a QA render. This wrapper runs that
and changes three things.

**Output routing.** Upstream writes into `assets_dep/mixamo_characters/`, which
is inside the read-only clone. Everything we generate belongs under
`MIBURI User/output/characters/` -- see `paths.py`.

**The SMPL-X forward-signature fix.** `chamfer_fit.py:277` is the only place in
the release that actually calls `smplx_model(...)`, and it calls it with
MIBURI's internal wrapper argument names, which the stock `smplx.SMPLX` swallows
in `**kwargs`. Without `_adapt_smplx_forward` below, the optimiser runs against
a constant. Details in that function's docstring.

**`--prescale`.** Optionally normalise the FBX to SMPL-X rest height before
fitting. The chamfer fit solves a per-axis scale of its own, so this is off by
default -- it exists so the two paths can be compared under names that say which
is which.

    U="MIBURI User"
    python "$U/fit_character.py" --fbx Remy.fbx --character-name remy_chamfer
    python "$U/fit_character.py" --fbx Remy.fbx --character-name remy_prescale_chamfer --prescale

Needs headless Blender (>= 3.0). Found on PATH, under the project's `tools/`,
or via `--blender-bin`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ASSETS, CHARACTER_DIR, PROJECT, ensure_repo_on_path  # noqa: E402

# Vertical extent of the SMPL-X neutral 2020 mesh in its rest pose, in metres.
# Only used as the `--prescale` target: the chamfer fit solves its own per-axis
# scale afterwards, so a few centimetres either way changes nothing but the
# optimiser's starting point. Override with `--target-height`.
SMPLX_REST_HEIGHT = 1.66

DEFAULT_SMPLX_MODEL = ASSETS / "smplx_2020" / "smplx" / "SMPLX_NEUTRAL_2020.npz"


def _find_blender(override: str | None) -> str | None:
    """Prefer an explicit path, then the project's own `tools/` Blender."""
    if override:
        return override
    for candidate in sorted(PROJECT.glob("tools/**/blender")):
        if candidate.is_file():
            return str(candidate)
    return None  # upstream falls back to `shutil.which("blender")`


def _adapt_smplx_forward(model):
    """Teach a stock `smplx.SMPLX` the argument names the fitter uses.

    `chamfer_fit.py:277` is the only code in the release that actually calls
    `smplx_model(...)`, and it calls it with MIBURI's internal wrapper API --
    `shape`, `global_rotation`, `global_translation`, a single 90-dim
    `hand_pose`, a single 9-dim `head_pose`. The PyPI package wants `betas`,
    `global_orient`, `transl`, split hands and split head, and its forward
    ends in `**kwargs`, so every wrong name is swallowed in silence.

    The effect is total: shape and pose have no influence on the output at
    all. Setting every beta to 5.0 moves the mesh 0.000000000 m; asking for
    the gradient raises "does not have a grad_fn", because the vertices were
    never connected to the parameters being optimised. So Adam ran for 900
    iterations against a constant. Everything else in the demo bypasses
    `forward()` and reimplements skinning by hand, which is why this never
    surfaced -- `make_smplx_model` even documents the assumption that
    forward is never called.
    """
    original_forward = model.forward

    def forward(
        shape=None,
        betas=None,
        expression=None,
        body_pose=None,
        hand_pose=None,
        head_pose=None,
        global_rotation=None,
        global_translation=None,
        **kwargs,
    ):
        left_hand = right_hand = None
        if hand_pose is not None:  # (B, 90) -> two (B, 45)
            left_hand, right_hand = hand_pose[:, :45], hand_pose[:, 45:]
        jaw = leye = reye = None
        if head_pose is not None:  # (B, 9) -> jaw, left eye, right eye
            jaw, leye, reye = head_pose[:, :3], head_pose[:, 3:6], head_pose[:, 6:9]

        return original_forward(
            betas=shape if shape is not None else betas,
            global_orient=global_rotation,
            transl=global_translation,
            body_pose=body_pose,
            expression=expression,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            return_verts=True,
            **kwargs,
        )

    model.forward = forward
    return model


def _prescale_loader(load_fbx_mesh, target_height: float = SMPLX_REST_HEIGHT):
    """Wrap `load_fbx_mesh` so the mesh arrives at roughly SMPL-X's scale.

    Mixamo exports are authored in centimetres and land ~100x too large for a
    metres-based SMPL-X. The chamfer fit does solve a per-axis scale, so this
    is not required -- but starting the optimiser two orders of magnitude off
    makes the alignment phase do all the work, and the first phase has only
    `--n_iters_align` steps to do it in.

    Scaling is uniform and about the mesh centroid, so it cannot change the
    shape the fit is trying to match -- only where the optimiser starts.
    Bone heads are moved with the vertices so skeleton retargeting stays
    consistent; everything else in the record is scale-free.
    """

    def loader(*args, **kwargs):
        mesh = load_fbx_mesh(*args, **kwargs)
        verts = np.asarray(mesh.verts, dtype=np.float32)
        height = float(verts[:, 1].max() - verts[:, 1].min())
        if height <= 0:
            print("[fit] prescale skipped: mesh has no vertical extent")
            return mesh

        factor = target_height / height
        print(f"[fit] prescale: mesh height {height:.4f} -> {target_height:.4f} "
              f"(x{factor:.5f})")
        centroid = verts.mean(axis=0, keepdims=True)
        scaled = centroid + (verts - centroid) * factor

        replacements = {"verts": scaled.astype(np.float32)}
        if mesh.bone_heads is not None:
            heads = np.asarray(mesh.bone_heads, dtype=np.float32)
            replacements["bone_heads"] = (
                centroid + (heads - centroid) * factor).astype(np.float32)
        return dataclasses.replace(mesh, **replacements)

    return loader


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fbx", required=True, type=Path,
                   help="Mixamo .fbx to fit (Adobe login needed to download one)")
    p.add_argument("--character-name", default=None,
                   help="output slug; defaults to the FBX stem. Name it for the "
                        "variant being run, e.g. remy_prescale_chamfer")
    p.add_argument("--out-dir", type=Path, default=CHARACTER_DIR,
                   help=f"bundle destination (default: {CHARACTER_DIR})")
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    p.add_argument("--prescale", action="store_true",
                   help="normalise the FBX to SMPL-X rest height before fitting")
    p.add_argument("--target-height", type=float, default=SMPLX_REST_HEIGHT,
                   help="prescale target in metres (default: %(default)s)")
    p.add_argument("--blender-bin", default=None,
                   help="headless Blender; defaults to tools/ then $PATH")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--n-iters-align", type=int, default=100)
    p.add_argument("--n-iters-joint", type=int, default=800)
    p.add_argument("--with-face", action="store_true",
                   help="also transfer face blendshapes + jaw skinning")
    p.add_argument("--skinning", choices=["native", "barycentric"], default="native")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed through to upstream verbatim")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.fbx.is_file():
        raise SystemExit(f"FBX not found: {args.fbx}")
    if not args.smplx_model.is_file():
        raise SystemExit(
            f"SMPL-X model not found: {args.smplx_model}\n"
            "It ships in the clone's assets_dep/ and needs its own registration.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.character_name or args.fbx.stem.lower().replace(" ", "_")

    ensure_repo_on_path()
    import torch
    from scripts import fit_mixamo_character as upstream
    from miburi.utils import viser_scene

    # Patch 1: make the stock smplx.SMPLX answer to the fitter's argument names.
    _make_smplx_model = viser_scene.make_smplx_model

    def make_smplx_model(*a, **kw):
        return _adapt_smplx_forward(_make_smplx_model(*a, **kw))

    viser_scene.make_smplx_model = make_smplx_model

    # Patch 2: optional prescale. `fit_mixamo_character` imported the loader by
    # name, so the module attribute is what has to be replaced.
    if args.prescale:
        upstream.load_fbx_mesh = _prescale_loader(
            upstream.load_fbx_mesh, target_height=args.target_height)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blender = _find_blender(args.blender_bin)

    argv = [
        "fit_mixamo_character",
        "--fbx_path", str(args.fbx),
        "--character_name", slug,
        "--smplx_model", str(args.smplx_model),
        "--out_dir", str(args.out_dir),
        "--device", device,
        "--n_iters_align", str(args.n_iters_align),
        "--n_iters_joint", str(args.n_iters_joint),
        "--skinning", args.skinning,
    ]
    if blender:
        argv += ["--blender_bin", blender]
    if args.with_face:
        argv += ["--with_face"]
    argv += args.extra

    print(f"[fit] slug        : {slug}")
    print(f"[fit] out dir     : {args.out_dir}")
    print(f"[fit] prescale    : {'on' if args.prescale else 'off'}")
    print(f"[fit] blender     : {blender or 'from $PATH'}")

    old_argv, sys.argv = sys.argv, argv
    try:
        upstream.main()
    finally:
        sys.argv = old_argv

    bundle = args.out_dir / f"{slug}.npz"
    print(f"[fit] wrote {bundle}"
          if bundle.is_file() else f"[fit] expected {bundle}, not found")
    print(f"[fit] render it with:  render.py --backend mixamo --character {slug}")


if __name__ == "__main__":
    main()
