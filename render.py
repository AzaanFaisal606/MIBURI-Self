"""Stage 3 -- motion .npz + wav -> chest-up frontal mp4.

Two interchangeable mesh backends behind one interface:

* ``smplx``  -- raw SMPL-X body. Flat colour, no clothes, but it is the only
  backend with a working face today (jaw + 100 expression coefficients).
* ``mixamo`` -- a retargeted Mixamo character with its baked per-vertex
  texture. Mixamo is not natively SMPL-X-compatible; ``scripts/
  fit_mixamo_character.py`` transfers SMPL-X skinning weights onto the Mixamo
  mesh offline, and ``pose_mixamo_character`` then drives that mesh from the
  *same* SMPL-X joint chain. Face motion requires the bundle to have been fit
  with ``--with_face``; ``y_bot.npz`` was not, so it is body-only.

Only the ``(verts, faces, colors)`` producer differs. Camera, lights,
renderer, encoder and audio mux are shared, which is the whole point of the
split.

    python "MIBURI User/render.py" \
        --npz   output/npz/stage1_wayne.npz \
        --wav   <clip>.wav \
        --out   stage1_wayne.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

# EGL must be selected before pyrender/OpenGL import anything.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import textures  # noqa: E402
from paths import ASSETS, CHARACTER_DIR, default_out, resolve_out  # noqa: E402

SMPLX_MODEL = ASSETS / "smplx_2020" / "smplx" / "SMPLX_NEUTRAL_2020.npz"
# Ours first, then the characters upstream ships (y_bot).
MIXAMO_DIRS = (
    CHARACTER_DIR,
    ASSETS / "mixamo_characters_release",
    ASSETS / "mixamo_characters",
)

MOTION_FPS = 25  # the npz says 30; it is wrong (docs/PROJECT.md invariant 6)
SMPLX_SKIN = (222, 178, 150, 255)

# Vertical FOV. Upstream's full-body camera uses 60 deg; at chest-up range that
# is a wide-angle lens ~0.8 m from the face, which distorts features and turns
# the model's forward/back lean into a large apparent size change. 28 deg is
# roughly a portrait lens: the camera sits further back and depth motion reads
# as lean rather than zoom.
DEFAULT_YFOV_DEG = 28.0

# SMPL-X pose layout inside the (T, 165) axis-angle block. Joint j occupies
# [3j, 3j+3): 0 pelvis, 1-21 body, 22 jaw, 23/24 eyes, 25-39 left hand,
# 40-54 right hand.
SL_GLOBAL = slice(0, 3)
SL_BODY = slice(3, 66)
SL_JAW = slice(66, 69)
SL_LEYE = slice(69, 72)
SL_REYE = slice(72, 75)
SL_LHAND = slice(75, 120)
SL_RHAND = slice(120, 165)
SL_HEAD = slice(66, 75)  # jaw + both eyes, the mixamo path's "head_pose"


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------


class Motion:
    """The arrays we need out of a MIBURI/BEATX .npz, on the target device."""

    def __init__(self, npz_path: str | Path, device: str, translation: str = "zero"):
        data = np.load(npz_path, allow_pickle=True)
        self.poses = torch.as_tensor(np.asarray(data["poses"], np.float32), device=device)
        n = self.poses.shape[0]

        expr = data["expressions"] if "expressions" in data.files else None
        self.expressions = (
            torch.as_tensor(np.asarray(expr, np.float32), device=device)
            if expr is not None
            else torch.zeros(n, 100, device=device)
        )

        trans = np.asarray(data["trans"], np.float32) if "trans" in data.files else np.zeros((n, 3), np.float32)
        if translation == "zero":
            trans = np.zeros_like(trans)
        elif translation == "first":
            trans = trans - trans[0:1]
        self.trans = torch.as_tensor(trans, device=device)

        # betas may be (300,) or (T, 300); SMPL-X shape is constant here.
        betas = np.asarray(data["betas"], np.float32) if "betas" in data.files else np.zeros(300, np.float32)
        if betas.ndim == 2:
            betas = betas[0]
        self.betas = torch.as_tensor(betas, device=device).unsqueeze(0)

        if self.poses.shape[1] != 165:
            raise ValueError(f"expected (T,165) poses, got {tuple(self.poses.shape)}")

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def slice(self, a: int, b: int) -> dict[str, torch.Tensor]:
        return {
            "poses": self.poses[a:b],
            "expressions": self.expressions[a:b],
            "trans": self.trans[a:b],
            "betas": self.betas.expand(b - a, -1),
        }


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class Backend(Protocol):
    name: str
    faces: np.ndarray
    colors: np.ndarray  # (V, 4) uint8, per-vertex RGBA

    def verts(self, chunk: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def meshes(self, verts: np.ndarray) -> list: ...


def _to_pyrender(pyrender, mesh):
    """Convert a trimesh to a pyrender mesh, de-metallising textured ones.

    trimesh hands pyrender a glTF MetallicRoughness material and the default
    `metallicFactor` is 1.0 -- a fully metal surface reflects almost nothing
    under directional lights, so a textured character renders nearly black.
    Cloth and skin are dielectric; forcing metallic to 0 is what makes the
    texture actually visible. Vertex-coloured meshes are left alone so the
    SMPL-X backend keeps the look earlier renders were judged against.
    """
    out = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    for primitive in out.primitives:
        material = primitive.material
        if getattr(material, "baseColorTexture", None) is not None:
            material.metallicFactor = 0.0
            material.roughnessFactor = 0.75
    return out


def _flat_mesh(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> list:
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = colors
    return [mesh]


def _make_smplx(device: str):
    from miburi.utils.viser_scene import make_smplx_model

    return make_smplx_model(SMPLX_MODEL).to(device)


class SmplxBackend:
    """Raw SMPL-X. Full face rig, no texture."""

    name = "smplx"

    def __init__(self, device: str, color: tuple[int, int, int, int] = SMPLX_SKIN):
        self.model = _make_smplx(device)
        self.faces = np.asarray(self.model.faces, np.int32)
        nv = int(self.model.get_num_verts())
        self.colors = np.tile(np.asarray(color, np.uint8), (nv, 1))

    def verts(self, chunk: dict[str, torch.Tensor]) -> torch.Tensor:
        p = chunk["poses"]
        out = self.model(
            betas=chunk["betas"],
            transl=chunk["trans"],
            expression=chunk["expressions"],
            global_orient=p[:, SL_GLOBAL],
            body_pose=p[:, SL_BODY],
            jaw_pose=p[:, SL_JAW],
            leye_pose=p[:, SL_LEYE],
            reye_pose=p[:, SL_REYE],
            left_hand_pose=p[:, SL_LHAND],
            right_hand_pose=p[:, SL_RHAND],
            return_verts=True,
        )
        return out.vertices

    def meshes(self, verts: np.ndarray) -> list:
        return _flat_mesh(verts, self.faces, self.colors)


class MixamoBackend:
    """A Mixamo mesh skinned by the SMPL-X joint chain, with baked texture."""

    name = "mixamo"

    def __init__(self, character: str, device: str, texture: str = "auto"):
        from miburi.utils.mixamo_character import load_mixamo_character, prepare_runtime_caches

        self.model = _make_smplx(device)
        path = resolve_character(character)
        self.char = load_mixamo_character(path, device=device)
        prepare_runtime_caches(self.char, self.model)

        self.faces = self.char.faces.detach().cpu().numpy().astype(np.int32)
        self.colors = _character_colors(self.char)
        self.has_face = self.char.expr_dirs_face is not None
        self.submeshes = _uv_submeshes(path, self.faces) if texture != "vertex" else []
        if texture == "uv" and not self.submeshes:
            raise SystemExit(
                f"--texture uv needs {textures.sidecar_path(path).name}; "
                f"create it with textures.py --fbx <file> --name {Path(path).stem}"
            )

    def meshes(self, verts: np.ndarray) -> list:
        if not self.submeshes:
            return _flat_mesh(verts, self.faces, self.colors)
        import trimesh

        return [
            trimesh.Trimesh(
                vertices=verts[sub["start"]:sub["end"]],
                faces=sub["faces"],
                visual=trimesh.visual.TextureVisuals(uv=sub["uv"], image=sub["image"]),
                process=False,
            )
            for sub in self.submeshes
        ]

    def verts(self, chunk: dict[str, torch.Tensor]) -> torch.Tensor:
        from miburi.utils.mixamo_character import pose_mixamo_character

        p = chunk["poses"]
        forward_kwargs = {
            "body_pose": p[:, SL_BODY],
            "head_pose": p[:, SL_HEAD],
            "hand_pose": torch.cat([p[:, SL_LHAND], p[:, SL_RHAND]], dim=1),
            "global_rotation": p[:, SL_GLOBAL],
            "global_translation": chunk["trans"],
            "expression": chunk["expressions"],
        }
        verts, _ = pose_mixamo_character(self.char, self.model, forward_kwargs)
        return verts


def resolve_character(name: str) -> Path:
    """Accept a path, or a bare slug to look up in the character dirs."""
    p = Path(name)
    if p.is_file():
        return p
    for d in MIXAMO_DIRS:
        cand = d / f"{p.stem}.npz"
        if cand.is_file():
            return cand
    searched = "  ".join(str(d) for d in MIXAMO_DIRS)
    raise FileNotFoundError(f"no Mixamo bundle {name!r}; looked in: {searched}")


def _uv_submeshes(character_npz: Path, faces: np.ndarray) -> list[dict]:
    """Split the mesh per submesh and pair each with its own texture image.

    Returns [] when no texture sidecar exists, which makes the caller fall
    back to per-vertex colours -- so this is purely additive.

    Each submesh owns a contiguous vertex range and its own UV layout, which
    is exactly why the fit pipeline gave up and baked to vertices. Drawing
    them as separate meshes sidesteps that: every one keeps its real texture.
    """
    import io

    from PIL import Image

    images = textures.load(character_npz)
    if not images:
        return []

    data = np.load(character_npz, allow_pickle=False)
    if "submesh_ranges" not in data.files or "uv_coords" not in data.files:
        return []

    uv_all = data["uv_coords"]
    names = [str(n) for n in data["submesh_names"]]
    out: list[dict] = []
    for name, (start, end) in zip(names, data["submesh_ranges"]):
        png = images.get(name)
        if png is None:
            continue
        # Keep only faces wholly inside this submesh, then rebase indices.
        owned = faces[(faces >= start).all(axis=1) & (faces < end).all(axis=1)]
        if len(owned) == 0:
            continue
        out.append({
            "name": name,
            "start": int(start),
            "end": int(end),
            "faces": (owned - start).astype(np.int32),
            "uv": uv_all[start:end],
            "image": Image.open(io.BytesIO(png)).convert("RGB"),
        })
    return out


def _character_colors(char) -> np.ndarray:
    """Per-vertex RGBA, mirroring motion_vis_server's fallback chain.

    Pre-baked ``vertex_colors`` first (the only option that survives a
    multi-submesh character, where body/hair/clothes each carry their own
    texture and UV layout), then a single-texture UV bake, then flat grey.
    """
    nv = char.num_verts
    if char.vertex_colors is not None:
        return np.asarray(char.vertex_colors, np.uint8).reshape(nv, 4)

    if char.uv_coords is not None and char.texture_png is not None:
        import io

        from PIL import Image

        tex = np.asarray(Image.open(io.BytesIO(char.texture_png)).convert("RGBA"))
        uv = char.uv_coords.detach().cpu().numpy()
        h, w = tex.shape[:2]
        # UV origin is bottom-left, image rows run top-down.
        px = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
        py = np.clip((1.0 - (uv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(np.int32)
        return tex[py, px].astype(np.uint8)

    return np.tile(np.array([200, 200, 200, 255], np.uint8), (nv, 1))


def make_backend(kind: str, device: str, character: str | None, texture: str = "auto") -> Backend:
    if kind == "smplx":
        return SmplxBackend(device)
    if kind == "mixamo":
        if not character:
            raise ValueError("--backend mixamo needs --character")
        return MixamoBackend(character, device, texture=texture)
    raise ValueError(f"unknown backend {kind!r}")


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


def chest_up_camera(
    sample_verts: np.ndarray,
    yfov: float,
    frame_fraction: float = 0.34,
    margin: float = 0.15,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Frame the top ``frame_fraction`` of the body, Zoom-call style.

    ``sample_verts`` is (S, V, 3) over frames spread across the clip, so the
    framing accounts for the whole sequence rather than a lucky first frame.
    Distance is solved from the vertical FOV, which makes the framing
    independent of resolution and of how tall the character happens to be --
    the same call frames SMPL-X and a Mixamo character identically.
    """
    y = sample_verts[..., 1]
    top = float(np.percentile(y, 99.9))
    bottom = float(y.min())
    crop_bottom = top - frame_fraction * (top - bottom)

    visible = sample_verts.reshape(-1, 3)
    visible = visible[visible[:, 1] >= crop_bottom]

    center = np.array(
        [
            float(np.median(visible[:, 0])),
            0.5 * (top + crop_bottom),
            float(np.median(visible[:, 2])),
        ],
        np.float32,
    )
    half_height = 0.5 * (top - crop_bottom) * (1.0 + margin)
    dist = float(half_height / np.tan(0.5 * yfov))

    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32) @ np.array(
        [[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32
    )

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = center + rot @ np.array([0.0, 0.0, dist], np.float32)
    return pose, {"top": top, "crop_bottom": crop_bottom, "dist": dist}


def _sample_verts(backend: Backend, motion: Motion, n: int = 24) -> np.ndarray:
    idx = np.unique(np.linspace(0, len(motion) - 1, min(n, len(motion))).astype(int))
    chunk = {k: v[idx] for k, v in motion.slice(0, len(motion)).items()}
    with torch.no_grad():
        return backend.verts(chunk).float().cpu().numpy()


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def render(
    motion: Motion,
    backend: Backend,
    out_path: Path,
    wav: str | None = None,
    width: int = 720,
    height: int = 720,
    fps: int = MOTION_FPS,
    frame_fraction: float = 0.34,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yfov_deg: float = DEFAULT_YFOV_DEG,
    bg: tuple[float, float, float] = (0.13, 0.14, 0.17),
    chunk_size: int = 64,
) -> Path:
    import cv2
    import pyrender
    import trimesh

    out_path.parent.mkdir(parents=True, exist_ok=True)

    yfov = float(np.deg2rad(yfov_deg))
    camera_pose, frame_info = chest_up_camera(
        _sample_verts(backend, motion),
        yfov=yfov,
        frame_fraction=frame_fraction,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )
    print(
        f"[render] framing: top y={frame_info['top']:.3f} "
        f"crop y={frame_info['crop_bottom']:.3f} camera dist={frame_info['dist']:.3f} m"
    )

    scene = pyrender.Scene(
        bg_color=np.array([*bg, 1.0], np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], np.float32),
    )
    scene.add(
        pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / height),
        pose=camera_pose,
    )
    # Key light on the camera axis, fill up and to the subject's left, so the
    # face is not a flat silhouette at this framing.
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[0, 3] += 1.5
    fill_pose[1, 3] += 1.5
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.6), pose=fill_pose)

    silent = out_path.with_suffix(".silent.mp4") if wav else out_path
    renderer = pyrender.OffscreenRenderer(width, height)
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open {silent} for writing")

    n = len(motion)
    try:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            with torch.no_grad():
                verts = backend.verts(motion.slice(start, end)).float().cpu().numpy()
            for v in verts:
                nodes = [
                    scene.add(_to_pyrender(pyrender, m)) for m in backend.meshes(v)
                ]
                color, _ = renderer.render(scene)
                writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                for node in nodes:
                    scene.remove_node(node)
            print(f"\r[render] {end}/{n} frames", end="", flush=True)
    finally:
        print()
        writer.release()
        renderer.delete()

    if wav:
        mux(silent, wav, out_path)
        silent.unlink(missing_ok=True)
    return out_path


def find_ffmpeg() -> str:
    """ffmpeg lives in the conda env, which is not on PATH when the
    interpreter is invoked by absolute path."""
    beside_python = Path(sys.executable).with_name("ffmpeg")
    if beside_python.is_file():
        return str(beside_python)
    found = shutil.which("ffmpeg")
    if found is None:
        raise RuntimeError(
            "ffmpeg not found next to the interpreter or on PATH; "
            "install it with `mamba install -n miburi -c conda-forge ffmpeg`"
        )
    return found


def mux(video: Path, audio: str, out_path: Path) -> Path:
    """Re-encode to H.264 and lay the source audio on top.

    cv2 writes mp4v, which many players and every browser refuse; this pass
    is what makes the file actually watchable, so it runs even though the
    video stream is already complete.
    """
    cmd = [
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="motion .npz from run_stage1.py")
    p.add_argument("--out", default=None,
                   help="output .mp4; a bare name lands in output/mp4/ (default: named after the npz)")
    p.add_argument("--wav", default=None, help="audio to mux on top")
    p.add_argument("--backend", default="smplx", choices=["smplx", "mixamo"])
    p.add_argument("--character", default=None, help="mixamo bundle: slug or path to .npz")
    p.add_argument("--texture", default="auto", choices=["auto", "uv", "vertex"],
                   help="auto = real UV textures when a _textures.npz sidecar exists, else "
                        "per-vertex colours; vertex forces the old baked-colour look")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=MOTION_FPS)
    p.add_argument("--frame-fraction", type=float, default=0.34,
                   help="fraction of body height to frame, from the head down (0.34 ~ chest-up)")
    p.add_argument("--yfov", type=float, default=DEFAULT_YFOV_DEG,
                   help="vertical field of view in degrees; lower = longer lens, less distortion")
    p.add_argument("--yaw", type=float, default=0.0, help="camera yaw in degrees, 0 = frontal")
    p.add_argument("--pitch", type=float, default=0.0, help="camera pitch in degrees, + looks down")
    p.add_argument("--translation", default="zero", choices=["none", "first", "zero"],
                   help="root translation: keep, subtract frame 0, or drop entirely. "
                        "Default zero -- generated drift is 0.65 m over 60 s, which walks "
                        "the avatar out of a chest-up frame")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    motion = Motion(args.npz, device=device, translation=args.translation)
    if args.max_seconds:
        keep = int(args.max_seconds * args.fps)
        for name in ("poses", "expressions", "trans"):
            setattr(motion, name, getattr(motion, name)[:keep])

    backend = make_backend(args.backend, device, args.character, args.texture)
    print(f"[render] backend={backend.name} verts={len(backend.colors)} faces={len(backend.faces)}")
    if isinstance(backend, MixamoBackend):
        if backend.submeshes:
            sizes = ", ".join(f"{s['name']} {s['image'].size[0]}px" for s in backend.submeshes)
            print(f"[render] UV textures: {sizes}")
        else:
            print("[render] per-vertex colours (no texture sidecar; see textures.py)")
    if isinstance(backend, MixamoBackend) and not backend.has_face:
        print("[render] note: this bundle was fit without --with_face; face will not move")

    out_path = resolve_out(args.out, ".mp4") if args.out else default_out(args.npz, ".mp4", f"_{backend.name}")
    out = render(
        motion, backend, out_path, wav=args.wav,
        width=args.width, height=args.height, fps=args.fps,
        frame_fraction=args.frame_fraction, yaw_deg=args.yaw, pitch_deg=args.pitch,
        yfov_deg=args.yfov,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"[render] wrote {out} ({len(motion)} frames, {len(motion) / args.fps:.1f}s, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
