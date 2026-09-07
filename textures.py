#!/usr/bin/env python
"""UV textures for a Mixamo character bundle, stored as a sidecar next to it.

The fitting pipeline bakes textures down to per-vertex RGBA, because a
multi-submesh character -- body, hair, hoodie, shoes -- carries one texture
image *and one UV layout* per submesh, and a single mesh cannot hold more than
one of each. Baking to vertices loses everything finer than the vertex spacing,
which on a Mixamo body is very visible at chest-up framing.

`render.py` sidesteps that by drawing each submesh as its own trimesh with its
own `TextureVisuals`. It just needs the images. This module keeps them in
`<character>_textures.npz`, one PNG blob per submesh name, so the character
bundle itself stays exactly what upstream writes.

    U="MIBURI User"
    python "$U/textures.py" --fbx Remy.fbx --name remy

writes `output/characters/remy_textures.npz` beside `remy.npz`. `render.py
--texture auto` then picks it up with no further flags; `--texture vertex`
ignores it and `--texture uv` fails loudly if it is missing.

Extraction runs `_blender_extract_textures.py` inside headless Blender, which
visits mesh objects in the same order `_blender_fbx_export.py` combines them,
so the names line up with `submesh_names` in the bundle.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHARACTER_DIR, PROJECT  # noqa: E402

_HELPER = Path(__file__).with_name("_blender_extract_textures.py")

SIDECAR_SUFFIX = "_textures.npz"


def sidecar_path(character_npz: str | Path) -> Path:
    """`.../remy.npz` -> `.../remy_textures.npz`."""
    p = Path(character_npz)
    return p.with_name(f"{p.stem}{SIDECAR_SUFFIX}")


def load(character_npz: str | Path) -> dict[str, bytes]:
    """Return `{submesh name: PNG bytes}`, or `{}` when there is no sidecar.

    Returning empty rather than raising is deliberate: it is what makes the
    texture path purely additive, so a bundle fitted before this existed still
    renders with per-vertex colours.
    """
    path = sidecar_path(character_npz)
    if not path.is_file():
        return {}

    with np.load(path, allow_pickle=False) as data:
        if "names" not in data.files:
            return {}
        names = [str(n) for n in data["names"]]
        return {
            name: data[f"blob_{i}"].tobytes()
            for i, name in enumerate(names)
            if f"blob_{i}" in data.files
        }


def save(character_npz: str | Path, images: dict[str, bytes]) -> Path:
    """Write the sidecar. Names are stored separately from the blobs so a
    submesh called `Beard.001` does not have to be a valid npz member name."""
    path = sidecar_path(character_npz)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(images)
    arrays = {f"blob_{i}": np.frombuffer(images[n], dtype=np.uint8)
              for i, n in enumerate(names)}
    np.savez(path, names=np.array(names, dtype=np.str_), **arrays)
    return path


def _find_blender(override: str | None) -> str:
    if override:
        if not Path(override).is_file():
            raise SystemExit(f"--blender-bin not found: {override}")
        return override
    for candidate in sorted(PROJECT.glob("tools/**/blender")):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("blender")
    if found is None:
        raise SystemExit(
            "`blender` not found. Put a headless build under tools/, on $PATH, "
            "or pass --blender-bin.")
    return found


def extract(fbx_path: str | Path, blender_bin: str | None = None) -> dict[str, bytes]:
    """Pull one diffuse PNG per mesh object out of an FBX, via headless Blender."""
    fbx_path = Path(fbx_path).resolve()
    if not fbx_path.is_file():
        raise SystemExit(f"FBX not found: {fbx_path}")
    if not _HELPER.is_file():
        raise SystemExit(f"Blender helper missing: {_HELPER}")

    blender = _find_blender(blender_bin)
    with tempfile.TemporaryDirectory(prefix="miburi-textures-") as tmp:
        cmd = [blender, "--background", "--python", str(_HELPER), "--",
               str(fbx_path), tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:])
            sys.stderr.write(proc.stderr[-4000:])
            raise SystemExit(f"blender failed ({proc.returncode}) on {fbx_path}")

        index = Path(tmp) / "index.txt"
        if not index.is_file():
            raise SystemExit(
                f"blender produced no index.txt for {fbx_path}; "
                "the FBX may have no mesh objects")

        images: dict[str, bytes] = {}
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            name, _, png = line.partition("\t")
            if not png:
                print(f"[textures] {name}: no diffuse image, skipped")
                continue
            blob = (Path(tmp) / png).read_bytes()
            images[name] = blob
            print(f"[textures] {name}: {len(blob) / 1024:.0f} KB")
        return images


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fbx", required=True, type=Path, help="source Mixamo .fbx")
    p.add_argument("--name", required=True,
                   help="character slug the sidecar belongs to, e.g. remy")
    p.add_argument("--character-dir", type=Path, default=CHARACTER_DIR,
                   help=f"where the bundle lives (default: {CHARACTER_DIR})")
    p.add_argument("--blender-bin", default=None)
    args = p.parse_args()

    bundle = args.character_dir / f"{args.name}.npz"
    if not bundle.is_file():
        print(f"[textures] note: {bundle} does not exist yet "
              "(fit_character.py writes it); the sidecar will wait for it")

    images = extract(args.fbx, args.blender_bin)
    if not images:
        raise SystemExit(f"[textures] no textures found in {args.fbx}")

    out = save(bundle, images)
    print(f"[textures] wrote {out} ({len(images)} submeshes)")
    print(f"[textures] render it with:  render.py --backend mixamo "
          f"--character {args.name} --texture uv")


if __name__ == "__main__":
    main()
