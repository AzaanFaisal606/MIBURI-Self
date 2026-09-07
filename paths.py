"""Where everything lives.

`miburi/` is a read-only reference clone. Nothing we produce belongs inside
it -- not motion, not video, not fitted characters, not ASR output. Every
path here is derived from this file's own location, so the tools run from any
working directory rather than only from the clone root.

Outputs are sorted by type under `MIBURI User/output/`:

    output/npz/         generated motion
    output/mp4/         rendered video
    output/json/        ASR transcripts
    output/characters/  fitted Mixamo bundles

The clone is still *read* from: `assets_dep/` (SMPL-X, the shipped y_bot,
Mimi/tokenizer cache) and `experiments/` (released checkpoints) are upstream
artifacts and stay where upstream expects them.
"""

from __future__ import annotations

import sys
from pathlib import Path

USER_DIR = Path(__file__).resolve().parent
PROJECT = USER_DIR.parent

REPO = PROJECT / "miburi"  # read-only upstream clone
ASSETS = REPO / "assets_dep"
EXPERIMENTS = REPO / "experiments"

OUTPUT = USER_DIR / "output"
NPZ_DIR = OUTPUT / "npz"
MP4_DIR = OUTPUT / "mp4"
JSON_DIR = OUTPUT / "json"
CHARACTER_DIR = OUTPUT / "characters"

_KINDS = {
    ".npz": NPZ_DIR,
    ".mp4": MP4_DIR,
    ".json": JSON_DIR,
}


def ensure_repo_on_path() -> None:
    """Make the clone's `scripts.*` package importable.

    `miburi` itself is pip-installed editable, but `scripts/` is not a
    package on the path -- it only resolves when the interpreter's working
    directory happens to be the clone root, which is exactly the coupling we
    are removing.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def resolve_out(value: str | Path, kind: str | None = None) -> Path:
    """Place a bare filename in the right output folder; respect real paths.

    ``--out run7.mp4``            -> output/mp4/run7.mp4
    ``--out subdir/run7.mp4``     -> subdir/run7.mp4, relative to cwd
    ``--out /tmp/run7.mp4``       -> exactly that

    So the common case needs no path at all, and an explicit path is never
    second-guessed.
    """
    p = Path(value)
    if p.is_absolute() or len(p.parts) > 1:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    base = _KINDS.get(kind or p.suffix.lower(), OUTPUT)
    base.mkdir(parents=True, exist_ok=True)
    return base / p.name


def default_out(source: str | Path, suffix: str, tag: str = "") -> Path:
    """Name an output after its input, e.g. `clip.wav` -> `output/npz/clip.npz`."""
    stem = Path(source).stem
    return resolve_out(f"{stem}{tag}{suffix}")
