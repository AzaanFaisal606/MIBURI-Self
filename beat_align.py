"""Measure whether generated motion is actually synchronised to the audio.

Mean absolute pose difference does NOT work for this: two motion sequences can
sit the same L1 distance apart whether or not they are in sync, and a 3-second
deliberate desync scores identically to none. Anything comparing motion-to-motion
elementwise is blind to timing.

Instead compare motion to the *audio*: cross-correlate per-frame motion velocity
against the audio onset-strength envelope, both resampled to the motion frame
rate, and report the lag at which they peak. A correctly-aligned clip peaks near
zero lag; a clip generated from a transcript shifted by X ms should peak near X.

Also reports BeatAlign in the style of the MIBURI paper: for each audio onset
peak, the distance to the nearest motion velocity peak.
"""

from __future__ import annotations

import numpy as np

MOTION_FPS = 25


def motion_velocity(poses: np.ndarray) -> np.ndarray:
    """Per-frame motion energy, length T-1."""
    return np.abs(np.diff(poses, axis=0)).mean(axis=1)


def audio_onset_envelope(wav_path: str, n_frames: int, fps: int = MOTION_FPS) -> np.ndarray:
    """Onset strength resampled onto the motion frame grid."""
    import librosa

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop = max(1, int(round(sr / fps)))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if len(env) < n_frames:
        env = np.pad(env, (0, n_frames - len(env)))
    return env[:n_frames]


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def peak_lag(
    poses: np.ndarray, wav_path: str, max_lag_s: float = 4.0, fps: int = MOTION_FPS
) -> tuple[float, float]:
    """Lag (seconds) maximising motion-vs-audio correlation, and that correlation.

    Positive lag means motion trails the audio.
    """
    vel = motion_velocity(poses)
    env = audio_onset_envelope(wav_path, len(vel), fps)
    v, e = _z(vel), _z(env)
    max_lag = int(max_lag_s * fps)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            a, b = v[-lag:], e[: len(e) + lag]
        elif lag > 0:
            a, b = v[: len(v) - lag], e[lag:]
        else:
            a, b = v, e
        n = min(len(a), len(b))
        corrs.append(float(np.dot(a[:n], b[:n]) / n) if n > 10 else -np.inf)
    corrs = np.asarray(corrs)
    best = int(np.argmax(corrs))
    return float(lags[best]) / fps, float(corrs[best])


def beat_align(
    poses: np.ndarray, wav_path: str, fps: int = MOTION_FPS, sigma: float = 0.1
) -> float:
    """MIBURI-style BeatAlign: mean exp(-dist^2 / 2 sigma^2) over audio beats."""
    from scipy.signal import find_peaks

    vel = motion_velocity(poses)
    env = audio_onset_envelope(wav_path, len(vel), fps)
    m_peaks, _ = find_peaks(vel, prominence=vel.std() * 0.3)
    a_peaks, _ = find_peaks(env, prominence=env.std() * 0.5)
    if len(m_peaks) == 0 or len(a_peaks) == 0:
        return float("nan")
    mt, at = m_peaks / fps, a_peaks / fps
    d = np.abs(at[:, None] - mt[None, :]).min(axis=1)
    return float(np.exp(-(d**2) / (2 * sigma**2)).mean())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True)
    p.add_argument("npz", nargs="+", help="motion .npz files to score")
    args = p.parse_args()

    print(f"{'file':<26}{'peak lag':>10}{'corr':>8}{'BeatAlign':>11}")
    for path in args.npz:
        poses = np.load(path)["poses"]
        lag, corr = peak_lag(poses, args.wav)
        ba = beat_align(poses, args.wav)
        print(f"{path.split('/')[-1]:<26}{lag:+9.2f}s{corr:8.3f}{ba:11.3f}")
