"""Trajectories, resampling, the style fingerprint, crop augmentation and encoder inputs.

Every trajectory is a 7-DOF joint-position series resampled to COMMON_RATE (15 Hz). The encoder
sees only the raw [q|v|a|j] series (`metric_series`, finite-differenced); the hand-crafted style
fingerprint (`fingerprint`) is the auxiliary regression TARGET during training and is never an
input, nor needed at inference.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import fft as sfft
from scipy.signal import find_peaks

COMMON_RATE = 15.0          # Hz -- everything resampled to this
N_JOINTS = 7
MIN_RAW_LEN = 30            # drop trajectories shorter than 2 s
MOVE_SPEED_THRESH = 0.02    # rad/s joint speed for "fraction moving"
_DT = 1.0 / COMMON_RATE
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Trajectory container + resampling                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Traj:
    dataset: str                  # source label ("droid")
    traj_id: str
    joint: np.ndarray             # (T,7) resampled joint angles [rad] @15 Hz
    raw_rate_hz: float
    n_raw: int
    _vel: np.ndarray = field(init=False, default=None, repr=False)

    @property
    def T(self) -> int:
        return self.joint.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_raw / self.raw_rate_hz

    @property
    def vel(self) -> np.ndarray:  # (T,7) finite-difference joint velocity [rad/s]
        if self._vel is None:
            self._vel = np.gradient(self.joint, _DT, axis=0)
        return self._vel

    def speed(self) -> np.ndarray:  # (T,) joint-space speed = ||vel||
        return np.linalg.norm(self.vel, axis=1)


def resample(joint: np.ndarray, t_orig: np.ndarray, target_rate=COMMON_RATE):
    """Linearly resample a (T,D) series sampled at times ``t_orig`` onto a uniform grid."""
    t0 = np.asarray(t_orig, np.float64) - t_orig[0]
    dur = float(t0[-1])
    n_new = max(2, int(round(dur * target_rate)) + 1)
    t_new = np.linspace(0.0, dur, n_new)
    return np.stack([np.interp(t_new, t0, joint[:, d]) for d in range(joint.shape[1])], axis=1)


# --------------------------------------------------------------------------- #
# Style fingerprint (the VAE's auxiliary training target)                     #
# --------------------------------------------------------------------------- #
def sparc(speed: np.ndarray, fs: float, fc: float = 10.0, amp_th: float = 0.05, padlevel: int = 4) -> float:
    """Spectral Arc Length smoothness of a speed profile (Balasubramanian 2015).
    Negative; closer to 0 = smoother, more negative = jerkier."""
    speed = np.asarray(speed, dtype=np.float64)
    if speed.size < 4 or np.allclose(speed, speed[0]):
        return np.nan
    fc = min(fc, fs / 2.0)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf / (Mf.max() + _EPS)
    n = min(len(f), len(Mf)); f, Mf = f[:n], Mf[:n]
    f_sel, Mf_sel = f[f <= fc], Mf[f <= fc]
    above = np.where(Mf_sel >= amp_th)[0]
    if above.size < 2:
        return np.nan
    f_sel, Mf_sel = f_sel[above[0]:above[-1] + 1], Mf_sel[above[0]:above[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0] + _EPS)
    dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))


def ldlj_speed(speed: np.ndarray, fs: float) -> float:
    """Log Dimensionless Jerk of a speed profile. Higher (->0) = smoother."""
    speed = np.asarray(speed, dtype=np.float64)
    if speed.size < 4:
        return np.nan
    dt = 1.0 / fs
    dur, vpeak = len(speed) * dt, np.max(np.abs(speed))
    if vpeak < _EPS:
        return np.nan
    d2 = np.gradient(np.gradient(speed, dt), dt)
    dlj = (dur ** 3 / vpeak ** 2) * (np.sum(d2 ** 2) * dt)
    return float(-np.log(dlj)) if dlj >= _EPS else np.nan


def speed_spectrum(speed: np.ndarray, fs: float):
    """Return (centroid, bandwidth, entropy) of the AC speed power spectrum."""
    s = np.asarray(speed, dtype=np.float64); s = s - s.mean()
    if s.size < 4 or np.allclose(s, 0):
        return np.nan, np.nan, np.nan
    F = np.abs(sfft.rfft(s)) ** 2
    freqs = sfft.rfftfreq(len(s), 1.0 / fs); F[0] = 0.0
    tot = F.sum()
    if tot < _EPS:
        return np.nan, np.nan, np.nan
    p = F / tot
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)) / np.log(len(p)))
    return centroid, bandwidth, entropy


# Physical energy bands (Hz) for the per-joint spectral fingerprint. At COMMON_RATE=15 Hz the
# Nyquist is 7.5 Hz. low = gross reach-and-place motion, mid = ~1-2 Hz corrections/wobble,
# high = fast micro-motion. How velocity power is spread across these bands is the main axis
# separating human teleoperation from planner motion.
BANDS_HZ = ((0.0, 0.5), (0.5, 2.0), (2.0, 7.5))
_BAND_TAGS = ("low", "mid", "high")


def band_fractions(x: np.ndarray, fs: float, bands=BANDS_HZ) -> list[float]:
    """Fraction of a 1-D signal's AC power falling in each frequency band (Hann-windowed rFFT).

    Amplitude-invariant (normalized by total power) and length-robust (bands are integrals), so it
    captures spectral *shape*, not motion size."""
    x = np.asarray(x, dtype=np.float64); x = x - x.mean()
    n = len(_BAND_TAGS)
    if x.size < 8 or np.allclose(x, 0):
        return [np.nan] * n
    P = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs); P[0] = 0.0     # drop DC
    tot = P.sum()
    if tot < _EPS:
        return [np.nan] * n
    return [float(P[(freqs >= lo) & (freqs < hi)].sum() / tot) for lo, hi in bands]


def fingerprint(tr: Traj) -> dict[str, float]:
    """Named, rate/length-invariant style descriptors for one trajectory (113 for 7 joints)."""
    q, v = tr.joint, tr.vel
    a = np.gradient(v, _DT, axis=0)
    j = np.gradient(a, _DT, axis=0)
    s = tr.speed()
    feats: dict[str, float] = {}
    for d in range(N_JOINTS):
        av, aa, aj = np.abs(v[:, d]), np.abs(a[:, d]), np.abs(j[:, d])
        feats[f"vel_mean_j{d}"] = float(av.mean())
        feats[f"vel_std_j{d}"] = float(av.std())
        feats[f"vel_max_j{d}"] = float(av.max())
        feats[f"vel_p95_j{d}"] = float(np.percentile(av, 95))
        feats[f"acc_mean_j{d}"] = float(aa.mean())
        feats[f"acc_std_j{d}"] = float(aa.std())
        feats[f"acc_max_j{d}"] = float(aa.max())
        feats[f"jerk_mean_j{d}"] = float(aj.mean())
        feats[f"jerk_max_j{d}"] = float(aj.max())
        feats[f"rom_j{d}"] = float(q[:, d].max() - q[:, d].min())
        sign = np.sign(v[:, d]); sign[sign == 0] = 1
        feats[f"vsign_j{d}"] = float(np.sum(np.abs(np.diff(sign)) > 0)) / (tr.duration_s + _EPS)
        # per-joint energy-band distribution of the joint velocity
        for tag, frac in zip(_BAND_TAGS, band_fractions(v[:, d], COMMON_RATE)):
            feats[f"bandfrac_{tag}_j{d}"] = frac
    path_len = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    feats["path_len_per_s"] = path_len / (tr.duration_s + _EPS)
    feats["straightness"] = float(np.linalg.norm(q[-1] - q[0])) / (path_len + _EPS)
    feats["speed_mean"] = float(s.mean())
    feats["speed_std"] = float(s.std())
    feats["speed_max"] = float(s.max())
    feats["sparc"] = sparc(s, COMMON_RATE)
    feats["ldlj"] = ldlj_speed(s, COMMON_RATE)
    feats["frac_moving"] = float(np.mean(s > MOVE_SPEED_THRESH))
    peaks, _ = find_peaks(s, prominence=0.05 * (s.max() + _EPS))
    feats["submov_per_sec"] = len(peaks) / (tr.duration_s + _EPS)
    cen, bw, ent = speed_spectrum(s, COMMON_RATE)
    feats["spec_centroid"], feats["spec_bandwidth"], feats["spec_entropy"] = cen, bw, ent
    for tag, frac in zip(_BAND_TAGS, band_fractions(s, COMMON_RATE)):   # scalar-speed band split
        feats[f"bandfrac_{tag}_speed"] = frac
    return feats


def feature_weights(names: list[str], band_w: float = 4.0) -> np.ndarray:
    """Per-target weights for the auxiliary regression.

    The band/spectral descriptors are a small minority of the fingerprint, so at equal weighting
    the 16-D latent under-encodes them. Up-weighting them by ``band_w`` forces the latent to carry
    the energy-band structure. Weights are renormalized to mean 1 so the aux-vs-KL balance (beta)
    is unchanged."""
    w = np.ones(len(names), dtype=np.float32)
    for i, n in enumerate(names):
        if n.startswith("bandfrac_") or n.startswith("spec_"):
            w[i] = band_w
    return w * (len(w) / w.sum())


# --------------------------------------------------------------------------- #
# Crop augmentation + batched fingerprints                                    #
# --------------------------------------------------------------------------- #
def _crop(tr: Traj, lo: float, hi: float, rng) -> Traj:
    """Length-invariant style-preserving augmentation: a random contiguous crop."""
    T = tr.T
    L = min(max(MIN_RAW_LEN, int(T * rng.uniform(lo, hi))), T)
    st = int(rng.integers(0, T - L + 1)) if T > L else 0
    return Traj(tr.dataset, tr.traj_id, tr.joint[st:st + L], COMMON_RATE, L)


def crop_variants(tr: Traj, n_crops: int, rng, lo: float = 0.4, hi: float = 1.0):
    """The trajectory itself plus ``n_crops`` random style-preserving crops."""
    return [tr] + [_crop(tr, lo, hi, rng) for _ in range(n_crops)]


def fp_rows(variants, names, workers: int = 0) -> np.ndarray:
    """(M, len(names)) fingerprint matrix for a list of trajectories, NaN -> column median.

    The per-trajectory fingerprint (FFT/SPARC/spectral) is the build bottleneck, so for large M it
    is computed across processes. ``workers``: 0 -> auto = min(24, cores), 1 -> serial. The result
    is identical to the serial path."""
    if not workers:
        workers = min(24, os.cpu_count() or 2)
    if workers > 1 and len(variants) >= 512:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            dicts = list(ex.map(fingerprint, variants, chunksize=64))
    else:
        dicts = [fingerprint(t) for t in variants]
    FP = np.array([[d[n] for n in names] for d in dicts], np.float64)
    return np.where(np.isnan(FP), np.nanmedian(FP, axis=0), FP)


# --------------------------------------------------------------------------- #
# Encoder input: raw [q|v|a|j] series + variable-length padding               #
# --------------------------------------------------------------------------- #
def metric_series(joint: np.ndarray) -> np.ndarray:
    """(T,7) joint positions @15 Hz -> (T, 28) per-timestep joint metric [q|v|a|j]."""
    v = np.gradient(joint, _DT, axis=0)
    a = np.gradient(v, _DT, axis=0)
    j = np.gradient(a, _DT, axis=0)
    return np.concatenate([joint, v, a, j], axis=1)


def pad_mask(batch, cmu, csd):
    """Pad a list of variable-length (T,C) series to (B,C,Tmax) standardized tensors,
    with a (B,1,Tmax) validity mask."""
    T = max(len(x) for x in batch)
    B, ch = len(batch), batch[0].shape[1]
    X = np.zeros((B, ch, T), np.float32)
    M = np.zeros((B, 1, T), np.float32)
    for i, x in enumerate(batch):
        X[i, :, :len(x)] = ((x - cmu) / csd).T
        M[i, 0, :len(x)] = 1.0
    return torch.tensor(X), torch.tensor(M)
