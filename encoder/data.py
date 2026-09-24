"""Training data for the VAE: full-DROID proprio (fetch + load).

DROID. `lerobot/droid_1.0.1` is a lerobot codebase-v3.0 dataset: many episodes are packed per
parquet file (split by `episode_index`), columns are dotted (`observation.state.joint_position`,
`action`, ...), images live in separate mp4 videos, and the native rate is 15 Hz. The repo lists
156 data parquet files, but `meta/episodes` references only file-000..file-085 (all 95,658
episodes); file-086..file-155 are orphaned duplicate re-exports and are skipped.

`fetch_droid_full` never downloads the parquet wholesale: it streams each canonical file reading
only the proprio columns (HfFileSystem + pyarrow column projection, ~46 MB/file, ~4 GB total),
splits rows by episode, and writes a resumable per-file shard
`<cache-dir>/droid_full_proprio/shard_{file:03d}.npz` (~1.7 GB in total). Re-running skips
finished shards, so a 429 (HF limit: 1000 requests / 5 min) or an interruption costs at most one
file. No HF token is needed.

    python -m vae.data fetch [--max-files N] [--overwrite] [--cache-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .features import MIN_RAW_LEN, Traj, resample

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

# --------------------------------------------------------------------------- #
# Full DROID (lerobot/droid_1.0.1, v3.0)                                      #
# --------------------------------------------------------------------------- #
DROID_FULL_REPO = "lerobot/droid_1.0.1"
DROID_SUBDIR = "droid_full_proprio"   # shard folder inside the cache dir
# Only the 86 canonical data files (file-000..file-085) hold the 95,658 unique episodes.
N_DATA_FILES = 86
NATIVE_RATE = 15.0                    # DROID v3.0 fps (meta/info.json)

# Proprio-only column projection.
_COLS = [
    "observation.state.joint_position",     # list<float>[7]
    "observation.state.gripper_position",   # float
    "action",                               # list<float>[8]  (7 joint deltas + gripper)
    "task_index",                           # int64
    "timestamp",                            # float  (resets to 0 each episode, step 1/15)
    "episode_index",                        # int64
]
_JOINT_W, _ACT_W = 7, 8


def droid_dir(cache_dir=DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / DROID_SUBDIR


def _list_2d(col, width: int) -> np.ndarray:
    """ChunkedArray of fixed-width list<float> -> (N, width) float32, fast path + fallback."""
    try:
        flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        a = np.asarray(flat, np.float32)
        if a.size % width == 0:
            return a.reshape(-1, width)
    except Exception:
        pass
    return np.asarray(col.to_pylist(), np.float32).reshape(-1, width)


def _scalar(col) -> np.ndarray:
    return np.asarray(col.combine_chunks().to_numpy(zero_copy_only=False))


def _group_bounds(ep: np.ndarray):
    """Contiguous [start, stop) row ranges, one per episode, from the ordered episode_index."""
    if len(ep) == 0:
        return []
    cut = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [len(ep)]])
    return list(zip(starts.tolist(), stops.tolist()))


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "too many request" in s or "rate limit" in s


def fetch_droid_full(cache_dir=DEFAULT_CACHE_DIR, sleep: float = 2.0, max_files: int | None = None,
                     overwrite: bool = False):
    """Stream the canonical v3.0 data parquet files (proprio columns only) into resumable shards.

    Throttled for the HF rate limit: a fixed inter-file ``sleep`` plus backoff on failures (90 s on
    a 429). Already-written shards are skipped unless ``overwrite``."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    out_dir = droid_dir(cache_dir)
    manifest_json = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    n_files = N_DATA_FILES if max_files is None else min(max_files, N_DATA_FILES)
    t_start = time.time()
    total_eps = 0
    for fi in range(n_files):
        shard = out_dir / f"shard_{fi:03d}.npz"
        if shard.exists() and not overwrite:
            try:
                total_eps += int(np.load(shard, allow_pickle=True)["n_eps"])
                print(f"[file {fi:3d}/{n_files}] shard exists, skip")
                continue
            except Exception:
                print(f"[file {fi:3d}] shard unreadable, refetching")
        path = f"datasets/{DROID_FULL_REPO}/data/chunk-000/file-{fi:03d}.parquet"

        tbl = None
        for attempt in range(6):
            try:
                with fs.open(path, "rb") as f:
                    tbl = pq.read_table(f, columns=_COLS)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                back = 90.0 if _is_rate_limit(e) else 3.0 * (attempt + 1)
                print(f"  [file {fi}] attempt {attempt + 1} failed ({type(e).__name__}: "
                      f"{str(e)[:80]}); backoff {back:.0f}s")
                time.sleep(back)

        ep_idx = _scalar(tbl["episode_index"]).astype(np.int64)
        joints_all = _list_2d(tbl["observation.state.joint_position"], _JOINT_W)
        grip_all = _scalar(tbl["observation.state.gripper_position"]).astype(np.float32)
        act_all = _list_2d(tbl["action"], _ACT_W)
        tidx_all = _scalar(tbl["task_index"]).astype(np.int64)
        ts_all = _scalar(tbl["timestamp"]).astype(np.float32)

        joints, grips, acts, tidx, eidx, durs = [], [], [], [], [], []
        for a, b in _group_bounds(ep_idx):
            joints.append(joints_all[a:b].astype(np.float32))
            grips.append(grip_all[a:b])
            acts.append(act_all[a:b].astype(np.float32))
            tidx.append(int(tidx_all[a]))
            eidx.append(int(ep_idx[a]))
            ts = ts_all[a:b]
            durs.append(float(ts[-1] - ts[0]) if b - a > 1 else 0.0)

        np.savez(
            shard,
            joints=np.array(joints, dtype=object), grippers=np.array(grips, dtype=object),
            actions=np.array(acts, dtype=object), task_index=np.asarray(tidx, np.int64),
            episode_index=np.asarray(eidx, np.int64), durations=np.asarray(durs, np.float32),
            n_eps=np.int64(len(joints)), fps=np.float32(NATIVE_RATE),
        )
        total_eps += len(joints)
        lens = [len(j) for j in joints]
        el = time.time() - t_start
        print(f"[file {fi:3d}/{n_files}] {len(joints):4d} eps  T:min={min(lens)} "
              f"med={int(np.median(lens))} max={max(lens)}  total_eps={total_eps}  {el:6.1f}s")
        time.sleep(sleep)

    manifest_json.write_text(json.dumps({
        "repo": DROID_FULL_REPO, "codebase_version": "v3.0", "n_files": n_files,
        "total_episodes": total_eps, "native_rate_hz": NATIVE_RATE, "columns": _COLS,
    }, indent=2))
    print(f"\n[done] {total_eps} episodes across {n_files} shards -> {out_dir} "
          f"({time.time() - t_start:.1f}s)")


def _shard_paths(cache_dir) -> list[Path]:
    return sorted(droid_dir(cache_dir).glob("shard_*.npz"))


def cache_ready(cache_dir=DEFAULT_CACHE_DIR) -> bool:
    return len(_shard_paths(cache_dir)) > 0


def load_droid_full(cache_dir=DEFAULT_CACHE_DIR) -> list[Traj]:
    """Every cached DROID episode with >= MIN_RAW_LEN frames -> list[Traj] @15 Hz, in shard order.

    Shards that fail to load (e.g. one still being written by a concurrent fetch) are skipped."""
    shards = _shard_paths(cache_dir)
    if not shards:
        raise FileNotFoundError(f"{droid_dir(cache_dir)} has no shards -- run: python -m vae.data fetch")
    out, dropped = [], 0
    for sp in shards:
        try:
            blob = np.load(sp, allow_pickle=True)
            J, E = blob["joints"], blob["episode_index"]
        except Exception as e:
            print(f"[droid] skip unreadable shard {sp.name}: {type(e).__name__}")
            continue
        for j in range(len(J)):
            q = np.asarray(J[j], np.float32)
            n_raw = len(q)
            if n_raw < MIN_RAW_LEN:
                dropped += 1
                continue
            # a near-identity resample: DROID is natively 15 Hz
            qr = resample(np.asarray(q, np.float64), np.arange(n_raw) / NATIVE_RATE)
            out.append(Traj("droid", f"droid_full/ep{int(E[j]):06d}", qr, NATIVE_RATE, n_raw))
    if dropped:
        print(f"[droid] dropped {dropped} short (<{MIN_RAW_LEN}) episodes")
    print(f"[droid] loaded {len(out)} episodes")
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vae.data", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stream full-DROID proprio into resumable shards")
    f.add_argument("--max-files", type=int, default=None,
                   help=f"fetch only the first N of the {N_DATA_FILES} data files")
    f.add_argument("--overwrite", action="store_true", help="refetch shards that already exist")
    f.add_argument("--sleep", type=float, default=2.0, help="pause between files [s]")
    s = sub.add_parser("status", help="summarize the local shard cache")
    for p in (f, s):
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                       help="cache root; shards go to <cache-dir>/%s/" % DROID_SUBDIR)
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        fetch_droid_full(args.cache_dir, sleep=args.sleep, max_files=args.max_files,
                         overwrite=args.overwrite)
    else:
        shards = _shard_paths(args.cache_dir)
        n = sum(int(np.load(sp, allow_pickle=True)["n_eps"]) for sp in shards)
        print(f"{droid_dir(args.cache_dir)}: {len(shards)} shards, {n} episodes")


if __name__ == "__main__":
    main()
