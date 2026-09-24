"""Assemble the successful episodes of a data-collection run into a LeRobot v3.0 dataset and push it.

    python data_collection/build_dataset.py runs/place_toys_on_plate --repo-id OWNER/NAME \
        [--no-push] [--private] [--max-episodes N] [--out-root DIR] [--prompt TEXT]

Environment: run it with the openpi environment's interpreter, ``submodules/openpi/.venv/bin/python``
(created by ``GIT_LFS_SKIP_SMUDGE=1 uv sync`` in ``submodules/openpi``), which already provides
everything this needs: numpy, av, opencv-python-headless, pyarrow and huggingface_hub. Pushing needs
a Hugging Face token (``HF_TOKEN``, or a cached ``huggingface-cli login``).

Reads every ``<run dir>/success/<timestamp>/`` episode that tiptop-run wrote (``robot_state.npz``,
``_meta.json`` and the camera mp4s) and writes a dataset with ``lerobot/droid_1.0.1``'s schema (15 fps,
180x320 videos) to ``<out-root>/<repo-id>``, by default ``$HF_LEROBOT_HOME/<repo-id>``, replacing
whatever was there. Unless ``--no-push``, the dataset is then uploaded to the Hub and tagged ``v3.0``.

``action.joint_velocity`` is the capture's ``action_joint_velocity`` (DROID's normalized joint velocity,
5 * (cmd_joint_position - joint_position)) clipped to [-1, 1]; episodes without it are skipped.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import av
import numpy as np
from lerobot_v3 import V3DatasetWriter

logger = logging.getLogger("build_dataset")

FPS = 15
_IMG_HW = (180, 320)  # DROID LeRobot image size (H, W)
JOINT_DIM = 7


def _clip_joint_velocity(cmd_jv: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip an (already DROID-scale) joint-velocity command to the [-1, 1] action box.

    Returns the clipped [N, joints] array and the fraction of elements clipped at the rail (a high value
    means the source motion routinely exceeds DROID's velocity envelope).
    """
    frac_clipped = float(np.mean(np.abs(cmd_jv) > 1.0)) if cmd_jv.size else 0.0
    return np.clip(cmd_jv, -1.0, 1.0).astype(np.float32), frac_clipped


# Camera filenames tiptop writes -> LeRobot common image key.
_REAL_CAMERAS = {
    "exterior_image_1_left": "external_cam.mp4",
    "exterior_image_2_left": "external_cam_2.mp4",
    "wrist_image_left": "hand_cam.mp4",
}
_REQUIRED_CAMERAS = ("exterior_image_1_left", "wrist_image_left")
# A single-exterior rig duplicates exterior_2 from exterior_1.
_DUPLICATE_FROM = {"exterior_image_2_left": "exterior_image_1_left"}


def _resample_indices(n_src: int, n_dst: int) -> np.ndarray:
    """Indices that evenly sample n_dst points from a sequence of length n_src (nearest)."""
    if n_dst <= 1 or n_src <= 1:
        return np.zeros(max(n_dst, 0), dtype=int)
    return np.clip(np.round(np.arange(n_dst) * (n_src - 1) / (n_dst - 1)), 0, n_src - 1).astype(int)


def _camera_indices(frame_time: np.ndarray, n_cam: int, record_start: float, record_stop: float) -> np.ndarray:
    """Camera-frame index per state frame, aligned by wall clock.

    Cameras start before execution begins and stop after it ends, so the recording window brackets the
    state window; mapping each state frame's ``frame_time`` onto the video by timestamp (rather than
    stretching the state timeline proportionally across the whole clip) keeps images in step with actions.
    ``frame_time`` is float64 epoch seconds -- never cast it to float32.
    """
    eff_fps = n_cam / (record_stop - record_start)
    idx = np.round((frame_time - record_start) * eff_fps)
    return np.clip(idx, 0, n_cam - 1).astype(int)


def _finite_float(x):
    """``float(x)`` when it is a finite real number, else ``None`` (for optional _meta.json fields)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _decode_resized(path: str, hw) -> list:
    """Decode an mp4 into an ordered list of HWC uint8 RGB frames, resized to (H, W)."""
    import cv2

    h, w = hw
    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    return frames


def _decode_cameras(ep_dir: Path):
    """Decoded {common_key -> frames} for the episode's cameras, or None if a required one is absent.

    exterior_1 and the wrist camera are required; exterior_2 is duplicated from exterior_1 when the rig
    has a single exterior camera.
    """
    decoded = {}
    for common_key, fname in _REAL_CAMERAS.items():
        mp4 = ep_dir / fname
        if not mp4.is_file():
            continue
        # An unreadable (e.g. zero-byte or truncated) or 0-frame mp4 is treated as a MISSING camera, so
        # one bad video skips just this episode instead of aborting the whole build.
        try:
            frames = _decode_resized(str(mp4), _IMG_HW)
        except (av.error.FFmpegError, IndexError, ValueError, OSError) as exc:
            logger.warning(f"{ep_dir.name}: {fname} unreadable ({exc!r}); treating camera as missing")
            continue
        if not frames:
            logger.warning(f"{ep_dir.name}: {fname} decoded to 0 frames; treating camera as missing")
            continue
        decoded[common_key] = frames
    missing = [k for k in _REQUIRED_CAMERAS if k not in decoded]
    if missing:
        logger.warning(f"{ep_dir.name}: missing required video(s) {missing}; skipping episode")
        return None
    for target, source in _DUPLICATE_FROM.items():
        if target not in decoded:
            decoded[target] = decoded[source]
    return decoded


def _lerobot_home() -> Path:
    """lerobot's ``HF_LEROBOT_HOME``: ``$HF_LEROBOT_HOME``, else ``$HF_HOME/lerobot`` (same resolution)."""
    from huggingface_hub.constants import HF_HOME

    if "LEROBOT_HOME" in os.environ:
        raise ValueError("LEROBOT_HOME is deprecated (lerobot refuses it); set HF_LEROBOT_HOME instead")
    return Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME) / "lerobot")).expanduser()


def _upload_dataset(dataset_root: Path, repo_id: str, private: bool) -> None:
    """Stream a finished local LeRobot dataset folder to the HF Hub with low RAM (HfApi.upload_folder)."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None  # None -> HfApi falls back to the cached token
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info(f"Uploading {dataset_root} -> HF dataset {repo_id} (streaming, low-RAM)")
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(dataset_root),
        commit_message="Add LeRobot dataset (streamed)",
    )
    # LeRobotDataset(repo_id) needs a git tag matching info.json's codebase_version, else it raises
    # RevisionNotFoundError; upload_folder doesn't create it.
    version = json.loads((dataset_root / "meta" / "info.json").read_text()).get("codebase_version")
    if version:
        api.create_tag(repo_id, tag=version, repo_type="dataset", exist_ok=True)
    logger.info(f"Uploaded {repo_id} and tagged {version}")


def build_dataset(
    run_dir, *, repo_id: str, out_root, push: bool, private: bool, max_episodes, prompt: str = ""
) -> int:
    """Build (and optionally push) the dataset from ``run_dir/success``. Returns episodes written.

    ``prompt`` is the task for episodes whose ``_meta.json`` has no instruction (tiptop always writes one).
    """
    success_dir = Path(run_dir) / "success"
    ep_dirs = []
    if success_dir.is_dir():
        ep_dirs = sorted(d for d in success_dir.glob("*") if d.is_dir() and (d / "robot_state.npz").is_file())
    if max_episodes is not None:
        ep_dirs = ep_dirs[:max_episodes]
    if not ep_dirs:
        logger.error(f"No success episodes with robot_state.npz under {success_dir}")
        return 0

    dataset_root = (Path(out_root) / repo_id) if out_root else (_lerobot_home() / repo_id)
    if dataset_root.exists():
        logger.info(f"Removing existing local dataset at {dataset_root}")
        shutil.rmtree(dataset_root)

    writer = None
    n_written = 0
    for ep in ep_dirs:
        with np.load(ep / "robot_state.npz") as sd:
            jp_raw = sd["joint_position"].astype(np.float32)
            if jp_raw.ndim == 2 and jp_raw.shape[1] != JOINT_DIM:
                logger.error(
                    f"{ep.name}: {jp_raw.shape[1]} joint columns, expected a single {JOINT_DIM}-DOF arm; skipping"
                )
                continue
            if "action_joint_velocity" not in sd.files:
                logger.error(f"{ep.name}: no action_joint_velocity in robot_state.npz; skipping")
                continue
            jp = jp_raw.reshape(-1, JOINT_DIM)
            gp = sd["gripper_position"].astype(np.float32).reshape(-1, 1)
            cmd_jp = sd["cmd_joint_position"].astype(np.float32).reshape(-1, JOINT_DIM)
            action_jv = sd["action_joint_velocity"].astype(np.float32).reshape(-1, JOINT_DIM)
            cmd_g_raw = sd["cmd_gripper"]
            cmd_g_shape = tuple(cmd_g_raw.shape)
            cmd_g = cmd_g_raw.astype(np.float32).reshape(-1, 1)
            # frame_time is the master wall-clock timeline (epoch seconds). Keep it float64: float32 near
            # the current epoch (~1.78e9) has ~128 s resolution and silently collapses every frame to one
            # timestamp. Never cast it to float32.
            frame_time = (
                np.asarray(sd["frame_time"], dtype=np.float64).reshape(-1) if "frame_time" in sd.files else None
            )
        n = len(jp)
        if n < 2:
            logger.warning(f"{ep.name}: only {n} frames; skipping")
            continue
        if not (len(gp) == len(cmd_jp) == len(cmd_g) == n):
            logger.warning(f"{ep.name}: state arrays disagree on length; skipping")
            continue
        if len(action_jv) != n:
            # Refuse rather than trust: a short/long action array would otherwise reach
            # np.concatenate and either throw or, worse, misalign actions against frames.
            logger.warning(
                f"{ep.name}: action_joint_velocity has {len(action_jv)} rows but the episode has "
                f"{n}; skipping (a partially written robot_state.npz)"
            )
            continue
        # Harden the action gripper: cmd_gripper becomes action[:, 7]. A non-[F]/[F,1] shape or a
        # non-binary value means the capture wrote a *continuous* gripper, so skip the episode loudly
        # rather than poison the dataset with a continuous gripper action.
        if len(cmd_g_shape) > 2 or (len(cmd_g_shape) == 2 and cmd_g_shape[1] != 1):
            logger.error(f"{ep.name}: cmd_gripper has shape {cmd_g_shape}, expected [F] or [F,1]; skipping episode")
            continue
        if not np.all((cmd_g == 0.0) | (cmd_g == 1.0)):
            bad = np.unique(cmd_g[(cmd_g != 0.0) & (cmd_g != 1.0)])
            logger.error(
                f"{ep.name}: cmd_gripper is not binary 0.0/1.0 (offending values {bad[:8].tolist()}); "
                f"skipping episode to avoid a continuous gripper action"
            )
            continue
        task = prompt
        record_start = record_stop = None
        meta_path = ep / "_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            task = meta.get("instruction", prompt)
            record_start = _finite_float(meta.get("record_start"))
            record_stop = _finite_float(meta.get("record_stop"))

        decoded = _decode_cameras(ep)
        if decoded is None:
            continue
        for common_key, frames in decoded.items():
            if abs(len(frames) - n) > 2:
                logger.warning(f"{ep.name}: {common_key} has {len(frames)} frames vs {n} state frames; resampling")

        # Align each camera to the state timeline. Prefer wall-clock alignment (frame_time against the
        # camera recording window in _meta.json); cameras can differ in frame count, so compute per camera.
        # Fall back to proportional resampling for legacy episodes without a usable recording window.
        use_wallclock = (
            frame_time is not None
            and len(frame_time) == n
            and record_start is not None
            and record_stop is not None
            and record_stop > record_start
        )
        if not use_wallclock:
            logger.warning(
                f"{ep.name}: no usable record_start/record_stop + frame_time in _meta.json; "
                f"falling back to proportional camera resampling"
            )
        aligned = {}
        for common_key in _REAL_CAMERAS:
            frames = decoded[common_key]
            if use_wallclock:
                idx = _camera_indices(frame_time, len(frames), record_start, record_stop)
            else:
                idx = _resample_indices(len(frames), n)
            aligned[common_key] = np.stack([frames[i] for i in idx])

        cmd_jv_norm, frac_clipped = _clip_joint_velocity(action_jv)
        if frac_clipped > 0.01:
            logger.warning(
                f"{ep.name}: {frac_clipped:.1%} of joint-velocity elements exceeded the [-1, 1] "
                f"envelope and were clipped"
            )
        actions = np.concatenate([cmd_jv_norm, cmd_g], axis=1).astype(np.float32)  # [N, joints + gripper]
        # Created only once an episode has passed every check, so an all-skipped run writes nothing.
        if writer is None:
            writer = V3DatasetWriter(dataset_root, FPS)
        writer.add_episode(
            images=aligned,
            joint_position=jp,
            gripper_position=gp,
            actions=actions,
            action_joint_position=cmd_jp,
            task=task,
        )
        n_written += 1
        logger.info(f"  [{n_written}/{len(ep_dirs)}] {ep.name}: {n} frames | task: {task!r}")

    if writer is None:
        logger.error("Every episode was skipped; nothing written")
        return 0
    writer.finalize()
    logger.info(f"Wrote {n_written} episodes to {dataset_root}")
    if push and n_written:
        _upload_dataset(dataset_root, repo_id, private)
    return n_written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build (and push) a LeRobot v3.0 dataset from a data-collection run.")
    ap.add_argument("run_dir", type=Path, help="tiptop-run output dir holding success/<timestamp>/ episodes")
    ap.add_argument("--repo-id", required=True, help="HF dataset repo, OWNER/NAME (also the local dataset dir name)")
    ap.add_argument("--no-push", dest="push", action="store_false", help="only build the local dataset")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    ap.add_argument("--max-episodes", type=int, default=None, help="use only the first N episodes (by timestamp)")
    ap.add_argument("--out-root", default=None, help="local build root (default: $HF_LEROBOT_HOME)")
    ap.add_argument("--prompt", default="", help="task for episodes whose _meta.json has no instruction")
    args = ap.parse_args()

    # --repo-id is also the local directory that gets replaced, so reject anything that is not exactly
    # OWNER/NAME (e.g. "", ".", "me", "me/", "a/../b") before anything is deleted.
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    owner, sep, name = args.repo_id.partition("/")
    if not (sep and owner and name and "/" not in name):
        ap.error(f"--repo-id must be OWNER/NAME, got {args.repo_id!r}")
    try:
        validate_repo_id(args.repo_id)
    except HFValidationError as exc:
        ap.error(f"--repo-id {args.repo_id!r} is not a valid Hugging Face repo id: {exc}")

    n = build_dataset(
        args.run_dir, repo_id=args.repo_id, out_root=args.out_root,
        push=args.push, private=args.private, max_episodes=args.max_episodes, prompt=args.prompt,
    )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
