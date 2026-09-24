"""Write LeRobot **v3.0** (video) datasets that match ``lerobot/droid_1.0.1``'s schema.

Used by ``build_dataset.py`` to emit datasets in the same format as DROID 1.0.1: camera frames are
stored as separate MP4 **videos** (not inline in the parquet), the low-dim columns use DROID's v3.0
names, and the training action is joint velocity + gripper. Datasets written here are consumed by
openpi's streaming loader (``openpi.training.streaming_dataset``, v3.0 path).

Design notes:
  * memory-bounded: camera frames are encoded to disk one frame at a time and the low-dim parquet is
    written incrementally (one row group per episode), so peak RAM is ~one episode's decoded images
    -- unlike lerobot's ``add_frame`` writer which buffered across episodes.
  * ``actions`` (joint velocity[7] + gripper[1]) is split into ``action.joint_velocity`` +
    ``action.gripper_position``; state (joint_position[7] + gripper[1]) into ``observation.state.*``.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Common frame-dict image key (as passed to ``add_episode``) -> DROID v3.0 video column name.
VIDEO_KEY_MAP = {
    "exterior_image_1_left": "observation.images.exterior_1_left",
    "exterior_image_2_left": "observation.images.exterior_2_left",
    "wrist_image_left": "observation.images.wrist_left",
}

# Fixed pyarrow schema for the v3.0 data parquet (DROID low-dim columns). Lists match DROID (variable
# list<float>), which openpi's v3.0 reader flattens per row.
_DATA_SCHEMA = pa.schema(
    [
        ("observation.state.joint_position", pa.list_(pa.float32())),
        ("observation.state.gripper_position", pa.list_(pa.float32())),
        ("observation.state", pa.list_(pa.float32())),
        ("action.joint_position", pa.list_(pa.float32())),
        ("action.joint_velocity", pa.list_(pa.float32())),
        ("action.gripper_position", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ]
)


def _feature_dict(
    height: int, width: int, video_keys=None, joint_dim: int = 7, gripper_dim: int = 1
) -> dict:
    """Feature spec. The defaults (``joint_dim``/``gripper_dim`` 7/1) are DROID's single 7-DOF arm."""
    def feat(dtype, shape):
        return {"dtype": dtype, "shape": list(shape), "names": None}

    keys = (video_keys or VIDEO_KEY_MAP).values()
    state_dim = joint_dim + gripper_dim
    return {
        **{v: feat("video", [height, width, 3]) for v in keys},
        "observation.state.joint_position": feat("float32", [joint_dim]),
        "observation.state.gripper_position": feat("float32", [gripper_dim]),
        "observation.state": feat("float32", [state_dim]),
        "action.joint_position": feat("float32", [joint_dim]),
        "action.joint_velocity": feat("float32", [joint_dim]),
        "action.gripper_position": feat("float32", [gripper_dim]),
        "action": feat("float32", [state_dim]),
        "timestamp": feat("float32", [1]),
        "frame_index": feat("int64", [1]),
        "episode_index": feat("int64", [1]),
        "index": feat("int64", [1]),
        "task_index": feat("int64", [1]),
    }


class V3DatasetWriter:
    """Incrementally assemble a LeRobot v3.0 (video) dataset in ``root`` (DROID schema).

    Call ``add_episode`` once per episode, then ``finalize``.
    """

    def __init__(
        self,
        root: str | Path,
        fps: int,
        *,
        height: int = 180,
        width: int = 320,
        robot_type: str = "panda",
        video_keys: dict[str, str] | None = None,
        joint_dim: int = 7,
        gripper_dim: int = 1,
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.height = int(height)
        self.width = int(width)
        self.robot_type = robot_type
        # Camera set and state width; the defaults are DROID's single 7-DOF arm.
        self.video_keys = dict(video_keys or VIDEO_KEY_MAP)
        self.joint_dim = int(joint_dim)
        self.gripper_dim = int(gripper_dim)
        self.action_dim = self.joint_dim + self.gripper_dim
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # One MP4 per camera (all episodes concatenated), encoded incrementally.
        self._encoders: dict[str, tuple] = {}
        for droid_key in self.video_keys.values():
            path = self.root / "videos" / droid_key / "chunk-000" / "file-000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(path), mode="w")
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20"}
            self._encoders[droid_key] = (container, stream)

        self._data_writer = pq.ParquetWriter(str(self.root / "data" / "chunk-000" / "file-000.parquet"), _DATA_SCHEMA)
        self._episode_rows: list[dict] = []
        self._tasks: dict[str, int] = {}
        self._global_index = 0
        self._frame_cursor = 0  # cumulative frames written to each camera MP4
        self._finalized = False

    # -- helpers ------------------------------------------------------------------------------
    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _encode(self, droid_key: str, image: np.ndarray) -> None:
        container, stream = self._encoders[droid_key]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image.astype(np.uint8)), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # -- main API -----------------------------------------------------------------------------
    def add_episode(
        self,
        *,
        images: dict[str, np.ndarray],
        joint_position: np.ndarray,
        gripper_position: np.ndarray,
        actions: np.ndarray,
        action_joint_position: np.ndarray,
        task: str,
    ) -> None:
        """Append one episode.

        Args:
            images: {common image key -> [N, H, W, 3] uint8}. Keys must be in ``self.video_keys``.
            joint_position: [N, joint_dim]; gripper_position: [N, gripper_dim];
                actions: [N, joint_dim + gripper_dim] (joint velocity, then gripper).
            action_joint_position: [N, joint_dim] COMMANDED/target joint positions (DROID 1.0.1 action.joint_position).
            task: language instruction for this episode.
        """
        joint_position = np.asarray(joint_position, dtype=np.float32).reshape(-1, self.joint_dim)
        gripper_position = np.asarray(gripper_position, dtype=np.float32).reshape(-1, self.gripper_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        action_joint_position = np.asarray(action_joint_position, dtype=np.float32).reshape(-1, self.joint_dim)
        n = len(joint_position)
        if n == 0:
            return
        if not (len(gripper_position) == n and len(actions) == n and len(action_joint_position) == n):
            raise ValueError(
                f"episode length mismatch: joint={n} gripper={len(gripper_position)} "
                f"act={len(actions)} act_jp={len(action_joint_position)}"
            )

        episode_index = len(self._episode_rows)
        task_index = self._task_index(task)
        from_ts = self._frame_cursor / self.fps

        # Encode video frames (per camera) for this episode.
        for common_key, droid_key in self.video_keys.items():
            frames = images[common_key]
            if len(frames) != n:
                raise ValueError(f"{common_key}: {len(frames)} frames vs {n} state rows")
            for i in range(n):
                self._encode(droid_key, np.asarray(frames[i]))
        self._frame_cursor += n
        to_ts = self._frame_cursor / self.fps

        # Low-dim rows -> one parquet row group.
        joint_velocity = actions[:, : self.joint_dim]
        gripper_action = actions[:, self.joint_dim :]
        state = np.concatenate([joint_position, gripper_position], axis=1)
        start = self._global_index
        table = pa.table(
            {
                "observation.state.joint_position": list(joint_position),
                "observation.state.gripper_position": list(gripper_position),
                "observation.state": list(state),
                "action.joint_position": list(action_joint_position),
                "action.joint_velocity": list(joint_velocity),
                "action.gripper_position": list(gripper_action),
                "action": list(actions),
                "timestamp": np.arange(n, dtype=np.float32) / self.fps,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, episode_index, dtype=np.int64),
                "index": np.arange(start, start + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            },
            schema=_DATA_SCHEMA,
        )
        self._data_writer.write_table(table)
        self._global_index += n

        ep_meta = {
            "episode_index": episode_index,
            "length": n,
            "tasks": [task],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
        }
        for droid_key in self.video_keys.values():
            ep_meta[f"videos/{droid_key}/chunk_index"] = 0
            ep_meta[f"videos/{droid_key}/file_index"] = 0
            ep_meta[f"videos/{droid_key}/from_timestamp"] = from_ts
            ep_meta[f"videos/{droid_key}/to_timestamp"] = to_ts
        self._episode_rows.append(ep_meta)

    def finalize(self) -> dict:
        """Flush encoders/parquet and write meta/info.json + meta/episodes + meta/tasks.parquet."""
        if self._finalized:
            return json.loads((self.root / "meta" / "info.json").read_text())
        for container, stream in self._encoders.values():
            for packet in stream.encode():  # flush
                container.mux(packet)
            container.close()
        self._data_writer.close()

        # Validate every encoded MP4 opens and has the expected frame count (a truncated / non-finalized
        # video would only fail later, with "Invalid data", at load time).
        for droid_key in self.video_keys.values():
            vpath = self.root / "videos" / droid_key / "chunk-000" / "file-000.mp4"
            try:
                with av.open(str(vpath)) as container:
                    n_frames = sum(1 for packet in container.demux(video=0) if packet.size)
            except Exception as exc:  # noqa: BLE001 - re-raise with context.
                raise RuntimeError(f"Encoded video {vpath} is unreadable ({exc!r}).") from exc
            if n_frames != self._global_index:
                raise RuntimeError(f"Encoded video {vpath} has {n_frames} frames, expected {self._global_index}.")

        pq.write_table(
            pa.Table.from_pylist(self._episode_rows),
            self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )
        ordered = sorted(self._tasks.items(), key=lambda kv: kv[1])
        pq.write_table(
            pa.table({"task_index": [i for _, i in ordered], "task": [t for t, _ in ordered]}),
            self.root / "meta" / "tasks.parquet",
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": len(self._episode_rows),
            "total_frames": self._global_index,
            "total_tasks": len(self._tasks),
            "fps": self.fps,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": _feature_dict(
                self.height, self.width, self.video_keys, self.joint_dim, self.gripper_dim
            ),
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        self._finalized = True
        return info
