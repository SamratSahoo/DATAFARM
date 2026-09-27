# DATAFARM

**DATAFARM** makes task and motion planning (TAMP) a useful data source for fine-tuning vision-language-action
(VLA) models. DATAFARM uses the pretraining distribution to guide trajectory generation, aligning joint
configurations, motion style and timing.

## Setup

**Requirements:** a DROID-style rig (Franka FR3 or Panda, Robotiq 2F-85, the NUC, a wrist and an external
ZED camera), a Linux x86-64 workstation with an NVIDIA GPU and CUDA 12, and [pixi](https://pixi.sh),
[uv](https://docs.astral.sh/uv/) and git-lfs.

Unless noted, every code block starts from the repository root on the workstation.

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/SamratSahoo/DATAFARM.git
cd DATAFARM
```

### 2. tiptop environment

```bash
cd submodules/tiptop
pixi install
pixi run setup-planners   # builds ../curobo and ../cuTAMP into the env
pixi run install-zed      # needs the ZED SDK in /usr/local/zed
pixi run tiptop-run -h    # import check
```

### 3. Perception servers

tiptop expects M2T2 on port 8123 and FoundationStereo on port 1234 on localhost (to use another machine, edit
`perception.*.url` in `submodules/tiptop/tiptop/config/tiptop.yml`). Run each in its own terminal:

```bash
cd submodules/M2T2
TORCH_CUDA_ARCH_LIST=8.9 bash build_server.sh   # set your GPU's compute capability
pixi run python server.py --port 8123
```

FoundationStereo needs pretrained weights. Download the `23-51-11` folder from
[Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf) and put the whole folder
in `submodules/FoundationStereo/pretrained_models/`, so that
`pretrained_models/23-51-11/model_best_bp2.pth` and `cfg.yaml` are both there. The server loads that path by default.
Without the weights it starts in "unconfigured" mode.

```bash
cd submodules/FoundationStereo
bash build_server.sh
pixi run python server.py --port 1234
```

### 4. Robot

The NUC runs two programs: DROID's server, which starts polymetis for the arm and gripper, and tiptop's
shim, which tiptop uses to control the arm. Run steps 1–4 on the NUC.

**1. Install DROID.** Follow DROID's NUC guide
([Docker](https://github.com/SamratSahoo/droid/blob/053e5b328e5550e49387798a0147ecdbde9e5f75/docs/software-setup/docker.md) or
[host](https://github.com/SamratSahoo/droid/blob/053e5b328e5550e49387798a0147ecdbde9e5f75/docs/software-setup/host-installation.md)) using the fork:

```bash
git clone --recurse-submodules https://github.com/SamratSahoo/droid.git
```

The guide's "Configure Parameters" step sets `robot_ip` (the arm's control box) and `sudo_password` in
[`droid/misc/parameters.py`](https://github.com/SamratSahoo/droid/blob/053e5b328e5550e49387798a0147ecdbde9e5f75/droid/misc/parameters.py). The server needs both.

**2. Add tiptop's shim.** In the DROID checkout, with DROID's polymetis environment active:

```bash
curl -LO https://raw.githubusercontent.com/SamratSahoo/tiptop/1d3dedf2e9880decb98b9d4669e050f4525c4157/bamboo_polymetis_shim.py
pip install pyzmq msgpack
```

**3. Start DROID's server.** In one terminal:

```bash
python scripts/server/run_server.py
```

This starts polymetis's robot server (port 50051) and gripper server (port 50052), replacing any that are
already running.

**4. Start the shim.** In a second terminal:

```bash
python bamboo_polymetis_shim.py
```

Its log should show `PolymetisGripper connected to localhost:50052`. It listens on ports 5555 (control),
5557 (state) and 5559 (gripper).

Keep both terminals running during calibration and collection. Repeat steps 3 and 4 after the NUC
restarts.

### 5. Configure and calibrate tiptop

```bash
export GOOGLE_API_KEY=...                 # Gemini (GEMINI_API_KEY also works)
export TIPTOP_ROBOT_HOST=<NUC IP>         # default 172.16.0.2
export TIPTOP_HAND_CAMERA_ID=<serial>     # wrist ZED
export TIPTOP_EXTERNAL_CAMERA_ID=<serial> # external ZED
```

Follow
[`docs/getting-started.md`](https://github.com/SamratSahoo/tiptop/blob/313c5da55c7a9992a9d9b884f6e4858666c6253d/docs/getting-started.md)
from `submodules/tiptop`, but skip "Start the Bamboo controller server" (the shim replaces it). Set the
workspace obstacles in `tiptop/workspace.py` and the capture pose `robot.q_capture` in
`tiptop/config/tiptop.yml`, then:

```bash
cd submodules/tiptop
pixi run calibrate-wrist-cam
pixi run compute-gripper-mask   # or: pixi run paint-gripper-mask
pixi run viz-calibration
```

See [Camera calibration](#camera-calibration) for where the wrist camera's extrinsics are stored and how\n`calibrate-wrist-cam` sets them.

### 6. openpi environment

```bash
cd submodules/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If `gsutil` is on `PATH` with an expired login, take it off `PATH` so openpi fetches `gs://openpi-assets`
anonymously.

## Camera calibration

DATAFARM perceives the scene through the wrist camera, so its extrinsics are the only ones you need to
set. The external camera is recorded for the dataset but never used for planning, so it needs no
calibration.

### Where the extrinsics live

tiptop reads extrinsics from `submodules/tiptop/tiptop/config/assets/calibration_info.json`, keyed by
camera serial (`TIPTOP_HAND_CAMERA_ID` for the wrist camera).

The wrist camera's entry is `ee_from_cam`, the pose of the left ZED lens relative to the end effector, as
`[x, y, z, roll, pitch, yaw]` in meters and radians (scipy `"xyz"` Euler angles):

```json
"13222437": {
  "pose": [0.0315, 0.0681, -0.1314, -0.3759, 0.0050, 3.1204],
  "timestamp": 1789669300.29
}
```

`tiptop-run` fails on startup if the wrist camera's serial has no entry.

### Setting them

Use `calibrate-wrist-cam` rather than editing the file by hand. It needs the robot (the NUC's DROID
server and the shim) running:

1. Fix the DROID ChArUco board to the table. The board size is set at the top of
   `submodules/tiptop/tiptop/scripts/calibrate_wrist_cam.py` (14 × 9 squares, 20 mm checkers, 15 mm
   markers); edit it if your board differs.
2. Put the arm in Programming mode in Franka Desk and guide it so the board is centered in the wrist
   camera's view, about 30–60 cm away. Then switch back to Execution mode.
3. Run the calibration:

   ```bash
   cd submodules/tiptop
   TIPTOP_CALIB_VIZ=1 pixi run calibrate-wrist-cam
   ```

   With `TIPTOP_CALIB_VIZ=1` it shows the camera feed and waits for `y` before moving. Without it, it runs
   headless and starts moving the arm 3 seconds after launch. The arm sweeps around its current pose for
   2–3 minutes, then the script writes the entry to `calibration_info.json`. If the fit isn't accurate
   enough, it raises an error and writes nothing.
4. Check the result:

   ```bash
   pixi run viz-calibration
   ```

   The point cloud should line up with the robot model and the table in Rerun.

Recalibrate whenever the wrist camera is bumped, remounted or swapped. A new unit has a new serial, so it
needs its own entry. The calibration file is part of the tiptop submodule, so commit changes there and
then update the submodule pointer in this repository.

## Pipeline

### 1. Trajectory encoder (optional)

The task configs already use `encoder/checkpoints/encoder.pt`. To train your own:

```bash
pip install -r encoder/requirements.txt
python -m encoder.data fetch     # DROID proprio, ~4 GB streamed into encoder/data_cache/
python -m encoder.train          # -> encoder/outputs/encoder.pt
```

To use it, point `encoder_path` in the task configs at the new checkpoint. See
[`encoder/README.md`](encoder/README.md) for the options.

### 2. Collect demonstrations

With the perception servers and the shim running (any Python 3.10+ with PyYAML):

```bash
python data_collection/collect.py data_collection/configs/place_toys_on_plate.yaml
# --output-dir DIR (default runs/<config>); arguments after -- go to tiptop-run, e.g. -- --max-planning-time 120
```

The robot moves as soon as the session starts, so keep a hand on the e-stop.

- After each rollout: `y` = success, `n` = failure, Enter = leave unlabelled.
- At the task prompt: Enter repeats the config's prompt, `q` ends the session (`home`, `open` and `reset`
  also work).
- Ctrl-C aborts the current rollout and returns to the task prompt (a motion segment already executing
  finishes first).

Successful episodes land in `runs/<config>/success/<timestamp>/`. Re-run the same command to continue.

### 3. Build the dataset

```bash
submodules/openpi/.venv/bin/python data_collection/build_dataset.py runs/place_toys_on_plate \
    --repo-id <hf-user>/place_toys_on_plate    # add --no-push to only build locally ($HF_LEROBOT_HOME/<repo-id>)
```

Pushing needs `HF_TOKEN` or `huggingface-cli login`.

### 4. Generate filters for a new dataset

The three task datasets and DROID already have filters in `vla/filters/`. For a new dataset:

```bash
cd vla
uv run --project ../submodules/openpi python ../submodules/openpi/examples/droid/compute_droid_nonidle_ranges_streaming.py \
    --repo-id <hf-user>/<dataset> --out filters/<name>.json
```

Then copy a config in `vla/configs/` and replace the task dataset's repo id in `repo_id`,
`nonidle_filter_paths` (pointing it at `filters/<name>.json`) and `sampling_weights`.

### 5. Train

```bash
vla/train.sh place_toys_on_plate --exp-name=my_run   # other flags go to openpi's train.py, e.g. --fsdp-devices=<n>
```

Checkpoints go to `vla/checkpoints/<config>/<exp-name>/<step>/`.

### 6. Run the VLA

Start the policy server on the workstation:

```bash
cd vla
OPENPI_CONFIG_DIR=$PWD/configs uv run --project ../submodules/openpi python ../submodules/openpi/scripts/serve_policy.py \
    policy:checkpoint --policy.config=place_toys_on_plate --policy.dir=checkpoints/place_toys_on_plate/my_run/19999
```

On the robot side, stop the shim and leave DROID's server running (Setup step 4). On the DROID control laptop, set
up openpi's DROID client as in
[`examples/droid/README.md`](https://github.com/SamratSahoo/openpi/blob/c4b2b1bf507e1ddd894902b2771bc6b224fb1873/examples/droid/README.md)
(step 2: install `packages/openpi-client` and copy `examples/droid/main.py` to `$DROID_ROOT/scripts/`),
then run it:

```bash
# on the DROID laptop, in the DROID conda env
cd $DROID_ROOT
python3 scripts/main.py --remote_host=<workstation IP> --remote_port=8000 \
    --left_camera_id=<serial> --right_camera_id=<serial> --wrist_camera_id=<serial> \
    --external_camera=left --max_timesteps=1800 --open_loop_horizon=6
```

At `Enter instruction:`, type the task's prompt from the table below, e.g.
`Place the toys on the plate with no collisions`. Start each rollout from tiptop's `robot.q_capture`, not
the DROID home pose the client resets to.

## Tasks and released artifacts

| Task | Prompt | Data config | Dataset | VLA Training Config |
|---|---|---|---|---|
| Place toys on plate | Place the toys on the plate with no collisions | [`place_toys_on_plate.yaml`](data_collection/configs/place_toys_on_plate.yaml) | [Link](https://huggingface.co/datasets/SamratSahoo/1_pp_toys_plate_vae_style_timing_apex_learned_posture_v2_prpl) | [`place_toys_on_plate`](vla/configs/place_toys_on_plate.yaml) |
| Sort fruits and toys | Sort the fruits into the green bowl and toys into the blue bowl | [`sort_fruits_and_toys.yaml`](data_collection/configs/sort_fruits_and_toys.yaml) | [Link](https://huggingface.co/datasets/SamratSahoo/3_pp_sort_fruits_toys_vae_style_timing_apex_learned_posture_v2_prpl) | [`sort_fruits_and_toys`](vla/configs/sort_fruits_and_toys.yaml) |
| Pack toys | Pack the toys onto the wooden tray | [`pack_toys.yaml`](data_collection/configs/pack_toys.yaml) | [Link](https://huggingface.co/datasets/SamratSahoo/4_pack_toys_vae_style_timing_apex_learned_posture_v2_prpl) | [`pack_toys`](vla/configs/pack_toys.yaml) |

## Licenses and acknowledgements

This repository is released under the [MIT License](LICENSE).
