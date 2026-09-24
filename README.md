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

```bash
cd submodules/FoundationStereo
bash build_server.sh
pixi run python server.py --port 1234
```

### 4. Robot

On the NUC, start polymetis's robot server (port 50051) and gripper server (port 50052), for example with
DROID's `droid/franka/launch_robot.sh` and `launch_gripper.sh`. Then copy
[`bamboo_polymetis_shim.py`](https://github.com/SamratSahoo/tiptop/blob/313c5da55c7a9992a9d9b884f6e4858666c6253d/bamboo_polymetis_shim.py)
to the NUC and run it in an environment with polymetis, pyzmq, msgpack, numpy and scipy:

```bash
# on the NUC
python bamboo_polymetis_shim.py
```

Check that its log shows `PolymetisGripper connected to localhost:50052`. Keep the shim running during
calibration and collection.

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

### 6. openpi environment

```bash
cd submodules/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If `gsutil` is on `PATH` with an expired login, take it off `PATH` so openpi fetches `gs://openpi-assets`
anonymously.

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

On the robot side, stop the shim and bring up DROID's standard NUC stack. On the DROID control laptop, set
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
