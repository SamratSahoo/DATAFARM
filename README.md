# DATAFARM

DATAFARM generates robot demonstrations with task-and-motion planning (TAMP) and uses them to fine-tune a
vision-language-action (VLA) policy. On a real Franka arm, TiPToP perceives the scene and grounds the
language task, cuTAMP plans the pick-and-place sequence, and cuRobo optimizes every motion. Planner motion
moves differently from human teleoperation, so cuRobo's trajectory optimization gets an extra cost from a
small trajectory-style VAE: the squared Mahalanobis distance of each motion's latent to the DROID latent
cluster, which pulls the motion toward DROID-like style and timing. The same VAE then re-times every gripper-to-gripper
stroke. Successful rollouts are recorded in DROID's format, converted to LeRobot datasets, and used to
fine-tune π0.5-DROID with openpi, co-trained 50/50 with DROID.

## Repository layout

```
DATAFARM/
├── data_collection/
│   ├── configs/             task configs: prompt, episode target, tiptop overrides
│   ├── collect.py           starts a recording tiptop-run session for one config
│   ├── build_dataset.py     successful episodes -> LeRobot v3.0 dataset (optionally pushed to the Hub)
│   └── lerobot_v3.py        LeRobot v3.0 writer used by build_dataset.py
├── vae/                     trajectory style VAE: DROID fetch, features, model, training (see vae/README.md)
│   └── checkpoints/vae.pt   VAE checkpoint the data-collection configs use (vae_path)
├── vla/
│   ├── configs/             openpi training configs (file name = config name)
│   ├── filters/             idle-frame filters for DROID and the three task datasets
│   └── train.sh             runs openpi training with these configs
└── submodules/
    ├── tiptop/              TiPToP: perception, planning, execution and recording (pixi env)
    ├── cuTAMP/              GPU task-and-motion planner, installed into the tiptop env
    ├── curobo/              GPU motion generation with the VAE manifold cost, installed into the tiptop env
    ├── M2T2/                grasp-prediction HTTP server
    ├── FoundationStereo/    stereo-depth HTTP server
    └── openpi/              π0.5 training and policy serving (uv env)
```

## Setup

**Hardware.** A DROID-style setup: a Franka FR3 or Panda with a Robotiq 2F-85 gripper, the NUC that runs
the arm controller, a wrist ZED camera and an external ZED camera, and a Linux x86-64 workstation with an
NVIDIA GPU and CUDA 12. The tiptop, M2T2 and FoundationStereo environments are linux-64 only. Full
fine-tuning of π0.5 needs more than 70 GB of GPU memory; `--fsdp-devices` shards it across GPUs.

**Tools.** [pixi](https://pixi.sh), [uv](https://docs.astral.sh/uv/) and git-lfs.

Unless noted, every code block below starts from the repository root on the workstation.

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
pixi run install-zed      # ZED Python API; needs the ZED SDK in /usr/local/zed
pixi run tiptop-run -h    # import check
```

### 3. Perception servers

tiptop expects M2T2 at `http://localhost:8123` and FoundationStereo at `http://localhost:1234`
(`perception.*.url` in `submodules/tiptop/tiptop/config/tiptop.yml`; edit the URLs if the servers run on
another machine). Each fork's `build_server.sh` creates the env, installs torch 2.7.1+cu128 and downloads
the weights (see [Licenses and acknowledgements](#licenses-and-acknowledgements)).

```bash
# M2T2 (grasps). TORCH_CUDA_ARCH_LIST is your GPU's compute capability (default 8.9).
cd submodules/M2T2
TORCH_CUDA_ARCH_LIST=8.9 bash build_server.sh
pixi run python server.py --port 8123
```

```bash
# FoundationStereo (depth), in a second terminal. Its server defaults to port 8124, so pass --port 1234.
cd submodules/FoundationStereo
bash build_server.sh
pixi run python server.py --port 1234
```

Both servers are required: tiptop gets ZED depth from FoundationStereo, and it checks that both servers are
`healthy` before every rollout.

### 4. Robot

The workstation drives the arm with the Bamboo client (the NUC at `TIPTOP_ROBOT_HOST`, default
`172.16.0.2`, ports 5555 and 5559). For data collection, run tiptop's
[`bamboo_polymetis_shim.py`](https://github.com/SamratSahoo/tiptop/blob/d193fee525b32363c68864effb2eebcebe2bc4d3/bamboo_polymetis_shim.py)
on the NUC. It serves the Bamboo protocol on top of polymetis: the arm goes through polymetis's robot server
(port 50051) and the Robotiq through its gripper server (`launch_gripper.py`, port 50052), so start both
first, as DROID's `droid/franka/launch_robot.sh` and `launch_gripper.sh` do. The shim also serves joint
states on port 5557, which the recorder samples during execution. Without that port, episodes are saved
without `robot_state.npz` and cannot be turned into a dataset. Copy the file to the NUC and run it in an
environment where `polymetis` is importable, with pyzmq, msgpack, numpy and scipy installed:

```bash
# on the NUC, in the directory you copied the file to
python bamboo_polymetis_shim.py
```

Check that its log shows `PolymetisGripper connected to localhost:50052`. Without the gripper server, the
default `--gripper-backend auto` falls back to driving `/dev/ttyUSB0` directly over Modbus (needs pymodbus
2.x, and reads the width on a slightly different scale than the released data). If that fails too, it logs
`using STUB gripper (motions are no-ops)` and keeps running with a gripper that never moves.

### 5. Configure and calibrate tiptop

```bash
export GOOGLE_API_KEY=...                 # Gemini (GEMINI_API_KEY also works)
export TIPTOP_ROBOT_HOST=<NUC IP>         # default 172.16.0.2
export TIPTOP_HAND_CAMERA_ID=<serial>     # wrist ZED
export TIPTOP_EXTERNAL_CAMERA_ID=<serial> # external ZED, recorded as DROID exterior_image_1
```

These override the lab rig's values in `submodules/tiptop/tiptop/config/tiptop.yml`, which remain the
defaults.

The shipped calibration files (under `submodules/tiptop/tiptop/config/`) are also the lab rig's: its
latest wrist extrinsics are in `assets/calibration_info_prpl.json`, which tiptop layers over
`assets/calibration_info.json` only when `DC_WORKSPACE=prpl` is set. Replace them with your own. With `DC_WORKSPACE` unset, `calibrate-wrist-cam`
writes, and tiptop-run reads, `calibration_info.json`. From `submodules/tiptop`, follow
[`docs/getting-started.md`](https://github.com/SamratSahoo/tiptop/blob/d193fee525b32363c68864effb2eebcebe2bc4d3/docs/getting-started.md),
skipping "Start the Bamboo controller server" (the shim from step 4 replaces it, and it must be running for
calibration too):

- `tiptop/workspace.py`: static workspace obstacles
- `robot.q_capture`: capture pose
- `pixi run calibrate-wrist-cam`: wrist-camera extrinsics, keyed by serial
- `pixi run compute-gripper-mask` (or `paint-gripper-mask`): gripper mask
- `pixi run viz-calibration`: check the result

### 6. openpi environment

```bash
cd submodules/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

`build_dataset.py` also runs in this environment. openpi fetches `gs://openpi-assets` (the π0.5-DROID base
weights and norm stats) with `gsutil` whenever `gsutil` is on `PATH`; if its login has expired, take it off
`PATH` to use anonymous access.

## Pipeline

### 1. Style VAE

`vae/checkpoints/vae.pt` is the checkpoint the data-collection configs use (`vae_path`), so this step is
optional. To train a VAE:

```bash
pip install -r vae/requirements.txt
python -m vae.data fetch     # DROID proprio, ~4 GB streamed, cached in vae/data_cache/
python -m vae.train          # -> vae/outputs/vae.pt + vae/outputs/vae_report.json
```

Training uses only DROID. The β-VAE is trained self-supervised on DROID episodes and random crops of them,
regressing each crop's style fingerprint, and the epoch with the lowest loss on held-out DROID episodes is
kept. The DROID latent mean, covariance and precision that the cost scores against are then baked into the
checkpoint. To use a newly trained checkpoint, point `vae_path` in the data-collection configs at it. See
[`vae/README.md`](vae/README.md) for the model, the training recipe and the checkpoint format cuRobo loads.

### 2. Collect demonstrations

With the perception servers and the NUC shim running, start a session for one task (any Python 3.10+
with PyYAML):

```bash
python data_collection/collect.py data_collection/configs/place_toys_on_plate.yaml
# options: --output-dir DIR (default runs/<config name>); arguments after -- go to tiptop-run,
# e.g. -- --max-planning-time 120
```

`collect.py` makes `vae_path` absolute, writes the overrides to `<output-dir>/tamp_overrides.json`, prints
how many episodes are collected out of `num_episodes`, and starts `tiptop-run --enable-recording` in the
tiptop env. The config's prompt is the first task. The robot moves as soon as the session starts, so keep a
hand on the e-stop. The operator:

- stages the scene; tiptop then perceives, plans, executes and records
- labels each rollout: `y` moves it to `success/`, `n` to `failure/`, and Enter leaves it in `eval/`
- at the next-task prompt, presses Enter to repeat the prompt or `q` to end the session (`home`, `open`
  and `reset` are also accepted)
- presses Ctrl-C to abort the current rollout and go back to the prompt; the session stays warm. A motion
  segment that is already executing still runs to its end.

Re-run the same command to continue a run. Each episode is written to
`runs/<config>/success/<timestamp>/`, with `robot_state.npz`, `tiptop_plan.json`, `_meta.json`,
`hand_cam.mp4` and `external_cam.mp4`.

### 3. Build the LeRobot dataset

```bash
submodules/openpi/.venv/bin/python data_collection/build_dataset.py runs/place_toys_on_plate \
    --repo-id <hf-user>/place_toys_on_plate    # add --no-push to only build locally
```

This converts every `success/` episode that has a `robot_state.npz` into a LeRobot v3.0 dataset with
`lerobot/droid_1.0.1`'s schema (15 fps, 180x320 video). The dataset is written to
`$HF_LEROBOT_HOME/<repo-id>` (`--out-root` to change), replacing any existing copy there, and then pushed
to the Hub and tagged `v3.0`. Pushing needs `HF_TOKEN` or `huggingface-cli login`.

### 4. Train and serve the VLA

```bash
vla/train.sh place_toys_on_plate --exp-name=my_run   # other flags go to openpi's train.py
```

Each config fine-tunes `pi05_droid` for 20,000 steps at batch 256, sampling the task dataset and DROID
50/50. Both are streamed from the Hub with idle frames removed (`vla/filters/`). Norm stats come from
`pi05_droid`, so there is no norm-stats step. `train.sh` runs from `vla/`, so checkpoints go to
`vla/checkpoints/<config>/<exp-name>/<step>/`; the last one is step 19999. W&B logging is on by default:
run `wandb login`, set `WANDB_MODE=offline`, or pass `--no-wandb-enabled`.

To train on your own dataset, copy a config, replace the task repo id in `repo_id`, `nonidle_filter_paths`
and `sampling_weights`, and generate its filter:

```bash
cd vla
uv run --project ../submodules/openpi python ../submodules/openpi/examples/droid/compute_droid_nonidle_ranges_streaming.py \
    --repo-id <hf-user>/<dataset> --out filters/<name>.json
```

To serve a released checkpoint, download `params/` and `assets/` (about 12.4 GB; `train_state/` is not
needed), then start openpi's policy server:

```bash
cd vla
uv run --project ../submodules/openpi huggingface-cli download \
    SamratSahoo/4_pack_toys_vae_style_timing_apex_learned_posture_v2_prpl \
    --include "params/*" "assets/*" --local-dir checkpoints/pack_toys
OPENPI_CONFIG_DIR=$PWD/configs uv run --project ../submodules/openpi python ../submodules/openpi/scripts/serve_policy.py \
    policy:checkpoint --policy.config=pack_toys --policy.dir=checkpoints/pack_toys
```

To serve your own checkpoint, pass `--policy.dir=checkpoints/<config>/<exp-name>/19999`. The server listens
on port 8000. On the robot side, stop the shim and bring up DROID's standard NUC stack, then run openpi's
DROID client
([`examples/droid/README.md`](https://github.com/SamratSahoo/openpi/blob/c4b2b1bf507e1ddd894902b2771bc6b224fb1873/examples/droid/README.md),
step 2) and give it the task's prompt. The released checkpoints were evaluated with
`--external_camera=left --max_timesteps=1800 --open_loop_horizon=6` (`10` for pack_toys), with each
rollout starting from tiptop's `robot.q_capture` rather than the DROID home pose the client resets to.

## Tasks and released artifacts

For each task, the dataset and the checkpoint share one Hugging Face id.

| Task | Prompt | Data config | Dataset | VLA config | Checkpoint |
|---|---|---|---|---|---|
| Place toys on plate | Place the toys on the plate with no collisions | [`place_toys_on_plate.yaml`](data_collection/configs/place_toys_on_plate.yaml) | [`SamratSahoo/1_pp_toys_plate_vae_style_timing_apex_learned_posture_v2_ep80_prpl`](https://huggingface.co/datasets/SamratSahoo/1_pp_toys_plate_vae_style_timing_apex_learned_posture_v2_ep80_prpl) (80 episodes) | [`place_toys_on_plate`](vla/configs/place_toys_on_plate.yaml) | [model](https://huggingface.co/SamratSahoo/1_pp_toys_plate_vae_style_timing_apex_learned_posture_v2_ep80_prpl) |
| Sort fruits and toys | Sort the fruits into the green bowl and toys into the blue bowl | [`sort_fruits_and_toys.yaml`](data_collection/configs/sort_fruits_and_toys.yaml) | [`SamratSahoo/3_pp_sort_fruits_toys_vae_style_timing_apex_learned_posture_v2_prpl`](https://huggingface.co/datasets/SamratSahoo/3_pp_sort_fruits_toys_vae_style_timing_apex_learned_posture_v2_prpl) (20 episodes) | [`sort_fruits_and_toys`](vla/configs/sort_fruits_and_toys.yaml) | [model](https://huggingface.co/SamratSahoo/3_pp_sort_fruits_toys_vae_style_timing_apex_learned_posture_v2_prpl) |
| Pack toys | Pack the toys onto the wooden tray | [`pack_toys.yaml`](data_collection/configs/pack_toys.yaml) | [`SamratSahoo/4_pack_toys_vae_style_timing_apex_learned_posture_v2_prpl`](https://huggingface.co/datasets/SamratSahoo/4_pack_toys_vae_style_timing_apex_learned_posture_v2_prpl) (20 episodes) | [`pack_toys`](vla/configs/pack_toys.yaml) | [model](https://huggingface.co/SamratSahoo/4_pack_toys_vae_style_timing_apex_learned_posture_v2_prpl) |

## Config reference

A data-collection config has three keys. `prompt` is the session's first task (Enter at the task prompt
repeats it), and each rollout's task becomes its episode's language instruction. `num_episodes` is the
collection target; `collect.py` only reports progress against it. `tamp_overrides` is passed to tiptop-run as
`--curobo-overrides`, and any key left out keeps tiptop's default.

| Key | Value | Effect | Default when omitted |
|---|---|---|---|
| `grasp_threshold` | 0.02 | M2T2 confidence floor (`mask_thresh`): more, less certain grasp candidates | 0.035 |
| `m2t2_num_runs` | 60 | M2T2 passes pooled into each object's grasp set | 5 |
| `voxel_downsample_size` | 0.005 | point-cloud voxel size (m) | 0.0075 |
| `num_particles` | 512 | cuTAMP particles per skeleton | 256 |
| `opt_steps_per_skeleton` | 600 | cuTAMP optimization steps per skeleton | 500 |
| `traj_length_norm` | `"inf"` | norm of cuTAMP's per-move joint-distance cost: max joint displacement; must stay a quoted string | 2 (Euclidean) |
| `grasp_center_weight` | 30 | turns on cuTAMP's off-center grasp cost with this weight | off |
| `grasp_rank_conf_weight` | 2 (pack only) | ranks satisfying particles by soft cost minus 2 × summed M2T2 confidence | confidence alone |
| `require_m2t2_grasps` | true (sort only) | fails planning when M2T2 finds no grasp for an object, instead of using heuristic grasps | false |
| `transit_apex_height` | 0.075 | plans Pick/Place transits through an apex this far (m) above the higher end-effector position | 0 (no apex) |
| `posture_selection_seeds` | 12 | each IK endpoint returns 12 branches and keeps the one that best matches DROID teleoperation postures | 0 (off) |
| `clear_goal_surfaces` | true (pack only) | first clears objects off an occupied goal surface, then plans the task | false |
| `vae_manifold_weight` | 25000 | weight of cuRobo's VAE manifold cost in trajectory optimization | 0 (off) |
| `vae_path` | `vae/checkpoints/vae.pt` | VAE checkpoint for the cost and the re-timing, relative to the repo root | cost: `$VAE_MANIFOLD_CKPT`; re-timing: none |
| `blend_trajectory` | true | merges the cuRobo segments between gripper events into one continuous stroke | false |
| `blend_mode` | `vae` | times each stroke by optimizing its VAE manifold score | `spline` (min-jerk) |
| `blend_smoothing` | 3.0e-4 | smoothing-spline penalty on the stroke path; keep the decimal point, because PyYAML reads `3e-4` as a string | 1e-4 |
| `blend_boundary_speed` | 0.07 | joint speed (rad/s) kept through gripper-adjacent stroke ends, so those frames are not idle | 0.15 |
| `time_dilation_factor_literal` | 1.0 | cuRobo time dilation; 1.0 means no slow-down (`time_dilation_factor: 1.0` would be treated as unset) | `robot.time_dilation_factor` in `tiptop.yml` (0.2) |

## Licenses and acknowledgements

DATAFARM builds on the projects below. Each submodule keeps the license in its own `LICENSE` file.

- [TiPToP](https://github.com/tiptop-robot/tiptop): MIT.
- [cuTAMP](https://github.com/NVlabs/cuTAMP), [cuRobo](https://github.com/NVlabs/curobo) 0.7 (via
  [williamshen-nz/curobo](https://github.com/williamshen-nz/curobo)), [M2T2](https://github.com/NVlabs/M2T2)
  and [FoundationStereo](https://github.com/NVlabs/FoundationStereo): NVIDIA License. The code and any
  derivative works, including these forks, may be used only non-commercially: for research or evaluation
  (cuTAMP, cuRobo, M2T2), or for research only (FoundationStereo).
- Perception weights fetched by `build_server.sh`: M2T2's (`wentao-yuan/m2t2`) are Apache-2.0 per their
  model card; FoundationStereo's pretrained models fall under its NVIDIA License (research only).
- [openpi](https://github.com/Physical-Intelligence/openpi): Apache-2.0. π0.5 is built on PaliGemma, so the
  π0.5-DROID weights and the released checkpoints are subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- DROID ([`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)): Apache-2.0, per its
  dataset card.

The full data-collection pipeline is therefore for non-commercial research use only. This repository's own
code (`vae/`, `data_collection/`, `vla/`) is released under the [MIT License](LICENSE).
