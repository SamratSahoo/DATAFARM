"""Start a tiptop-run data-collection session for one task config.

    python data_collection/collect.py data_collection/configs/place_toys_on_plate.yaml \
        [--output-dir DIR] [-- EXTRA_TIPTOP_RUN_ARGS ...]

Writes the config's ``tamp_overrides`` to ``<output-dir>/tamp_overrides.json`` (with ``vae_path``
made absolute) and then replaces this process with

    pixi run --manifest-path submodules/tiptop/pixi.toml tiptop-run \
        --output-dir <output-dir> --enable-recording --curobo-overrides <output-dir>/tamp_overrides.json

run from ``submodules/tiptop``. The session is therefore exactly a tiptop-run session started by
hand: after each rollout it asks whether it succeeded (``y`` / ``n`` / Enter to leave it
unlabelled), then for the next task (Enter repeats the config's prompt, ``q`` quits), and Ctrl-C
aborts only the rollout in progress. Labelled rollouts are moved to ``<output-dir>/success/`` and
``<output-dir>/failure/``; ``build_dataset.py`` turns the successes into a LeRobot dataset.

Environment passed to tiptop-run: ``TIPTOP_TASK`` (the config's prompt, used as the first task),
``TIPTOP_CONFIG_ID`` (the config name, recorded in each episode's ``_meta.json``) and
``VAE_MANIFOLD_CKPT`` (the resolved ``vae_path``) are set here; everything else is inherited, e.g.
``GOOGLE_API_KEY``, ``TIPTOP_ROBOT_HOST``, the ``TIPTOP_*_CAMERA_ID`` serials and ``DC_WORKSPACE`` (which makes tiptop layer
``calibration_info_<workspace>.json`` over its default ``calibration_info.json``).

Needs only the standard library and PyYAML.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TIPTOP_DIR = REPO_ROOT / "submodules" / "tiptop"
RUNS_DIR = REPO_ROOT / "runs"
OVERRIDES_FILE = "tamp_overrides.json"
CONFIG_KEYS = ("prompt", "num_episodes", "tamp_overrides")


def load_config(path: Path) -> dict:
    """The ``prompt``, ``num_episodes`` and ``tamp_overrides`` of a task config."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    unknown = sorted(set(data) - set(CONFIG_KEYS))
    if unknown:
        print(f"warning: {path.name}: ignoring unknown key(s) {unknown}", file=sys.stderr)
    overrides = data.get("tamp_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: tamp_overrides must be a mapping")
    return {
        "prompt": data.get("prompt") or "",
        "num_episodes": int(data.get("num_episodes", 0)),
        "tamp_overrides": overrides,
    }


def resolve_vae_path(overrides: dict) -> dict:
    """``overrides`` with ``vae_path`` made absolute; a relative path is relative to the repo root.

    tiptop resolves a relative ``vae_path`` against its own install location, which is not this
    repository's root when tiptop is a submodule, so it is always handed an absolute path.
    """
    if overrides.get("vae_path") is None:
        return overrides
    path = Path(os.path.expanduser(str(overrides["vae_path"])))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(os.path.abspath(path))
    if not path.is_file():
        raise FileNotFoundError(f"vae_path {overrides['vae_path']!r} resolves to {path}, which does not exist")
    return {**overrides, "vae_path": str(path)}


def count_collected(output_dir: Path) -> int:
    """Successful episodes that have both the robot state and the plan (i.e. are usable)."""
    success_dir = output_dir / "success"
    if not success_dir.is_dir():
        return 0
    return sum(
        1
        for d in success_dir.iterdir()
        if d.is_dir() and (d / "robot_state.npz").is_file() and (d / "tiptop_plan.json").is_file()
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        description="Start a tiptop-run data-collection session for a task config.",
        epilog="Arguments after `--` are passed to tiptop-run unchanged.",
    )
    parser.add_argument("config", type=Path, help="task config, e.g. data_collection/configs/pack_toys.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="where episodes are written (default: runs/<config name>)"
    )
    args = parser.parse_args(argv)

    name = args.config.stem
    try:
        config = load_config(args.config)
        overrides = resolve_vae_path(config["tamp_overrides"])
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"error: {exc}")
    if not (TIPTOP_DIR / "pixi.toml").is_file():
        sys.exit(f"error: {TIPTOP_DIR} is missing; run `git submodule update --init --recursive`")

    # Absolute, because tiptop-run runs from the tiptop submodule, not from the caller's cwd.
    output_dir = Path(os.path.abspath((args.output_dir or RUNS_DIR / name).expanduser()))
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["pixi", "run", "--manifest-path", str(TIPTOP_DIR / "pixi.toml"), "tiptop-run"]
    cmd += ["--output-dir", str(output_dir), "--enable-recording"]
    if overrides:
        overrides_path = output_dir / OVERRIDES_FILE
        overrides_path.write_text(json.dumps(overrides, indent=2, allow_nan=False) + "\n")
        cmd += ["--curobo-overrides", str(overrides_path)]
    cmd += extra

    env = dict(os.environ)
    env["TIPTOP_TASK"] = config["prompt"]
    env["TIPTOP_CONFIG_ID"] = name
    if overrides.get("vae_path") is not None:
        env["VAE_MANIFOLD_CKPT"] = overrides["vae_path"]

    collected, target = count_collected(output_dir), config["num_episodes"]
    print(f"{name}: {collected}/{target} episodes collected in {output_dir}")
    if target and collected >= target:
        print(f"{name}: target reached; new rollouts are still recorded")
    print("$ " + shlex.join(cmd))
    sys.stdout.flush()
    sys.stderr.flush()

    os.chdir(TIPTOP_DIR)
    try:
        os.execvpe(cmd[0], cmd, env)
    except FileNotFoundError:
        sys.exit("error: `pixi` was not found on PATH (see https://pixi.sh)")


if __name__ == "__main__":
    main()
