# Filterbank VAE (motion-style prior)

A small β-VAE that maps a 7-DOF Franka joint trajectory of any length to a 16-D *style* latent.
During data collection, cuRobo's `VaeManifoldCost` encodes each trajectory-optimization segment and
penalizes the squared Mahalanobis distance of its latent to the DROID latent cluster, which pulls
planned motion toward the timing and style of human teleoperation.

- **Input:** the raw joint series at 15 Hz as `[q|v|a|j]` (28 channels; velocity, acceleration and
  jerk are finite differences). No images.
- **Model** (`model.py`): a learnable filterbank of dilated convolutions plus a strided temporal
  branch, then masked global pooling, then the latent. An auxiliary head regresses a 113-D
  hand-crafted style fingerprint (`features.py`: velocity, acceleration and jerk statistics, range
  of motion, SPARC/LDLJ smoothness, speed spectrum, per-joint energy-band fractions). The
  fingerprint is a training target only.
- **Training** (`train.py`): self-supervised, on DROID trajectories only. Batches are drawn
  uniformly from the training episodes and 6 random crops of each (40–100% of its length, windowed
  to at most 384 frames). The loss is a feature-weighted MSE on each crop's fingerprint
  (band/spectral targets ×4) plus β·KL with free bits (β = 0.005, linear warm-up).
- **Model selection:** every third epoch and at the last, the same loss (eval mode, no input noise,
  full β, a fixed-seed latent sample) is computed on up to 1,200 held-out DROID episodes. The epoch
  with the lowest validation loss is kept.
- **DROID cluster:** the kept encoder embeds trajopt-length sub-segments (50–95 frames) of all
  cached DROID episodes. Their latent mean, covariance and precision are stored in the checkpoint
  for the cost's Mahalanobis distance.

Run everything from the repository root: `pip install -r vae/requirements.txt`.

## Data

**DROID** (public, `lerobot/droid_1.0.1`). Only the proprio columns are streamed; no HF token is
needed. The fetch is resumable. It streams about 4 GB and writes about 1.7 GB of shards to
`vae/data_cache/droid_full_proprio/`:

```bash
python -m vae.data fetch            # --max-files N for a quick subset, --cache-dir DIR to relocate
python -m vae.data status
```

Each DROID data file holds about 1,100 episodes. When training on a quick subset, pass
`--max-droid` below the fetched episode count so that some episodes are left for validation (e.g.
`fetch --max-files 4`, then `train --max-droid 3000`).

## Train

```bash
python -m vae.train                 # -> vae/outputs/vae.pt + vae/outputs/vae_report.json
```

Defaults:

- 30,000 DROID training episodes (`--max-droid`); up to 1,200 of the remaining episodes form the
  validation set
- 60 epochs × 800 steps, batch 128, 6 crops per episode
- d = 16, β = 0.005 with a 10-epoch KL warm-up and 0.1 free bits per dimension, dropout 0.3, band
  weight 4.0
- Adam at lr 7e-4 (cosine-annealed) with weight decay 1e-4, gradient clipping at 5.0, input noise
  σ = 0.1, seed 0
- DROID cluster statistics from up to 120,000 sub-segments of all cached episodes that have at
  least 30 frames

Use a GPU (`--device` defaults to `cuda` when available). See `python -m vae.train --help` for all
flags. The log prints the training and validation loss at every evaluated epoch, and
`vae_report.json` records the validation loss, the selected epoch, `kl_droid_mean` and
`maha2_droid_mean`.

## Checkpoint contract

A checkpoint is a `torch.save` dict. cuRobo's `load_vae_manifold`
(`submodules/curobo/src/curobo/rollout/cost/vae_manifold_cost.py`) loads it with
`torch.load(..., map_location="cpu", weights_only=False)` and then `load_state_dict(strict=True)`
into its own copy of `FilterbankVAE`.

It reads these keys:

| Key | Contents |
|---|---|
| `state_dict` | encoder weights |
| `ch` | 28 input channels |
| `latent` | latent dimension (16) |
| `n_feat` | 113 |
| `n_joints` | 7 |
| `chan_mu`, `chan_sd` | input standardization |
| `droid_latent_mean`, `droid_latent_precision` | the DROID cluster |

Checkpoints written by `python -m vae.train` also store `feat_names`, `feat_mu`/`feat_sd`,
`droid_latent_cov`, `kl_droid_*`, `maha2_droid_mean` and a record of the run (`hparams`,
`train_protocol`, `metrics` with the validation loss and the selected epoch, episode counts).

cuRobo keeps its own copy of the model class and of the 15 Hz `[q|v|a|j]` preprocessing in
`features.metric_series`. Do not rename or reorder the layers in `model.py`, because cuRobo's copy
must load the `state_dict` strictly.

`vae/checkpoints/vae.pt` is the checkpoint the data-collection configs use: they turn the cost on
with `vae_manifold_weight` and point both the cost and the stroke re-timing (`blend_mode: vae`) at
it with `vae_path: vae/checkpoints/vae.pt`, relative to the repository root. To use a newly trained
checkpoint, point `vae_path` at it and re-tune `vae_manifold_weight`: the cost is that weight times
the squared Mahalanobis distance, whose size on planned motion differs between checkpoints.
