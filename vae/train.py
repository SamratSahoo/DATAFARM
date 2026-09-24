"""Train the FilterbankVAE on full DROID -> one checkpoint.

Training is self-supervised and uses only DROID trajectories. The encoder sees only the raw
[q|v|a|j] trajectory and is trained as a beta-VAE: (a) a feature-weighted MSE regression of each
crop's 113-D style fingerprint (band/spectral targets up-weighted) plus (b) a beta-weighted KL term
with free bits and a linear KL warm-up. Batches are sampled uniformly from a DROID subsample
(--max-droid) and random crops of its episodes.

Every third epoch and at the last, the same loss (eval mode, no input noise, full beta) is computed
on up to 1200 held-out DROID episodes, and the epoch with the lowest validation loss is kept. The
DROID latent-cluster statistics that cuRobo's VaeManifoldCost scores against (mean, precision, ...)
are then computed from trajopt-length sub-segments of ALL cached DROID episodes and baked into the
checkpoint.

    python -m vae.train [--cache-dir DIR] [--out vae/outputs/vae.pt]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import DEFAULT_CACHE_DIR, cache_ready, load_droid_full
from .features import (COMMON_RATE, N_JOINTS, crop_variants, feature_weights, fingerprint, fp_rows,
                       metric_series, pad_mask)
from .model import FilterbankVAE, beta_at, kl_freebits

DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "vae.pt"

MAXLEN = 384            # training crops longer than this are randomly windowed to it (frames @15 Hz)
NOISE = 0.10            # input noise std (standardized units)
N_VAL_DROID = 1200      # held-out DROID episodes in the validation set
EVAL_BATCH = 256        # padding changes the strided branch slightly, so batch sizes are fixed
# cuRobo's VaeManifoldCost encodes a trajopt segment (horizon 32 @ base_dt 0.15 s) resampled to
# 15 Hz -> ~70 steps, so the DROID cluster is calibrated on DROID sub-segments of that length.
SEG_LEN = (50, 95)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def build_crops(trajs, n_crops, seed):
    """Each trajectory plus ``n_crops`` random crops -> ([q|v|a|j] series, cropped Traj) lists."""
    rng = np.random.default_rng(seed)
    series, variants = [], []
    for t in trajs:
        for tv in crop_variants(t, n_crops, rng, 0.4):
            X = metric_series(tv.joint).astype(np.float32)
            if len(X) > MAXLEN:
                st = rng.integers(0, len(X) - MAXLEN + 1); X = X[st:st + MAXLEN]
            series.append(X); variants.append(tv)
    return series, variants


def objective(pred, y, mu, logvar, fw, beta, free_bits):
    """Feature-weighted fingerprint MSE + beta * KL with free bits."""
    return (fw * (pred - y) ** 2).mean() + beta * kl_freebits(mu, logvar, free_bits)


def val_loss(model, series, Y, cmu, csd, fw, beta, free_bits, device, seed=0) -> float:
    """``objective`` over a held-out set: eval mode, no input noise, fixed EVAL_BATCH batches.

    The latent sample uses its own fixed-seed generator, so the loss depends only on the weights
    (every evaluated epoch sees the same noise) and the training RNG is left untouched."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    preds, mus, logvars = [], [], []
    with torch.no_grad():
        for i in range(0, len(series), EVAL_BATCH):
            X, M = pad_mask(series[i:i + EVAL_BATCH], cmu, csd)
            mu, logvar = model.encode(X.to(device), M.to(device))
            eps = torch.randn(mu.shape, generator=gen).to(device)
            preds.append(model.aux(mu + eps * torch.exp(0.5 * logvar))); mus.append(mu); logvars.append(logvar)
    return float(objective(torch.cat(preds), Y.to(device), torch.cat(mus), torch.cat(logvars),
                           fw, beta, free_bits))


def train_encoder(droid_tr, val_droid, names, *, d, beta_max, batch, dropout, n_crops, epochs,
                  warmup, free_bits, lr, seed, steps_per_epoch, band_w, fp_workers, device):
    """Train on crop-augmented DROID with uniform batches; return the lowest-val-loss epoch."""
    tr_s, tr_v = build_crops(droid_tr, n_crops, seed)
    print(f"[train] fingerprinting {len(tr_s)} crops + {len(val_droid)} validation episodes ...",
          flush=True)
    tr_fp = fp_rows(tr_v, names, workers=fp_workers)
    allX = np.concatenate([s.reshape(-1, s.shape[1]) for s in tr_s], 0)
    cmu, csd = allX.mean(0), allX.std(0) + 1e-6
    fmu, fsd = tr_fp.mean(0), tr_fp.std(0); fsd[fsd < 1e-9] = 1.0
    Y = torch.tensor((tr_fp - fmu) / fsd, dtype=torch.float32)
    val_s = [metric_series(t.joint).astype(np.float32) for t in val_droid]
    val_Y = torch.tensor((fp_rows(val_droid, names, workers=fp_workers) - fmu) / fsd, dtype=torch.float32)
    fw = torch.tensor(feature_weights(names, band_w), device=device)   # up-weight band/spectral targets
    ch, n_feat = tr_s[0].shape[1], len(names)

    torch.manual_seed(seed)
    model = FilterbankVAE(ch, d, n_feat, p=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)

    best = {"val_loss": float("inf")}
    for ep in range(epochs):
        model.train()
        beta = beta_at(ep, beta_max, warmup)
        run = 0.0
        for _ in range(steps_per_epoch):
            idx = rng.choice(len(tr_s), batch, replace=len(tr_s) < batch)
            X, M = pad_mask([tr_s[j] for j in idx], cmu, csd)
            X = (X + NOISE * torch.randn_like(X)).to(device)
            mu, logvar, pred = model(X, M.to(device))
            loss = objective(pred, Y[idx].to(device), mu, logvar, fw, beta, free_bits)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            run += loss.detach()
        sched.step()
        if ep % 3 == 2 or ep == epochs - 1:
            val = val_loss(model, val_s, val_Y, cmu, csd, fw, beta_max, free_bits, device)
            if val < best["val_loss"]:
                best = {"val_loss": val, "ep": ep, "cmu": cmu, "csd": csd,
                        "fmu": fmu, "fsd": fsd, "ch": ch, "n_feat": n_feat,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            print(f"[train] ep {ep:3d}  beta={beta:.4f}  train_loss={float(run) / steps_per_epoch:.4f}  "
                  f"val_loss={val:.4f}  best={best['val_loss']:.4f}", flush=True)
    return best


def droid_latent_stats(state, ch, d, n_feat, cmu, csd, droid, *, seed, max_segments, batch, device):
    """Encode DROID sub-segments (~trajopt-segment length) and summarize the latent cloud.

    With at least ``max_segments`` episodes, a uniform subset of that many episodes contributes one
    segment each; otherwise every episode contributes ``max_segments // len(droid)`` segments."""
    model = FilterbankVAE(ch, d, n_feat).to(device)
    model.load_state_dict(state)
    model.eval()
    rng = np.random.default_rng(seed)
    if len(droid) >= max_segments:
        keep = sorted(rng.choice(len(droid), size=max_segments, replace=False).tolist())
        droid = [droid[i] for i in keep]
        n_seg = 1
    else:
        n_seg = max(1, max_segments // len(droid))
    series = []
    for t in droid:
        q, T = t.joint, t.T
        for _ in range(n_seg):
            L = min(int(rng.integers(SEG_LEN[0], SEG_LEN[1] + 1)), T)
            st = int(rng.integers(0, T - L + 1)) if T > L else 0
            series.append(metric_series(q[st:st + L]).astype(np.float32))
    mus, kls = [], []
    with torch.no_grad():
        for i in range(0, len(series), batch):
            X, M = pad_mask(series[i:i + batch], cmu, csd)
            mu, logvar = model.encode(X.to(device), M.to(device))
            mus.append(mu.cpu().numpy())
            kls.append((0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1)).cpu().numpy())
    Z, KL = np.concatenate(mus), np.concatenate(kls)
    cov = np.cov(Z, rowvar=False) + 1e-4 * np.eye(d)
    prec = np.linalg.inv(cov)
    dz = Z - Z.mean(0)
    return {
        "droid_latent_mean": Z.mean(0).astype(np.float32),
        "droid_latent_cov": cov.astype(np.float32),
        "droid_latent_precision": prec.astype(np.float32),
        "kl_droid_mean": float(KL.mean()), "kl_droid_median": float(np.median(KL)),
        "maha2_droid_mean": float(np.einsum("ni,ij,nj->n", dz, prec, dz).mean()),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def pos_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vae.train", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = ap.add_argument_group("data / output")
    io.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="DROID cache root written by `python -m vae.data fetch`")
    io.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="checkpoint path; a <stem>_report.json summary is written next to it")
    tr = ap.add_argument_group("training")
    tr.add_argument("--max-droid", type=pos_int, default=30000,
                    help="DROID episodes the encoder trains on; up to 1200 of the remaining "
                         "episodes form the validation set")
    tr.add_argument("--epochs", type=pos_int, default=60, help="training epochs")
    tr.add_argument("--steps", type=pos_int, default=800, help="optimizer steps per epoch")
    tr.add_argument("--batch", type=pos_int, default=128, help="batch size (crops sampled uniformly)")
    tr.add_argument("--crops-droid", type=int, default=6, help="random crops per DROID episode")
    tr.add_argument("--latent-dim", type=int, default=16, help="latent dimension")
    tr.add_argument("--beta", type=float, default=0.005, help="max KL weight")
    tr.add_argument("--warmup", type=int, default=10, help="KL warm-up epochs")
    tr.add_argument("--free-bits", type=float, default=0.10, help="per-dimension KL floor")
    tr.add_argument("--dropout", type=float, default=0.3, help="dropout in the encoder head")
    tr.add_argument("--lr", type=float, default=7e-4, help="Adam learning rate (cosine-annealed)")
    tr.add_argument("--band-weight", type=float, default=4.0,
                    help="weight of the band/spectral fingerprint targets in the aux loss")
    tr.add_argument("--seed", type=int, default=0, help="seed for the splits, crops, init and batches")
    st = ap.add_argument_group("DROID cluster statistics")
    st.add_argument("--stats-segments", type=pos_int, default=120000,
                    help="max DROID sub-segments encoded for the latent-cluster statistics")
    st.add_argument("--stats-batch", type=pos_int, default=256, help="encoding batch size for the statistics")
    rt = ap.add_argument_group("runtime")
    rt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="torch device")
    rt.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="torch CPU threads")
    rt.add_argument("--fp-workers", type=int, default=0,
                    help="processes for the fingerprint build (0 = min(24, cores), 1 = serial)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    seed, max_droid = args.seed, args.max_droid
    d, beta, dropout = args.latent_dim, args.beta, args.dropout
    print(f"[device] {device} | torch {torch.__version__} | max_droid={max_droid}")
    if not cache_ready(args.cache_dir):
        raise SystemExit(f"full-DROID cache missing under {args.cache_dir} -- run: python -m vae.data fetch")

    droid_all = load_droid_full(args.cache_dir)                       # all episodes (cluster stats)
    if not droid_all:
        raise SystemExit(f"no usable DROID episodes in {args.cache_dir} -- re-run: python -m vae.data fetch")
    rng = np.random.default_rng(seed)
    if max_droid < len(droid_all):
        tr_idx = sorted(rng.choice(len(droid_all), size=max_droid, replace=False).tolist())
    else:
        tr_idx = list(range(len(droid_all)))
    tr_set = set(tr_idx)
    droid_tr = [droid_all[i] for i in tr_idx]
    held = [i for i in range(len(droid_all)) if i not in tr_set]
    if len(held) < 2:
        raise SystemExit("no held-out DROID episodes left for validation -- lower --max-droid")
    val_droid = [droid_all[held[i]] for i in rng.choice(len(held), min(N_VAL_DROID, len(held)), replace=False)]

    names = list(fingerprint(droid_all[0]).keys())
    print(f"[data] encoder: {len(droid_tr)} droid | val {len(val_droid)} held-out droid | "
          f"cluster stats over {len(droid_all)} droid")

    best = train_encoder(droid_tr, val_droid, names, d=d, beta_max=beta, batch=args.batch,
                         dropout=dropout, n_crops=args.crops_droid, epochs=args.epochs,
                         warmup=args.warmup, free_bits=args.free_bits, lr=args.lr, seed=seed,
                         steps_per_epoch=args.steps, band_w=args.band_weight,
                         fp_workers=args.fp_workers, device=device)
    if "state" not in best:
        raise SystemExit("validation loss was never finite -- no epoch to keep")
    print(f"[train] val_loss={best['val_loss']:.4f}  best_epoch={best['ep']}")

    stats_segs = args.stats_segments
    print(f"[stats] encoding DROID sub-segments over all {len(droid_all)} episodes (<= {stats_segs}) ...")
    stats = droid_latent_stats(best["state"], best["ch"], d, int(best["n_feat"]), best["cmu"], best["csd"],
                               droid_all, seed=seed, max_segments=stats_segs, batch=args.stats_batch,
                               device=device)
    print(f"[stats] kl_droid_mean={stats['kl_droid_mean']:.2f}  maha2_droid_mean={stats['maha2_droid_mean']:.2f}")

    # Checkpoint schema read by cuRobo's VaeManifoldCost: state_dict (strict), ch, latent, n_feat,
    # n_joints, chan_mu/chan_sd and droid_latent_mean/precision. The rest is provenance.
    blob = {
        "kind": "filterbank", "state_dict": best["state"], "ch": int(best["ch"]),
        "latent": d, "n_feat": int(best["n_feat"]), "feat_names": names,
        "chan_mu": np.asarray(best["cmu"], np.float32), "chan_sd": np.asarray(best["csd"], np.float32),
        "feat_mu": np.asarray(best["fmu"], np.float32), "feat_sd": np.asarray(best["fsd"], np.float32),
        "source_rate": COMMON_RATE, "n_joints": N_JOINTS, "embed_dim": d,
        "hparams": {"d": d, "beta_max": beta, "dropout": dropout, "batch": args.batch,
                    "n_crops_d": args.crops_droid, "band_w": args.band_weight},
        "train_protocol": ("full DROID lerobot/droid_1.0.1 (v3.0) only; self-supervised "
                           "feature-weighted fingerprint regression + beta-KL with free bits on "
                           "uniform batches of DROID crops; encoder on a DROID subsample, "
                           "selected by held-out DROID validation loss; cluster stats over ALL "
                           "episodes"),
        "metrics": {"val_loss": float(best["val_loss"]), "selected_epoch": int(best["ep"])},
        "embed": "encoder mu of raw [q|v|a|j] trajectory (filterbank, fingerprint regression)",
        "dataset": "lerobot/droid_1.0.1", "dataset_codebase_version": "v3.0",
        "n_droid_train": int(len(droid_tr)), "n_droid_val": int(len(val_droid)),
        "n_droid_stats": int(len(droid_all)), "stats_max_segments": int(stats_segs),
    }
    blob.update(stats)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    print(f"[saved] {out}")
    report = out.with_name(out.stem + "_report.json")
    report.write_text(json.dumps({
        "dataset": "lerobot/droid_1.0.1", "n_droid_train": len(droid_tr), "n_droid_val": len(val_droid),
        "n_droid_stats": len(droid_all), "d": d, "beta": beta, "batch": args.batch,
        "epochs": args.epochs, "metrics": blob["metrics"],
        "kl_droid_mean": stats["kl_droid_mean"], "maha2_droid_mean": stats["maha2_droid_mean"],
    }, indent=2))
    print(f"[saved] {report}")


if __name__ == "__main__":
    main()
