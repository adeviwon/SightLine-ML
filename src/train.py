"""
SightLine ML — Train the DnCNN-lite denoiser.

Usage:
    python3 src/train.py [--epochs 60] [--n-per-condition 40] [--seed 123]

Outputs:
    models/dncnn_lite.pt        — best checkpoint (by val PSNR)
    models/train_history.json   — per-epoch metrics for documentation
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from synthdata import make_dataset
from model import DnCNNLite, PatchDataset


def psnr(pred, target, eps=1e-8, ink_only=False):
    if ink_only:
        # PSNR measured on ink (text) pixels only — the metric that tracks OCR quality.
        # Text pixels: target luma < 0.5 (dark strokes on white docs).
        m = target < 0.5
        if m.sum() < 8:
            return psnr(pred, target, eps, ink_only=False)
        mse = ((pred[m] - target[m]) ** 2).mean()
    else:
        mse = ((pred - target) ** 2).mean()
    if mse < eps:
        return 40.0
    return float(10.0 * math.log10(1.0 / mse))


def evaluate(model, loader):
    model.eval()
    tot_full, tot_ink = 0.0, 0.0
    with torch.no_grad():
        for deg, clean in loader:
            out = model(deg)
            tot_full += psnr(out.numpy(), clean.numpy())
            tot_ink += psnr(out.numpy(), clean.numpy(), ink_only=True)
    n = max(1, len(loader))
    return tot_ink / n, tot_full / n  # (headline=ink PSNR, secondary=full PSNR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-condition", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset cache: regenerating 200 renders costs 60-90s per chunk; with
    # chunked resume training this dominates. Cache patches to .npy once.
    import hashlib
    cache_key = hashlib.md5(f"v1-{args.n_per_condition}-{args.seed}".encode()).hexdigest()[:10]
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    cfile_x = cache_dir / f"X_{cache_key}.npy"
    cfile_y = cache_dir / f"Y_{cache_key}.npy"
    if cfile_x.exists() and cfile_y.exists():
        X = np.load(cfile_x)
        Y = np.load(cfile_y)
        print(f"dataset cache hit: {X.shape[0]} patches from {cfile_x.name}", flush=True)
    else:
        data = make_dataset(n_per_condition=args.n_per_condition, seed=args.seed)
        pds_tmp = PatchDataset(data)
        X = np.stack([p[0].numpy()[0] for p in [(pds_tmp[i][0], pds_tmp[i][1]) for i in range(len(pds_tmp))]])
        Y = np.stack([p[1].numpy()[0] for p in [(pds_tmp[i][0], pds_tmp[i][1]) for i in range(len(pds_tmp))]])
        np.save(cfile_x, X)
        np.save(cfile_y, Y)
        print(f"dataset built+cached: {X.shape[0]} patches -> {cfile_x.name}", flush=True)

    n_train = int(X.shape[0] * 0.8)
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X[:n_train]).unsqueeze(1), torch.from_numpy(Y[:n_train]).unsqueeze(1))
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X[n_train:]).unsqueeze(1), torch.from_numpy(Y[n_train:]).unsqueeze(1))
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)

    model = DnCNNLite()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Masked MSE: weight loss toward ink (text) regions — the white background
    # dominates plain MSE on document patches (verified: train_mse identical across
    # epochs, PSNR pinned ~14.6 dB — model was predicting ~input). The mask is the
    # inverted clean-patch luma: 1.0 where text, ~0.04 floor elsewhere (keeps
    # background consistent too, just 25x down-weighted).
    BG_FLOOR = 0.04

    history = []
    best_psnr = -1.0
    best_state = None
    start_epoch = 0
    ckpt_path = Path("models/dncnn_lite.pt")
    if ckpt_path.exists() and args.resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"])
        start_epoch = ck.get("epoch", 0)
        best_psnr = ck.get("val_psnr", -1)
        best_state = copy.deepcopy(model.state_dict())
        hist_file = Path("models/train_history.json")
        if hist_file.exists():
            history = json.load(open(hist_file))["history"]
        print(f"resumed from epoch {start_epoch}, best_psnr {best_psnr:.2f}", flush=True)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = 0.0
        for deg, clean in train_ld:
            opt.zero_grad()
            out = model(deg)
            with torch.no_grad():
                ink = (1.0 - clean).pow(2)  # clean text is dark -> (1-clean)^2 peaks on strokes
                mask = BG_FLOOR + (1.0 - BG_FLOOR) * ink
            loss = (mask * (out - clean).pow(2)).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * deg.size(0)
        sched.step()

        ink_psnr, full_psnr = evaluate(model, val_ld)
        rec = {
            "epoch": epoch + 1,
            "train_mse": round(ep_loss / len(train_ds), 6),
            "val_ink_psnr": round(ink_psnr, 3),
            "val_full_psnr": round(full_psnr, 3),
            "lr": round(sched.get_last_lr()[0], 8),
        }
        history.append(rec)
        if ink_psnr > best_psnr:
            best_psnr = ink_psnr
            best_state = copy.deepcopy(model.state_dict())
        if best_state is not None:
            Path("models").mkdir(exist_ok=True)
            torch.save({"state_dict": best_state, "val_psnr": best_psnr,
                        "arch": "DnCNNLite-8-64", "seed": args.seed,
                        "epoch": epoch + 1, "history": history,
                        "train_patches": len(train_ds), "val_patches": len(val_ds),
                        "seconds": round(time.time() - t0, 1)}, ckpt_path)
            with open("models/train_history.json", "w") as f:
                json.dump({"history": history, "best_val_psnr": round(best_psnr, 3),
                           "epochs_done": epoch + 1}, f, indent=2)
        print(f"epoch {epoch+1:3d}/{args.epochs}  mse {rec['train_mse']:.5f}  ink_psnr {ink_psnr:.2f} dB  full_psnr {full_psnr:.2f} dB", flush=True)

    Path("models").mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "val_psnr": best_psnr,
                "arch": "DnCNNLite-8-64", "seed": args.seed}, "models/dncnn_lite.pt")
    with open("models/train_history.json", "w") as f:
        json.dump({"history": history, "best_val_psnr": round(best_psnr, 3),
                   "epochs": args.epochs, "train_patches": len(train_ds),
                   "val_patches": len(val_ds), "seconds": round(time.time() - t0, 1)}, f, indent=2)

    print(f"\nbest val PSNR: {best_psnr:.2f} dB  (saved models/dncnn_lite.pt, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
