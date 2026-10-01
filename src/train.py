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


def psnr(pred, target, eps=1e-8):
    mse = ((pred - target) ** 2).mean()
    if mse < eps:
        return 40.0
    return float(10.0 * math.log10(1.0 / mse))


def evaluate(model, loader):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for deg, clean in loader:
            out = model(deg)
            tot += psnr(out.numpy(), clean.numpy())
    return tot / max(1, len(loader))


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

    data = make_dataset(n_per_condition=args.n_per_condition, seed=args.seed)
    n_train = int(len(data) * 0.8)
    train_ds = PatchDataset(data[:n_train])
    val_ds = PatchDataset(data[n_train:])
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)

    model = DnCNNLite()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

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
            loss = loss_fn(out, clean)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * deg.size(0)
        sched.step()

        val_psnr = evaluate(model, val_ld)
        rec = {
            "epoch": epoch + 1,
            "train_mse": ep_loss / len(train_ds),
            "val_psnr": round(val_psnr, 3),
            "lr": sched.get_last_lr()[0],
        }
        history.append(rec)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 5 == 0 and best_state is not None:
            Path("models").mkdir(exist_ok=True)
            torch.save({"state_dict": best_state, "val_psnr": best_psnr,
                        "arch": "DnCNNLite-8-64", "seed": args.seed,
                        "epoch": epoch + 1, "history": history,
                        "train_patches": len(train_ds), "val_patches": len(val_ds),
                        "seconds": round(time.time() - t0, 1)}, ckpt_path)
            with open("models/train_history.json", "w") as f:
                json.dump({"history": history, "best_val_psnr": round(best_psnr, 3),
                           "epochs_done": epoch + 1}, f, indent=2)
        print(f"epoch {epoch+1:3d}/{args.epochs}  mse {rec['train_mse']:.5f}  val_psnr {val_psnr:.2f} dB", flush=True)

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
