"""
SightLine ML — PyTorch denoising model (DnCNN-lite).

8-layer residual CNN that learns to invert the synthetic degradation family
(noise / blur / contrast loss) on 64x256 document patches. Trained on-device
classical baselines (median, unsharp) are the comparison baseline.

Architecture (deliberately small — trains in minutes on CPU, exports to ONNX
for the mobile pipeline):
  Conv3x3(1->64)-ReLU, then 6x [Conv3x3(64->64)-BN-ReLU], final Conv3x3(64->1)
  global residual: output = input - predicted_noise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNNLite(nn.Module):
    def __init__(self, channels=64, depth=8):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.BatchNorm2d(channels),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.body(x)
        return x - noise  # residual learning: predict the degradation


class PatchDataset(torch.utils.data.Dataset):
    """Yields (degraded_patch, clean_patch) float32 [0,1] tensors."""

    def __init__(self, data, patch_w=256, patch_h=64):
        self.items = []
        for d in data:
            deg = d["degraded"].convert("L")
            clean = d["clean"].convert("L")
            W, H = deg.size
            # 3 non-overlapping text-band patches per doc
            for band in range(3):
                y0 = 90 + band * 64
                if y0 + patch_h > H: break
                deg_p = deg.crop((24, y0, 24 + patch_w, y0 + patch_h))
                clean_p = clean.crop((24, y0, 24 + patch_w, y0 + patch_h))
                self.items.append((
                    np.asarray(deg_p, dtype=np.float32) / 255.0,
                    np.asarray(clean_p, dtype=np.float32) / 255.0,
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        deg, clean = self.items[i]
        return (torch.from_numpy(deg).unsqueeze(0),
                torch.from_numpy(clean).unsqueeze(0))
