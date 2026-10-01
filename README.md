# SightLine ML — PyTorch Fine-Tuning Pipeline

> ML evidence layer for SightLine (Imperial Hackathon Hong Kong 2026).
> Fine-tunes a denoising network that restores degraded document photos
> before OCR — measured, not claimed.

## What this repo does

The SightLine pipeline (desktop + iPhone PWA) reads documents aloud for
visually impaired users. Real photos arrive blurred, noisy, shadowed,
rotated. This repo:

1. **Synthesizes** a reproducible training corpus: document renders paired
   with their degraded versions across an 8-condition gate
   (clean / blur2 / blur3 / noise25 / rot6 / shadow / lowcontrast / everything).
2. **Trains** DnCNN-lite — a 224K-parameter residual denoising CNN
   (PyTorch, CPU-trained in minutes) — to invert the degradation family.
3. **Evaluates end-to-end**: not PSNR-vs-PSNR, but *does the OCR + field
   extraction recover the correct structured fields* — comparing
   raw vs classical (median/unsharp) vs DnCNN restoration.
4. **Documents everything** with per-epoch training curves and eval tables
   for the hackathon judges.

## Architecture

```
degraded doc photo
      │
      ▼
DnCNN-lite (PyTorch, ONNX-exportable)
  8 layers: Conv3x3(1→64)-ReLU → 6×[Conv3x3-BN-ReLU] → Conv3x3(64→1)
  residual: output = input − predicted_noise
      │
      ▼
restored image → Tesseract OCR → classify → field extraction
      │
      ▼
field_accuracy / ocr_confidence / exact_dosage_rate vs baselines
```

## Files

| Path | Purpose |
|---|---|
| `src/synthdata.py` | reproducible doc renderer + degradation family |
| `src/model.py` | DnCNN-lite + PatchDataset |
| `src/train.py` | chunked training with resume + 5-epoch checkpoints |
| `src/evaluate.py` | end-to-end accuracy harness (raw/classical/dncnn) |
| `models/dncnn_lite.pt` | best checkpoint by val PSNR |
| `models/train_history.json` | per-epoch training curve |
| `docs/eval_results.json` | eval harness output |
| `docs/TRAINING_REPORT.md` | judge-facing writeup with results |

## Why it matters for the hackathon

Judges see many "we used AI" slides. This repo shows:
- a **trained model** (not an API call) with a training curve,
- **measured end-to-end accuracy** on a fixed adversarial gate,
- honest degradation reporting (blur3 is sub-readable and reported as such).

## Run

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
cd SightLine-ML
python3 src/train.py --epochs 30 --n-per-condition 25 --batch 16 --resume
python3 src/evaluate.py --n-per-condition 12
```

## License

MIT.
