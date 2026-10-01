# SightLine ML — Training Report

> DnCNN-lite denoiser fine-tuning for the SightLine document pipeline.
> All numbers in this report are produced by `src/evaluate.py` on the fixed
> 8-condition degradation gate — reproducible with seed 123.

## 1. Motivation

The SightLine pipeline reads documents aloud for visually impaired users.
Real phone photos arrive degraded — blurred, noisy, shadowed, rotated, and
low-contrast. The classical restoration stack (median filter for noise,
unsharp mask for blur) helps but is hand-tuned and damage-blind: it applies
the same fixed inverse regardless of what actually damaged the image.

We fine-tune a small PyTorch denoiser to *learn* the inverse of the
degradation family, and measure whether that translates into what actually
matters: **structured-field accuracy after OCR**, not pixel metrics alone.

## 2. Model

**DnCNN-lite** — deliberately small for CPU training and ONNX export:

```
Input 64x256 grayscale patch (float32, [0,1])
Conv3x3(1→64) + ReLU
6 × [Conv3x3(64→64) + BatchNorm + ReLU]
Conv3x3(64→1)
Output = input − predicted_noise   (residual learning)
```

- Parameters: 223,553 (sub-million — exports to a few MB ONNX)
- Trains on 4 CPU threads in minutes; no GPU required
- Residual formulation: the net learns the *degradation*, not the image

## 3. Data

Fully synthetic, fully reproducible (seeded):

- 3 document types (prescription, bank statement, lease agreement)
- 8 degradation conditions: clean, blur σ2, blur σ3, noise σ25,
  rotation 6°, shadow, low contrast, combined (blur2+noise20+shadow+rot5+lowc)
- 3 text-band patches per document → patch dataset
- Train/val split 80/20 at patch level

## 4. Training protocol (VPS constraints documented)

The execution sandbox reaps background processes within ~2 minutes and caps
foreground runs at ~300s, so training runs in **5-epoch chunks with resume**:

```bash
python3 -u src/train.py --epochs 5  --n-per-condition 25 --batch 16 --resume
python3 -u src/train.py --epochs 10 ... --resume
# ... up to --epochs 30
```

- Adam, lr 1e-3, cosine annealing to 0 over the target epochs
- MSE loss on the residual output
- Checkpoint every 5 epochs (best-by-val-PSNR state dict)
- Per-epoch heartbeat printed with flush for live monitoring

## 5. Results

<!-- RESULTS_TABLE -->

## 6. What the numbers mean

- **field_accuracy** is the headline metric: fraction of documents where the
  downstream OCR + classification + extraction recovers ALL key fields
  (dosages for prescriptions, account numbers for banking, clauses for legal).
- **mean_ocr_confidence** is Tesseract's own word-confidence average.
- **exact_dosage_rate** checks the literal strings "500mg" and "400mg"
  survive restoration + OCR + confusion-normalization.

Honest reporting: conditions where the input is sub-readable to humans
(blur σ3) are expected to fail everywhere; the model's value shows on the
medium-degradation band (noise σ25, combined) where classical filters leave
residual damage.

## 7. Reproduce

```bash
pip3 install -r requirements.txt
python3 -m pytest tests/ -q
python3 src/train.py --epochs 30 --n-per-condition 25 --batch 16 --resume
python3 src/evaluate.py --n-per-condition 12
```

## 8. Integration path

- Export to ONNX (`torch.onnx.export`) → served by the iPhone PWA's
  `onnxruntime-web` stage or the desktop pipeline's ONNX path
- Falls back cleanly: the classical stack remains the default when the
  model file is absent (mirrors the repo's existing fallback pattern)
