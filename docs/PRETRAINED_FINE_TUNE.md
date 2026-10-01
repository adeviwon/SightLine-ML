# Pretrained Fine-Tune: all-MiniLM-L6-v2 Document Classifier

> Part of the SightLine ML evidence layer (Imperial Hackathon Hong Kong 2026).

## Why pretrained beats from-scratch

Training a text classifier from scratch needs millions of examples. Transfer
learning gets us there with **97 unique document templates**:

| | TF-IDF baseline (production) | MiniLM + head (this repo) |
|---|---|---|
| Representation | 500-char n-gram counts | 384-dim semantic embedding |
| Pretraining | none | ~1B sentence pairs (22M params) |
| Trainable params | full model (~500×4) | 49,796 (2-layer MLP head) |
| Encoder | — | **frozen** (no fine-tune drift) |
| Val accuracy | 87.5% (n=8) | **93.3%** (n=15), best ckpt 100% |
| Macro F1 | 0.867 | **0.927** |
| Training time | seconds | **8.7s** (CPU) |
| Per-category | — | medical 100% · legal 100% · general 100% · banking 83.3% |

The frozen encoder is the point: semantic understanding (paraphrase,
word-order, synonym invariance) comes free from pretraining; we train only
a 49K-parameter head, which is why 8.7 seconds of CPU suffices and why the
model cannot "forget" its language knowledge (no catastrophic forgetting).

## Model card

- **Encoder**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  — 6-layer MiniLM, 22M params, 384 output dims, mean pooling, normalised
- **Head**: Linear(384→128) → ReLU → Dropout(0.1) → Linear(128→4)
- **Classes**: banking · medical · legal · general
- **Loss**: cross-entropy on the head only
- **Optimizer**: AdamW, lr 2e-3, batch 8

## Corpus design (97 unique templates, deduped)

- 4 categories × ~24 unique templates
- Varied phrasings, field orders, boilerplate — realistic document fragments
- **Deduplicated before splitting** — an earlier run accidentally had repeated
  templates across train/val (leakage, inflated to 88.9%); the shipped number
  (93.3%) is measured with guaranteed disjoint sets, seed 123
- 85/15 stratified split → 82 train / 15 val

## Methodology honesty

- n_val = 15 is small: one flipped sample moves accuracy 6.7 points. The best
  checkpoint hit 100% (epoch 7) but we report the final-epoch 93.3% as headline
  with the training curve as context.
- banking at 83.3%: the hardest category (free-form statements vs structured
  legal/medical templates). More varied banking templates is the known fix.
- The TF-IDF baseline comparison is directional (different val sizes: 8 vs 15).

## Training curve (deduped run)

```
epoch  loss    val_acc
1      1.19    86.7%
2      0.83    86.7%
3      0.53    86.7%
4      0.37    86.7%
5      0.29    86.7%
6      0.29    86.7%
7      0.19    100.0%   <- best checkpoint saved
8      0.13    93.3%    <- final (headline)
```

## Integration path

### Desktop (`SightLine`)
Replace the TF-IDF pipeline in `src/offscan/classifier.py`:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
head = torch.load("models/minilm_head.pt", weights_only=True)  # load head

def classify(text):
    emb = encoder.encode([text])              # 384-dim, frozen
    logits = head.net(torch.tensor(emb))
    idx = int(logits.argmax())
    return CATEGORIES[idx]                    # banking/medical/legal/general
```

### Mobile (`SightLine-Mobile`)
Export once:

```python
torch.onnx.export(head_net, dummy_384, "minilm_head.onnx")
```

Then in the PWA: embed with `transformers.js` (MiniLM ONNX, quantized ~23MB,
cached by the existing service worker) → `onnxruntime-web` runs the head
(<1ms). Falls back to the current keyword-density classifier if the model
files are absent — same pattern as the desktop ONNX fallback.

## Reproduce

```bash
pip3 install sentence-transformers scikit-learn
cd SightLine-ML
python3 src/finetune_minilm.py --epochs 8          # ~9s CPU
cat models/minilm_eval.json                        # 93.3% / 0.927 F1
python3 src/finetune_minilm.py --epochs 8 --resume # resume from checkpoint
```
