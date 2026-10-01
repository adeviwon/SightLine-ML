# SightLine — Offline Document Reader for the Visually Impaired

> **Your eyes, offline. Nothing leaves your phone.**
> Imperial Hackathon Hong Kong & Macau 2026 — Open Challenge track.

Three repos, one product:

| Repo | What it is | Role in the pitch |
|---|---|---|
| [SightLine](https://github.com/adeviwon/SightLine) | Desktop pipeline (Python: OpenCV + Tesseract + ONNX + scikit-learn + espeak) | The reference architecture — 37 tests, production preprocessing, full docs |
| [SightLine-Mobile](https://github.com/adeviwon/SightLine-Mobile) | iPhone PWA (JS: canvas preprocessing + tesseract.js WASM + Web Speech) | **The demo** — installable, works in airplane mode |
| [SightLine-ML](https://github.com/adeviwon/SightLine-ML) | PyTorch fine-tuning (DnCNN denoiser + MiniLM classifier) | **The ML evidence** — trained models with measured accuracy |

## The 60-second explanation

**Problem.** 43M blind people worldwide (2M in the UK, 50,000+ in Hong Kong)
can't read their own bank statements, prescriptions, or legal letters.
Every existing app (Google Lens, Seeing AI, Be My Eyes) uploads the document
photo to a cloud server. For sensitive documents that is a privacy violation
blind users are forced to accept for their independence.

**Solution.** SightLine reads documents aloud with everything running on the
device. No internet needed after install. No cloud. Zero bytes leave the phone
— provable from the service-worker code in minutes.

**How (pipeline).**

```
photo/PDF (up to 4K)
   │  1. PREPROCESS — grayscale, quality check (blur/brightness/noise detectors),
   │     deskew (projection sweep), tile-CLAHE contrast, adaptive restore
   │     (median for noise / unsharp for blur), Otsu binarize
   ▼
   │  2. OCR — tesseract.js (WASM) multi-pass: PSM 6 → 3 → 11 across
   │     preprocessing-stage candidates; garbage rejection; early-exit ≥75%
   ▼
   │  3. UNDERSTAND — normalize OCR confusions (S00mg→500mg) →
   │     classify (banking/medical/legal/general) → extract fields
   │     (dosages, accounts, IBANs, clauses) → NER (drugs, money, dates)
   ▼
   │  4. SPEAK — natural-language summary + full text via offline speech synthesis
   ▼
spoken summary with confidence scores, 0 bytes sent
```

**ML (why it's not just "we used Tesseract").**

1. **DnCNN denoiser** — trained in `SightLine-ML` on a synthetic degradation
   corpus to invert noise/blur before OCR (measured end-to-end against
   classical restoration, seed-reproducible).
2. **MiniLM classifier** — pretrained sentence-transformers encoder
   (frozen) + trained MLP head for document classification, replacing
   keyword/TF-IDF matching (transfer learning, ONNX-exportable).
3. Every stage outputs a **confidence score**; unreadable input degrades
   honestly with a retake prompt instead of hallucinating.

**Policy alignment (HK 2026 Policy Address).** Para 108 (AI in Healthcare),
Para 113 (AI for Welfare Lab, $300M), Para 376-377 (persons with disabilities),
Para 398-400 (Gerontechnology Promotion Scheme, $100M), Para 441-442
(cybersecurity — zero-network is secure by design).

## Demo (judges' moment)

1. Install the PWA on an iPhone (Add to Home Screen — one-time 40MB download)
2. Turn on **airplane mode** — on camera
3. Scan a real prescription → spoken summary with dosages
4. Scan a bank letter → account/card details extracted
5. Show the zero-network guarantee from `sw.js`

## Docs per repo

- `SightLine/docs/` — architecture, privacy, hackathon guide, victory plan
- `SightLine-Mobile/docs/` — ARCHITECTURE (pipeline detail + robustness table),
  PRIVACY (5 verifiable mechanisms), HACKATHON (submission kit)
- `SightLine-ML/docs/` — TRAINING_REPORT (denoiser), PRETRAINED_FINE_TUNE
  (MiniLM classifier), eval_results.json
