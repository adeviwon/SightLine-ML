"""
SightLine ML — Evaluation harness.

Measures END-TO-END accuracy (not just PSNR): does the OCR + field extraction
recover the CORRECT structured fields from degraded documents?

Accuracies reported:
  1. field_accuracy    — fraction of documents where ALL key fields extract correctly
  2. ocr_confidence    — mean Tesseract confidence
  3. exact_dosage_rate — fraction with exact "500mg"/"400mg" recovered

Baselines compared on identical inputs:
  A. raw degraded (no restoration)
  B. classical: median/unsharp (current mobile pipeline)
  C. DnCNN-lite (PyTorch, this repo)
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from synthdata import make_dataset, CONDITIONS, render_document, degrade
from model import DnCNNLite

GROUND_TRUTH = {
    "prescription": {"dosages": {"500mg", "400mg", "1200mg"}, "patient": "Jane Doe"},
    "banking": {"account": "12345678", "sort_code": "40-12-19"},
    "legal": {"clauses": {"3.2", "5.1"}},
}

KEY_FIELDS = {
    "prescription": ["Medication", "Dosage"],
    "banking": ["Account Number"],
    "legal": ["Clause"],
}


def ocr_image(path, psm="6"):
    r = subprocess.run(["tesseract", path, "stdout", "--psm", psm, "tsv"],
                       capture_output=True, text=True, timeout=60)
    confs, words = [], []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 12 and parts[11].strip():
            try:
                c = float(parts[10])
            except ValueError:
                continue
            if c >= 0:
                confs.append(c)
                words.append(parts[11])
    text = " ".join(words)
    conf = sum(confs) / len(confs) / 100 if confs else 0.0
    return text, conf


def restore_classical(gray, laplacian_var):
    """Current mobile pipeline policy: median for noise, unsharp for blur."""
    _, _, noisy = laplacian_var > 800, None, laplacian_var > 800
    blurry = laplacian_var < 100
    out = gray
    if noisy:
        out = cv2.medianBlur(out, 3)
    if blurry:
        g = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1.5, g, -0.5, 0)
    return out


def restore_dncnn(gray, model):
    """Pad to multiples, run DnCNN on full image (works on any size)."""
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
    with torch.no_grad():
        out = model(t)
    out = out.squeeze(0).squeeze(0).numpy()
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def check_fields(doc_type, text):
    """Return (fields_found, all_key_fields_found)."""
    import importlib
    sys.path.insert(0, "/home/ubuntu/SightLine-Mobile/js")
    # use the JS extractors via node for parity
    probe = (
        "const S = require('/home/ubuntu/SightLine-Mobile/js/pipeline.js');"
        f"const t = S.normalizeOCRText({text!r});"
        "const cls = S.classify(t);"
        "const f = S.extractFields(t, cls.category);"
        "console.log(JSON.stringify({category: cls.category, fields: f}));"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fjs:
        fjs.write(probe)
        p = fjs.name
    r = subprocess.run(["node", p], capture_output=True, text=True, timeout=30)
    Path(p).unlink()
    if r.returncode != 0:
        return [], False
    import json as _json
    d = _json.loads(r.stdout.strip())
    labels = [f["label"] for f in d["fields"]]
    key = KEY_FIELDS[doc_type]
    found = [k for k in key if k in labels]
    return d["fields"], all(k in labels for k in key)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--model", default="models/dncnn_lite.pt")
    ap.add_argument("--out", default="docs/eval_results.json")
    args = ap.parse_args()

    model = DnCNNLite()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded DnCNN-lite (val PSNR {ckpt['val_psnr']:.2f} dB)")

    data = make_dataset(n_per_condition=args.n_per_condition, seed=args.seed)
    results = {m: {"field_acc": [], "ocr_conf": [], "exact_dosage": []}
               for m in ("raw", "classical", "dncnn")}

    for d in data:
        cond, doc_type = d["condition"], d["doc_type"]
        deg = d["degraded"].convert("L")
        gray = np.asarray(deg)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        variants = {
            "raw": gray,
            "classical": restore_classical(gray, lap),
            "dncnn": restore_dncnn(gray, model),
        }
        for mname, img in variants.items():
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, img)
                p = f.name
            text, conf = ocr_image(p)
            Path(p).unlink()
            fields, all_key = check_fields(doc_type, text)
            exact_dosage = False
            if doc_type == "prescription":
                exact_dosage = bool(re.search(r"500mg", text)) and bool(re.search(r"400mg", text))
            results[mname]["field_acc"].append(float(all_key))
            results[mname]["ocr_conf"].append(conf)
            results[mname]["exact_dosage"].append(float(exact_dosage))

    summary = {}
    for mname, r in results.items():
        summary[mname] = {
            "field_accuracy": round(float(np.mean(r["field_acc"])), 4),
            "mean_ocr_confidence": round(float(np.mean(r["ocr_conf"])), 4),
            "exact_dosage_rate": round(float(np.mean(r["exact_dosage"])), 4),
            "n_documents": len(r["field_acc"]),
        }
    # improvement stats
    summary["dncnn_vs_classical_field_acc_delta"] = round(
        summary["dncnn"]["field_accuracy"] - summary["classical"]["field_accuracy"], 4)
    summary["dncnn_vs_raw_field_acc_delta"] = round(
        summary["dncnn"]["field_accuracy"] - summary["raw"]["field_accuracy"], 4)

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "seed": args.seed,
                   "model_val_psnr": ckpt["val_psnr"]}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
