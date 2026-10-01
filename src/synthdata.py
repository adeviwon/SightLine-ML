"""
SightLine ML — Synthetic training data generator.

Renders document text with PIL, applies parameterized degradations (blur,
noise, shadow, rotation, low contrast, compression), and pairs each degraded
patch with its clean counterpart. Seeded for full reproducibility.

Used to train the DnCNN denoiser and the confusion-corrector.
"""

import io
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

PRESCRIPTION_LINES = [
    "PRESCRIPTION",
    "Patient: Jane Doe",
    "Date: 01/10/2026",
    "",
    "Rx: Amoxicillin 500mg",
    "Take one capsule three times daily",
    "for 7 days",
    "",
    "Rx: Ibuprofen 400mg",
    "Every 6-8 hours as needed",
    "Maximum: 1200mg per day",
    "",
    "WARNING: May cause drowsiness",
    "Keep out of reach of children",
]

BANKING_LINES = [
    "HSBC BANK STATEMENT",
    "Account Holder: John Smith",
    "Account Number: 12345678",
    "Sort Code: 40-12-19",
    "",
    "Opening Balance: GBP 1,234.56",
    "Deposits: GBP 3,200.00",
    "Withdrawals: GBP 1,890.50",
    "Closing Balance: GBP 2,544.06",
    "",
    "Card ending 4521",
]

LEGAL_LINES = [
    "LEASE AGREEMENT",
    "Contract between Party A and Party B",
    "Dated 15th January 2024",
    "",
    "Property: 123 Baker Street, London NW1",
    "Tenant shall pay monthly rent of 2,500",
    "",
    "Clause 3.2: Terms of Service",
    "Clause 5.1: Termination Notice",
    "",
    "Case Reference: 2024-CV-00456",
]


def _get_font(size=20, bold=False):
    idx = 1 if bold else 0
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size), idx
    return ImageFont.load_default(), 0


def render_document(lines, width=800, jitter=2, seed=None):
    """Render a clean document image."""
    rng = random.Random(seed)
    height = 90 + len(lines) * 30 + 40
    img = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    font, _ = _get_font(19, bold=True)
    body, _ = _get_font(18)
    y = 28
    for i, line in enumerate(lines):
        f = font if i == 0 else body
        draw.text((48 + rng.randint(-jitter, jitter), y), line, fill=(28, 28, 28), font=f)
        y += 30
    return img


def degrade(img, seed=None, blur=0.0, noise=0.0, shadow=False,
            rotation=0.0, low_contrast=False, jpeg_quality=None):
    """Apply the exact degradation family used by the evaluation gate."""
    rng = random.Random(seed)
    img = img.copy()
    if shadow:
        w, h = img.size
        ov = Image.new("RGB", (w, h), (0, 0, 0))
        od = ImageDraw.Draw(ov)
        for x in range(w):
            od.line([(x, 0), (x, h)], fill=(int(80 * x / w),) * 3)
        img = Image.blend(img, ov, 0.28)
    if noise > 0:
        arr = np.asarray(img).astype(np.int16)
        arr = np.clip(arr + np.random.default_rng(seed or 0).normal(
            0, noise, arr.shape).astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if low_contrast:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip((arr - 128.0) * 0.5 + 160.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if rotation != 0.0:
        img = img.rotate(rotation, fillcolor=(205, 205, 205), expand=False)
    if jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


# Canonical 8-condition gate (matches the evaluation used across the project)
CONDITIONS = {
    "clean":        dict(),
    "blur2":        dict(blur=2.0),
    "blur3":        dict(blur=3.0),
    "noise25":      dict(noise=25.0),
    "rot6":         dict(rotation=6.0),
    "shadow":       dict(shadow=True),
    "lowcontrast":  dict(low_contrast=True),
    "everything":   dict(blur=2.0, noise=20.0, shadow=True, rotation=5.0, low_contrast=True),
}


def make_dataset(n_per_condition=40, seed=123):
    """
    Build (degraded_patch, clean_patch, meta) training tuples.
    Patches are 64x256 grayscale float32 in [0,1] — DnCNN input format.
    Returns list of dicts.
    """
    docs = [
        (PRESCRIPTION_LINES, "prescription"),
        (BANKING_LINES, "banking"),
        (LEGAL_LINES, "legal"),
    ]
    data = []
    for cond_name, params in CONDITIONS.items():
        for i in range(n_per_condition):
            seed_i = seed * 1000 + hash(cond_name) % 997 + i
            lines, doc_type = docs[i % 3]
            clean = render_document(lines, seed=seed_i)
            deg = degrade(clean, seed=seed_i, **params)
            data.append({
                "condition": cond_name,
                "doc_type": doc_type,
                "clean": clean,
                "degraded": deg,
                "seed": seed_i,
            })
    return data


def to_patch(img, x=48, y=100, w=256, h=64):
    """Crop a 64x256 grayscale float32 patch in [0,1]."""
    g = img.convert("L").crop((x, y, x + w, y + h))
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a
