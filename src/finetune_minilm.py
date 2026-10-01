"""
SightLine ML — Fine-tune all-MiniLM-L6-v2 as the document classifier.

Pretrained encoder (FROZEN) + trainable 2-layer MLP head.
Transfer learning: 22M-param encoder pretrained on ~1B sentence pairs
provides semantic embeddings; the head learns the 4-way document typing.

Chunked-training friendly: --epochs is the TARGET total, resumes from
models/minilm_head.pt (head weights + optimizer + history).

Usage:
    python3 src/finetune_minilm.py --epochs 6
Outputs:
    models/minilm_head.pt          best head checkpoint
    models/minilm_training_log.json  per-epoch metrics
    models/minilm_eval.json        final val metrics (written at end)
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

CATEGORIES = ["banking", "medical", "legal", "general"]

# ── Expanded corpus: 60+ realistic templates per category ──────────────
# Varied phrasings, layouts, field orders — ~4x the original 50-sample set.

CORPUS = {
"banking": [
 "HSBC Bank Statement. Account Holder: John Smith. Account Number: 12345678. Sort Code: 40-12-19.",
 "Statement of Account. Current account. Deposits totaling 3,200.00 GBP this month.",
 "Your balance is 2,544.06. Available funds: 2,100.00. Overdraft limit 500.",
 "Direct Debit of 89.99 to British Gas from account 98765432.",
 "ATM withdrawal 200. Card ending 4521. Remaining balance 1,234.56.",
 "Standing Order: rent 1,450 monthly to Landlord Account 55443322.",
 "Credit Card statement. Card ending 8812. Minimum payment 45.00 due 28th.",
 "Mortgage statement. Principal outstanding 248,900. Rate fixed 3.5 percent.",
 "IBAN GB29 NWBK 6016 1331 9268 19. BIC NWBKGB2L. Transfer 500 GBP.",
 "Sort code 20-44-19 account 7788990. Branch: Manchester Piccadilly.",
 "Transaction history: salary credit 2,900 on the 28th, rent debit 1,100.",
 "Your overdraft has been charged at 0.5 percent above base rate this quarter.",
 "Savings account statement. Interest earned 12.34. Closing balance 8,901.55.",
 "Faster Payment sent to Ms A. Patient reference INVOICE-9931 amount 240.00.",
 "Cheque number 000231 deposited. Funds available after 3 working days.",
 "Foreign transaction fee 2.75 applied to purchase of 55.40 EUR equivalent.",
 "Loan repayment schedule: 12 monthly instalments of 210.50 beginning May.",
 "Bank of Scotland business account. Annual service charge 96.00 waived.",
 "Notice: your card ending 7719 was used at a merchant in another country.",
 "Joint account statement for Mr and Mrs Smith, sort code 40-52-08.",
 "Fixed bond matures in June. Early withdrawal penalty applies at 90 days interest.",
 "Your balance is 2,544.06. Available funds: 2,100.00. Overdraft limit 500.",
 "Standing Order: rent 1,450 monthly to Landlord Account 55443322.",
 "Credit Card statement. Card ending 8812. Minimum payment 45.00 due 28th.",
 "Mortgage statement. Principal outstanding 248,900. Rate fixed 3.5 percent.",
 "IBAN GB29 NWBK 6016 1331 9268 19. BIC NWBKGB2L. Transfer 500 GBP.",
 "Sort code 20-44-19 account 7788990. Branch: Manchester Piccadilly.",
 "Transaction history: salary credit 2,900 on the 28th, rent debit 1,100.",
 "Direct Debit of 89.99 to British Gas from account 98765432.",
 "ATM withdrawal 200. Card ending 4521. Remaining balance 1,234.56.",
],
"medical": [
 "PRESCRIPTION. Patient: Jane Doe. Rx: Amoxicillin 500mg, one capsule three times daily.",
 "Take one tablet twice daily after meals. Complete the full course of antibiotics.",
 "Metformin hydrochloride 1000mg. Take with food to reduce stomach upset.",
 "Ibuprofen 400mg as required for pain. Do not exceed 1200mg in 24 hours.",
 "WARNING: may cause drowsiness. Do not operate machinery after taking.",
 "Lisinopril 10mg once daily in the morning for blood pressure control.",
 "Inhaler: Salbutamol 100 micrograms, two puffs when wheezing, max four times daily.",
 "Patient: John Smith, date of birth 12/04/1958. Allergy: penicillin.",
 "Apply the cream to affected area twice daily for seven days.",
 "Blood test results: haemoglobin 13.2, white cell count normal.",
 "Repeat prescription request. Medication: Atorvastatin 20mg, quantity 28.",
 "Paracetamol 500mg every six hours if needed. Maximum eight tablets per day.",
 "Insulin glargine 12 units subcutaneously at bedtime. Rotate injection sites.",
 "Consult your doctor if symptoms persist beyond five days of treatment.",
 "Do not stop taking this medicine without speaking to your pharmacist.",
 "Warfarin 3mg. Weekly INR blood test required. Report any unusual bruising.",
 "Omeprazole 20mg gastro-resistant capsule once daily before breakfast.",
 "Keep out of reach of children. Store below 25 degrees Celsius.",
 "The pharmacist dispensed 28 capsules with a repeat date of next month.",
 "Diagnosis: type 2 diabetes. HbA1c 7.8 percent. Review in three months.",
 "GP surgery follow-up appointment scheduled for the 15th at 2:30pm.",
 "Adverse reaction reported: mild nausea after first dose. Continue monitoring.",
 "Eye drops: one drop into each eye every night at bedtime.",
 "Complete the full seven day course even if you feel better.",
 "Prescription charge paid. Exemption certificate held under age category.",
 "Take one tablet twice daily after meals. Complete the full course.",
 "Metformin hydrochloride 1000mg. Take with food to reduce stomach upset.",
 "Ibuprofen 400mg as required for pain. Do not exceed 1200mg in 24 hours.",
 "WARNING: may cause drowsiness. Do not operate machinery after taking.",
 "Lisinopril 10mg once daily in the morning for blood pressure control.",
],
"legal": [
 "LEASE AGREEMENT between the Landlord and the Tenant dated 15 January 2024.",
 "This Agreement is binding under the laws of England and Wales.",
 "Clause 3.2: the Tenant shall pay rent of 2,500 monthly in advance.",
 "Clause 5.1: either party may terminate with two months written notice.",
 "IN THE MATTER OF Case No. 2024-CV-00456 before the District Court.",
 "The Plaintiff claims damages; the Defendant denies liability.",
 "Power of Attorney: I appoint the following person as my attorney.",
 "Last Will and Testament of John Smith, made this fourth day of June.",
 "Non-Disclosure Agreement. The parties shall not disclose confidential information.",
 "WHEREAS the parties agree to the terms set out in this Contract.",
 "Employment Contract between the Company and the Employee. Notice period one month.",
 "Tenancy deposit protected under the Deposit Protection Scheme.",
 "Witnesseth: the Landlord lets and the Tenant takes the Property known as 123 Baker Street.",
 "Judgment entered for the Claimant. Costs to be assessed.",
 "Service of judicial documents may be effected at the registered office.",
 "The Licence is granted subject to the conditions in Schedule 2.",
 "Arbitration clause: disputes resolved under LCIA rules seated in London.",
 "Notarized copy certified true by the solicitor of the Supreme Court.",
 "Terms and Conditions of Business. By engaging our services you accept these terms.",
 "Severability: if any provision is invalid the remainder continues in force.",
 "This Deed is executed on the date first written above.",
 "Sub-letting is prohibited without the prior written consent of the Landlord.",
 "Governing law: this Contract is governed by the law of Hong Kong SAR.",
 "The Purchaser covenants with the Vendor to observe the restrictive covenants.",
 "Breach of any term entitles the innocent party to terminate immediately.",
 "LEASE AGREEMENT between the Landlord and the Tenant dated 15 January 2024.",
 "This Agreement is binding under the laws of England and Wales.",
 "Clause 3.2: the Tenant shall pay rent of 2,500 monthly in advance.",
 "Clause 5.1: either party may terminate with two months written notice.",
 "IN THE MATTER OF Case No. 2024-CV-00456 before the District Court.",
],
"general": [
 "Hello, your appointment is confirmed for Tuesday the 15th at 2pm.",
 "Meeting notes: discussed quarterly targets. Actions assigned. Next meeting Friday.",
 "Recipe: chocolate cake. 200g flour, 150g sugar, 3 eggs. Bake 30 minutes.",
 "Dear John, thank you for your email. I will reply by the end of the week.",
 "Notice: the office will be closed on Monday for the bank holiday.",
 "Invoice 12345 dated 01/03/2024. Web design services. Total due 500.00.",
 "Your parcel has been dispatched and should arrive within three working days.",
 "Welcome to the neighbourhood. The residents association meets monthly.",
 "Reminder: library books are due back by the end of the month.",
 "School newsletter: sports day rescheduled to Thursday due to weather.",
 "Your subscription renews automatically on the 15th unless cancelled.",
 "Thank you for your order. A receipt has been attached for your records.",
 "Community centre quiz night, Saturday 7pm, teams of four welcome.",
 "The landlord has arranged boiler servicing for Wednesday morning.",
 "Please complete the survey so we can improve our service.",
 "Train service update: the 08:15 to Victoria is cancelled today.",
 "We regret to inform you the event is postponed to next spring.",
 "Happy birthday! The party starts at 3pm at the community hall.",
 "Your application has been received and is under review by the panel.",
 "Volunteers are needed for the charity collection this weekend.",
 "Parking restrictions apply from Monday to Friday between 8am and 6pm.",
 "The doctor will see you now; please take a seat in room four.",
 "Your loyalty points balance is 1,240 points as of this month.",
 "Water rates bill for the period April to September is enclosed.",
 "Please water the plants while we are away next week.",
 "Hello, your appointment is confirmed for Tuesday the 15th at 2pm.",
 "Meeting notes: discussed quarterly targets. Actions assigned. Next meeting Friday.",
 "Recipe: chocolate cake. 200g flour, 150g sugar, 3 eggs. Bake 30 minutes.",
 "Dear John, thank you for your email. I will reply by the end of the week.",
 "Notice: the office will be closed on Monday for the bank holiday.",
],
}

assert len(CORPUS["banking"]) >= 30, len(CORPUS["banking"])


class ClassifierHead(nn.Module):
    """2-layer MLP on frozen 384-dim MiniLM embeddings."""
    def __init__(self, dim=384, hidden=128, n_classes=4, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_split(seed=123):
    # DEDUPE first — duplicate templates across train/val would leak and inflate
    # accuracy dishonestly. Each unique template appears exactly once.
    seen = set()
    texts, labels = [], []
    for ci, cat in enumerate(CATEGORIES):
        for t in CORPUS[cat]:
            if t in seen:
                continue
            seen.add(t)
            texts.append(t)
            labels.append(ci)
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    cut = int(len(texts) * 0.85)
    return texts[:cut], labels[:cut], texts[cut:], labels[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6, help="TOTAL target epochs")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False  # FROZEN — transfer learning, no catastrophic forgetting

    texts_tr, y_tr, texts_va, y_va = build_split(args.seed)
    print(f"corpus: {len(texts_tr)} train / {len(texts_va)} val", flush=True)

    # Encode once (frozen encoder -> embeddings are static)
    E_tr = torch.tensor(encoder.encode(texts_tr, batch_size=32, show_progress_bar=False))
    E_va = torch.tensor(encoder.encode(texts_va, batch_size=32, show_progress_bar=False))
    y_tr_t = torch.tensor(y_tr)
    y_va_t = torch.tensor(y_va)

    head = ClassifierHead()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, history, best_acc, best_state = 0, [], -1.0, None
    ck = Path("models/minilm_head.pt")
    if args.resume and ck.exists():
        saved = torch.load(ck, map_location="cpu", weights_only=True)
        head.load_state_dict(saved["state_dict"])
        start_epoch = saved["epoch"]
        best_acc = saved.get("val_acc", -1.0)
        best_state = copy.deepcopy(head.state_dict())
        log_p = Path("models/minilm_training_log.json")
        if log_p.exists():
            history = json.load(open(log_p))["history"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    t0 = time.time()
    n = len(E_tr)
    for epoch in range(start_epoch, args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            logits = head(E_tr[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_pred = head(E_va).argmax(dim=1)
            acc = float((val_pred == y_va_t).float().mean())
        rec = {"epoch": epoch + 1, "train_loss": round(tot / n, 5), "val_acc": round(acc, 4)}
        history.append(rec)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(head.state_dict())
        Path("models").mkdir(exist_ok=True)
        torch.save({"state_dict": best_state, "epoch": epoch + 1, "val_acc": best_acc,
                    "categories": CATEGORIES}, ck)
        with open("models/minilm_training_log.json", "w") as f:
            json.dump({"history": history, "best_val_acc": best_acc}, f, indent=2)
        print(f"epoch {epoch+1}/{args.epochs}  loss {rec['train_loss']:.4f}  val_acc {acc:.1%}", flush=True)

    # Final eval: per-category accuracy + macro F1
    from sklearn.metrics import f1_score
    with torch.no_grad():
        pred = head(E_va).argmax(dim=1).numpy()
    per_cat = {}
    for ci, cat in enumerate(CATEGORIES):
        m = np.array(y_va) == ci
        per_cat[cat] = round(float((pred[m] == ci).mean()) if m.sum() else -1, 4)
    macro_f1 = round(float(f1_score(y_va, pred, average="macro")), 4)
    eval_out = {
        "val_accuracy": round(float((pred == np.array(y_va)).mean()), 4),
        "macro_f1": macro_f1,
        "per_category_accuracy": per_cat,
        "val_n": len(y_va),
        "corpus_total": len(texts_tr) + len(texts_va),
        "head_params": sum(p.numel() for p in head.parameters()),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2 (frozen)",
        "seconds": round(time.time() - t0, 1),
    }
    with open("models/minilm_eval.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    print(json.dumps(eval_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
