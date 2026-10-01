import torch
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m.eval()

class Wrapped(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.auto = st[0].auto_model
    def forward(self, input_ids, attention_mask):
        out = self.auto(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        emb = summed / counts
        return torch.nn.functional.normalize(emb, p=2, dim=1)

w = Wrapped(m)
ids = torch.ones(1, 32, dtype=torch.long)
am = torch.ones(1, 32, dtype=torch.long)
torch.onnx.export(w, (ids, am), "/tmp/minilm_encoder.onnx",
                  input_names=["input_ids", "attention_mask"],
                  output_names=["sentence_embedding"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                "attention_mask": {0: "batch", 1: "seq"},
                                "sentence_embedding": {0: "batch"}})
import os
print("exported:", os.path.getsize("/tmp/minilm_encoder.onnx"), "bytes")

import onnxruntime as ort
sess = ort.InferenceSession("/tmp/minilm_encoder.onnx", providers=["CPUExecutionProvider"])
e = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy()})[0]
print("ort embedding:", e.shape, "norm:", round(float((e[0]**2).sum()**0.5), 4))

# parity vs sentence-transformers on a real sentence
ref = m.encode(["Account Number: 12345678 sort code 40-12-19"])
tok = m.tokenizer(["Account Number: 12345678 sort code 40-12-19"], padding="max_length",
                  max_length=32, return_tensors="np")
e2 = sess.run(None, {"input_ids": tok["input_ids"].astype(np.int64),
                     "attention_mask": tok["attention_mask"].astype(np.int64)})[0]
cos = float((e2[0]*ref[0]).sum()/(((e2[0]**2).sum()**0.5)*((ref[0]**2).sum()**0.5)))
print("parity cosine vs ST:", round(cos, 5))
