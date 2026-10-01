import onnxruntime as ort, numpy as np
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sess = ort.InferenceSession("/tmp/minilm_encoder.onnx", providers=["CPUExecutionProvider"])
tok = m.tokenizer(["Account Number: 12345678 sort code 40-12-19"],
                  padding="max_length", max_length=32, return_tensors="np")
e = sess.run(None, {"input_ids": tok["input_ids"].astype(np.int64),
                    "attention_mask": tok["attention_mask"].astype(np.int64)})[0]
ref = m.encode(["Account Number: 12345678 sort code 40-12-19"])
cos = float((e[0]*ref[0]).sum()/(((e[0]**2).sum()**0.5)*((ref[0]**2).sum()**0.5)))
print("parity cosine vs sentence-transformers:", round(cos, 5))
