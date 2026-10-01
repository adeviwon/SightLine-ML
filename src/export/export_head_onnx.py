import torch, torch.nn as nn, json, sys
sys.path.insert(0, "/home/ubuntu/SightLine-ML/src")
from finetune_minilm import ClassifierHead, CATEGORIES

head = ClassifierHead()
ck = torch.load("/home/ubuntu/SightLine-ML/models/minilm_head.pt", map_location="cpu", weights_only=True)
head.load_state_dict(ck["state_dict"])
head.eval()

dummy = torch.randn(1, 384)
torch.onnx.export(head.net, dummy, "/tmp/minilm_head.onnx",
                  input_names=["embedding"], output_names=["logits"],
                  dynamic_axes={"embedding": {0: "batch"}})
print("exported:", __import__("os").path.getsize("/tmp/minilm_head.onnx"), "bytes")

import onnxruntime as ort
sess = ort.InferenceSession("/tmp/minilm_head.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"embedding": dummy.numpy()})[0]
print("ort output shape:", out.shape, "argmax:", int(out.argmax()))
print("matches torch:", int(out.argmax()) == int(head(dummy).argmax()))
