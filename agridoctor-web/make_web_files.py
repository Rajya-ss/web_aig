"""
Run once on your PC, from your AgriDoctor project root (with the .venv active):

    python make_web_files.py

It reads  runs/model_run/  (model.onnx or best.pt, class_names.json)
and writes, next to index.html:  model.onnx  labels.json  kb.json
Put this script in the project root or pass paths:  python make_web_files.py <model_dir> <out_dir>
"""
import json, shutil, sys
from pathlib import Path

model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/model_run")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
out_dir.mkdir(exist_ok=True)

labels = json.loads((model_dir / "class_names.json").read_text(encoding="utf-8"))
(out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
print("labels.json:", len(labels), "classes")

onnx = model_dir / "model.onnx"
if onnx.is_file():
    shutil.copy(onnx, out_dir / "model.onnx")
else:
    import timm, torch
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    net = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(labels))
    net.load_state_dict(ckpt["model_state_dict"]); net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, 224, 224), str(out_dir / "model.onnx"),
                      input_names=["input"], output_names=["logits"], opset_version=17,
                      dynamic_axes=None)
print("model.onnx: %.1f MB" % ((out_dir / "model.onnx").stat().st_size / 1e6))

for cand in [Path("knowledge_base.json"), model_dir / "knowledge_base.json", model_dir.parent / "knowledge_base.json"]:
    if cand.is_file():
        shutil.copy(cand, out_dir / "kb.json"); print("kb.json copied from", cand); break
else:
    print("No knowledge_base.json found; the library will show no guidance.")
