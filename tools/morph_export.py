"""
Подготовка морфологического разбора иврита для приложения (запускается в GitHub Actions вручную).

Модель: dicta-il/dictabert-tiny-joint (Dicta, лицензия CC BY 4.0 — использование разрешено с указанием авторства).
Скрипт:
  1. скачивает модель (только головы lex / morph / prefix — синтаксис и NER не нужны);
  2. сохраняет исходный код модели и конфиг (чтобы повторить разбор в приложении);
  3. прогоняет примеры через model.predict — эталон для проверки приложения;
  4. экспортирует модель в ONNX и сжимает до int8 (~45–50 МБ) для работы на телефоне без интернета;
  5. кладёт всё в папку models/morph/ (её публикует workflow).
"""
import dataclasses
import inspect
import json
import os
import shutil
import sys
import traceback

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "dicta-il/dictabert-tiny-joint"
OUT = "models/morph"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "export_log.txt"), "w", encoding="utf-8")


def say(*a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)


SAMPLES = [
    "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות",
    "היא לא ידעה מה לומר אז היא שתקה",
    "סגרי את הדלת בבקשה",
    "אנחנו נרקוד מחר בחוג",
    "הילדים הלכו לבית הספר והמורה שאלה אותם שאלות",
    "כשהגעתי הביתה ראיתי שהחתול אכל את האוכל",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, do_syntax=False, do_ner=False)
model.eval()
say("model class:", type(model).__module__, type(model).__name__)
say(model)

# 1) Исходный код модели и конфиг — по ним приложение повторяет разбор.
mod = sys.modules[type(model).__module__]
with open(os.path.join(OUT, "model_source.py"), "w", encoding="utf-8") as f:
    f.write(inspect.getsource(mod))
for name in list(sys.modules):
    if "BertForJointParsing" in name or "dictabert" in name.lower() or name.startswith("transformers_modules"):
        m = sys.modules[name]
        try:
            src = inspect.getsource(m)
            fn = name.replace(".", "_") + ".py"
            with open(os.path.join(OUT, "src_" + fn), "w", encoding="utf-8") as f:
                f.write(src)
            say("saved source:", name)
        except Exception as e:
            say("no source for", name, e)
model.config.to_json_file(os.path.join(OUT, "config.json"))
tok.save_pretrained(OUT)  # vocab.txt, tokenizer.json, tokenizer_config.json

# 2) Эталонный разбор.
with torch.no_grad():
    ref = model.predict(SAMPLES, tok, output_style="json")
with open(os.path.join(OUT, "reference.json"), "w", encoding="utf-8") as f:
    json.dump(ref, f, ensure_ascii=False, indent=1)
say("reference written")

# 3) Выходы forward — какие тензоры возвращает модель.
enc = tok(SAMPLES[:2], return_tensors="pt", padding=True)
with torch.no_grad():
    raw = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])


def flatten(obj, prefix="out"):
    """Все тензоры из выхода модели: dataclass / dict / tuple → плоский список (имя, тензор)."""
    items = []
    if torch.is_tensor(obj):
        items.append((prefix, obj))
    elif dataclasses.is_dataclass(obj) or hasattr(obj, "to_tuple"):
        d = obj.__dict__ if dataclasses.is_dataclass(obj) else dict(obj.items())
        for k, v in d.items():
            items += flatten(v, f"{prefix}.{k}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            items += flatten(v, f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            items += flatten(v, f"{prefix}.{i}")
    return items


flat = flatten(raw)
say("forward outputs:")
for n, t in flat:
    say(" ", n, tuple(t.shape), t.dtype)
names = [n for n, _ in flat]
with open(os.path.join(OUT, "outputs.json"), "w", encoding="utf-8") as f:
    json.dump([{"name": n, "shape": list(t.shape), "dtype": str(t.dtype)} for n, t in flat], f, indent=1)


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return tuple(t for _, t in flatten(out))


# 4) ONNX + сжатие int8.
onnx_path = os.path.join(OUT, "morph.onnx")
try:
    out_names = ["o%d" % i for i in range(len(names))]
    dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}
    for i, (_, t) in enumerate(flat):
        axes = {0: "batch"}
        if t.dim() >= 2:
            axes[1] = "seq"
        dyn[out_names[i]] = axes
    torch.onnx.export(
        Wrapper(model), (enc["input_ids"], enc["attention_mask"]), onnx_path,
        input_names=["input_ids", "attention_mask"], output_names=out_names,
        dynamic_axes=dyn, opset_version=17, do_constant_folding=True,
    )
    with open(os.path.join(OUT, "onnx_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(out_names, names)), f, indent=1)
    say("onnx exported:", os.path.getsize(onnx_path) // 1_000_000, "MB")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = os.path.join(OUT, "morph.int8.onnx")
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    say("int8:", os.path.getsize(q_path) // 1_000_000, "MB")
    os.remove(onnx_path)  # большой fp32 не публикуем

    # Проверка: int8 даёт те же argmax, что и PyTorch.
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(q_path)
    res = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})
    for (n, t), r in zip(flat, res):
        if t.dim() >= 3 and t.dtype.is_floating_point:
            agree = float((t.argmax(-1).numpy() == r.argmax(-1)).mean())
            say(f"  check {n}: argmax agreement {agree:.3f}")
except Exception:
    say("ONNX FAILED:\n" + traceback.format_exc())

say("done")
log.close()
