"""
Морфологический разбор иврита для приложения — подготовка модели (GitHub Actions, запуск вручную, один раз).

Модель: dicta-il/dictabert-tiny-joint (Dicta, CC BY 4.0 — можно использовать с указанием авторства).
Нужны только головы lex (начальная форма) и morph (часть речи, род, число, лицо, время, приставки, суффикс).

Результат в models/morph/:
  morph.int8.onnx   — модель для телефона (выходы уже «разобраны»: номера меток, а не вероятности);
  vocab.txt         — словарь WordPiece;
  pretokenizer.txt  — регулярное выражение разбиения на слова (как в токенизаторе Dicta);
  normalizer.txt    — допустимые символы (остальное → [UNK]);
  labels.json       — названия меток;
  report.txt        — проверка: наш токенизатор и разбор совпадают с оригиналом Dicta.
"""
import json
import os
import re as stdre
import traceback
import unicodedata

import regex  # \p{P}, как в Rust-токенизаторе
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "dicta-il/dictabert-tiny-joint"
OUT = "models/morph"
os.makedirs(OUT, exist_ok=True)
for old in os.listdir(OUT):  # старые файлы первой попытки
    if old.startswith("src_") or old in ("model_source.py", "export_log.txt", "outputs.json", "onnx_outputs.json"):
        os.remove(os.path.join(OUT, old))
rep = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")


def say(*a):
    print(*a, flush=True)
    print(*a, file=rep, flush=True)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, do_syntax=False, do_ner=False, do_prefix=False)
model.eval()
src = __import__("sys").modules[type(model).__module__]
morph_mod = __import__("sys").modules[type(model.morph).__module__]
ALL_POS = morph_mod.ALL_POS
ALL_PREFIX_POS = morph_mod.ALL_PREFIX_POS
ALL_SUFFIX_POS = morph_mod.ALL_SUFFIX_POS
ALL_FEATURES = morph_mod.ALL_FEATURES
json.dump(dict(pos=ALL_POS, prefix=ALL_PREFIX_POS, suffix=ALL_SUFFIX_POS, features=ALL_FEATURES),
          open(os.path.join(OUT, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# --- токенизатор: выгружаем правила и проверяем свою копию алгоритма (её повторяет приложение на Kotlin) ---
tj = json.loads(tok.backend_tokenizer.to_str())
say("normalizer:", json.dumps(tj["normalizer"], ensure_ascii=False)[:300])
pre = tj["pre_tokenizer"]
say("pre_tokenizer:", {k: v for k, v in pre.items() if k != "pattern"})
PRE = pre["pattern"]["Regex"]
open(os.path.join(OUT, "pretokenizer.txt"), "w", encoding="utf-8").write(PRE)
norm_regex = [n for n in tj["normalizer"]["normalizers"] if n["type"] == "Replace" and "Regex" in n["pattern"]][0]["pattern"]["Regex"]
open(os.path.join(OUT, "normalizer.txt"), "w", encoding="utf-8").write(norm_regex)
tok.save_pretrained(OUT)
vocab = {}
for i, line in enumerate(open(os.path.join(OUT, "vocab.txt"), encoding="utf-8")):
    vocab[line.rstrip("\n")] = i
PRE_RE = regex.compile(PRE)
NORM_RE = regex.compile(norm_regex)


def my_normalize(s):
    s = unicodedata.normalize("NFD", s).lower()
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("<foreign>", "[UNK]")
    return NORM_RE.sub("[UNK]", s)


def wordpiece(word):
    if len(word) > 100:
        return ["[UNK]"]
    out, start = [], 0
    while start < len(word):
        end, cur = len(word), None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:
            return ["[UNK]"]
        out.append(cur)
        start = end
    return out


def my_tokenize(text):
    pieces = []
    for m in PRE_RE.finditer(my_normalize(text)):
        w = m.group(0)
        pieces += ["[UNK]"] if w == "[UNK]" else wordpiece(w)
    return ["[CLS]"] + pieces + ["[SEP]"]


sentences = []
try:
    feed = json.load(open("feed/latest.json", encoding="utf-8"))
    for item in feed["items"]:
        sentences += [s["plain"] for s in item["sentences"]]
except Exception as e:
    say("no feed:", e)
sentences += [
    "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות",
    "היא לא ידעה מה לומר אז היא שתקה.", "סגרי את הדלת, בבקשה!", "אנחנו נרקוד מחר בחוג היפ הופ.",
    "כשהגעתי הביתה ראיתי שהחתול אכל את האוכל", "ה\"ביץ של המורה", "מה שלומך? 100% טוב!",
]
same = 0
for s in sentences:
    ref = tok.convert_ids_to_tokens(tok(s)["input_ids"])
    mine = my_tokenize(s)
    if ref == mine:
        same += 1
    elif same < 3 or True:
        say("TOKENIZER DIFF:", s, "\n  hf:", ref, "\n  my:", mine)
say(f"tokenizer: {same}/{len(sentences)} sentences identical")

# --- эталонный разбор Dicta ---
with torch.no_grad():
    ref_out = model.predict(sentences, tok, output_style="json", compute_syntax_mst=None)
json.dump(ref_out[-7:], open(os.path.join(OUT, "reference.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


class Wrapper(torch.nn.Module):
    """Выход — сразу номера меток (argmax / top-3), чтобы телефону не возиться с большими массивами."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        mo = out.morph_logits
        lex = torch.topk(out.lex_logits, 3, dim=-1).indices
        pos = mo.pos_logits.argmax(-1)
        feats = torch.stack([f.argmax(-1) for f in mo.features_logits], dim=-1)
        prefix = (mo.prefix_logits > 0.5).to(torch.int64)
        suffix = mo.suffix_logits.argmax(-1)
        sfeats = torch.stack([f.argmax(-1) for f in mo.suffix_features_logits], dim=-1)
        return lex, pos, feats, prefix, suffix, sfeats


OUTS = ["lex", "pos", "feats", "prefix", "suffix", "suffix_feats"]
enc = tok(sentences[:2], return_tensors="pt", padding=True)
onnx_path = os.path.join(OUT, "morph.onnx")
q_path = os.path.join(OUT, "morph.int8.onnx")
try:
    torch.onnx.export(
        Wrapper(model), (enc["input_ids"], enc["attention_mask"]), onnx_path,
        input_names=["input_ids", "attention_mask"], output_names=OUTS,
        dynamic_axes={n: {0: "batch", 1: "seq"} for n in ["input_ids", "attention_mask"] + OUTS},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    say("onnx fp32:", os.path.getsize(onnx_path) // 1_000_000, "MB")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    say("onnx int8:", os.path.getsize(q_path) // 1_000_000, "MB")
    os.remove(onnx_path)
except Exception:
    say("ONNX FAILED:\n" + traceback.format_exc())
    raise

# --- проверка: наш токенизатор + int8-модель + наш разбор == разбор Dicta ---
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(q_path)
lex_tokens = tok.convert_ids_to_tokens(list(range(len(vocab))))


def pick_lex(token, lexemes):  # как pick_lex_by_letters у Dicta
    possible = set(c for c in token if c not in "אהוי")
    for lx in lexemes:
        if sum(c in possible for c in lx) >= min([2, len(possible), len([c for c in lx if c not in "אהוי"])]):
            return lx
    return "[BLANK]"


tot = ok_pos = ok_lex = ok_feats = ok_pref = 0
special = {"[CLS]", "[SEP]", "[PAD]"}
for s, ref in zip(sentences, ref_out):
    toks = my_tokenize(s)
    ids = np.array([[vocab[t] for t in toks]], dtype=np.int64)
    lex, pos, feats, prefix, suffix, sfeats = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    words = []
    for i, t in enumerate(toks):
        if t in special:
            continue
        if t.startswith("##"):
            words[-1]["token"] += t[2:]
            continue
        words.append(dict(
            token=t,
            lex=[lex_tokens[j] for j in lex[0, i]],
            pos=ALL_POS[pos[0, i]],
            feats={n: v[feats[0, i, k]] for k, (n, v) in enumerate(ALL_FEATURES) if v[feats[0, i, k]] != "none"},
            prefixes=[ALL_PREFIX_POS[k] for k in range(len(ALL_PREFIX_POS)) if prefix[0, i, k] > 0],
        ))
    for w, r in zip(words, ref["tokens"]):
        tot += 1
        ok_pos += w["pos"] == r["morph"]["pos"]
        ok_feats += w["feats"] == r["morph"]["feats"]
        ok_pref += w["prefixes"] == r["morph"]["prefixes"]
        ok_lex += pick_lex(w["token"], w["lex"]) == r["lex"]
say(f"words {tot}: pos {ok_pos/tot:.3f}, feats {ok_feats/tot:.3f}, prefixes {ok_pref/tot:.3f}, lex {ok_lex/tot:.3f}")
say("done")
rep.close()
