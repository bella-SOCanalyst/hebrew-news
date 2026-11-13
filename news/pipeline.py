#!/usr/bin/env python3
"""
Ежедневный выпуск учебных новостей на иврите для приложения «Иврит».

Каждый день — 3 новости: Общество, Технологии, Культура. Уровни по кругу (алеф / бет / гимель):
у каждой темы каждый день свой, за три дня тема проходит все три уровня.

Как соблюдаются права:
  * с новостных сайтов берутся ТОЛЬКО факты — заголовки из RSS Google Новостей (факты не охраняются
    авторским правом); текст статей и фото не копируются;
  * учебный текст пишет Gemini — своими словами, упрощённо, с русским заголовком и переводом;
  * картинка — только со свободной лицензией из Wikimedia Commons (CC0 / PD / CC BY / CC BY-SA)
    с указанием автора; если подходящей нет — приложение покажет значок темы;
  * в выпуске сохраняются ссылки на исходные сообщения.

Дальше: никуд — модель Dicta (dicta-il/dictabert-large-char-menaked, CC BY 4.0),
голос — Google Cloud TTS, Chirp 3 HD «Iapetus» (как в приложении).

Ключи — только из переменных окружения (секреты GitHub): GEMINI_API_KEY, GOOGLE_TTS_API_KEY.

  python news/pipeline.py                 # сделать выпуск на сегодня
  python news/pipeline.py --date 2026-11-15 --no-nikud --no-audio   # быстрый прогон без модели и голоса
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hebrew import audio_name, strip_nikud, words_plain  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "feed"
AUDIO = ROOT / "audio"
KEEP_DAYS = 7          # в latest.json — последние 7 дней
AUDIO_KEEP_DAYS = 30   # MP3 старше месяца удаляются, чтобы репозиторий не рос бесконечно
UA = "HebrewLearningNews/1.0 (personal learning app; github actions)"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = "he-IL-Chirp3-HD-Iapetus"

# Темы: (ключ, русское название, раздел Google Новостей)
CATEGORIES = [
    ("society", "Общество", "NATION"),
    ("tech", "Технологии", "TECHNOLOGY"),
    ("culture", "Культура", "ENTERTAINMENT"),
]

LEVELS = {
    1: dict(name="алеф", cefr="A1–A2", spec=(
        "Level ALEF (absolute beginner, ulpan alef): 4–6 very short sentences (5–8 words each). "
        "Only the most frequent ~1000 words. Mostly present tense, simple subject–verb–object. "
        "No idioms, no construct chains (smichut) except very common ones, no passive.")),
    2: dict(name="бет", cefr="A2–B1", spec=(
        "Level BET (ulpan bet): 6–8 sentences (7–12 words each). Common vocabulary, past and future tense "
        "allowed, simple connectors (כי, אבל, לכן, ש־). Avoid rare words and long smichut chains.")),
    3: dict(name="гимель", cefr="B1–B2", spec=(
        "Level GIMEL (ulpan gimel): 8–10 sentences (10–16 words each). Natural newspaper style, "
        "smichut, relative clauses, all tenses and binyanim, typical news vocabulary.")),
}


def levels_for(day: dt.date) -> dict[str, int]:
    """Уровень каждой темы на дату: круговой сдвиг, за 3 дня каждая тема — алеф, бет и гимель."""
    n = day.toordinal()
    return {key: (n + i) % 3 + 1 for i, (key, _, _) in enumerate(CATEGORIES)}


# ---------------------------------------------------------------- сеть

def http(url: str, body: dict | None = None, timeout: int = 60, headers: dict | None = None) -> bytes:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    h = {"User-Agent": UA}
    if body is not None:
        h["Content-Type"] = "application/json; charset=utf-8"
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------- 1. факты (только заголовки)

def headlines(topic: str, limit: int = 12) -> list[dict]:
    """Заголовки Google Новостей (Израиль, иврит) по разделу. Это факты, тексты статей не берём."""
    url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=he&gl=IL&ceid=IL:he"
    return parse_rss(http(url).decode("utf-8"), limit)


def parse_rss(xml_text: str, limit: int = 12) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "").strip()
        # «Заголовок - Источник» → отделяем источник
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        if title:
            out.append({"title": html.unescape(title), "source": source, "link": link})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 2. текст (Gemini)

ARTICLE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chosen_index": {"type": "INTEGER"},
        "title_he": {"type": "STRING"},
        "title_ru": {"type": "STRING"},
        "sentences": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"he": {"type": "STRING"}, "ru": {"type": "STRING"}},
            "required": ["he", "ru"],
        }},
        "image_query": {"type": "STRING"},
        "glossary": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "form": {"type": "STRING"},
                "lemma": {"type": "STRING"},
                "pos": {"type": "STRING", "enum": [
                    "NOUN", "VERB", "ADJECTIVE", "ADVERB", "PRONOUN", "PREPOSITION",
                    "CONJUNCTION", "PARTICLE", "NUMERAL", "PROPER_NOUN", "PHRASE"]},
                "ru": {"type": "STRING"},
                "root": {"type": "STRING"},
                "binyan": {"type": "STRING", "enum": [
                    "NONE", "PAAL", "NIFAL", "PIEL", "PUAL", "HIFIL", "HUFAL", "HITPAEL"]},
            },
            "required": ["form", "lemma", "pos", "ru"],
        }},
    },
    "required": ["chosen_index", "title_he", "title_ru", "sentences", "image_query", "glossary"],
}


def prompt_for(category_ru: str, level: int, heads: list[dict]) -> str:
    lv = LEVELS[level]
    lines = "\n".join(f"{i}. {h['title']} ({h['source']})" for i, h in enumerate(heads))
    return f"""You write short ORIGINAL news texts in Hebrew for Russian-speaking adult learners of Hebrew.

Topic: {category_ru}. Today's Israeli headlines (facts only):
{lines}

1. Choose ONE headline (chosen_index) that suits learners: prefer neutral or positive, everyday,
   understandable stories; avoid graphic violence, casualties, crime details and partisan politics.
2. Write the story in YOUR OWN WORDS — do not reproduce any outlet's wording. Use only facts from the
   headlines plus widely known background; do not invent names, numbers, quotes or details.
3. {lv['spec']}
4. Spelling: modern Israeli full spelling (כתיב מלא) WITHOUT nikud, like a newspaper.
5. title_he: short Hebrew headline (no nikud). title_ru: natural Russian headline.
6. sentences: each Hebrew sentence with an accurate natural Russian translation.
7. image_query: 2–4 English words for a free stock photo on Wikimedia Commons (a generic scene or object,
   never a real person's name), e.g. "electric bus city", "theater stage".
8. glossary: every content word of the text once (nouns, verbs, adjectives, adverbs, numerals, phrases;
   skip names of people). form = exactly as in the text but WITHOUT prefixes ו/ה/ב/ל/כ/מ/ש;
   lemma = dictionary form without nikud (verbs: infinitive, nouns: singular, adjectives: masculine singular);
   ru = short Russian translation of the lemma; root = consonants with dashes like "כ-ת-ב" or "" ;
   binyan for verbs, otherwise NONE.
"""


def gemini(api_key: str, prompt: str, schema: dict) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.6,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    resp = json.loads(http(url, body, timeout=180))
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def clean_hebrew(s: str) -> str:
    """Убираем никуд (если модель всё-таки поставила) и лишние пробелы."""
    return re.sub(r"\s+", " ", strip_nikud(s)).strip()


# ---------------------------------------------------------------- 3. картинка (Wikimedia Commons)

FREE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [0-9.]+)", re.IGNORECASE)


def commons_image(query: str) -> dict | None:
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": "6",
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": "15",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": "960",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    data = json.loads(http(url, timeout=30))
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not FREE.match(lic.strip()) or info.get("width", 0) < 600 or info.get("width", 0) < info.get("height", 1):
            continue
        artist = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")).strip() or "автор не указан"
        return {
            "url": info.get("thumburl") or info.get("url"),
            "author": html.unescape(artist)[:80],
            "license": lic,
            "page": info.get("descriptionurl"),
        }
    return None


# ---------------------------------------------------------------- 4. никуд (Dicta)

class Nikud:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer  # импорт только при запуске
        mid = "dicta-il/dictabert-large-char-menaked"
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid, trust_remote_code=True)
        self.model.eval()

    def vocalize(self, plain: list[str]) -> list[str]:
        # '*' помечает буквы-огласовки (ктив мале): убираем знак, буквы оставляем — как в «שער למתחיל».
        out = []
        for i in range(0, len(plain), 16):
            out.extend(self.model.predict(plain[i:i + 16], self.tok, mark_matres_lectionis="*"))
        return [s.replace("*", "") for s in out]


class NoNikud:
    def vocalize(self, plain: list[str]) -> list[str]:
        return list(plain)


def align_forms(plain: str, vocal: str) -> dict[str, str]:
    """Слово без никуда → то же слово с никудом (буквы и пробелы у Dicta совпадают с исходником)."""
    pw, vw = plain.split(" "), vocal.split(" ")
    if len(pw) != len(vw):
        return {}
    out = {}
    for p, v in zip(pw, vw):
        key = words_plain(p)
        if key and strip_nikud(words_plain(v)) == key:
            out[key] = words_plain(v)
    return out


# ---------------------------------------------------------------- 5. голос (Google TTS)

def synthesize(api_key: str, text: str) -> bytes:
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "he-IL", "name": VOICE},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.0},
    }
    return base64.b64decode(json.loads(http(url, body))["audioContent"])


# ---------------------------------------------------------------- сборка выпуска

def make_item(day: dt.date, key: str, cat_ru: str, topic: str, level: int, gemini_key: str) -> dict | None:
    heads = headlines(topic)
    if not heads:
        print(f"[{cat_ru}] нет заголовков", flush=True)
        return None
    art = gemini(gemini_key, prompt_for(cat_ru, level, heads), ARTICLE_SCHEMA)
    chosen = heads[min(max(art.get("chosen_index", 0), 0), len(heads) - 1)]
    sentences = [{"plain": clean_hebrew(s["he"]), "ru": s["ru"].strip()} for s in art["sentences"] if s.get("he")]
    image = None
    try:
        image = commons_image(art.get("image_query", ""))
    except Exception as e:  # картинка не обязательна
        print(f"[{cat_ru}] картинка: {e}", flush=True)
    return {
        "id": f"{day.isoformat()}-{key}",
        "date": day.isoformat(),
        "category": cat_ru,
        "level": level,
        "cefr": LEVELS[level]["cefr"],
        "title_ru": art["title_ru"].strip(),
        "title_plain": clean_hebrew(art["title_he"]),
        "sentences": sentences,
        "glossary": [
            {k: (g.get(k) or "").strip() for k in ("form", "lemma", "pos", "ru", "root", "binyan")}
            for g in art.get("glossary", []) if g.get("lemma")
        ],
        "image": image,
        "sources": [{"title": chosen["title"], "outlet": chosen["source"], "url": chosen["link"]}],
    }


def add_nikud(items: list[dict], nikud) -> None:
    for it in items:
        plains = [it["title_plain"]] + [s["plain"] for s in it["sentences"]]
        vocal = nikud.vocalize(plains)
        it["title_he"] = vocal[0]
        forms: dict[str, str] = {}
        for s, v in zip(it["sentences"], vocal[1:]):
            s["he"] = v
            forms.update(align_forms(s["plain"], v))
        lemmas = [clean_hebrew(g["lemma"]) for g in it["glossary"]]
        lemma_vocal = nikud.vocalize(lemmas) if lemmas else []
        for g, lv in zip(it["glossary"], lemma_vocal):
            g["lemma"] = clean_hebrew(g["lemma"])
            g["lemma_nikud"] = lv
            g["form"] = clean_hebrew(g["form"])
            g["form_nikud"] = forms.get(g["form"], "")


def add_audio(items: list[dict], tts_key: str, day: dt.date) -> None:
    folder = AUDIO / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    for it in items:
        for s in it["sentences"]:
            name = audio_name(strip_nikud(s["he"])) + ".mp3"
            path = folder / name
            if not path.exists():
                path.write_bytes(synthesize(tts_key, s["plain"]))
            s["audio"] = f"audio/{day.isoformat()}/{name}"


def write_feed(day: dt.date, items: list[dict]) -> None:
    (FEED / "archive").mkdir(parents=True, exist_ok=True)
    (FEED / "archive" / f"{day.isoformat()}.json").write_text(
        json.dumps({"date": day.isoformat(), "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    # latest.json — все выпуски за последние KEEP_DAYS дней (приложение берёт то, чего у него нет).
    all_items = []
    for d in range(KEEP_DAYS):
        f = FEED / "archive" / f"{(day - dt.timedelta(days=d)).isoformat()}.json"
        if f.exists():
            all_items += json.loads(f.read_text(encoding="utf-8"))["items"]
    (FEED / "latest.json").write_text(json.dumps({
        "format": 1,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "items": all_items,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def prune_audio(day: dt.date) -> None:
    if not AUDIO.exists():
        return
    for d in AUDIO.iterdir():
        try:
            if (day - dt.date.fromisoformat(d.name)).days > AUDIO_KEEP_DAYS:
                shutil.rmtree(d)
        except ValueError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="дата выпуска YYYY-MM-DD (по умолчанию — сегодня по Израилю)")
    ap.add_argument("--no-nikud", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--force", action="store_true", help="пересоздать выпуск, даже если он уже есть")
    args = ap.parse_args(argv)

    israel = dt.timezone(dt.timedelta(hours=2))
    day = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(israel).date()
    if (FEED / "archive" / f"{day.isoformat()}.json").exists() and not args.force:
        print(f"Выпуск {day} уже есть.")
        return 0

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    tts_key = os.environ.get("GOOGLE_TTS_API_KEY", "").strip()
    if not gemini_key:
        print("Нет GEMINI_API_KEY (секрет репозитория).")
        return 2

    lv = levels_for(day)
    items = []
    for key, cat_ru, topic in CATEGORIES:
        try:
            it = make_item(day, key, cat_ru, topic, lv[key], gemini_key)
            if it:
                items.append(it)
                print(f"[{cat_ru}] уровень {LEVELS[lv[key]]['name']}: {it['title_ru']}", flush=True)
        except Exception as e:  # одна тема упала — остальные всё равно выйдут
            print(f"[{cat_ru}] ошибка: {e}", flush=True)
    if not items:
        print("Не получилось ни одной новости.")
        return 1

    add_nikud(items, NoNikud() if args.no_nikud else Nikud())
    if not args.no_audio and tts_key:
        add_audio(items, tts_key, day)
    write_feed(day, items)
    prune_audio(day)
    print(f"Готово: {len(items)} новости за {day}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
