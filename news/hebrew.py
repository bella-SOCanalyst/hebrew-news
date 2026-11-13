"""Иврит: те же правила, что в приложении (HebrewText) и в tools/tts — чтобы имена MP3 совпадали."""
from __future__ import annotations

import hashlib

VOICE = "he-IL-Chirp3-HD-Iapetus"


def is_mark(c: str) -> bool:
    return ("֑" <= c <= "ֽ") or c in "ׇֿׁׂׅׄ"


def is_bidi(c: str) -> bool:
    return c in "‎‏" or "‪" <= c <= "‮" or "⁦" <= c <= "⁩"


def is_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def strip_nikud(text: str) -> str:
    """Точно как HebrewText.stripNikud в приложении."""
    return "".join(c for c in text if not is_mark(c) and not is_bidi(c))


def words_plain(token: str) -> str:
    """Слово без знаков препинания (буквы и никуд), напр. «הַבַּיִת,» → «הַבַּיִת»."""
    return "".join(c for c in token if is_letter(c) or is_mark(c))


def audio_name(plain: str, voice: str = VOICE) -> str:
    """Как SpeechPlayer.audioName: первые 16 hex SHA-1("голос|текст без никуда")."""
    return hashlib.sha1(f"{voice}|{plain}".encode("utf-8")).hexdigest()[:16]
