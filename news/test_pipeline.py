import datetime as dt
import json
import unittest

import pipeline as p
from hebrew import audio_name, strip_nikud


class LevelsTest(unittest.TestCase):
    def test_every_day_has_alef_bet_gimel(self):
        for d in range(30):
            lv = p.levels_for(dt.date(2026, 11, 1) + dt.timedelta(days=d))
            self.assertEqual(sorted(lv.values()), [1, 2, 3])

    def test_each_topic_rotates_through_all_levels_in_3_days(self):
        start = dt.date(2026, 11, 14)
        for key, _, _ in p.CATEGORIES:
            seen = {p.levels_for(start + dt.timedelta(days=d))[key] for d in range(3)}
            self.assertEqual(seen, {1, 2, 3})


class QuotaTest(unittest.TestCase):
    def test_quota_switches_model_without_waiting(self):
        calls = []

        def fake_http(url, body=None, timeout=60, headers=None):
            calls.append(url)
            if "big-flash" in url:
                raise RuntimeError('HTTP 429 ... { "error": { "code": 429, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED" } }')
            return json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]}).encode()

        old_http, old_cache, old_sleep = p.http, list(p._model_cache), p.time.sleep
        p.http = fake_http
        p._model_cache[:] = ["big-flash", "small-flash"]
        p._exhausted.clear()
        p.time.sleep = lambda s: self.fail("не должно быть ожидания при исчерпанной квоте")
        try:
            self.assertEqual(p.gemini("k", "x", {}), {"ok": 1})
            self.assertEqual(p.gemini("k", "y", {}), {"ok": 1})
            self.assertEqual(sum("big-flash" in c for c in calls), 1)  # второй раз исчерпанную модель не трогаем
        finally:
            p.http, p.time.sleep = old_http, old_sleep
            p._model_cache[:] = old_cache
            p._exhausted.clear()


class LengthTest(unittest.TestCase):
    def test_count_words_skips_punctuation(self):
        self.assertEqual(p.count_words([{"plain": "שלום עולם , היום - יום יפה."}, {"plain": "כן"}]), 6)

    def test_prompt_demands_min_words(self):
        prompt = p.prompt_for("Общество", 1, [{"title": "כותרת", "source": "ynet"}])
        self.assertIn(str(p.MIN_WORDS), prompt)


class RssTest(unittest.TestCase):
    def test_source_suffix_removed(self):
        xml = """<rss><channel>
        <item><title>רכבת חדשה לחיפה - ynet</title><link>https://x/1</link><source url="https://ynet">ynet</source></item>
        <item><title>פסטיבל בתל אביב &amp; ביפו - כאן</title><link>https://x/2</link><source>כאן</source></item>
        </channel></rss>"""
        items = p.parse_rss(xml)
        self.assertEqual(items[0]["title"], "רכבת חדשה לחיפה")
        self.assertEqual(items[0]["source"], "ynet")
        self.assertEqual(items[1]["title"], "פסטיבל בתל אביב & ביפו")


class AlignTest(unittest.TestCase):
    def test_forms_mapped_with_punctuation(self):
        m = p.align_forms("הספרייה נפתחה, היום", "הַסִּפְרִיָּה נִפְתְּחָה, הַיּוֹם")
        self.assertEqual(m["נפתחה"], "נִפְתְּחָה")
        self.assertEqual(m["היום"], "הַיּוֹם")

    def test_length_mismatch_gives_nothing(self):
        self.assertEqual(p.align_forms("א ב", "אַ"), {})


class AudioNameTest(unittest.TestCase):
    def test_same_as_app(self):
        # То же имя, что считает SpeechPlayer.audioName(stripNikud(textWithNikud)).
        self.assertEqual(audio_name(strip_nikud("שָׁלוֹם")), audio_name("שלום"))
        self.assertEqual(len(audio_name("שלום")), 16)


if __name__ == "__main__":
    unittest.main()
