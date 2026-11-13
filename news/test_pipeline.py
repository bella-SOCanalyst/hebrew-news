import datetime as dt
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
