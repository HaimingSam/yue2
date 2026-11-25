import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yue2-cover-test'))
SPEC = importlib.util.spec_from_file_location('phrasing', ROOT / 'yue2-cover-test/prepare_hjf3_lyrics_phrasing.py')
phrasing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phrasing)


class PhraseBreathTests(unittest.TestCase):
    def test_both_detectors_support_cut(self):
        self.assertTrue(phrasing.allow_breath(8.11, 8.97, 9.15, 8.842, 0.929, [0.056, 0.111]))

    def test_word_not_finished(self):
        self.assertFalse(phrasing.allow_breath(4.0, 4.5, 4.67, 4.581, 0.9, [0, 0]))

    def test_one_detector_finds_voice(self):
        self.assertFalse(phrasing.allow_breath(8, 8.8, 9, 8.7, 0.9, [0, 0.3]))

    def test_low_word_score(self):
        self.assertFalse(phrasing.allow_breath(8, 8.8, 9, 8.7, 0.4, [0, 0]))

    def test_do_not_destroy_short_note(self):
        self.assertFalse(phrasing.allow_breath(8.7, 8.8, 9, 8.7, 0.9, [0, 0]))

    def test_missing_frames_fail_closed(self):
        self.assertEqual(phrasing.voiced_fraction(np.array([0]), np.array([440]), np.array([1]), 2, 3), 1)


if __name__ == '__main__':
    unittest.main()
