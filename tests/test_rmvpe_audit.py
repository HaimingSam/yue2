import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    'rmvpe_audit', Path(__file__).resolve().parents[1] / 'yue2-cover-test/audit_rmvpe.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RmvpeAuditTests(unittest.TestCase):
    def stats(self, pitch, activation=0.9):
        times = np.arange(100) * 0.01
        f0 = np.full(100, 440 * 2 ** ((pitch - 69) / 12))
        return audit.summarize(times, f0, np.full(100, activation), 0, 1)

    def test_stable_semitone_correction(self):
        stats = self.stats(70)
        self.assertTrue(stats['stable'])
        self.assertEqual(audit.decide(69, stats, stats), (70, 'auto_corrected'))

    def test_preserve_supported_original(self):
        stats = self.stats(69)
        self.assertEqual(audit.decide(69, stats, stats), (69, 'supported_original'))

    def test_octave_requires_review(self):
        stats = self.stats(81)
        self.assertEqual(audit.decide(69, stats, stats), (69, 'review_large_interval'))

    def test_disagreement_requires_review(self):
        self.assertEqual(audit.decide(69, self.stats(70), self.stats(71)), (69, 'review_disagreement'))

    def test_unvoiced_and_low_activation(self):
        stats = self.stats(70, activation=0.01)
        self.assertFalse(stats['stable'])
        self.assertEqual(audit.decide(69, stats, stats), (69, 'review_unvoiced'))

    def test_split_pitch_is_not_stable(self):
        times = np.arange(100) * 0.01
        pitches = np.concatenate([np.full(50, 69), np.full(50, 72)])
        stats = audit.summarize(times, 440 * 2 ** ((pitches - 69) / 12), np.ones(100), 0, 1)
        self.assertFalse(stats['stable'])

    def test_short_note_requires_review(self):
        times = np.arange(10) * 0.01
        stats = audit.summarize(times, np.full(10, 440), np.ones(10), 0, 0.08)
        self.assertFalse(stats['stable'])

    def test_boundary_glides_excluded(self):
        times = np.arange(100) * 0.01
        f0 = np.full(100, 440.0)
        f0[:8] = 880
        f0[92:] = 220
        stats = audit.summarize(times, f0, np.ones(100), 0, 1)
        self.assertTrue(stats['stable'])
        self.assertEqual(stats['candidate'], 69)


if __name__ == '__main__':
    unittest.main()
