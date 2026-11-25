import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    'hjf3_comparison', Path(__file__).resolve().parents[1] / 'yue2-cover-test/run_hjf3_comparison.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
MINOR_SPEC = importlib.util.spec_from_file_location(
    'hjf3_bb_minor', Path(__file__).resolve().parents[1] / 'yue2-cover-test/run_hjf3_bb_minor.py')
minor = importlib.util.module_from_spec(MINOR_SPEC)
MINOR_SPEC.loader.exec_module(minor)


class ScaleSnapTests(unittest.TestCase):
    def test_scale_notes_unchanged(self):
        for pitch in range(128):
            if pitch % 12 in comparison.D_MAJOR:
                self.assertEqual(comparison.snap_d_major(pitch, pitch + 0.4), pitch)

    def test_all_results_in_scale(self):
        for pitch in range(128):
            result = comparison.snap_d_major(pitch, pitch)
            self.assertIn(result % 12, comparison.D_MAJOR)
            self.assertLessEqual(abs(result - pitch), 1)

    def test_tie_uses_f0_direction(self):
        self.assertEqual(comparison.snap_d_major(70, 70.2), 71)
        self.assertEqual(comparison.snap_d_major(70, 69.8), 69)

    def test_exact_tie_is_deterministic(self):
        self.assertEqual(comparison.snap_d_major(70, 70.0), 69)

    def test_bb_natural_minor(self):
        self.assertEqual(minor.SCALE, {10, 0, 1, 3, 5, 6, 8})
        for pitch in range(128):
            result = minor.snap(pitch, pitch)
            self.assertIn(result % 12, minor.SCALE)
            self.assertLessEqual(abs(result - pitch), 1)
            if pitch % 12 in minor.SCALE:
                self.assertEqual(result, pitch)

    def test_bb_minor_tie_uses_f0(self):
        self.assertEqual(minor.snap(69, 69.2), 70)
        self.assertEqual(minor.snap(69, 68.8), 68)


if __name__ == '__main__':
    unittest.main()
