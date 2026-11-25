import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "yue2-cover-mashup/make_mashup_generic.py"
SPEC = importlib.util.spec_from_file_location("make_mashup_generic", SCRIPT)
mashup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mashup
SPEC.loader.exec_module(mashup)


def source_score(denominator, bpm, key="C"):
    note = "C8D8E8F8|" if denominator == 32 else "C4D4E4F4|"
    return ("X:1\nT:\nM:4/4\n" + f"L:1/{denominator}\nQ:1/4={bpm}\n" +
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n' +
            'V: Ins clef=treble name="Ins Melody" snm="Inst."\n' + f"K:{key}\n" +
            "% intro\nV: Vocal\nZ|\nV: Ins\nZ|\n" +
            f"% verse\nV: Vocal\n{note}\nV: Ins\nZ|\n")


class MashupGenericTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root / "a.abc").write_text(source_score(32, 60), encoding="utf-8")
        (self.root / "b.abc").write_text(source_score(16, 120), encoding="utf-8")
        (self.root / "a.json").write_text(json.dumps({"lyrics": [
            {"start": 5, "end": 6, "text": "A lyric"}]}), encoding="utf-8")
        (self.root / "b.json").write_text(json.dumps({"data": [
            {"text": "B lyric", "words": [{"word": "B", "start": 2250, "end": 2750}]}]}), encoding="utf-8")
        self.config = {"id": "test_ab", "style": "test", "target_bpm": 60, "target_denominator": 32,
                       "max_tempo_ratio": 2.0,
                       "sources": {"A": {"score": "a.abc", "lyrics": "a.json"},
                                   "B": {"score": "b.abc", "lyrics": "b.json", "timestamp_unit": "milliseconds"}},
                       "recipe": [{"source": "A", "section": "intro", "instrumental": True},
                                  {"source": "A", "section": "verse"},
                                  {"source": "B", "section": "verse"}]}
        self.config_path = self.root / "config.json"

    def run_build(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        return mashup.build(self.config_path, self.root / "result")

    def test_rearranges_score_and_remaps_timestamped_lyrics(self):
        summary = self.run_build()
        self.assertEqual(summary["duration_seconds"], 12)
        self.assertEqual(summary["lyric_lines"], 2)
        score = (self.root / "result/score.abc").read_text(encoding="utf-8")
        self.assertIn("% B verse 1\nV: Vocal\nM:4/4\nC8D8E8F8|", score)
        aligned = json.loads((self.root / "result/aligned_lyrics.json").read_text(encoding="utf-8"))
        self.assertEqual([(item["target_start"], item["target_end"]) for item in aligned],
                         [(5, 6), (8.5, 9.5)])
        request = json.loads((self.root / "result/request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["cot"], "melody")
        self.assertIn("A lyric", request["lyrics"])
        self.assertIn("B lyric", request["lyrics"])

    def test_minimal_config_infers_request_and_recipe(self):
        self.config = {"sources": self.config["sources"]}
        summary = self.run_build()
        self.assertGreater(summary["sections"], 0)
        request = json.loads((self.root / "result/request.json").read_text(encoding="utf-8"))
        self.assertTrue(request["id"].endswith("_mashup"))
        self.assertIn("mashup duet", request["style"])

    def test_compressed_mode_records_mode_and_keeps_both_sources(self):
        self.config = {"sources": self.config["sources"], "compression": "compressed"}
        summary = self.run_build()
        self.assertGreater(summary["sections"], 1)
        manifest = json.loads((self.root / "result/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["compression"], "compressed")
        self.assertEqual({item["source"] for item in manifest["sections"]}, {"A", "B"})

    def test_rejects_nonempty_output(self):
        self.run_build()
        with self.assertRaises(FileExistsError):
            self.run_build()

    def test_rejects_missing_occurrence_without_writing(self):
        self.config["recipe"][2]["occurrence"] = 2
        with self.assertRaisesRegex(ValueError, "has no verse #2"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_transposes_different_keys(self):
        (self.root / "b.abc").write_text(source_score(16, 120, "G"), encoding="utf-8")
        self.run_build()
        score = (self.root / "result/score.abc").read_text(encoding="utf-8")
        self.assertIn("K:C", score)
        self.assertIn("% B verse 1\nV: Vocal\nM:4/4\nF8G8A8B8|", score)
        parsed = mashup.load_abc_tools().parse_abc(score)
        self.assertEqual([pitch for _, pitch, _ in parsed.voices["Vocal"].notes[-4:]], [65, 67, 69, 71])
        manifest = json.loads((self.root / "result/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["transposition_semitones"], {"A": 0, "B": 5})
        self.assertEqual(manifest["sections"][-1]["transpose_semitones"], 5)

    def test_vocal_range_selects_feasible_key(self):
        (self.root / "b.abc").write_text(source_score(16, 120, "G"), encoding="utf-8")
        self.config["vocal_pitch_range"] = [48, 70]
        self.run_build()
        manifest = json.loads((self.root / "result/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["target_key"], "G")
        self.assertEqual(manifest["transposition_semitones"], {"A": -5, "B": 0})

    def test_rejects_explicit_key_outside_vocal_range(self):
        (self.root / "b.abc").write_text(source_score(16, 120, "G"), encoding="utf-8")
        self.config.update(target_key="C", vocal_pitch_range=[48, 70])
        with self.assertRaisesRegex(ValueError, "transposed Vocal exceeds vocal_pitch_range"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_preserves_tied_accidentals_across_bars(self):
        original = source_score(16, 120, "G")
        (self.root / "b.abc").write_text(
            original.replace("C4D4E4F4|\nV: Ins\nZ|", "=F16-|F16|\nV: Ins\nZ2|"),
            encoding="utf-8")
        self.run_build()
        score = (self.root / "result/score.abc").read_text(encoding="utf-8")
        parsed = mashup.load_abc_tools().parse_abc(score)
        self.assertEqual([pitch for _, pitch, _ in parsed.voices["Vocal"].notes[-1:]], [70])
        self.assertEqual(parsed.voices["Vocal"].notes[-1][2], 8)

    def test_transposition_preserves_bar_accidentals(self):
        original = source_score(16, 120, "G")
        (self.root / "b.abc").write_text(
            original.replace("C4D4E4F4|", "G4^G4=G4G4|"), encoding="utf-8")
        self.run_build()
        score = (self.root / "result/score.abc").read_text(encoding="utf-8")
        parsed = mashup.load_abc_tools().parse_abc(score)
        self.assertEqual([pitch for _, pitch, _ in parsed.voices["Vocal"].notes[-4:]], [72, 73, 72, 72])

    def test_rejects_tie_crossing_section_boundary(self):
        original = source_score(32, 60)
        original = original.replace("% intro\nV: Vocal\nZ|", "% intro\nV: Vocal\nC32-|")
        (self.root / "a.abc").write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cuts a tied note at a boundary"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_keeps_meter_changes_within_a_section(self):
        original = source_score(16, 120).replace("M:4/4", "M:3/4", 1)
        original = original.replace(
            "% verse\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|",
            "% verse\nV: Vocal\nC4D4E4|\nV: Ins\nZ|\n"
            "V: Vocal\nM:4/4\nC4D4E4F4|\nV: Ins\nM:4/4\nZ|\n"
            "V: Vocal\nC4D4E4F4|\nV: Ins\nZ|")
        (self.root / "b.abc").write_text(original, encoding="utf-8")
        self.run_build()
        output = (self.root / "result/score.abc").read_text(encoding="utf-8")
        parsed = mashup.load_abc_tools().parse_abc(output)
        self.assertEqual([bar[2] for bar in parsed.voices["Vocal"].bars[-3:]],
                         [(3, 4), (4, 4), (4, 4)])
        manifest = json.loads((self.root / "result/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sections"][-1]["duration_quarters"], "11")
        self.assertEqual(manifest["sections"][-1]["meter_sequence"], ["3/4", "4/4", "4/4"])

    def test_rejects_meter_change_when_disabled(self):
        original = source_score(16, 120).replace("M:4/4", "M:3/4", 1)
        original = original.replace("C4D4E4F4|", "C4D4E4|")
        (self.root / "b.abc").write_text(original, encoding="utf-8")
        self.config["allow_meter_changes"] = False
        with self.assertRaisesRegex(ValueError, "allow_meter_changes is false"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_rejects_large_tempo_change(self):
        self.config["max_tempo_ratio"] = 1.5
        with self.assertRaisesRegex(ValueError, "exceeds max_tempo_ratio"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_rejects_changed_event_even_if_rebuilt_score_parses(self):
        self.run_build()
        abc_tools = mashup.load_abc_tools()
        score = (self.root / "result/score.abc").read_text(encoding="utf-8")
        corrupted = score.replace("C8D8E8F8|", "D8D8E8F8|", 1)
        source_a = mashup.load_source("A", self.config["sources"]["A"], self.root, abc_tools)
        source_b = mashup.load_source("B", self.config["sources"]["B"], self.root, abc_tools)
        emissions = [(source_a.sections[("intro", 1)], Fraction(0), 0),
                     (source_a.sections[("verse", 1)], Fraction(4), 0),
                     (source_b.sections[("verse", 1)], Fraction(8), 0)]
        with self.assertRaisesRegex(ValueError, "note onsets, pitches or durations changed"):
            mashup.verify_sections(abc_tools.parse_abc(corrupted), emissions)

    def test_reports_syllable_to_multiple_notes(self):
        (self.root / "a.json").write_text(json.dumps({"data": [{"text": "你好", "start": 4, "end": 8,
            "words": [{"word": "你", "start": 4, "end": 6},
                      {"word": "好", "start": 6, "end": 8}]}]}), encoding="utf-8")
        with mock.patch.dict(sys.modules, {"pypinyin": None}):
            self.run_build()
        data = json.loads((self.root / "result/syllable_alignment.json").read_text(encoding="utf-8"))
        report = next(line for line in data["lines"] if line["text"] == "你好")
        self.assertEqual(report["time_basis"], "word_timestamps")
        self.assertEqual(report["status"], "review")
        self.assertEqual([entry["text"] for entry in report["syllables"]], ["你", "好"])
        self.assertTrue(any(len(entry["notes"]) > 1 for entry in report["syllables"]))
        self.assertIsNone(report["syllables"][0]["pronunciation_estimate"])

    def test_uses_installed_pinyin_for_chinese(self):
        if importlib.util.find_spec("pypinyin") is None:
            self.skipTest("pypinyin is not installed")
        (self.root / "a.json").write_text(json.dumps({"lyrics": [{"text": "你好", "start": 4, "end": 8,
            "words": [{"word": "你", "start": 4, "end": 6},
                      {"word": "好", "start": 6, "end": 8}]}]}), encoding="utf-8")
        self.run_build()
        data = json.loads((self.root / "result/syllable_alignment.json").read_text(encoding="utf-8"))
        report = next(line for line in data["lines"] if line["text"] == "你好")
        self.assertEqual(report["pronunciation_source"], "pypinyin_estimate")
        self.assertEqual([entry["pronunciation_estimate"] for entry in report["syllables"]], ["ni3", "hao3"])

    def test_reports_rests_and_insufficient_notes(self):
        (self.root / "a.abc").write_text(source_score(32, 60).replace("C8D8E8F8|", "C8z8D8z8|"),
                                          encoding="utf-8")
        (self.root / "a.json").write_text(json.dumps({"lyrics": [{"text": "你好", "start": 4, "end": 8}]}),
                                          encoding="utf-8")
        self.run_build()
        data = json.loads((self.root / "result/syllable_alignment.json").read_text(encoding="utf-8"))
        report = next(line for line in data["lines"] if line["text"] == "你好")
        self.assertEqual(report["rests_seconds"], [[5, 6], [7, 8]])
        self.assertEqual([len(entry["notes"]) for entry in report["syllables"]], [1, 1])

    def test_strict_mode_accepts_mocked_pronunciation_and_words(self):
        (self.root / "a.json").write_text(json.dumps({"lyrics": [{"text": "你好", "start": 4, "end": 8,
            "words": [{"word": "你", "start": 4, "end": 6},
                      {"word": "好", "start": 6, "end": 8}]}]}), encoding="utf-8")
        self.config["syllable_alignment"] = "strict"
        self.config["recipe"][2]["instrumental"] = True
        fake_pypinyin = types.SimpleNamespace(Style=types.SimpleNamespace(TONE3=object()),
                                              lazy_pinyin=lambda text, style: ["ni3", "hao3"])
        with mock.patch.dict(sys.modules, {"pypinyin": fake_pypinyin}):
            self.run_build()
        data = json.loads((self.root / "result/syllable_alignment.json").read_text(encoding="utf-8"))
        report = next(line for line in data["lines"] if line["text"] == "你好")
        self.assertEqual(report["status"], "aligned_estimate")
        self.assertEqual([entry["pronunciation_estimate"] for entry in report["syllables"]], ["ni3", "hao3"])

    def test_strict_mode_rejects_unverified_pronunciation(self):
        (self.root / "a.json").write_text(json.dumps({"lyrics": [{"text": "A lyric", "start": 4, "end": 8}]}),
                                          encoding="utf-8")
        self.config["syllable_alignment"] = "strict"
        with self.assertRaisesRegex(ValueError, "needs manual alignment"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_reports_too_few_notes_without_inventing_syllables(self):
        (self.root / "a.abc").write_text(source_score(32, 60).replace("C8D8E8F8|", "C16D16|"),
                                          encoding="utf-8")
        (self.root / "a.json").write_text(json.dumps({"lyrics": [{"text": "你好吗", "start": 4, "end": 8}]}),
                                          encoding="utf-8")
        self.run_build()
        report = next(line for line in json.loads((self.root / "result/syllable_alignment.json").read_text())["lines"]
                      if line["text"] == "你好吗")
        self.assertEqual(report["syllables"], [])
        self.assertIn("3 个音节只有 2 个", report["suggestion"])

    def test_can_disable_syllable_alignment(self):
        self.config["syllable_alignment"] = "off"
        self.run_build()
        data = json.loads((self.root / "result/syllable_alignment.json").read_text(encoding="utf-8"))
        self.assertEqual((data["mode"], data["line_count"], data["lines"]), ("off", 0, []))

    def test_word_timestamp_maps_to_target_time(self):
        (self.root / "b.json").write_text(json.dumps({"data": [{"text": "B", "words": [
            {"word": "B", "start": 2250, "end": 2750}]}]}), encoding="utf-8")
        self.run_build()
        report = next(line for line in json.loads((self.root / "result/syllable_alignment.json").read_text())["lines"]
                      if line["text"] == "B")
        self.assertEqual(report["time_basis"], "word_timestamps")
        self.assertEqual(report["syllables"][0]["expected_seconds"], [8.5, 9.5])
        self.assertEqual(report["syllables"][0]["onset_error_ms"], -500)

    def test_rejects_chords_in_melody_mode(self):
        (self.root / "a.abc").write_text(source_score(32, 60).replace("C8D8E8F8|", '"C"C8D8E8F8|'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "chord-free input"):
            self.run_build()
        self.assertFalse((self.root / "result").exists())

    def test_audio_offset_shifts_selection_without_shifting_output(self):
        self.config["sources"]["B"]["audio_offset_seconds"] = 1
        (self.root / "b.json").write_text(json.dumps({"lyrics": [
            {"start": 3.25, "end": 3.75, "text": "shifted lyric"}]}), encoding="utf-8")
        self.config["sources"]["B"]["timestamp_unit"] = "seconds"
        self.run_build()
        aligned = json.loads((self.root / "result/aligned_lyrics.json").read_text(encoding="utf-8"))
        self.assertEqual((aligned[-1]["target_start"], aligned[-1]["target_end"]), (8.5, 9.5))

    def test_empty_section_does_not_reuse_unrelated_lyrics(self):
        (self.root / "b.json").write_text(json.dumps({"lyrics": [
            {"start": 0.25, "end": 0.75, "text": "intro only"}]}), encoding="utf-8")
        self.config["sources"]["B"]["timestamp_unit"] = "seconds"
        self.run_build()
        request = json.loads((self.root / "result/request.json").read_text(encoding="utf-8"))
        self.assertNotIn("intro only", request["lyrics"])
        self.assertEqual(request["lyrics"].split("\n\n")[-1], "[B verse 1]")


if __name__ == "__main__":
    unittest.main()
