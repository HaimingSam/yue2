import importlib.util
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("evaluate_abc_yue2_continuation",
                                              TOOLS / "evaluate_abc_yue2_continuation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_open_prefix_keeps_abc_end_open():
    from yue2.protocol import ABC_END, ABC_START, EOD, SongRequest

    class Tokenizer:
        def encode(self, text):
            return list(text.encode())

    vocal = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
             "% chorus\nV: Vocal\nC4D4E4F4|\n")
    request = SongRequest(style="pop", lyrics="la", cot="full")
    prefix, native = module.open_prefix(request, Tokenizer(), vocal)
    assert prefix == [EOD] + list(request.text().encode()) + [ABC_START] + list(native.encode())
    assert ABC_END not in prefix
    assert native.endswith("V: Ins\nZ|\n")


def test_score_continuation_measures_new_notes_and_repeated_chorus():
    vocal = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
             "% chorus\nV: Vocal\nC4D4E4F4|\n")
    native = module.restore_ins(vocal)
    completion = "% verse\nV: Vocal\nG4A4B4c4|\nV: Ins\nZ|\n% chorus\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n"
    report = module.score_continuation(native, completion, module.load_validator())
    assert report["valid"] and report["exact_chorus_repeated"]
    assert report["song_like"]
    assert report["generated_vocal_notes"] == 8
    assert report["sections"] == ["chorus", "verse", "chorus"]
    bad = module.score_continuation(native, "% verse\nV: Vocal\nG4A4B4c6|\nV: Ins\nZ|\n",
                                    module.load_validator())
    assert not bad["valid"]
