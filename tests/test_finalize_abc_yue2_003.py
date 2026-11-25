import importlib.util
import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("finalize_abc_yue2_003",
                                              TOOLS / "finalize_abc_yue2_003.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_finalization_copies_and_audits_fallback(tmp_path):
    from abc_vocal_format import restore_ins

    vocal = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
             "% chorus\nV: Vocal\nC4D4E4F4|\n")
    inputs, output, fallback = (tmp_path / name for name in ("inputs", "output", "fallback"))
    (inputs / "first").mkdir(parents=True)
    (inputs / "first" / "vocals_dereverb.wav").touch()
    output.mkdir()
    fallback.mkdir()
    (fallback / "first.abc").write_text(restore_ins(vocal), encoding="utf-8")
    (output / "batch_manifest.jsonl").write_text(
        json.dumps({"status": "failed", "input": str(inputs / "first" / "vocals_dereverb.wav"),
                    "error": "empty Vocal"}) + "\n", encoding="utf-8")
    mapping = {"first": (fallback, "vocals.wav")}

    summary = module.finalize(inputs, output, mapping, copy_fallbacks=True)
    assert summary["abc_files"] == 1
    assert summary["missing"] == []
    assert summary["recovered_this_run"][0]["actual_audio_input"] == str(inputs / "first" / "vocals.wav")
    assert (output / "first.abc").is_file()
    assert module.finalize(inputs, output, mapping, copy_fallbacks=True)["recovered_this_run"] == []
    assert len((output / "batch_manifest.jsonl").read_text().splitlines()) == 2
