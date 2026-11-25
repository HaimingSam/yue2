import importlib.util
import sys
from pathlib import Path
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("generate_abc_two_stage", TOOLS / "generate_abc_two_stage.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_parse_structure_and_extract_section():
    plan = "<STRUCTURE>\n% intro\n% verse\n% chorus\n</STRUCTURE>\n"
    assert module.parse_structure(plan) == ["intro", "verse", "chorus"]
    score = "% verse\nV: Vocal\nC4D4|\n% chorus\nV: Vocal\nE4F4|\n"
    assert module.extract_generated_section(score, "chorus") == ["V: Vocal", "E4F4|"]
    assert module.parse_counted_structure("% intro groups=2\n% verse groups=5\n") == [("intro", 2), ("verse", 5)]


def test_invalid_generated_section_keeps_raw_debug_files(tmp_path, monkeypatch):
    import pytest

    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    output_path = tmp_path / "result.abc"
    responses = iter(("<STRUCTURE>\n% verse\n% chorus\n</STRUCTURE>\n",
                      "% verse\nV: Vocal\n" + "C4D4E4F4|" * 5 + "\n"))
    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    monkeypatch.setattr(module, "generate", lambda *args: next(responses))

    with pytest.raises(ValueError, match="Expected 1–4 measures, got 5.*assembled.vocal.abc"):
        module.run(input_path, output_path, "stage1", "stage2", "cpu", 100, 100)

    assert not output_path.exists()
    assert (tmp_path / "result.debug/02-verse.txt").is_file() is False
    assert (tmp_path / "result.debug/01-verse.txt").read_text().startswith("% verse")
    assert "% chorus" in (tmp_path / "result.debug/assembled.vocal.abc").read_text()


def test_chunked_inference_uses_position_and_keeps_input_chorus(tmp_path, monkeypatch):
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    output_path = tmp_path / "result.abc"
    prompts = []
    def fake_generate(model, tokenizer, prompt, device, max_tokens):
        prompts.append(prompt)
        if "<GROUP_INDEX>1</GROUP_INDEX>" in prompt:
            return "<MORE_GROUP>\nV: Vocal\nG4A4B4c4|\n"
        return "<END_SECTION>\nV: Vocal\nG4A4B4c4|\n"
    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    monkeypatch.setattr(module, "generate", fake_generate)
    report = module.run(input_path, output_path, None, "stage2", "cpu", 100, 100,
                        chunked=True, structure_override="<STRUCTURE>\n% verse\n% chorus\n</STRUCTURE>\n")
    assert report["generated_sections"] == [{"section": "verse", "source": "model", "groups": 2},
                                             {"section": "chorus", "source": "input"}]
    assert "<SECTION_INDEX>1</SECTION_INDEX>" in prompts[0]
    assert "<PREVIOUS_GROUP>\nV: Vocal\nG4A4B4c4|" in prompts[1]
    assert "C4D4E4F4|" in output_path.read_text()
    assert report["structure_source"] == "provided"


def test_counted_plan_stops_when_model_never_emits_end(tmp_path, monkeypatch):
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    def fake_generate(model, tokenizer, prompt, device, max_tokens):
        calls.append(prompt)
        return "<MORE_GROUP>\nV: Vocal\nG4A4B4c4|\n"
    monkeypatch.setattr(module, "generate", fake_generate)
    report = module.run(input_path, tmp_path / "out.abc", None, "stage2", "cpu", 100, 100,
                        chunked=True, structure_override="% verse groups=2\n% chorus groups=1\n")
    assert len(calls) == 2
    assert report["planned_groups"] == [2, 1]
    assert report["generated_sections"][0]["groups"] == 2


def test_chunked_generation_rejects_five_bar_group_immediately(tmp_path, monkeypatch):
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    monkeypatch.setattr(module, "generate", lambda *args: "<END_SECTION>\nV: Vocal\n" + "C4D4E4F4|" * 5)
    with pytest.raises(ValueError, match="group 1 failed.*1–4 measures"):
        module.run(input_path, tmp_path / "out.abc", None, "stage2", "cpu", 100, 100,
                   chunked=True, structure_override="% verse\n% chorus\n")
    assert (tmp_path / "out.debug/01-verse-01.txt").is_file()
    assert not (tmp_path / "out.abc").exists()


def test_sampling_selects_diverse_valid_group_and_copies_chorus(tmp_path, monkeypatch):
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    responses = iter(("<MORE_GROUP>\nV: Vocal\nC4D4E4F4G4|\n",
                      "<END_SECTION>\nV: Vocal\ne2e2e2e2e2e2e2e2|\n",
                      "<END_SECTION>\nV: Vocal\nG4A4B4c4|\n"))
    options = []

    def fake_generate(model, tokenizer, prompt, device, max_tokens, **kwargs):
        options.append(kwargs)
        return next(responses)

    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    monkeypatch.setattr(module, "generate", fake_generate)
    output_path = tmp_path / "result.abc"
    report = module.run(input_path, output_path, None, "stage2", "cpu", 100, 100,
                        chunked=True, structure_override="% verse groups=1\n% chorus groups=1\n",
                        sample_candidates=3, temperature=0.8, top_p=0.9)
    assert "G4A4B4c4|" in output_path.read_text()
    assert "C4D4E4F4|" in output_path.read_text()
    assert report["sample_candidates"] == 3
    assert options == [{"temperature": 0.8, "top_p": 0.9}] * 3
    assert "candidate-03" in (tmp_path / "result.debug/01-verse-01-selected.txt").read_text()


def test_sampling_fails_when_all_candidates_are_invalid(tmp_path, monkeypatch):
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\n")
    input_path = tmp_path / "input.abc"
    input_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(module, "load_model", lambda path, device: (None, None))
    monkeypatch.setattr(module, "generate", lambda *args, **kwargs: "<END_SECTION>\nV: Vocal\nC4D4E4F4G4|\n")
    with pytest.raises(ValueError, match="group 1 failed.*candidate-01"):
        module.run(input_path, tmp_path / "out.abc", None, "stage2", "cpu", 100, 100,
                   chunked=True, structure_override="% verse groups=1\n% chorus groups=1\n",
                   sample_candidates=2)
    assert (tmp_path / "out.debug/01-verse-01-candidate-02.txt").exists()
    assert not (tmp_path / "out.abc").exists()
