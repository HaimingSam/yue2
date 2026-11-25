from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path('/data/cmq/YuE')
OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'f0_strong_bb_minor'
ORIGINAL = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3/notation'
PAPER = ROOT / 'yue2-cover-test/work/transcriptions_tuning/f8e2_hjf3_full_paper/notation'
REQUEST = ROOT / 'yue2-cover-test/work/requests/f8e2_hjf3.json'
LYRICS = ROOT / 'yue2-cover-test/lyrics/hjf3.json'
sys.path.insert(0, str(ROOT / 'yue2-cover-test'))
sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
from audit_rmvpe import make_midi, notation_module
from abc_tools import parse_abc, report, json_value, compare


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value) + '\n')


def main():
    native = notation_module()
    anchors = np.loadtxt(BASE / 'melody.lab')
    original = rows(ORIGINAL / 'song_beats.txt')
    paper = rows(PAPER / 'song_beats.txt')
    boundary = 19.35
    selected = [row for row in paper if float(row[0]) < boundary]
    selected += [row for row in original if float(row[0]) >= boundary]
    keys = [[start, end, 'Bb:minor'] for start, end, key in rows(ORIGINAL / 'song_keys.txt')]
    structures = rows(ORIGINAL / 'song_structures.txt')
    midi = make_midi(anchors)
    baseline_text, baseline_native = native.generate_abc_from_data(
        midi, original, [], keys, structures, melody_only=True)
    assert baseline_text == (BASE / 'score.abc').read_text()
    rebuilt = native.build_rebuilt_abc_score_from_data(
        midi, selected, [], keys, structures, melody_only=True)
    estimated_tempo = native.estimate_tempo(rebuilt)
    with patch.object(native, 'estimate_tempo', return_value=95.0):
        text = native.score_to_abc(rebuilt)
        native.validate_serialized_abc(text, rebuilt)
    before, after = parse_abc(baseline_text), parse_abc(text)
    old_notes = before.voices['Vocal'].notes
    new_notes = after.voices['Vocal'].notes
    assert len(old_notes) == len(new_notes) == len(anchors) == 110
    assert [note[1] for note in old_notes] == [note[1] for note in new_notes] == anchors[:, 2].astype(int).tolist()
    assert all(note[1] % 12 in {10, 0, 1, 3, 5, 6, 8} for note in new_notes)
    assert before.bpm == after.bpm == 95
    assert before.unit == after.unit
    assert not after.voices['Ins'].notes
    assert all(not voice.chords for voice in after.voices.values())
    assert '\nK:Bbm\n' in text
    assert all(bar[2] == (4, 4) for bar in after.voices['Vocal'].bars)
    assert [line for line in text.splitlines() if line.startswith('%')] == [line for line in baseline_text.splitlines() if line.startswith('%')]
    events = []
    for index, (anchor, old, new) in enumerate(zip(anchors, old_notes, new_notes), 1):
        start, end, pitch = anchor
        if start >= boundary:
            assert new[0] - old[0] == 1 and new[2] == old[2]
        events.append(dict(note_index_1based=index, midi_pitch=int(pitch),
                           source_start_seconds=float(start), source_end_seconds=float(end),
                           before_onset_quarters=old[0], after_onset_quarters=new[0],
                           before_duration_quarters=old[2], after_duration_quarters=new[2],
                           onset_delta_quarters=new[0] - old[0], duration_delta_quarters=new[2] - old[2],
                           before_nominal_start_seconds=float(old[0] * 60 / 95),
                           after_nominal_start_seconds=float(new[0] * 60 / 95)))
    bars = []
    for index, (old, new) in enumerate(zip(before.voices['Vocal'].bars, after.voices['Vocal'].bars), 1):
        old_measure = baseline_native.measures[index - 1]
        new_measure = rebuilt.measures[index - 1]
        bars.append(dict(bar_1based=index, before=dict(onset_quarters=old[0], length_quarters=old[1], meter=old[2]),
                         after=dict(onset_quarters=new[0], length_quarters=new[1], meter=new[2]),
                         before_source_span_seconds=[float(baseline_native.subbeat_times[old_measure.start_t]), float(baseline_native.subbeat_times[old_measure.end_t])],
                         after_source_span_seconds=[float(rebuilt.subbeat_times[new_measure.start_t]), float(rebuilt.subbeat_times[new_measure.end_t])]))
    assert len(before.voices['Vocal'].bars) == len(after.voices['Vocal'].bars) == 24
    sources = [BASE / 'score.abc', BASE / 'melody.lab', BASE / 'generation/audio.flac', REQUEST, LYRICS,
               ORIGINAL / 'song_beats.txt', PAPER / 'song_beats.txt', ORIGINAL / 'song_keys.txt',
               ORIGINAL / 'song_structures.txt', ROOT / 'models/SheetSage2/notation_sheetsage2.py',
               ROOT / 'models/SheetSage2/io_sheetsage2.py', ROOT / 'yue2-cover-test/audit_rmvpe.py',
               ROOT / 'skills/yue2-music/scripts/abc_tools.py']
    (OUT / 'score.abc').write_text(text)
    (OUT / 'baseline.abc').write_bytes((BASE / 'score.abc').read_bytes())
    (OUT / 'request.json').write_bytes(REQUEST.read_bytes())
    (OUT / 'melody.lab').write_bytes((BASE / 'melody.lab').read_bytes())
    (OUT / 'selected_beats.tsv').write_text('\n'.join('\t'.join(row) for row in selected) + '\n')
    assert (OUT / 'request.json').read_bytes() == REQUEST.read_bytes()
    save('validation.json', dict(native_serialized_validation='passed', baseline_native_reproduction='byte-exact',
                               ordered_110_pitches='exact', source_second_anchors='byte-exact LAB',
                               request='byte-exact', structure=report(after), strict_comparison=compare(before, after)))
    lyric_data = json.loads(LYRICS.read_text())
    save('edit_manifest.json', dict(
        decision='Exploratory structural consistency edit; no confirmed objective quality improvement.',
        contract=dict(pitches='ordered 110 exactly, original high register', key='Bb natural minor',
                      tempo=95, cot='melody', instrumental_notes=0, request='byte-identical',
                      allowed_edit='early beat grid and resulting quantized rhythms/meters only'),
        method='Native rebuild from unchanged LAB seconds, paper beats before 19.35s, original beats thereafter; native tempo estimator temporarily overridden to prescribed 95 BPM during serialization/validation; no ABC token patching.',
        sources_sha256={str(path): digest(path) for path in sources},
        outputs_sha256={name: digest(OUT / name) for name in ['score.abc', 'baseline.abc', 'request.json', 'melody.lab', 'selected_beats.tsv', 'prepare.py']},
        boundary_source_seconds=boundary, native_estimated_tempo_before_override=estimated_tempo,
        timing_caveat='Source anchors remain unchanged input evidence, not exact output timing. Quantization and fixed-tempo rendering change score seconds. Events at/after 19.35s retain durations and local beat relationships but shift +1 quarter globally (60/95 seconds).',
        parent_agent_grid_check=dict(scope='starts before 19.35s', word_start_mean_error_ms=dict(original=35.18, paper=36.35),
                                             lab_mean_error_ms=dict(original='approximately 0; already quantized', paper=5.88), paper_lab_max_error_ms=60),
        lyric_timestamp_top_level_fields=list(lyric_data),
        lyric_phrase_context=[dict(start=row['start'], end=row['end'], text=row['text']) for row in lyric_data['subtitle_lyrics']],
        lyrics_policy='Context and breath-boundary evidence only; no forced one-word-per-note mapping and no lyric edits.',
        source_beat_rows=original, selected_beat_rows=selected, diagnostics=rebuilt.diagnostics,
        notes=events, bars=bars,
        changed_onset_count=sum(old[0] != new[0] for old, new in zip(old_notes, new_notes)),
        changed_duration_count=sum(old[2] != new[2] for old, new in zip(old_notes, new_notes)),
        audio_generated=False, audio_listened=False))
    print(json.dumps(dict(notes=len(new_notes), bars=len(bars), tempo=after.bpm,
                          duration_quarters=str(after.voices['Vocal'].time),
                          changed_durations=sum(old[2] != new[2] for old, new in zip(old_notes, new_notes)))))


if __name__ == '__main__':
    main()
