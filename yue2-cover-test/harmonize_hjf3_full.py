"""Constrained automatic chord assignment for the existing Bb-minor melody."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from audit_rmvpe import digest, make_midi, notation_module, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'yue2-cover-test/generated3/hjf3/f0_strong_bb_minor'
OUT = ROOT / 'yue2-cover-test/generated3/hjf3/full_auto_chords_bb_minor'
CHORDS = [
    ('Bbm', 'Bb:min', 10, (10, 1, 5)),
    ('Bbm7', 'Bb:min7', 10, (10, 1, 5, 8)),
    ('Db', 'Db:maj', 1, (1, 5, 8)),
    ('Dbmaj7', 'Db:maj7', 1, (1, 5, 8, 0)),
    ('Ebm', 'Eb:min', 3, (3, 6, 10)),
    ('Ebm7', 'Eb:min7', 3, (3, 6, 10, 1)),
    ('Fm', 'F:min', 5, (5, 8, 0)),
    ('Fm7', 'F:min7', 5, (5, 8, 0, 3)),
    ('Gb', 'Gb:maj', 6, (6, 10, 1)),
    ('Gbmaj7', 'Gb:maj7', 6, (6, 10, 1, 5)),
    ('Ab', 'Ab:maj', 8, (8, 0, 3)),
    ('Ab7', 'Ab:7', 8, (8, 0, 3, 6)),
]


def compatibility(notes, onset, length, chord):
    tones = chord[3]
    score = 0.0
    for start, pitch, duration in notes:
        overlap = max(0.0, min(float(start + duration), onset + length) - max(float(start), onset))
        if not overlap:
            continue
        weight = overlap * (1.3 if float(start) <= onset < float(start + duration) else 1.0)
        distance = min(min((pitch - tone) % 12, (tone - pitch) % 12) for tone in tones)
        score += weight * (1.0 if pitch % 12 in tones else (-0.65 if distance == 1 else -0.2))
    return score - (0.35 if len(tones) == 4 else 0)


def transition(previous, current):
    if previous == current:
        return -0.15
    if CHORDS[previous][2] == CHORDS[current][2]:
        return -0.25
    common = len(set(CHORDS[previous][3]) & set(CHORDS[current][3]))
    root_distance = (CHORDS[current][2] - CHORDS[previous][2]) % 12
    return 0.08 * common - 0.25 + (0.2 if root_distance in (5, 7) else 0)


def main():
    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import compare, json_value, parse_abc, report

    if OUT.exists():
        raise FileExistsError(OUT)
    before = parse_abc((BASE / 'score.abc').read_text())
    bars = before.voices['Vocal'].bars
    notes = before.voices['Vocal'].notes
    emissions = np.array([[compatibility(notes, float(start), float(length), chord) for chord in CHORDS]
                          for start, length, meter in bars])
    emissions[0, :2] += 0.6
    scores = emissions.copy()
    back = np.zeros(scores.shape, dtype=int)
    for bar in range(1, len(bars)):
        for current in range(len(CHORDS)):
            choices = [scores[bar - 1, previous] + transition(previous, current) for previous in range(len(CHORDS))]
            previous = int(np.argmax(choices))
            back[bar, current] = previous
            scores[bar, current] += choices[previous]
    selected = [int(np.argmax(scores[-1]))]
    for bar in range(len(bars) - 1, 0, -1):
        selected.append(int(back[bar, selected[-1]]))
    selected.reverse()
    source = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3/notation'
    rows = lambda name: [line.split() for line in (source / name).read_text().splitlines() if line.strip()]
    keys = [[start, end, 'Bb:minor'] for start, end, unused in rows('song_keys.txt')]
    native = notation_module()
    melody = make_midi(np.loadtxt(BASE / 'melody.lab'))
    baseline_text, grid = native.generate_abc_from_data(melody, rows('song_beats.txt'), [], keys, rows('song_structures.txt'), melody_only=True)
    assert baseline_text == (BASE / 'score.abc').read_text()
    intervals, audit = [], []
    for index, (measure, chord_index) in enumerate(zip(grid.measures, selected)):
        start = float(grid.subbeat_times[measure.start_t])
        end = float(grid.subbeat_times[measure.end_t])
        chord = CHORDS[chord_index]
        intervals.append([start, end, chord[1]])
        audit.append(dict(bar=index + 1, start_quarters=float(bars[index][0]), length_quarters=float(bars[index][1]),
                          source_start=start, source_end=end, chord=chord[0],
                          compatibility=float(emissions[index, chord_index]),
                          candidates={item[0]:float(emissions[index, rank]) for rank,item in enumerate(CHORDS)}))
    text, rebuilt = native.generate_abc_from_data(melody, rows('song_beats.txt'), intervals, keys, rows('song_structures.txt'), melody_only=False)
    after = parse_abc(text)
    comparison = compare(before, after)
    assert comparison['match'], comparison
    assert len(after.voices['Vocal'].notes) == 110
    assert after.voices['Vocal'].chords and not after.voices['Ins'].notes
    OUT.mkdir()
    (OUT / 'score.abc').write_text(text)
    (OUT / 'baseline.abc').write_text(baseline_text)
    (OUT / 'chords.lab').write_text(''.join(f'{start:.6f}\t{end:.6f}\t{chord}\n' for start,end,chord in intervals))
    request = json.loads((ROOT / 'yue2-cover-test/work/requests/f8e2_hjf3.json').read_text())
    request['cot'] = 'full'
    write_json(OUT / 'request.json', request)
    write_json(OUT / 'chord_plan.json', audit)
    write_json(OUT / 'validation.json', json.loads(json.dumps(dict(comparison=comparison, score=report(after)), default=json_value)))
    write_json(OUT / 'edit_manifest.json', dict(method='Deterministic duration/accent-weighted chord compatibility and dynamic programming transitions, following local skill. Not MIDI-SAG inference or original chord recovery.',
        source_sha256=digest(BASE / 'score.abc'), result_sha256=digest(OUT / 'score.abc'),
        melody_notes=110, changed_pitches=0, changed_onsets=0, changed_durations=0,
        tempo=95, key='Bb natural minor', cot='full', progression=[CHORDS[index][0] for index in selected],
        assumptions='Diatonic triads and seventh chords; one chord candidate per bar, including original 3/4 bar. No forced tonic ending because final melody Gb conflicts with Bb-minor triad.',
        uncertainty='Heuristic scores are not calibrated musical quality. Input-score preservation does not guarantee generated vocal adherence. No listening claim.',
        diagnostics=rebuilt.diagnostics))
    print(json.dumps(dict(progression=[CHORDS[index][0] for index in selected], comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
