"""Restore the baseline beat grid and make evidence-gated phrase-end breaths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from audit_rmvpe import digest, make_midi, notation_module, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'yue2-cover-test/generated3/hjf3/f0_strong_bb_minor'
OUTPUT = ROOT / 'yue2-cover-test/generated3/hjf3/lyrics_phrasing_bb_minor'
LYRICS = ROOT / 'yue2-cover-test/lyrics/hjf3.json'
REQUEST = ROOT / 'yue2-cover-test/work/requests/f8e2_hjf3.json'
NOTATION = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3/notation'
AUDIT = ROOT / 'yue2-cover-test/work/f0_audit/f8e2_hjf3_round1'


def voiced_fraction(times, f0, confidence, start, end):
    selected = (times >= start) & (times < end)
    if not selected.any():
        return 1.0
    return float(np.mean((f0[selected] > 0) & (confidence[selected] >= 0.10)))


def allow_breath(start, cut, end, word_end, word_probability, fractions):
    return (start + 0.15 <= cut < end and cut >= word_end and word_probability >= 0.6
            and len(fractions) == 2 and all(value <= 0.2 for value in fractions))


def main(resume_incomplete=False):
    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import json_value, parse_abc, report

    if (OUTPUT / 'edit_manifest.json').exists() or (OUTPUT / 'generation').exists():
        raise FileExistsError('Refusing to overwrite a completed or generated edit')
    if (OUTPUT / 'score.abc').exists() and not resume_incomplete:
        raise FileExistsError('Refusing to overwrite existing phrasing score')
    plan = json.loads((OUTPUT / 'lyrics_plan.json').read_text())
    request = json.loads(REQUEST.read_text())
    chars = lambda text: ''.join(re.findall(r'[\u4e00-\u9fff]', text))
    if chars(plan['proposed_lyrics']) != chars(request['lyrics']):
        raise ValueError('Lyric characters or order changed')
    if len(plan['proposed_lyrics'].splitlines()) != 12:
        raise ValueError('Expected twelve phrase lines')
    phrases = json.loads(LYRICS.read_text())['subtitle_lyrics']
    original = np.loadtxt(BASE / 'melody.lab')
    notes = original.copy()
    rmvpe = np.load(AUDIT / 'f0.npz')
    pyin = np.load(AUDIT / 'pyin.npz')
    read_rows = lambda name: [line.split() for line in (NOTATION / name).read_text().splitlines() if line.strip()]
    native = notation_module()
    beats, structures = read_rows('song_beats.txt'), read_rows('song_structures.txt')
    keys = [[start, end, 'Bb:minor'] for start, end, unused in read_rows('song_keys.txt')]
    baseline_text, baseline_native = native.generate_abc_from_data(make_midi(original), beats, [], keys, structures, melody_only=True)
    if baseline_text != (BASE / 'score.abc').read_text():
        raise ValueError('Baseline native reconstruction is not exact')
    phrase_reviews = []
    for index, phrase in enumerate(phrases):
        word = phrase['words'][-1]
        note_index = int(np.argmin(abs(original[:, 0] - word['start'])))
        start, end, pitch = original[note_index]
        boundary = int(np.argmin(abs(baseline_native.subbeat_times - end)))
        if boundary == 0:
            raise ValueError('Missing note-end grid boundary')
        cut = float(baseline_native.subbeat_times[boundary - 1])
        fractions = [voiced_fraction(rmvpe['times'], rmvpe['vocal_f0'], rmvpe['vocal_activation'], cut, end),
                     voiced_fraction(pyin['times'], pyin['f0'], pyin['activation'], cut, end)]
        eligible = allow_breath(start, cut, end, word['end'], word['prob'], fractions)
        if eligible:
            notes[note_index, 1] = cut
        next_onset = float(original[note_index + 1, 0]) if note_index + 1 < len(original) else None
        phrase_reviews.append(dict(phrase=index + 1, text=phrase['text'], final_word=word['word'],
                                    word_end=word['end'], word_probability=word['prob'],
                                    note=note_index + 1, onset=float(start), pitch=int(pitch),
                                    old_end=float(end), proposed_end=cut, actual_end=float(notes[note_index, 1]),
                                    rmvpe_tail_voiced_fraction=fractions[0], pyin_tail_voiced_fraction=fractions[1],
                                    changed=bool(eligible), next_note_onset=next_onset,
                                    source_gap_before=None if next_onset is None else next_onset-float(end),
                                    source_gap_after=None if next_onset is None else next_onset-float(notes[note_index, 1])))
    text, rebuilt = native.generate_abc_from_data(make_midi(notes), beats, [], keys, structures, melody_only=True)
    before, after = parse_abc(baseline_text), parse_abc(text)
    changed = []
    assert before.bpm == after.bpm == 95
    assert len(before.voices['Vocal'].notes) == len(after.voices['Vocal'].notes) == 110
    assert not after.voices['Ins'].notes
    for voice in ['Vocal', 'Ins']:
        assert before.voices[voice].bars == after.voices[voice].bars
        assert before.voices[voice].time == after.voices[voice].time
        for index, (old, new) in enumerate(zip(before.voices[voice].notes, after.voices[voice].notes), 1):
            assert old[:2] == new[:2], 'Onset or pitch changed'
            if old[2] != new[2]:
                assert old[2] - new[2] == 1 / 4
                changed.append(dict(note=index, old_duration=str(old[2]), new_duration=str(new[2])))
    assert all(note[1] % 12 in {10, 0, 1, 3, 5, 6, 8} for note in after.voices['Vocal'].notes)
    assert {item['note'] for item in changed} == {item['note'] for item in phrase_reviews if item['changed']}
    if (OUTPUT / 'score.abc').exists() and (OUTPUT / 'score.abc').read_text() != text:
        raise ValueError('Incomplete output does not match recomputed score')
    (OUTPUT / 'score.abc').write_text(text)
    (OUTPUT / 'baseline.abc').write_text(baseline_text)
    make_midi(notes).write(str(OUTPUT / 'melody.mid'))
    np.savetxt(OUTPUT / 'melody.lab', notes, fmt=['%.6f', '%.6f', '%d'], delimiter='\t')
    request['lyrics'] = plan['proposed_lyrics']
    write_json(OUTPUT / 'request.json', request)
    write_json(OUTPUT / 'phrase_review.json', phrase_reviews)
    write_json(OUTPUT / 'abc_check.json', json.loads(json.dumps(report(after), default=json_value)))
    write_json(OUTPUT / 'edit_manifest.json', dict(
        baseline=str(BASE / 'score.abc'), request_source=str(REQUEST),
        source_hashes={str(path): digest(path) for path in [LYRICS, REQUEST, BASE/'score.abc', BASE/'melody.lab',
                       AUDIT/'f0.npz', AUDIT/'pyin.npz', OUTPUT/'lyrics_plan.json']},
        changes=changed, phrases=12, pitches=110, out_of_scale_notes=0,
        preservation='Compared with pre-agent Bb-minor baseline: all note onsets, pitches, bars, total beats, tempo, style, seed and Chinese lyric characters unchanged.',
        rollback='Do not use previous agent_optimized_bb_minor 4/4 regridding or its extended phrase ends.',
        rule='Only phrase-final notes: shorten at most one sixteenth-grid step if both F0 estimators have <=20% voiced frames in removed tail, cut >= lyric word end, prob >=0.6, remaining duration >=150ms.',
        caveats=['Word prob is an uncalibrated input score, not probability of timing correctness.',
                 'Newlines/punctuation are conditioning only, not hard lyric alignment.',
                 'F0 detection may miss weak singing; improvement needs listening.',
                 'Both lyric formatting and two note tails change, so not a single-variable experiment.'],
        diagnostics=rebuilt.diagnostics))
    print(json.dumps(dict(changes=changed, lyrics=request['lyrics']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-incomplete', action='store_true')
    args = parser.parse_args()
    main(args.resume_incomplete)
