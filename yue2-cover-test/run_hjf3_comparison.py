"""Prepare and render source-register conservative and aggressive F0 covers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'yue2-cover-test/generated3/hjf3'


def prepare(output):
    from audit_rmvpe import make_midi, notation_module, render_preview, write_json
    import soundfile as sf

    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import parse_abc

    source = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3'
    audit = ROOT / 'yue2-cover-test/work/f0_audit/f8e2_hjf3_round1'
    original = np.loadtxt(source / 'melody_vocal.lab')
    conservative = np.loadtxt(audit / 'octave_aligned/melody_vocal.corrected.lab')
    conservative[:, 2] -= 12
    aggressive = original.copy()
    curves = np.load(audit / 'f0.npz')
    pyin = np.load(audit / 'pyin.npz')
    lyrics = json.loads((ROOT / 'yue2-cover-test/lyrics/hjf3.json').read_text())
    words = [word for sentence in lyrics['subtitle_lyrics'] for word in sentence['words']]
    rows = []
    for index, (start, end, pitch) in enumerate(original):
        edge = min(0.08, 0.15 * (end - start))
        selected = ((curves['times'] >= start + edge) & (curves['times'] < end - edge)
                    & (curves['vocal_f0'] > 0) & (curves['vocal_activation'] >= 0.10))
        values = 69 + 12 * np.log2(curves['vocal_f0'][selected] / 440)
        candidate = int(np.floor(np.median(values) + 0.5)) if len(values) >= 3 else int(conservative[index, 2])
        independent = ((pyin['times'] >= start + edge) & (pyin['times'] < end - edge)
                       & (pyin['f0'] > 0) & (pyin['activation'] >= 0.10))
        independent_values = 69 + 12 * np.log2(pyin['f0'][independent] / 440)
        nearest = min(words, key=lambda word: abs(word['start'] - start))
        aggressive[index, 2] = candidate
        rows.append(dict(note=index + 1, start=float(start), end=float(end),
                         nearest_word=nearest['word'], nearest_word_start=nearest['start'],
                         original=int(pitch), original_lower_octave=int(pitch) - 12,
                         conservative=int(conservative[index, 2]), aggressive=candidate,
                         rmvpe_frames=len(values), rmvpe_median=float(np.median(values)) if len(values) else None,
                         pyin_median=float(np.median(independent_values)) if len(independent_values) else None,
                         policy='rounded_rmvpe_median' if len(values) >= 3 else 'conservative_fallback'))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'note_changes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    native = notation_module()
    original_score = parse_abc((source / 'score.abc').read_text())
    read_rows = lambda name: [line.split() for line in (source / 'notation' / name).read_text().splitlines() if line.strip()]
    request = json.loads((ROOT / 'yue2-cover-test/work/requests/f8e2_hjf3.json').read_text())
    reports = {}
    for label, notes in [('conservative', conservative), ('f0_strong', aggressive)]:
        folder = output / label
        folder.mkdir()
        midi = make_midi(notes)
        text, rebuilt = native.generate_abc_from_data(midi, read_rows('song_beats.txt'), [],
                read_rows('song_keys.txt'), read_rows('song_structures.txt'), melody_only=True)
        parsed = parse_abc(text)
        if parsed.bpm != original_score.bpm:
            raise ValueError('BPM changed')
        for voice in ['Vocal', 'Ins']:
            before, after = original_score.voices[voice], parsed.voices[voice]
            if before.bars != after.bars or [(note[0], note[2]) for note in before.notes] != [(note[0], note[2]) for note in after.notes]:
                raise ValueError('Score timing or note count changed')
        if [note[1] for note in parsed.voices['Vocal'].notes] != notes[:, 2].astype(int).tolist():
            raise ValueError('Pitch round trip failed')
        (folder / 'score.abc').write_text(text)
        midi.write(str(folder / 'melody.mid'))
        np.savetxt(folder / 'melody.lab', notes, fmt=['%.6f', '%.6f', '%d'], delimiter='\t')
        render_preview(notes, folder / 'melody_preview.wav', 59.194, sf)
        write_json(folder / 'request.json', dict(request, id='hjf3_' + label))
        reports[label] = dict(changes_after_octave_correction=int(np.sum(notes[:, 2] != original[:, 2] - 12)),
                              note_count=len(notes), diagnostics=rebuilt.diagnostics)
    report = dict(versions=reports, both_lowered_one_octave=True, rhythm_changed=False,
                  strong_rule='Round median vocal RMVPE MIDI pitch in each original note core; >=3 active frames; otherwise conservative fallback. No scale snapping or strict stability gate.',
                  seed=request['seed'], lyrics_source='Original corrected generation request; timestamp words provide audit labels only.',
                  pair_different_notes=int(np.sum(conservative[:, 2] != aggressive[:, 2])),
                  limitation='Strong version may quantize glides/off-key singing incorrectly; neither version changes beat grid or note segmentation.')
    write_json(output / 'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


D_MAJOR = frozenset([2, 4, 6, 7, 9, 11, 1])
RESTORED_LABELS = ['f0_strong_original_register', 'f0_strong_d_major']


def snap_d_major(pitch, reference):
    candidates = [value for value in range(128) if value % 12 in D_MAJOR]
    return min(candidates, key=lambda value: (abs(value - pitch), abs(value - reference), value))


def prepare_restored(output):
    from audit_rmvpe import make_midi, notation_module, render_preview, write_json
    import soundfile as sf

    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import parse_abc

    for label in RESTORED_LABELS:
        if (output / label).exists() or (output / (label + '.wav')).exists():
            raise FileExistsError(label)
    source = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3'
    previous = output / 'f0_strong'
    low = np.loadtxt(previous / 'melody.lab')
    restored = low.copy()
    restored[:, 2] += 12
    if np.any(restored[:, 2] > 127):
        raise ValueError('Restored pitches exceed MIDI range')
    changes = list(csv.DictReader((output / 'note_changes.csv').open()))
    if len(changes) != len(restored):
        raise ValueError('F0 audit and melody have different note counts')
    snapped = restored.copy()
    audit_rows = []
    for index, (start, end, pitch) in enumerate(restored):
        evidence = changes[index]
        if int(evidence['aggressive']) != int(low[index, 2]) or abs(float(evidence['start']) - start) > 1e-5:
            raise ValueError('F0 evidence does not match input melody')
        reference = float(evidence['rmvpe_median']) + 12 if evidence['rmvpe_median'] else float(pitch)
        target = snap_d_major(int(pitch), reference)
        snapped[index, 2] = target
        audit_rows.append(dict(note=index + 1, start=float(start), end=float(end),
                               word=evidence['nearest_word'], restored=int(pitch),
                               snapped=target, f0_reference=reference))
    with (output / 'd_major_changes.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    native = notation_module()
    baseline = parse_abc((previous / 'score.abc').read_text())
    read_rows = lambda name: [line.split() for line in (source / 'notation' / name).read_text().splitlines() if line.strip()]
    request = json.loads((previous / 'request.json').read_text())
    reports = {}
    for label, notes in zip(RESTORED_LABELS, [restored, snapped]):
        folder = output / label
        folder.mkdir()
        midi = make_midi(notes)
        text, rebuilt = native.generate_abc_from_data(midi, read_rows('song_beats.txt'), [],
                read_rows('song_keys.txt'), read_rows('song_structures.txt'), melody_only=True)
        parsed = parse_abc(text)
        if baseline.bpm != parsed.bpm:
            raise ValueError('Tempo changed')
        for voice in ['Vocal', 'Ins']:
            before, after = baseline.voices[voice], parsed.voices[voice]
            if before.bars != after.bars or [(note[0], note[2]) for note in before.notes] != [(note[0], note[2]) for note in after.notes]:
                raise ValueError('Timing, meter or note count changed')
        if [note[1] for note in parsed.voices['Vocal'].notes] != notes[:, 2].astype(int).tolist():
            raise ValueError('ABC pitch round trip failed')
        (folder / 'score.abc').write_text(text)
        midi.write(str(folder / 'melody.mid'))
        np.savetxt(folder / 'melody.lab', notes, fmt=['%.6f', '%.6f', '%d'], delimiter='\t')
        render_preview(notes, folder / 'melody_preview.wav', 59.194, sf)
        write_json(folder / 'request.json', dict(request, id='hjf3_' + label))
        reports[label] = dict(note_count=len(notes), outside_d_major=sum(int(pitch) % 12 not in D_MAJOR for pitch in notes[:, 2]),
                              diagnostics=rebuilt.diagnostics)
    write_json(output / 'restored_preparation.json', dict(versions=reports,
               source='f0_strong/melody.lab', restored_semitones=12, seed=request['seed'],
               snapped_notes=int(np.sum(restored[:, 2] != snapped[:, 2])),
               snapping='Nearest D-major MIDI note; equal distances resolved by original floating F0, then lower pitch.',
               rhythm_changed=False, caution='D major applies to input melody, not a hard constraint on generated accompaniment or vocals.'))
    print(json.dumps(reports, indent=2))


def generate(output, restored=False):
    import soundfile as sf

    environment = dict(os.environ, HF_HOME='/data/cmq/hf-cache', PYTHONDONTWRITEBYTECODE='1')
    results = {}
    labels = RESTORED_LABELS if restored else ['conservative', 'f0_strong']
    report_name = 'restored_generation_report.json' if restored else 'generation_report.json'
    for label in labels:
        if (output / label / 'generation').exists() or (output / (label + '.wav')).exists():
            raise FileExistsError(label)
    for label in labels:
        folder = output / label
        destination = folder / 'generation'
        if destination.exists():
            raise FileExistsError(destination)
        command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'skills/yue2-music/scripts/run_yue2.py'),
                   'generate', '--request', str(folder / 'request.json'), '--abc-file', str(folder / 'score.abc'),
                   '--cot', 'melody', '--output', str(destination), '--device', 'cuda', '--offline']
        with (folder / 'generation.log').open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            results[label] = dict(status='failed', returncode=completed.returncode)
        else:
            result = json.loads((destination / 'run.json').read_text())['results'][0]
            audio, rate = sf.read(destination / 'audio.flac', dtype='float32', always_2d=True)
            if not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
                raise ValueError('Invalid generated audio')
            sf.write(output / (label + '.wav'), audio, rate, subtype='PCM_24')
            results[label] = dict(result, sample_rate=rate, channels=audio.shape[1],
                                  peak=float(np.max(np.abs(audio))), duration=len(audio) / rate)
        (output / report_name).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(label, results[label], flush=True)
    if any(result['status'] != 'complete' for result in results.values()):
        raise RuntimeError('Generation incomplete; see generation_report.json and per-version logs')
    if restored:
        (output / 'listen_restored.html').write_text('''<!doctype html><meta charset="utf-8"><title>F0 strong 原音区对照</title>
<h1>F0 strong 原音区 / 吸附 D 大调</h1><p>两版均恢复高一个八度；相同歌词、风格、seed，节拍不变。</p>
<h2>原音区：仅恢复八度</h2><audio controls src="f0_strong_original_register.wav"></audio>
<h2>原音区：吸附 D 大调</h2><audio controls src="f0_strong_d_major.wav"></audio>
<p>D 大调版本修改了 44 个音符，每个升或降一个半音；约束只施加于输入谱面。</p>''')
        return
    (output / 'listen.html').write_text('''<!doctype html><meta charset="utf-8"><title>HJF3 cover 对照</title>
<h1>保守版 / F0 强修版</h1><p>两版均已降回原唱音区。相同歌词、风格、seed；未修改节拍。</p>
<h2>保守版</h2><audio controls src="conservative.wav"></audio>
<h2>F0 强修版</h2><audio controls src="f0_strong.wav"></audio>
<p>强修版按 F0 中位数量化，可能保留原唱偏音或误判滑音，不保证更好听。</p>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate', 'prepare-restored', 'generate-restored'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.stage == 'prepare-restored':
        prepare_restored(args.output.resolve())
    elif args.stage == 'generate-restored':
        generate(args.output.resolve(), restored=True)
    else:
        (prepare if args.stage == 'prepare' else generate)(args.output.resolve())


if __name__ == '__main__':
    main()
