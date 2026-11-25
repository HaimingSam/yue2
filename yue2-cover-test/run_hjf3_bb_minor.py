"""Snap the original-register F0 strong score to Bb natural minor and render."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'yue2-cover-test/generated3/hjf3'
LABEL = 'f0_strong_bb_minor'
SCALE = frozenset([10, 0, 1, 3, 5, 6, 8])


def snap(pitch, reference):
    return min((candidate for candidate in range(128) if candidate % 12 in SCALE),
               key=lambda candidate: (abs(candidate - pitch), abs(candidate - reference), candidate))


def prepare():
    import soundfile as sf
    from audit_rmvpe import make_midi, notation_module, render_preview, write_json, digest

    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import parse_abc

    folder = OUTPUT / LABEL
    if folder.exists() or (OUTPUT / (LABEL + '.wav')).exists():
        raise FileExistsError(folder)
    previous = OUTPUT / 'f0_strong_original_register'
    source = ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3/notation'
    notes = np.loadtxt(previous / 'melody.lab')
    evidence = list(csv.DictReader((OUTPUT / 'note_changes.csv').open()))
    if len(notes) != len(evidence):
        raise ValueError('F0 evidence count mismatch')
    corrected = notes.copy()
    changes = []
    for index, (start, end, pitch) in enumerate(notes):
        row = evidence[index]
        if int(row['aggressive']) + 12 != pitch or abs(float(row['start']) - start) > 1e-5:
            raise ValueError('F0 evidence is not aligned')
        reference = float(row['rmvpe_median']) + 12 if row['rmvpe_median'] else float(pitch)
        target = snap(int(pitch), reference)
        corrected[index, 2] = target
        changes.append(dict(note=index + 1, start=float(start), end=float(end), word=row['nearest_word'],
                            original=int(pitch), snapped=target, shift=target - int(pitch), f0_reference=reference))
    native = notation_module()
    read_rows = lambda name: [line.split() for line in (source / name).read_text().splitlines() if line.strip()]
    keys = [[start, end, 'Bb:minor'] for start, end, unused_key in read_rows('song_keys.txt')]
    midi = make_midi(corrected)
    text, rebuilt = native.generate_abc_from_data(midi, read_rows('song_beats.txt'), [], keys,
                                                 read_rows('song_structures.txt'), melody_only=True)
    before, after = parse_abc((previous / 'score.abc').read_text()), parse_abc(text)
    assert before.bpm == after.bpm
    for voice in ['Vocal', 'Ins']:
        assert before.voices[voice].bars == after.voices[voice].bars
        assert [(note[0], note[2]) for note in before.voices[voice].notes] == [(note[0], note[2]) for note in after.voices[voice].notes]
    assert [note[1] for note in after.voices['Vocal'].notes] == corrected[:, 2].astype(int).tolist()
    assert all(note[1] % 12 in SCALE for note in after.voices['Vocal'].notes)
    assert '\nK:Bbm\n' in text
    folder.mkdir()
    (folder / 'score.abc').write_text(text)
    midi.write(str(folder / 'melody.mid'))
    np.savetxt(folder / 'melody.lab', corrected, fmt=['%.6f', '%.6f', '%d'], delimiter='\t')
    render_preview(corrected, folder / 'melody_preview.wav', 59.194, sf)
    request = json.loads((previous / 'request.json').read_text())
    write_json(folder / 'request.json', dict(request, id='hjf3_' + LABEL))
    with (folder / 'note_changes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    outside = [note for note in before.voices['Vocal'].notes if note[1] % 12 not in SCALE]
    names = {2: 'D', 4: 'E', 7: 'G', 9: 'A', 11: 'B'}
    report = dict(scale='Bb natural minor', pitch_classes=sorted(SCALE), key_header='K:Bbm',
                  input_notes=len(notes), outside_before=len(outside), outside_after=0,
                  outside_before_by_pitch=dict(Counter(names[note[1] % 12] for note in outside)),
                  outside_duration_fraction=sum(float(note[2]) for note in outside) / sum(float(note[2]) for note in before.voices['Vocal'].notes),
                  changed_notes=sum(row['shift'] != 0 for row in changes),
                  shift_counts=dict(Counter(row['shift'] for row in changes)),
                  seed=request['seed'], octave_change=0, rhythm_changed=False,
                  source_score_sha256=digest(previous / 'score.abc'), diagnostics=rebuilt.diagnostics,
                  rule='Nearest scale note; ties use floating RMVPE F0 then lower pitch.',
                  caveat='Counts concern the conditioning score, not a retranscription of generated audio. Natural minor chosen, not harmonic/melodic minor. Key follows user hypothesis, not an independent key detection.')
    write_json(folder / 'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def generate():
    import soundfile as sf
    from yue2.storage import verify_result

    folder = OUTPUT / LABEL
    destination = folder / 'generation'
    wav_path = OUTPUT / (LABEL + '.wav')
    if destination.exists() or wav_path.exists():
        raise FileExistsError(destination)
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'skills/yue2-music/scripts/run_yue2.py'),
               'generate', '--request', str(folder / 'request.json'), '--abc-file', str(folder / 'score.abc'),
               '--cot', 'melody', '--output', str(destination), '--device', 'cuda', '--offline']
    environment = dict(os.environ, HF_HOME='/data/cmq/hf-cache', PYTHONDONTWRITEBYTECODE='1')
    with (folder / 'generation.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_result(destination)
    assert (folder / 'score.abc').read_bytes() == (destination / 'score.abc').read_bytes()
    result = json.loads((destination / 'run.json').read_text())['results'][0]
    if result['status'] != 'complete':
        raise RuntimeError(result)
    audio, rate = sf.read(destination / 'audio.flac', dtype='float32', always_2d=True)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise ValueError('Invalid generated audio')
    sf.write(wav_path, audio, rate, subtype='PCM_24')
    result.update(sample_rate=rate, channels=audio.shape[1], peak=float(np.max(np.abs(audio))))
    (folder / 'generation_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUTPUT / 'listen_bb_minor.html').write_text('''<!doctype html><meta charset="utf-8"><title>Bb 小调对照</title>
<h1>F0 strong 原音区 / B♭自然小调吸附</h1>
<h2>未吸附的原音区版</h2><audio controls src="f0_strong_original_register.wav"></audio>
<h2>B♭自然小调版</h2><audio controls src="f0_strong_bb_minor.wav"></audio>
<p>音区、节拍及生成参数不变。输入谱调号改为 Bbm，38 个调外音吸附后为零；生成音频未重新转录统计。</p>''')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate'])
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else generate)()
