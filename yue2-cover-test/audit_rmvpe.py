"""Offline, pitch-only RMVPE audit; never overwrites a transcription."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RULES = dict(min_frames=10, min_coverage=0.75, min_support=0.80,
             max_mad=0.22, max_center_error=0.25, max_auto_shift=2,
             frame_activation=0.10, edge_fraction=0.15, max_edge_seconds=0.08)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def summarize(times, f0, activation, start, end):
    edge = min(RULES['max_edge_seconds'], RULES['edge_fraction'] * (end - start))
    selected = (times >= start + edge) & (times < end - edge)
    valid = selected & (f0 > 0) & (activation >= RULES['frame_activation'])
    pitches = 69 + 12 * np.log2(f0[valid] / 440)
    result = dict(frames=int(valid.sum()), coverage=float(valid.sum() / max(1, selected.sum())),
                  candidate=None, median=None, mad=None, support=0.0, stable=False)
    if not len(pitches):
        return result
    median = float(np.median(pitches))
    candidate = int(np.rint(median))
    mad = float(np.median(np.abs(pitches - median)))
    support = float(np.mean(np.abs(pitches - candidate) <= 0.45))
    result.update(candidate=candidate, median=median, mad=mad, support=support)
    result['stable'] = bool(len(pitches) >= RULES['min_frames']
                            and result['coverage'] >= RULES['min_coverage']
                            and support >= RULES['min_support']
                            and mad <= RULES['max_mad']
                            and abs(median - candidate) <= RULES['max_center_error'])
    return result


def decide(original, vocal, mix):
    candidate = vocal['candidate']
    if candidate is None:
        return original, 'review_unvoiced'
    if not vocal['stable'] or not mix['stable']:
        return original, 'review_unstable'
    if candidate != mix['candidate']:
        return original, 'review_disagreement'
    if candidate == original:
        return original, 'supported_original'
    if abs(candidate - original) > RULES['max_auto_shift']:
        return original, 'review_large_interval'
    return candidate, 'auto_corrected'


def extract(args):
    import librosa
    import soundfile as sf
    import torch

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Extraction needs a fresh or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.separator_repo))
    from utils import demix, get_model_from_config

    config_path = args.separator_repo / 'ckpts/config_melbandroformer_inst.yaml'
    checkpoint = args.separator_repo / 'ckpts/inst_v1e_plus.ckpt'
    model, config = get_model_from_config('mel_band_roformer', str(config_path))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    config.inference.batch_size = 1
    config.training.use_amp = args.device != 'cpu'
    model.eval().to(args.device)
    audio, rate = sf.read(args.audio, always_2d=True, dtype='float32')
    audio = librosa.resample(audio.T, orig_sr=rate, target_sr=44100)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if audio.shape[0] != 2:
        raise ValueError('Only mono/stereo source audio is supported')
    separated = demix(config, model, audio, torch.device(args.device), 'mel_band_roformer', pbar=True)
    vocal = audio - separated['other']
    if vocal.shape != audio.shape:
        raise ValueError('Separation changed audio alignment')
    sf.write(args.output / 'vocals.wav', vocal.T, 44100, subtype='FLOAT')
    del model, separated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.path.insert(0, str(args.rmvpe_repo))
    from rmvpe.inference import RMVPE

    estimator = RMVPE(str(args.checkpoint), device=args.device)
    state = torch.load(args.checkpoint, map_location='cpu')['model']
    estimator.model.load_state_dict(state, strict=True)
    del state
    curves = {}
    for name, waveform in [('mix', audio), ('vocal', vocal)]:
        samples = librosa.resample(waveform.mean(axis=0), orig_sr=44100, target_sr=16000)
        f0_parts, activation_parts = [], []
        for start in range(0, len(samples), 160000):
            end = min(start + 160000, len(samples))
            left, right = max(0, start - 32000), min(len(samples), end + 32000)
            tensor = torch.from_numpy(samples[left:right]).float().unsqueeze(0).to(args.device)
            with torch.inference_mode():
                hidden = estimator.mel2hidden(estimator.mel_extractor(tensor, center=True))
                f0, unused_cents = estimator.decode(hidden, thred=RULES['frame_activation'])
            first = (start - left) // 160
            count = (end - start + 159) // 160
            f0_parts.append(f0[0, first:first + count])
            activation_parts.append(hidden[0, first:first + count].max(dim=-1).values.cpu().numpy())
        curves[name + '_f0'] = np.concatenate(f0_parts)
        curves[name + '_activation'] = np.concatenate(activation_parts)
        print(name, 'frames:', len(curves[name + '_f0']), flush=True)
    curves['times'] = np.arange(len(curves['mix_f0'])) * 0.01
    np.savez(args.output / 'f0.npz', **curves)
    provenance = dict(audio=str(args.audio.resolve()), audio_sha256=digest(args.audio),
                      checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                      separator_checkpoint=str(checkpoint), separator_sha256=digest(checkpoint),
                      separator_config_sha256=digest(config_path), device=args.device,
                      torch_version=torch.__version__, rules=RULES,
                      caveat='Two inputs to the SAME estimator, not independent ground truth.',
                      rmvpe_sources={path.name: digest(path) for path in (args.rmvpe_repo / 'rmvpe').glob('*.py')})
    write_json(args.output / 'extraction.json', provenance)


def notation_module():
    package = types.ModuleType('audit_sheetsage')
    package.__path__ = [str(ROOT / 'models/SheetSage2')]
    sys.modules[package.__name__] = package
    return importlib.import_module('audit_sheetsage.notation_sheetsage2')


def verify(args):
    import librosa
    import soundfile as sf

    destination = args.output / 'pyin.npz'
    if destination.exists():
        raise FileExistsError(destination)
    waveform, rate = sf.read(args.output / 'vocals.wav', always_2d=True)
    waveform = librosa.resample(waveform.mean(axis=1), orig_sr=rate, target_sr=16000)
    f0, voiced, probability = librosa.pyin(waveform, sr=16000, fmin=65, fmax=900,
                                          frame_length=1024, hop_length=160)
    f0 = np.where(voiced & np.isfinite(f0), f0, 0)
    np.savez(destination, times=np.arange(len(f0)) * 0.01, f0=f0,
             activation=np.where(voiced, probability, 0))
    write_json(args.output / 'pyin_config.json', dict(librosa=librosa.__version__, sample_rate=16000,
               fmin=65, fmax=900, frame_length=1024, hop_length=160,
               vocals_sha256=digest(args.output / 'vocals.wav')))
    print('Independent pYIN frames:', len(f0), flush=True)


def make_midi(notes):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(resolution=960)
    for track, name in [(0, 'Vocal'), (1, 'Ins')]:
        instrument = pretty_midi.Instrument(0, name=name)
        if track == 0:
            for start, end, pitch in notes:
                instrument.notes.append(pretty_midi.Note(100, int(pitch), float(start), float(end)))
        midi.instruments.append(instrument)
    return midi


def write_review_page(output, rows, reference_octave):
    players = [('original_preview.wav', '原谱音高：沿用原始 LAB 时间轴'),
               ('corrected_preview.wav', '保守修正版：保留原谱音区，仅修改高置信度音高')]
    if reference_octave:
        players.append(('source_register_preview.wav', '原唱音区候选：保守修正版整体调整一个八度'))
    sections = ''.join(f'<section><h2>{title}</h2><audio controls preload="metadata" src="{name}"></audio></section>'
                       for name, title in players)
    changes = ''.join(f'<tr><td>{row["note"]}</td><td>{row["start"]:.2f}–{row["end"]:.2f}</td>'
                      f'<td>{row["original"]} → {row["corrected"]}</td></tr>'
                      for row in rows if row['action'] == 'auto_corrected')
    title = html.escape(output.name)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>F0 转谱复核 · {title}</title><style>
body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f6f7f9;color:#17212e}}
section{{background:white;padding:12px 20px;margin:12px 0;border-radius:10px}}
audio{{width:100%}}img{{width:100%}}td,th{{padding:8px 18px;text-align:left}}button{{padding:8px;margin:4px}}
</style><h1>第一轮 F0 辅助修谱</h1>
<p>这些是简单合成音色的旋律试听，不是重新生成的 cover，也不是 ABC 固定速度回放。
使用原始 LAB 秒级时间轴，以隔离音高差异。图中 F0 为对齐谱面显示而偏移 {reference_octave:+d} 个半音。</p>
<p>仅在 RMVPE 混音、人声及可用 pYIN 证据一致时修正。指标代表 F0 一致性，不是人工标注准确率。</p>
<button data-time="9.8">全部定位到 9.8 秒</button><button data-time="53.8">全部定位到 53.8 秒</button>
{sections}<h2>保守修正列表</h2><table><tr><th>音符序号</th><th>原音频秒数</th><th>MIDI 音高</th></tr>{changes}</table>
<h2>音高对照</h2><img src="pitch_comparison.png" alt="原谱、修正版和 F0 音高对照">
<p><a href="note_audit.csv">全部音符 CSV</a> · <a href="report.json">结构化报告</a> ·
<a href="corrected.abc">保守修正版 ABC</a> · <a href="rhythm_review.md">未修改的节奏疑点</a></p>
<script>
document.querySelectorAll('button').forEach(button => button.onclick = () => {{
  document.querySelectorAll('audio').forEach(audio => {{audio.pause(); audio.currentTime = Number(button.dataset.time);}});
}});
document.querySelectorAll('audio').forEach(player => player.onplay = () => {{
  document.querySelectorAll('audio').forEach(other => {{if (other !== player) other.pause();}});
}});
</script></html>'''
    (output / 'listen.html').write_text(page)


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf

    sys.path.insert(0, str(ROOT / 'skills/yue2-music/scripts'))
    from abc_tools import parse_abc

    extraction_root = args.output
    extraction = json.loads((extraction_root / 'extraction.json').read_text())
    manifest = json.loads((args.transcription / 'input.json').read_text())
    if extraction['audio_sha256'] != manifest['source_audio_sha256']:
        raise ValueError('F0 source is not the audio used by this transcription')
    if args.analysis_name:
        args.output = extraction_root / args.analysis_name
        args.output.mkdir(exist_ok=False)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Analysis already exists; use a fresh output directory')
    curves = np.load(extraction_root / 'f0.npz')
    independent = np.load(extraction_root / 'pyin.npz') if (extraction_root / 'pyin.npz').exists() else None
    if args.reference_octave and independent is None:
        raise ValueError('Octave-aligned analysis requires independent pYIN verification')
    notes = np.loadtxt(args.transcription / 'melody_vocal.lab')
    corrected = notes.copy()
    rows = []
    for index, (start, end, original) in enumerate(notes):
        stats = {name: summarize(curves['times'], curves[name + '_f0'] * 2 ** (args.reference_octave / 12), curves[name + '_activation'], start, end)
                 for name in ('vocal', 'mix')}
        if independent is not None:
            stats['pyin'] = summarize(independent['times'], independent['f0'] * 2 ** (args.reference_octave / 12), independent['activation'], start, end)
        pitch, action = decide(int(original), stats['vocal'], stats['mix'])
        if action == 'auto_corrected' and independent is not None:
            if not stats['pyin']['stable'] or stats['pyin']['candidate'] != pitch:
                pitch, action = int(original), 'review_independent_disagreement'
        corrected[index, 2] = pitch
        row = dict(note=index + 1, start=float(start), end=float(end), original=int(original),
                   corrected=pitch, action=action)
        for name, values in stats.items():
            row.update({name + '_' + key: value for key, value in values.items()})
        rows.append(row)
    with (args.output / 'note_audit.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'note_audit.json', rows)
    np.savetxt(args.output / 'melody_vocal.corrected.lab', corrected, fmt=['%.6f', '%.6f', '%d'], delimiter='\t')
    native = notation_module()
    data_rows = lambda name: [line.split() for line in (args.transcription / 'notation' / name).read_text().splitlines() if line.strip()]
    grids = data_rows('song_beats.txt')
    keys, structures = data_rows('song_keys.txt'), data_rows('song_structures.txt')
    before = parse_abc((args.transcription / 'score.abc').read_text())
    diagnostics = []
    for label, events in [('original', notes), ('corrected', corrected)]:
        midi = make_midi(events)
        midi.write(str(args.output / (label + '.mid')))
        text, rebuilt = native.generate_abc_from_data(midi, grids, [], keys, structures, melody_only=True)
        parsed = parse_abc(text)
        for voice in ('Vocal', 'Ins'):
            old, new = before.voices[voice], parsed.voices[voice]
            if old.bars != new.bars or len(old.notes) != len(new.notes):
                raise ValueError('Native re-export changed note count or bars')
            for previous, current in zip(old.notes, new.notes):
                if (previous[0], previous[2]) != (current[0], current[2]):
                    raise ValueError('Native re-export changed symbolic timing')
            if label == 'original' and old.notes != new.notes:
                raise ValueError('Baseline round trip failed')
        if [note[1] for note in parsed.voices['Vocal'].notes] != events[:, 2].astype(int).tolist():
            raise ValueError('ABC pitch round trip failed')
        if before.bpm != parsed.bpm:
            raise ValueError('Tempo changed')
        (args.output / (label + '.abc')).write_text(text)
        diagnostics = rebuilt.diagnostics
        render_preview(events, args.output / (label + '_preview.wav'), float(curves['times'][-1] + 0.02), sf)
    if args.reference_octave:
        source_register = corrected.copy()
        source_register[:, 2] -= args.reference_octave
        if np.any((source_register[:, 2] < 0) | (source_register[:, 2] > 127)):
            raise ValueError('Octave adjustment exceeds MIDI pitch range')
        midi = make_midi(source_register)
        midi.write(str(args.output / 'source_register.mid'))
        text, rebuilt = native.generate_abc_from_data(midi, grids, [], keys, structures, melody_only=True)
        parsed = parse_abc(text)
        for old, new in zip(before.voices['Vocal'].notes, parsed.voices['Vocal'].notes):
            if (old[0], old[2]) != (new[0], new[2]):
                raise ValueError('Source-register export changed timing')
        if [note[1] for note in parsed.voices['Vocal'].notes] != source_register[:, 2].astype(int).tolist():
            raise ValueError('Source-register pitch round trip failed')
        (args.output / 'source_register.abc').write_text(text)
        render_preview(source_register, args.output / 'source_register_preview.wav', float(curves['times'][-1] + 0.02), sf)
    figure, axes = plt.subplots(4, 1, figsize=(18, 13))
    for panel, axis in enumerate(axes):
        lower, upper = panel * 15, (panel + 1) * 15
        selected = (curves['times'] >= lower) & (curves['times'] < upper)
        for name, color in [('mix', '#aaaaaa'), ('vocal', '#2878b5')]:
            valid = selected & (curves[name + '_f0'] > 0)
            axis.scatter(curves['times'][valid], 69 + 12 * np.log2(curves[name + '_f0'][valid] / 440) + args.reference_octave,
                         s=2, color=color, alpha=0.5, label=name + ' RMVPE')
        if independent is not None:
            valid = (independent['times'] >= lower) & (independent['times'] < upper) & (independent['f0'] > 0)
            axis.scatter(independent['times'][valid], 69 + 12 * np.log2(independent['f0'][valid] / 440) + args.reference_octave,
                         s=1, color='#9c27b0', alpha=0.4, label='independent pYIN')
        for row in rows:
            if row['end'] < lower or row['start'] > upper:
                continue
            axis.hlines(row['original'], row['start'], row['end'], color='#e58606', linewidth=2)
            if row['action'] == 'auto_corrected':
                axis.hlines(row['corrected'], row['start'], row['end'], color='#1b9e77', linewidth=3)
                axis.text(row['start'], row['corrected'] + 0.6, str(row['note']), fontsize=8)
        voiced = curves['vocal_f0'][curves['vocal_f0'] > 0]
        minimum = min(float(notes[:, 2].min()), float(np.percentile(69 + 12 * np.log2(voiced / 440), 1)) + args.reference_octave)
        axis.set(xlim=(lower, upper), ylim=(minimum - 2, float(notes[:, 2].max()) + 3), ylabel='MIDI pitch', xlabel='Source seconds')
        axis.grid(alpha=0.2)
    axes[0].legend(loc='upper right')
    figure.suptitle(f'Orange: original | Green: corrections | F0 display offset: {args.reference_octave:+d} semitones | Numbers: note indices')
    figure.tight_layout()
    figure.savefig(args.output / 'pitch_comparison.png', dpi=150)
    plt.close(figure)
    counts = {action: sum(row['action'] == action for row in rows) for action in sorted({row['action'] for row in rows})}
    metrics = {}
    for name in (('vocal', 'mix', 'pyin') if independent is not None else ('vocal', 'mix')):
        selected = [row for row in rows if row[name + '_stable']]
        metrics[name] = dict(stable_notes=len(selected),
                             original_agree=sum(row['original'] == row[name + '_candidate'] for row in selected),
                             corrected_agree=sum(row['corrected'] == row[name + '_candidate'] for row in selected))
    report = dict(notes=len(notes), actions=counts, rules=RULES, f0_agreement=metrics,
                  reference_octave=args.reference_octave,
                  validation='Note counts, onsets, durations, bars, tempo and pitches checked.',
                  changes=[row for row in rows if row['action'] == 'auto_corrected'],
                  diagnostics=diagnostics, transcription=str(args.transcription.resolve()),
                  source_hashes={name: digest(args.transcription / name) for name in ['score.abc', 'melody_vocal.lab']},
                  limitation='F0 agreement is not labeled accuracy or listening quality. pYIN is an independent algorithm, not ground truth. Rhythm untouched.')
    write_json(args.output / 'report.json', report)
    write_review_page(args.output, rows, args.reference_octave)
    (args.output / 'rhythm_review.md').write_text(
        '# Rhythm review\n\nPitch-only run: no timing or meter changes.\n\n'
        + '\n'.join('- ' + message for message in diagnostics)
        + ('\n\nFor f8e2_hjf3: inspect 17.43–19.35 s (current three-beat measure); '
        'the existing paper variant uses 16.79–19.35 s and four beats. '
        'Neither grid is accepted as ground truth. Review preceding downbeats, not only the meter label.\n'
        if args.transcription.name == 'f8e2_hjf3' else '\n'))
    print(json.dumps(dict(actions=counts, f0_agreement=metrics), indent=2))


def render_preview(notes, path, duration, soundfile):
    rate = 24000
    audio = np.zeros(int((duration + 0.1) * rate), dtype=np.float32)
    for start, end, pitch in notes:
        first, last = int(start * rate), int(end * rate)
        times = np.arange(last - first) / rate
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        tone = np.sin(2 * np.pi * frequency * times) + 0.25 * np.sin(4 * np.pi * frequency * times)
        envelope = np.minimum(times / 0.012, 1) * np.minimum((len(times) / rate - times) / 0.025, 1)
        audio[first:last] += (0.18 * tone * envelope).astype(np.float32)
    soundfile.write(path, audio, rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'verify', 'analyze'])
    parser.add_argument('--audio', type=Path, default=ROOT / 'yue2-cover-test/download/f8e2_hjf3.mp3')
    parser.add_argument('--transcription', type=Path, default=ROOT / 'yue2-cover-test/work/transcriptions/f8e2_hjf3')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rmvpe-repo', type=Path, default=Path('/data/cmq/MIDI-SAG/MuseControlLite'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/data/cmq/MIDI-SAG/MIDI-SAG_checkpoints/rmvpe_model.pt'))
    parser.add_argument('--separator-repo', type=Path, default=Path('/data/cmq/audioalgo-master-URV/URV'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--analysis-name', help='Fresh analysis subdirectory; keep extraction and initial analysis intact')
    parser.add_argument('--reference-octave', type=int, choices=[-12, 0, 12], default=0,
                        help='Explicit octave normalization for comparison only; corrected.abc retains input register')
    args = parser.parse_args()
    {'extract': extract, 'verify': verify, 'analyze': analyze}[args.stage](args)


if __name__ == '__main__':
    main()
