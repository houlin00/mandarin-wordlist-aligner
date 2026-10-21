"""
run_speaker.py  --  whole repair pipeline for one speaker.

  ~/miniconda3/envs/lenition/bin/python run_speaker.py 2          # S2
  ~/miniconda3/envs/lenition/bin/python run_speaker.py 2 3 4 5    # several

Input : rough_trascription/S{n}_REFINED.TextGrid (tier refined_word, automatic),
        recording/S{n}.WAV, wordlist_csv/speaker_{n}_list.csv
Output: context_mfa/S{n}/S{n}_final.TextGrid (+ _words.csv, S{n}_not_read.csv)
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import context_mfa as cm

ROOT = os.path.dirname(HERE)
BASE_DICT = os.path.expanduser('~/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict')
MFA_ENV = os.path.expanduser('~/miniconda3/envs/mfa_env')


def mfa_align(corpus, dict_path, out, tries=3):
    """MFA occasionally dies at start-up (import/config errors); retry."""
    env = dict(os.environ, PATH=f'{MFA_ENV}/bin:' + os.environ['PATH'])
    msg = ''
    for t in range(tries):
        subprocess.run(['rm', '-rf', out])
        r = subprocess.run([f'{MFA_ENV}/bin/mfa', 'align', corpus, dict_path, 'mandarin_mfa', out,
                            '--clean', '--no_tokenization', '--beam', '100', '--retry_beam', '400',
                            '--single_speaker', '-j', '4'], env=env, capture_output=True, text=True)
        if os.path.isdir(out) and any(f.endswith('.TextGrid') for f in os.listdir(out)):
            return
        msg = r.stdout[-1500:] + r.stderr[-1500:]
        print(f'mfa align failed (try {t + 1}/{tries}), retrying')
    raise RuntimeError('mfa align failed:\n' + msg)


def not_read_table(gaps, allw, out_csv):
    import ast
    rec = allw[allw.source == 'recovered']
    rows = []
    for r in gaps.itertuples():
        ws = ast.literal_eval(r.words) if isinstance(r.words, str) else list(r.words)
        got = ((rec.new_start >= r.gap_start - 0.1) & (rec.new_end <= r.gap_end + 0.1)).sum()
        if r.has_speech and got >= len(ws):
            continue
        rows.append(dict(missing_words=' '.join(ws), gap_start=round(r.gap_start, 2),
                         gap_end=round(r.gap_end, 2), speech_s=r.speech_s,
                         status=('no speech in gap -> probably not read' if not r.has_speech else
                                 'speech in gap but not recovered -> check')))
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    return pd.DataFrame(rows)


def run(n):
    S = f'S{n}'
    return run_paths(S, os.path.join(ROOT, 'rough_trascription', f'{S}_REFINED.TextGrid'),
                     os.path.join(ROOT, 'recording and scripts', 'recording', f'{S}.WAV'),
                     os.path.join(ROOT, 'wordlist_csv', f'speaker_{n}_list.csv'),
                     os.path.join(ROOT, 'context_mfa', S))


def run_paths(S, tg, wav, wl, out):
    os.makedirs(out, exist_ok=True)
    P = lambda *a: os.path.join(out, *a)
    print(f'===== {S} =====')
    cm.prep_windows(tg, wav, S, P('win'), BASE_DICT, P('win_dict.txt'), wordlist_csv=wl)
    mfa_align(P('win'), P('win_dict.txt'), P('win_out'))
    rep = cm.repair_intervals(P('win_out'), P('win'), S, P(f'{S}_repaired.csv'))
    gaps, anc = cm.find_gaps(rep, wl, wav)
    gaps.to_csv(P(f'{S}_gaps.csv'), index=False, encoding='utf-8-sig')
    gw = pd.DataFrame(columns=['word', 'new_start', 'new_end', 'gap'])
    todo = gaps[gaps.has_speech] if len(gaps) else gaps
    if len(todo):
        cm.prep_gap_windows(gaps, anc, wav, S, P('gapwin'), BASE_DICT, P('gap_dict.txt'))
        if os.path.exists(P('gapwin', f'{S}_gapman.csv')) and \
                len(pd.read_csv(P('gapwin', f'{S}_gapman.csv'))):
            mfa_align(P('gapwin'), P('gap_dict.txt'), P('gap_out'))
            gw = cm.collect_gap_words(P('gap_out'), P('gapwin'), S)
    allw = cm.combine(rep, gw, P(f'{S}_all_words.csv'))
    allw = cm.trim_ends(allw, wav)
    allw.to_csv(P(f'{S}_all_words.csv'), index=False, encoding='utf-8-sig')
    nr = not_read_table(gaps, allw, P(f'{S}_not_read.csv'))
    cm.prep_repaired(allw, wav, S, P('clips'), BASE_DICT, P('clip_dict.txt'))
    mfa_align(P('clips'), P('clip_dict.txt'), P('clip_out'))
    res = cm.final_tg(P('clip_out'), P('clips'), S, allw, P(f'{S}_final.TextGrid'))
    fl = res['flags'].fillna('').str.split().explode().value_counts()
    summary = dict(speaker=S, intervals=int((rep.status == 'ok').sum()),
                   windows_failed=int((rep.status != 'ok').sum()),
                   recovered=len(gw), not_read=int((nr.status.str.startswith('no speech')).sum())
                   if len(nr) else 0, to_check_not_read=int(nr.status.str.startswith('speech').sum())
                   if len(nr) else 0, words_final=len(res), flagged=int((res['flags'] != '').sum()),
                   poor_fit=int(fl.get('poor_fit(misread/lenition?)', 0)),
                   erhua=int(fl.get('erhua', 0)), long_dur=int(fl.get('odd_duration(2', 0)))
    return summary


if __name__ == '__main__':
    rows = []
    for n in sys.argv[1:]:
        try:
            rows.append(run(int(n)))
        except Exception as e:                       # keep going with the other speakers
            print(f'S{n} FAILED: {e}')
            rows.append(dict(speaker=f'S{n}', error=str(e)[:200]))
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'context_mfa', 'batch_summary.csv'),
                                  index=False, encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_string())
