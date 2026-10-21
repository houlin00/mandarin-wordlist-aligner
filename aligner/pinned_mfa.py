"""
pinned_mfa.py
=============
Per-word MFA alignment that cannot move your manually corrected word boundaries.

Input : a word-level TextGrid you have already corrected (tier `refined_word`,
        one Chinese word per interval) + the recording.
Output: TextGrid with `words` + `phones` tiers whose word edges == your edges,
        and a CSV measuring how far MFA *wanted* to move them (free uncertainty
        signal / aligner metric).

Run in two environments (mfa_env has no pandas/textgrid):

  # 1. conda activate lenition
  from pinned_mfa import *
  prep(GOLD_TG, WAV, 'S1', CORPUS_DIR, BASE_DICT, OUT_DICT)

  # 2. conda activate mfa_env   (terminal)
  mfa align CORPUS_DIR OUT_DICT mandarin_mfa MFA_OUT --clean --beam 100 \
            --retry_beam 400 --single_speaker

  # 3. conda activate lenition
  merge(MFA_OUT, CORPUS_DIR, 'S1', FINAL_TG)

Because every clip holds exactly one word (+ padding), MFA has nothing to do
except place phones inside that word.  Padding lets it see closure / frication
that your edge might clip, but merge() re-pins the first/last phone to your edge.
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from textgrid import TextGrid, IntervalTier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_list_aligner import write_mfa_dictionary  # reuse word-level dictionary

import re
_TONE_MARK = {'˥': '1', '˧˥': '2', '˨˩˦': '3', '˥˩': '4', '˩': '3', '˧': '1', '˨˩': '3'}


def _tone_of(pron):
    m = re.search(r'[˥˧˨˩˦]+', pron)
    return _TONE_MARK.get(m.group(0)) if m else None


def build_syl_map(base_dict):
    """pinyin-with-tone -> MFA phones, from single-character dictionary entries.
    The tone mark inside each pronunciation picks which pypinyin reading of a
    heteronym it belongs to (好 hao3 / 猫 mao1 / 宅 zhai2); when one tone has
    several readings (大 da4 / dai4) the most probable pronunciation goes to the
    first (most frequent) pypinyin reading."""
    from pypinyin import pinyin, Style
    m = {}
    prons = {}
    for line in open(base_dict, encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 6 and len(parts[0]) == 1:
            prons.setdefault(parts[0], []).append((float(parts[1]), parts[-1].strip()))
    for ch, ps in prons.items():
        hets = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        ps = sorted(dict.fromkeys(ps), key=lambda x: -x[0])
        for t in '12345':
            cand = [h for h in hets if h[-1:] == t]
            prs = [pr for _, pr in ps if _tone_of(pr) == t]
            for h, pr in zip(cand, prs):
                m.setdefault(h, pr)
    return m


def build_word_prons(words, base_dict, erhua_variants=False):
    """word -> list of (pronunciation, prob): citation tones, third-tone-sandhi
    variant, and (optionally) a final-ɻ erhua variant of each for non-erhua words."""
    from pypinyin import pinyin, Style
    syl2ph = build_syl_map(base_dict)

    def conv(sylls, w):
        out = []
        for k, sy in enumerate(sylls):
            ph = syl2ph.get(sy)
            if ph is None and sy.endswith('5'):          # neutral tone -> citation tone shape
                cit = pinyin(w[k], style=Style.TONE3)[0][0]
                ph = syl2ph.get(cit)
            if ph is None:
                return None
            out.append(ph)
        return ' '.join(out)

    res = {}
    for w in words:
        erhua = len(w) > 2 and w.endswith('儿')
        wc = w[:-1] if erhua else w
        base = [x[0] for x in pinyin(wc, style=Style.TONE3, neutral_tone_with_five=True)]
        sand = list(base)                                   # 3-3 -> 2-3, right to left
        for i in range(len(sand) - 2, -1, -1):
            if sand[i].endswith('3') and sand[i + 1].endswith('3'):
                sand[i] = sand[i][:-1] + '2'
        vs = [conv(base, wc)]
        if sand != base:
            vs.append(conv(sand, wc))
        vs = [v + ' ɻ' if erhua else v for v in dict.fromkeys(vs) if v]
        if vs:
            out = [(v, 0.9) for v in vs]
            if erhua_variants and not erhua:
                out += [(v + ' ɻ', 0.1) for v in vs]
            res[w] = out
    return res


PAD_S = 0.10          # padding on each side of a word clip
GOLD_TIER = 'refined_word'


def read_gold(gold_tg, tier=GOLD_TIER):
    tg = TextGrid.fromFile(gold_tg)           # handles Praat's UTF-16 files
    t = next(x for x in tg.tiers if x.name == tier)
    return [(iv.minTime, iv.maxTime, iv.mark.strip()) for iv in t if iv.mark.strip()]


def write_token_dict(pron, tok, out_dict):
    with open(out_dict, 'w', encoding='utf-8') as f:
        for w, vs in pron.items():
            for v, pr in vs:
                f.write(f'{tok[w]}\t{pr}\t0.25\t1.0\t1.0\t{v}\n')


def prep(gold_tg, wav_path, prefix, corpus_dir, base_dict, out_dict,
         items=None, erhua_variants=True, tier=GOLD_TIER):
    """Each word gets an ASCII token (w0001...) in the .lab and the dictionary:
    MFA's text normaliser otherwise re-segments Chinese strings (e.g. 出界分 ->
    出界 分) and the forced word pronunciation is lost."""
    import librosa
    gold = items if items is not None else read_gold(gold_tg, tier)
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    dur = len(y) / sr
    os.makedirs(corpus_dir, exist_ok=True)
    gold = [g for g in gold if g[2] != '(off-list)']
    uniq = list(dict.fromkeys(w for _, _, w in gold))
    tok = {w: f'w{i:04d}' for i, w in enumerate(uniq)}
    pron = build_word_prons(uniq, base_dict, erhua_variants)
    write_token_dict(pron, tok, out_dict)
    missing = [w for w in uniq if w not in pron]
    if missing:
        print('no pronunciation (clips skipped):', missing)
    rows = []
    for k, (s, e, w) in enumerate(gold):
        if w not in pron:
            continue
        lo = s - PAD_S
        hi = e + PAD_S
        if k > 0:
            lo = max(lo, (gold[k - 1][1] + s) / 2)       # don't swallow the neighbour
        if k + 1 < len(gold):
            hi = min(hi, (e + gold[k + 1][0]) / 2)
        lo, hi = max(0.0, lo), min(dur, hi)
        if hi - lo < 0.05:
            continue
        cid = f'{prefix}_{k:05d}'
        sf.write(os.path.join(corpus_dir, cid + '.wav'), y[int(lo * sr):int(hi * sr)], sr)
        with open(os.path.join(corpus_dir, cid + '.lab'), 'w', encoding='utf-8') as f:
            f.write(tok[w])
        rows.append(dict(clip=cid, word=w, gold_start=s, gold_end=e, clip_start=lo, clip_end=hi))
    pd.DataFrame(rows).to_csv(os.path.join(corpus_dir, f'{prefix}_manifest.csv'),
                              index=False, encoding='utf-8')
    print(f'{len(rows)} clips -> {corpus_dir}')


def merge(mfa_out, corpus_dir, prefix, out_tg, min_phone_s=0.005):
    man = pd.read_csv(os.path.join(corpus_dir, f'{prefix}_manifest.csv'))
    words, phones, rows, free = [], [], [], []
    for _, r in man.iterrows():
        p = os.path.join(mfa_out, r['clip'] + '.TextGrid')
        if not os.path.exists(p):
            rows.append(dict(clip=r['clip'], word=r['word'], status='no_output'))
            continue
        tg = TextGrid.fromFile(p)
        wt = next(t for t in tg.tiers if 'word' in t.name.lower())
        pt = next(t for t in tg.tiers if 'phone' in t.name.lower())
        ws = [iv for iv in wt if iv.mark.strip() and iv.mark not in ('<eps>',)]
        if not ws:
            rows.append(dict(clip=r['clip'], word=r['word'], status='no_word'))
            continue
        off = r['clip_start']
        w0, w1 = ws[0].minTime + off, ws[-1].maxTime + off
        ph = [(q.minTime + off, q.maxTime + off, q.mark) for q in pt
              if q.mark.strip() and q.mark not in ('sil', 'sp', '<eps>')
              and q.maxTime + off > w0 - 1e-6 and q.minTime + off < w1 + 1e-6]
        spn = bool(ph) and all(m == 'spn' for _, _, m in ph)
        gs, ge = r['gold_start'], r['gold_end']
        free.append((w0, w1, r['word']))
        # how far MFA wanted to move the edges (negative = MFA earlier than gold)
        rows.append(dict(clip=r['clip'], word=r['word'], status='spn' if spn else 'ok',
                         mfa_start=w0, mfa_end=w1, gold_start=gs, gold_end=ge,
                         d_on_ms=1000 * (w0 - gs), d_off_ms=1000 * (w1 - ge),
                         n_phones=len(ph), last_phone=ph[-1][2] if ph else '',
                         first_phone_ms=1000 * (ph[0][1] - ph[0][0]) if ph else np.nan,
                         erhua=bool(ph) and ph[-1][2] == 'ɻ' and not r['word'].endswith('儿')))
        if not ph:
            continue
        # pin: clip phones to gold edges, first/last phone stretched to the edge
        pinned = []
        for a, b, m in ph:
            a, b = max(a, gs), min(b, ge)
            if b - a >= min_phone_s:
                pinned.append([a, b, m])
        if not pinned:
            continue
        pinned[0][0], pinned[-1][1] = gs, ge
        words.append((gs, ge, r['word']))
        phones.extend((a, b, m) for a, b, m in pinned)

    dur = max([e for _, e, _ in words] + [0]) + 0.1
    tg = TextGrid(minTime=0, maxTime=dur)
    for name, ivs in (('words', words), ('phones', phones), ('mfa_free', free)):
        tier = IntervalTier(name, 0, dur)
        last = 0.0
        for s, e, lab in sorted(ivs):
            s = max(s, last)
            if e - s > 1e-4:
                tier.add(s, e, lab)
                last = e
        tg.append(tier)
    tg.write(out_tg)
    res = pd.DataFrame(rows)
    res.to_csv(out_tg.replace('.TextGrid', '_edges.csv'), index=False, encoding='utf-8')
    ok = res[res['status'] == 'ok']
    print(res['status'].value_counts().to_string())
    for col, name in (('d_on_ms', 'onset'), ('d_off_ms', 'offset')):
        a = ok[col].abs()
        print(f'MFA free {name} vs your edge: median {a.median():.0f} ms; '
              f'<=20 ms {(a <= 20).mean():.1%}; <=50 ms {(a <= 50).mean():.1%}; '
              f'mean bias {ok[col].mean():+.0f} ms')
    print(f'-> {out_tg}')
    return res


def fix_dictionary(words, base_dict, out_dict):
    """Second pass for words write_mfa_dictionary could not build (syllables whose
    only dictionary characters are heteronyms).  For a missing syllable use the
    first pronunciation of any single character whose pypinyin readings include
    that syllable and which has exactly one pronunciation in the base dictionary."""
    from pypinyin import pinyin, Style
    from known_list_aligner import _read_mfa_dict, word_tone3
    _, prons = _read_mfa_dict(base_dict)
    have = {l.split('\t')[0] for l in open(out_dict, encoding='utf-8')}
    loose = {}
    for ch, ps in prons.items():
        if len(ch) != 1 or len(set(ps)) != 1:
            continue
        for syl in pinyin(ch, style=Style.TONE3, heteronym=True,
                          neutral_tone_with_five=True)[0]:
            loose.setdefault(syl, ps[0])
    add = []
    for w in dict.fromkeys(words):
        if w in have:
            continue
        ph = [loose.get(s) or loose.get(s.rstrip('5') + '1') for s in word_tone3(w)]
        if all(ph):
            add.append(f"{w}\t0.99\t0.25\t1.0\t1.0\t{' '.join(ph)}")
        else:
            print('still missing:', w, word_tone3(w))
    with open(out_dict, 'a', encoding='utf-8') as f:
        f.write('\n'.join(add) + '\n')
    print(f'fix_dictionary added {len(add)} words')


def review_table(edges_csv, analysis_csv, out_csv, out_tg=None, top_n=250):
    """Rank words by how likely the INPUT interval is wrong (not MFA).
    Signals: MFA wants to move an edge a lot (>100 ms), low per-clip likelihood,
    odd phone durations.  If out_tg is given, adds a `review` tier (top_n words)
    to that TextGrid so you can jump through them in Praat."""
    e = pd.read_csv(edges_csv)
    a = pd.read_csv(analysis_csv).rename(columns={'file': 'clip'})
    e = e.merge(a[['clip', 'speech_log_likelihood', 'phone_duration_deviation']],
                on='clip', how='left')
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    e['edge_move'] = e[['d_on_ms', 'd_off_ms']].abs().max(axis=1)
    e['score'] = (e['edge_move'] / 100.0
                  + (-z(e['speech_log_likelihood'])).clip(lower=0)
                  + z(e['phone_duration_deviation']).clip(lower=0))
    def why(r):
        out = []
        if r['d_on_ms'] > 100: out.append('MFA onset later: leading noise/silence or previous word in interval?')
        if r['d_on_ms'] < -100: out.append('MFA onset earlier: interval starts too late?')
        if r['d_off_ms'] < -100: out.append('MFA offset earlier: trailing silence/breath/next word in interval?')
        if r['d_off_ms'] > 100: out.append('MFA offset later: interval ends too early?')
        return '; '.join(out)
    e['why'] = e.apply(why, axis=1)
    e = e.sort_values('score', ascending=False).reset_index(drop=True)
    e.to_csv(out_csv, index=False, encoding='utf-8-sig')
    if out_tg:
        tg = TextGrid.fromFile(out_tg)
        tg.tiers = [t for t in tg.tiers if t.name != 'review']
        tier = IntervalTier('review', 0, tg.maxTime)
        for _, r in e.head(top_n).sort_values('gold_start').iterrows():
            tier.add(r['gold_start'], r['gold_end'], f"CHECK {r['word']} {r['why'][:40]}")
        tg.append(tier)
        tg.write(out_tg)
    print(f'{(e.edge_move > 100).sum()} words with an edge move > 100 ms; csv -> {out_csv}')
    return e
