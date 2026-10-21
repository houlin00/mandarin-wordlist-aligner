"""
context_mfa.py
==============
Repair word intervals that were cut wrongly by the energy-based segmentation
(truncated words, neighbour spill-over, merged words, breath at the start).

Idea: instead of trusting an interval, align a WINDOW that also contains the
previous and next word of the list.  MFA (with optional silence between
tokens) finds where the target word really is.  Your interval is only the
starting guess.

Pipeline (envs: lenition = python, mfa_env = the `mfa` command)

  # lenition
  from context_mfa import *
  prep_windows(GOLD_TG, WAV, 'S1', WIN_DIR, BASE_DICT, WIN_DICT)
  # mfa_env
  mfa align WIN_DIR WIN_DICT mandarin_mfa WIN_OUT --clean --no_tokenization \
            --beam 100 --retry_beam 400 --single_speaker -j 4
  # lenition
  rep = repair_intervals(WIN_OUT, WIN_DIR, 'S1', 'S1_repaired.csv')
  prep_repaired(rep, WAV, 'S1', CLIP_DIR, BASE_DICT, CLIP_DICT)
  # mfa_env
  mfa align CLIP_DIR CLIP_DICT mandarin_mfa CLIP_OUT --clean --no_tokenization \
            --beam 100 --retry_beam 400 --single_speaker -j 4
  # lenition
  final_tg(CLIP_OUT, CLIP_DIR, 'S1', rep, 'S1_final.TextGrid')
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from textgrid import TextGrid, IntervalTier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pinned_mfa as pm

CTX_PAD_S = 0.10      # extra audio before the previous / after the next word
MAX_CTX_S = 1.5       # never look further than this from the target interval
MAX_GAP_S = 1.5       # a neighbour farther away than this is left out


def prep_windows(gold_tg, wav_path, prefix, win_dir, base_dict, out_dict, wordlist_csv=None,
                 max_between=4):
    """With wordlist_csv the window is built from the WORDLIST ORDER: intervals are first
    matched to wordlist positions, and each window contains every list word from the
    previous interval's word to the next interval's word.  Words that are missing
    in between (hidden inside a merged interval) are therefore in the transcript and
    cannot be swallowed by the target word.  Without it, only neighbouring labels
    are used."""
    import librosa
    gold = [g for g in pm.read_gold(gold_tg) if g[2] != '(off-list)']
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    dur = len(y) / sr
    os.makedirs(win_dir, exist_ok=True)
    wl = idx = None
    if wordlist_csv:
        wl = pd.read_csv(wordlist_csv).sort_values('order')['word'].astype(str).str.strip().tolist()
        idx = _match_to_wordlist([g[2] for g in gold], wl)
    uniq = list(dict.fromkeys([w for _, _, w in gold] + (wl or [])))
    tok = {w: f'w{i:04d}' for i, w in enumerate(uniq)}
    pron = pm.build_word_prons(uniq, base_dict, erhua_variants=True)
    pm.write_token_dict(pron, tok, out_dict)
    rows = []
    for k, (s, e, w) in enumerate(gold):
        if w not in pron:
            continue
        left, right, lo, hi = [], [], s - CTX_PAD_S, e + CTX_PAD_S
        if k > 0 and gold[k - 1][2] in pron and s - gold[k - 1][1] <= MAX_GAP_S:
            lo = gold[k - 1][0] - CTX_PAD_S
            pi, ci = (idx[k - 1], idx[k]) if idx else (None, None)
            if pi is not None and ci is not None and 0 < ci - pi <= max_between:
                left = [x for x in wl[pi:ci] if x in pron]
            else:
                left = [gold[k - 1][2]]
        if k + 1 < len(gold) and gold[k + 1][2] in pron and gold[k + 1][0] - e <= MAX_GAP_S:
            hi = gold[k + 1][1] + CTX_PAD_S
            ci, ni = (idx[k], idx[k + 1]) if idx else (None, None)
            if ci is not None and ni is not None and 0 < ni - ci <= max_between:
                right = [x for x in wl[ci + 1:ni + 1] if x in pron]
            else:
                right = [gold[k + 1][2]]
        lo = max(0.0, s - MAX_CTX_S, lo)
        hi = min(dur, e + MAX_CTX_S, hi)
        toks = [tok[x] for x in left] + [tok[w]] + [tok[x] for x in right]
        cid = f'{prefix}_{k:05d}'
        sf.write(os.path.join(win_dir, cid + '.wav'), y[int(lo * sr):int(hi * sr)], sr)
        with open(os.path.join(win_dir, cid + '.lab'), 'w', encoding='utf-8') as f:
            f.write(' '.join(toks))
        rows.append(dict(clip=cid, k=k, word=w, gold_start=s, gold_end=e,
                         win_start=lo, win_end=hi, n_prev=len(left), n_tok=len(toks)))
    pd.DataFrame(rows).to_csv(os.path.join(win_dir, f'{prefix}_manifest.csv'),
                              index=False, encoding='utf-8')
    print(f'{len(rows)} windows -> {win_dir}')


def repair_intervals(win_out, win_dir, prefix, out_csv):
    """Read the target word's interval out of every window alignment."""
    man = pd.read_csv(os.path.join(win_dir, f'{prefix}_manifest.csv'))
    rows = []
    for _, r in man.iterrows():
        p = os.path.join(win_out, r['clip'] + '.TextGrid')
        base = dict(k=r['k'], word=r['word'], gold_start=r['gold_start'],
                    gold_end=r['gold_end'], n_tok=r['n_tok'])
        if not os.path.exists(p):
            rows.append(dict(base, status='no_output'))
            continue
        tg = TextGrid.fromFile(p)
        wt = next(t for t in tg.tiers if 'word' in t.name.lower())
        ws = [iv for iv in wt if iv.mark.strip() and iv.mark != '<eps>']
        if len(ws) != r['n_tok']:
            rows.append(dict(base, status=f'word_count {len(ws)}!={r["n_tok"]}'))
            continue
        iv = ws[int(r['n_prev'])]
        s, e = iv.minTime + r['win_start'], iv.maxTime + r['win_start']
        rows.append(dict(base, status='ok', new_start=s, new_end=e,
                         d_start_ms=1000 * (s - r['gold_start']),
                         d_end_ms=1000 * (e - r['gold_end']),
                         new_dur=e - s))
    df = pd.DataFrame(rows)
    ok = df[df.status == 'ok']
    print(df.status.value_counts().to_string())
    print(f'start moved >50 ms: {(ok.d_start_ms.abs() > 50).mean():.1%}; '
          f'end moved >50 ms: {(ok.d_end_ms.abs() > 50).mean():.1%}; '
          f'implausible duration (<0.15 or >1.5 s): '
          f'{((ok.new_dur < 0.15) | (ok.new_dur > 1.5)).sum()}')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return df


def prep_repaired(rep, wav_path, prefix, clip_dir, base_dict, out_dict):
    """Single-word clips on the repaired intervals -> final phones, per-word
    likelihood, erhua detection."""
    ok = rep[rep.status == 'ok'].sort_values('new_start')
    items = [(r.new_start, r.new_end, r.word) for r in ok.itertuples()]
    pm.prep(None, wav_path, prefix, clip_dir, base_dict, out_dict, items=items)
    # map clip index (position in items) back to original k
    man_p = os.path.join(clip_dir, f'{prefix}_manifest.csv')
    man = pd.read_csv(man_p)
    ks = ok.k.tolist()
    man['k'] = [ks[int(c.split('_')[-1])] for c in man['clip']]
    man.to_csv(man_p, index=False, encoding='utf-8')


def final_tg(clip_out, clip_dir, prefix, rep, out_tg):
    res = pm.merge(clip_out, clip_dir, prefix, out_tg)
    man = pd.read_csv(os.path.join(clip_dir, f'{prefix}_manifest.csv'))[['clip', 'k']]
    res = res.merge(man, on='clip', how='left')
    an = pd.read_csv(os.path.join(clip_out, 'alignment_analysis.csv')).rename(columns={'file': 'clip'})
    res = res.merge(an[['clip', 'speech_log_likelihood', 'phone_duration_deviation']],
                    on='clip', how='left')
    res = res.merge(rep[['k', 'gold_start', 'gold_end', 'd_start_ms', 'd_end_ms', 'new_dur', 'source']]
                    .rename(columns={'gold_start': 'orig_start', 'gold_end': 'orig_end'}),
                    on='k', how='left')
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    res['fit_z'] = z(res['speech_log_likelihood'])

    def flags(r):
        f = []
        if r['fit_z'] < -1.5: f.append('poor_fit(misread/lenition?)')
        if r.get('erhua') == True: f.append('erhua')
        if r['new_dur'] < 0.15 or r['new_dur'] > 1.2: f.append('odd_duration(2 reps?)')
        if r.get('source') == 'recovered': f.append('recovered')
        if r['first_phone_ms'] > 200: f.append('long_first_phone(breath?)')
        if max(abs(r['d_start_ms']), abs(r['d_end_ms'])) > 150: f.append('moved>150ms')
        return ' '.join(f)

    res['flags'] = res.apply(flags, axis=1)
    out_csv = out_tg.replace('.TextGrid', '_words.csv')
    res.to_csv(out_csv, index=False, encoding='utf-8-sig')
    tg = TextGrid.fromFile(out_tg)
    tier = IntervalTier('flags', 0, tg.maxTime)
    last = 0.0
    for r in res[res['flags'] != ''].sort_values('gold_start').itertuples():
        s, e = max(r.gold_start, last), r.gold_end
        if e - s > 1e-4:
            tier.add(s, e, f'{r.word} {r.flags}')
            last = e
    tg.append(tier)
    tg.write(out_tg)
    print(f"{(res['flags'] != '').sum()} words flagged -> {out_csv}")
    return res


# ---------------------------------------------------------------------------
# Gap filling: words that have no interval (second repetition / preceding word
# of a merged interval) are recovered from the wordlist order; gaps without
# speech are reported as probably not read.
# ---------------------------------------------------------------------------
def _match_to_wordlist(labels, wl):
    """Needleman-Wunsch: returns list, per label, of wordlist index or None."""
    n, m = len(labels), len(wl)
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    S[:, 0] = -np.arange(n + 1)
    S[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        li = labels[i - 1]
        Si, Sp = S[i], S[i - 1]
        for j in range(1, m + 1):
            sc = 2 if li == wl[j - 1] else -1
            Si[j] = max(Sp[j - 1] + sc, Sp[j] - 1, Si[j - 1] - 1)
    out = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        sc = 2 if labels[i - 1] == wl[j - 1] else -1
        if S[i, j] == S[i - 1, j - 1] + sc:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif S[i, j] == S[i - 1, j] - 1:
            i -= 1
        else:
            j -= 1
    return out


def _speech_mask(y, sr, hop=0.01, above_db=12.0):
    import librosa
    h = int(hop * sr)
    rms = librosa.feature.rms(y=y, frame_length=int(0.025 * sr), hop_length=h)[0]
    db = 20 * np.log10(rms + 1e-9)
    floor = np.percentile(db, 10)
    return db > floor + above_db, h / sr


def find_gaps(rep, wordlist_csv, wav_path, min_speech_s=0.12):
    """-> (gaps DataFrame, anchors DataFrame with wordlist index)."""
    import librosa
    wl = pd.read_csv(wordlist_csv).sort_values('order')['word'].astype(str).str.strip().tolist()
    a = rep[rep.status == 'ok'].sort_values('new_start').reset_index(drop=True)
    idx = _match_to_wordlist(a.word.tolist(), wl)
    a['wl_idx'] = idx
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    mask, fr = _speech_mask(y, sr)
    rows = []
    last = a[a.wl_idx.notna()].index.tolist()

    def speech_in(g0, g1):
        seg = mask[int(g0 / fr):int(g1 / fr)]
        best = run = 0
        for v in seg:
            run = run + 1 if v else 0
            best = max(best, run)
        return best * fr

    for p, q in zip(last[:-1], last[1:]):
        i0, i1 = int(a.at[p, 'wl_idx']), int(a.at[q, 'wl_idx'])
        if i1 - i0 <= 1:
            continue
        miss = wl[i0 + 1:i1]
        # candidate regions (left anchor pos, right anchor pos).  Repeated words make
        # the sequence match ambiguous (which repetition did the interval belong to?),
        # so when a missing word equals a neighbouring anchor's word, also try the
        # region on the other side of that anchor and keep the one with most speech.
        cands = [(p, q)]
        if q + 1 < len(a) and a.at[q, 'word'] == miss[-1]:
            cands.append((q, q + 1))
        if p - 1 >= 0 and a.at[p, 'word'] == miss[0]:
            cands.append((p - 1, p))
        scored = [(speech_in(a.at[l, 'new_end'], a.at[r, 'new_start']), l, r) for l, r in cands]
        sp, l, r = max(scored, key=lambda x: x[0])
        g0, g1 = a.at[l, 'new_end'], a.at[r, 'new_start']
        rows.append(dict(prev_pos=l, next_pos=r, wl_from=i0 + 1, wl_to=i1 - 1,
                         words=miss, gap_start=g0, gap_end=g1, gap_s=g1 - g0,
                         speech_s=sp, has_speech=sp >= min_speech_s,
                         moved_side=(l, r) != (p, q)))
    return pd.DataFrame(rows), a


def prep_gap_windows(gaps, anchors, wav_path, prefix, win_dir, base_dict, out_dict,
                     max_s_per_word=1.6):
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    todo = gaps[gaps.has_speech & (gaps.gap_s <= max_s_per_word * gaps.words.apply(len) + 1.0)]
    allw = {w for ws in todo.words for w in ws}
    allw |= set(anchors.word)
    uniq = sorted(allw)
    tok = {w: f'w{i:04d}' for i, w in enumerate(uniq)}
    pron = pm.build_word_prons(uniq, base_dict, erhua_variants=True)
    pm.write_token_dict(pron, tok, out_dict)
    os.makedirs(win_dir, exist_ok=True)
    rows = []
    for gi, g in todo.iterrows():
        ws = [w for w in g.words if w in pron]
        if not ws:
            continue
        pa, na = anchors.loc[g.prev_pos], anchors.loc[g.next_pos]
        toks = [tok[pa.word]] + [tok[w] for w in ws] + [tok[na.word]]
        lo, hi = max(0.0, pa.new_start - 0.05), min(len(y) / sr, na.new_end + 0.05)
        cid = f'{prefix}_gap{gi:04d}'
        sf.write(os.path.join(win_dir, cid + '.wav'), y[int(lo * sr):int(hi * sr)], sr)
        with open(os.path.join(win_dir, cid + '.lab'), 'w', encoding='utf-8') as f:
            f.write(' '.join(toks))
        rows.append(dict(clip=cid, gap=gi, words='|'.join(ws), win_start=lo, n_tok=len(toks)))
    pd.DataFrame(rows).to_csv(os.path.join(win_dir, f'{prefix}_gapman.csv'),
                              index=False, encoding='utf-8')
    print(f'{len(rows)} gap windows -> {win_dir}')


def collect_gap_words(win_out, win_dir, prefix):
    man = pd.read_csv(os.path.join(win_dir, f'{prefix}_gapman.csv'))
    rows = []
    for _, r in man.iterrows():
        p = os.path.join(win_out, r['clip'] + '.TextGrid')
        if not os.path.exists(p):
            continue
        tg = TextGrid.fromFile(p)
        wt = next(t for t in tg.tiers if 'word' in t.name.lower())
        ws = [iv for iv in wt if iv.mark.strip() and iv.mark != '<eps>']
        if len(ws) != r['n_tok']:
            continue
        for iv, w in zip(ws[1:-1], str(r['words']).split('|')):
            rows.append(dict(word=w, new_start=iv.minTime + r['win_start'],
                             new_end=iv.maxTime + r['win_start'], gap=r['gap']))
    return pd.DataFrame(rows)


def combine(rep, gapwords, out_csv):
    """Repaired anchors + recovered words in one table for the final pass."""
    a = rep[rep.status == 'ok'].copy()
    a['source'] = 'repaired'
    g = gapwords.copy()
    g['k'] = 100000 + np.arange(len(g))
    g['gold_start'], g['gold_end'] = g.new_start, g.new_end
    g['d_start_ms'] = g['d_end_ms'] = 0.0
    g['new_dur'] = g.new_end - g.new_start
    g['status'], g['source'] = 'ok', 'recovered'
    cols = ['k', 'word', 'gold_start', 'gold_end', 'new_start', 'new_end',
            'd_start_ms', 'd_end_ms', 'new_dur', 'status', 'source']
    allw = pd.concat([a[cols], g[cols]]).sort_values('new_start').reset_index(drop=True)
    allw.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'{len(a)} repaired + {len(g)} recovered = {len(allw)} words')
    return allw


def trim_ends(allw, wav_path, drop_db=12.0, min_keep_s=0.06, flat_tail_ms=60, flat_slope=-0.3):
    """Cut each word end back to the last frame within `drop_db` of the word's own
    energy peak.  The recordings have room echo that follows words by 100-300 ms; the
    hand annotation stops where the direct speech stops.  (12 dB chosen on S4/S5 gold.)

    Exception: if the removed tail is >= flat_tail_ms long and its energy is NOT decaying
    (slope >= flat_slope dB per 10 ms frame), the tail is probably real speech (e.g. a
    final vowel with continuing voicing), so the word keeps its untrimmed end.  Echo
    decays; speech does not.  On S4/S5 this cut the share of ends that are >50 ms too early
    from 22.4/19.5 % to 18.8/17.5 % without adding late ends."""
    import librosa
    import scipy.ndimage as nd
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(r + 1e-9)
    out = allw.copy()
    out['new_end_raw'] = out['new_end']
    ne, n_kept = [], 0
    for s, e in zip(out.new_start, out.new_end):
        a, b = int(s * 100), min(int(e * 100), len(db))
        if b - a < 4:
            ne.append(e)
            continue
        seg = nd.uniform_filter1d(db[a:b], 3)
        idx = np.where(seg >= seg.max() - drop_db)[0]
        if len(idx) == 0:
            ne.append(e)
            continue
        cut = max(s + min_keep_s, min(e, (a + idx[-1] + 1) / 100))
        i1 = int(cut * 100)
        if (e - cut) * 1000 >= flat_tail_ms and b - i1 >= 3:
            if np.polyfit(np.arange(b - i1), db[i1:b], 1)[0] >= flat_slope:
                cut = e
                n_kept += 1
        ne.append(cut)
    out['new_end'] = ne
    out['gold_end'] = out['new_end']                     # downstream tables use this column
    out['new_dur'] = out['new_end'] - out['new_start']
    print(f"end trimmed by >=30 ms for {((out.new_end_raw - out.new_end) >= 0.03).sum()} of {len(out)} words; "
          f"{n_kept} words kept untrimmed because the tail was not decaying")
    return out


def trim_ends_floor(allw, wav_path, above_floor_db=10.0, floor_pct=10, min_keep_s=0.06):
    """Gentler end-trim (2026-10-19): cut each word end back only to the last frame that is
    >= (recording noise floor + above_floor_db), the floor being the `floor_pct` percentile of the
    whole recording's frame level.  The old peak-relative 12 dB rule (trim_ends) was tuned on S4/S5
    and cut real speech on other recordings (A12: 73% of ends trimmed).  On 97 held-out hand-
    corrected tokens of 5 other speakers: ends within 50 ms 0.57 (12 dB rule) -> 0.70 (this rule).
    Only trims; never extends (energy-based extension overshoots into room echo)."""
    import librosa
    import scipy.ndimage as nd
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(r + 1e-9)
    thr = np.percentile(db, floor_pct) + above_floor_db
    out = allw.copy()
    if 'new_end_raw' not in out:
        out['new_end_raw'] = out['new_end']
    ne = []
    for s, e in zip(out.new_start, out.new_end_raw):
        a, b = int(s * 100), min(int(e * 100), len(db))
        if b - a < 4:
            ne.append(e); continue
        idx = np.where(nd.uniform_filter1d(db[a:b], 3) >= thr)[0]
        ne.append(e if len(idx) == 0 else max(s + min_keep_s, min(e, (a + idx[-1] + 1) / 100)))
    out['new_end'] = ne
    out['gold_end'] = out['new_end']
    out['new_dur'] = out['new_end'] - out['new_start']
    print(f"floor trim: threshold {thr:.1f} dB; end trimmed by >=30 ms for "
          f"{((out.new_end_raw - out.new_end) >= 0.03).sum()} of {len(out)} words")
    return out
