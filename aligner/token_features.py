"""
token_features.py -- one row per word token with features for
  (1) disfluency flags (hesitation / lengthening, to exclude from duration analyses)
  (2) a boundary-confidence model

  build_tokens(final_tg, final_csv, wav, recording)  ->  DataFrame

Everything here is computed from the pipeline output (final TextGrid + final_words.csv) and the audio.
Deliberately NOT used for confidence: the aligner's likelihood / poor_fit flag, because a lenited
retroflex fits the retroflex model badly -- using it would drop exactly the lenited tokens.
"""
import re
import numpy as np
import pandas as pd
from textgrid import TextGrid

TONE = re.compile(r'[˥˧˨˩˦]')
NASAL_CODA = {'n', 'ŋ'}


def split_syllables(phones):
    """phones: [(start, end, label)] of one word -> list of syllables (list of phone tuples).
    A syllable ends at its tone-marked nucleus plus an optional nasal coda."""
    syls, cur, i = [], [], 0
    n = len(phones)
    while i < n:
        cur.append(phones[i])
        if TONE.search(phones[i][2]):
            if i + 1 < n and phones[i + 1][2] in NASAL_CODA:
                nxt_is_nucleus = i + 2 < n and TONE.search(phones[i + 2][2])
                if phones[i + 1][2] == 'ŋ' or not nxt_is_nucleus:
                    cur.append(phones[i + 1]); i += 1
            if i + 1 < n and phones[i + 1][2] == 'ɻ' and not (i + 2 < n and TONE.search(phones[i + 2][2])):
                cur.append(phones[i + 1]); i += 1             # erhua coda; an ɻ followed by a nucleus is the NEXT syllable's onset
                                                             # (日 ri, or a lenited retroflex), fixed 2026-10-20
            syls.append(cur); cur = []
        i += 1
    if cur:
        if syls:
            syls[-1] += cur
        else:
            syls.append(cur)
    return syls


def _speech_mask(y, sr=16000, hop=160, above_db=12.0):
    import librosa
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=hop)[0]
    db = 20 * np.log10(r + 1e-9)
    floor = np.percentile(db, 10)
    return db, floor


def build_tokens(final_tg, final_csv, wav, recording):
    import librosa
    f = pd.read_csv(final_csv)
    f = f[f.status == 'ok'].sort_values('gold_start').reset_index(drop=True)
    tg = TextGrid.fromFile(final_tg)
    pt = next(t for t in tg.tiers if t.name == 'phones')
    ph = [(i.minTime, i.maxTime, i.mark) for i in pt if i.mark.strip()]
    ph_starts = np.array([p[0] for p in ph])

    t = pd.DataFrame({'recording': recording, 'word': f.word, 'start': f.gold_start, 'end': f.gold_end,
                      'mfa_start': f.mfa_start, 'mfa_end': f.mfa_end, 'source': f.source,
                      'repair_move_start_ms': f.d_start_ms, 'repair_move_end_ms': f.d_end_ms,
                      'free_vs_word_start_ms': 1000 * (f.mfa_start - f.gold_start),
                      'free_vs_word_end_ms': 1000 * (f.mfa_end - f.gold_end),
                      'first_phone_ms': f.first_phone_ms, 'n_phones': f.n_phones,
                      'erhua': f.erhua.fillna(False).astype(bool), 'old_flags': f['flags'].fillna(''),
                      'fit_z': f.fit_z})
    t['dur'] = t.end - t.start
    # --- syllables from the phones tier
    sd = []
    for r in t.itertuples():
        a = np.searchsorted(ph_starts, r.start - 1e-4)
        b = np.searchsorted(ph_starts, r.end - 1e-4)
        P = ph[a:b]
        S = split_syllables(P)
        want = len(re.findall(r'[一-鿿]', r.word)) - r.word.count('儿')   # final and mid-word erhua
        sd.append([1000 * (s[-1][1] - s[0][0]) for s in S] if len(S) == want and want in (3, 4) else [])
    t['syl_ms'] = sd
    # --- per speaker/word reference durations (median over this speaker's reads of the same word)
    g = t.groupby('word')
    t['n_reads'] = g.dur.transform('size')
    t['dur_ref'] = g.dur.transform('median')
    # --- speech-rate block: the list is read twice (first half / second half = slow / fast block).
    # Which half is slow is inferred from durations; each token's durations are divided by its block's
    # scale so that slow tokens are not flagged as 'long' just for being slow.
    t['block'] = (np.arange(len(t)) >= len(t) / 2).astype(int)
    m = t.groupby(['word', 'block']).dur.mean().unstack()
    both = m.dropna()
    first_longer = float((both[0] > both[1]).mean()) if len(both) else 0.5
    scale = {b: float(np.median(both[b] / both.mean(axis=1))) for b in (0, 1)} if len(both) else {0: 1.0, 1: 1.0}
    t['condition'] = np.where(t.block == (0 if first_longer > 0.5 else 1), 'slow', 'fast')
    t['cond_scale'] = t.block.map(scale)
    t['dur_ratio'] = np.where(t.n_reads >= 3, t.dur / (t.dur_ref * t.cond_scale), np.nan)
    ratios = []
    for w, idx in g.groups.items():
        sub = t.loc[idx]; L = [x for x in sub.syl_ms if len(x)]
        ref = np.median(np.array(L), axis=0) if len(L) >= 3 else None
        for i in idx:
            x = t.at[i, 'syl_ms']
            ratios.append((i, list(np.array(x) / ref / t.at[i, 'cond_scale']) if (ref is not None and len(x) == len(ref)) else []))
    sr_ = dict(ratios)
    t['syl_ratio'] = [sr_.get(i, []) for i in t.index]
    t['syl_ratio_max'] = [max(x) if len(x) else np.nan for x in t.syl_ratio]
    t['syl_ratio_argmax'] = [int(np.argmax(x)) + 1 if len(x) else np.nan for x in t.syl_ratio]
    # --- repetition structure
    prev_same = t.word.eq(t.word.shift(1))
    t['rep'] = np.where(prev_same & ~t.word.shift(1).eq(t.word.shift(2)), 2, 1)
    d2 = t.dur.shift(-1)
    t['pair_asym'] = np.where((t.rep == 1) & t.word.eq(t.word.shift(-1)), (t.dur - d2).abs() / ((t.dur + d2) / 2), np.nan)
    t['pair_asym'] = t.pair_asym.fillna(pd.Series(np.where((t.rep == 2), ((t.dur - t.dur.shift(1)).abs() / ((t.dur + t.dur.shift(1)) / 2)), np.nan)))
    t['extra_rep'] = [int(((t.word == w) & (abs(t.start - s) < 6)).sum() > 2) for w, s in zip(t.word, t.start)]
    # --- audio: pause before, non-lexical sound before, SNR
    y, _ = librosa.load(wav, sr=16000, mono=True)
    db, floor = _speech_mask(y)
    t['pre_gap'] = t.start - t.end.shift(1)
    ig = t.loc[t.rep == 1, 'pre_gap']
    med, mad = ig.median(), (ig - ig.median()).abs().median() + 1e-6
    t['pre_gap_z'] = np.where(t.rep == 1, (t.pre_gap - med) / (1.4826 * mad), np.nan)
    sp, snr = [], []
    for r in t.itertuples():
        a, b = int((r.start - r.pre_gap) * 100) if r.pre_gap == r.pre_gap else int(r.start * 100) - 50, int(r.start * 100)
        a, b = a + 5, b - 3                                   # keep away from the word edges
        run = best = 0
        for v in (db[max(a, 0):max(b, 0)] > floor + 12):
            run = run + 1 if v else 0
            best = max(best, run)
        sp.append(best / 100)
        i0, i1 = int(r.start * 100), min(int(r.end * 100), len(db))
        snr.append(float(db[i0:i1].max() - floor) if i1 > i0 else np.nan)
    t['sound_before_s'] = np.where(t.rep == 1, sp, 0.0)
    t['snr_db'] = snr
    return t


def disfluency_flags(t, long_word=1.5, lengthened_syl=1.7, pause_z=3.5, pause_min_s=1.5,
                     sound_before_s=0.15, pair_asym=0.4):
    """Candidate hesitation / lengthening flags (thresholds are first guesses; calibrate by ear)."""
    f = pd.DataFrame(index=t.index)
    f['long_word'] = t.dur_ratio >= long_word
    f['lengthened_syllable'] = t.syl_ratio_max >= lengthened_syl
    f['pause_before'] = (t.pre_gap_z >= pause_z) & (t.pre_gap >= pause_min_s)
    f['sound_before'] = t.sound_before_s >= sound_before_s
    f['extra_repetition'] = t.extra_rep == 1
    f['rep_asymmetry'] = t.pair_asym >= pair_asym
    f['any'] = f.any(axis=1)
    return f
