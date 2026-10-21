"""
hybrid_labels.py -- improve the Whisper labels of a rough_transcription TextGrid.

Whisper (rough_additional.py / your notebook) gives a `refined_word` tier of intervals with
labels.  A Chinese wav2vec2-CTC model scores the KNOWN list words against every interval
(known_list_aligner.CTCScorer) and, where it assigns a word, that label replaces Whisper's.
Intervals the CTC step does not assign keep Whisper's label.  Measured on S4/S5 (against the
hand annotation) this raised label precision 92.0 -> 94.8 % (S4) and 85.0 -> 90.6 % (S5).

Run in the lenition environment (needs transformers + torch).
"""
import os, sys
import numpy as np
import pandas as pd
from textgrid import TextGrid, IntervalTier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import known_list_aligner as k


def hybrid_tier(rough_tg, wav, wordlist_csv, out_tg, cache_csv=None):
    tg = TextGrid.fromFile(rough_tg)
    tier = next(t for t in tg.tiers if t.name == 'refined_word')
    ivs = [(i.minTime, i.maxTime, i.mark.strip()) for i in tier if i.mark.strip()]
    segs = [(s, e) for s, e, w in ivs if e - s >= 0.05]
    words = k.load_wordlist(wordlist_csv)
    if cache_csv and os.path.exists(cache_csv):
        df = pd.read_csv(cache_csv)
    else:
        y, sr = k.load_audio(wav)
        scorer = k.CTCScorer()
        S = scorer.score_all(y, sr, segs, words)
        path = k.align(S, words)
        scorer.score_multi(S, words, path)
        path = k.align(S, words)
        df = k.alignment_table(path, segs, words, S)
        if cache_csv:
            df.to_csv(cache_csv, index=False, encoding='utf-8-sig')
    a = df[df.expected_idx.notna() & (df.status != 'missing')].copy()
    a['key'] = list(zip(a.start.round(3), a.end.round(3)))
    best = a.sort_values('expected_idx').groupby('key').word.first().to_dict()
    n_changed = n_ctc = 0
    out = IntervalTier('refined_word', 0, tg.maxTime)
    last = 0.0
    for s, e, w in ivs:
        new = best.get((round(s, 3), round(e, 3)))
        if new:
            n_ctc += 1
            n_changed += new != w
            w = new
        s2 = max(s, last)
        if e - s2 > 1e-4:
            out.add(s2, e, w)
            last = e
    tg2 = TextGrid(minTime=0, maxTime=tg.maxTime)
    tg2.append(out)
    tg2.write(out_tg)
    print(f'{len(ivs)} intervals; CTC assigned {n_ctc}, changed the label of {n_changed}')
    return out_tg
