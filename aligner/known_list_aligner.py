"""
known_list_aligner.py
=====================
Replaces "rough segmentation + free Whisper transcription + 3-character chunks + fuzzy
matching" with "rough segmentation + scoring against the known word list + dynamic
programming that allows merges and splits + MFA for the final boundaries".

Idea
----
1. For a read word list we already know what each speaker said and in what order
   (the wordlist CSV), so ASR does not have to guess the words. The only question is:
   "which list word does this stretch of audio sound most like?" That is a scoring
   problem, not a transcription problem.
2. A character-level Chinese CTC model (wav2vec2) scores every segment x candidate word
   (CTC log-likelihood, a few milliseconds per candidate). It will not turn the non-word
   "shi zhi he" into a real phrase the way a free-running recognizer does.
3. Dynamic programming aligns the segment sequence to the list sequence, allowing:
     MATCH   1 segment = 1 word
     MERGE   1 segment = the same word read twice with no pause in between (most of the
             words the first speaker's rough pass missed were first repetitions)
     SPLIT   2 segments = 1 word (the VAD cut at a voiceless fricative or stop closure)
     INSERT  a segment that matches no word (cough, slip, talking)
     DELETE  a list word that was not read or not found
4. The rough segmentation only finds *where* the words are. Final word and phone
   boundaries come from MFA (one short clip per word, word-level pronunciations that force
   the list's reading). Hand-tuned thresholds are not used to fix boundaries.
5. Hand-annotated TextGrids are used as the gold standard to measure accuracy and tune
   the parameters below.

Dependencies
------------
  pip install torch transformers pypinyin textgrid librosa soundfile pandas
  (optional) pip install silero-vad

Usage (in a notebook)
---------------------
  from known_list_aligner import *
  y, sr = load_audio(AUDIO_PATH)
  segs  = segments_from_textgrid(SEG_TG)          # an existing rough segmentation
  # or: segs = segments_from_silero(y, sr)
  words = load_wordlist(WORDLIST_CSV)
  scorer = CTCScorer()                             # downloads the model on first run
  S = scorer.score_all(y, sr, segs, words)
  path = align(S, words)                           # pass 1: 1 segment ~ 1 word
  scorer.score_multi(S, words, path)               # pass 2: segments holding 2-6 words
  path = align(S, words)                           #   (fast speech merges words)
  df = alignment_table(path, segs, words, S)
  df.to_csv(OUT_CSV, index=False)
  write_rough_textgrid(df, segs, len(y)/sr, OUT_TG)
  export_mfa_clips(df, y, sr, MFA_CORPUS, prefix='S1')
  write_mfa_dictionary(words, BASE_DICT, MFA_DICT)
  # terminal: mfa align MFA_CORPUS MFA_DICT mandarin_mfa MFA_OUT --clean --beam 100 --retry_beam 400
  tg = import_mfa_results(MFA_OUT, MFA_CORPUS, df, len(y)/sr, FINAL_TG)
  evaluate(GOLD_TG, FINAL_TG)                      # when a gold standard exists
"""

import os
import re
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters (run with the defaults first, then tune against the gold standard with evaluate())
# ---------------------------------------------------------------------------
SCORE_PAD_S   = 0.10   # seconds of extra audio on each side when scoring (so the model hears fricative onsets)
MAX_SPLIT_GAP = 0.30   # only consider "2 segments = 1 word" if the gap between them is below this
MATCH_BONUS   = 4.0    # reward for each matched word
INS_PEN       = -1.0   # cost of skipping a segment
DEL_PEN       = -1.5   # cost of skipping a list word
SPLIT_PEN     = -0.5   # extra cost of a split
SCORE_FLOOR   = -15.0  # floor for the per-character normalized score
OK_THRESH     = -2.0   # per-character score >= this -> ok
CHECK_THRESH  = -4.0   # between the two -> check; lower -> review
MFA_PAD_S     = 0.15   # padding (s) around each clip exported for MFA


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def load_audio(path, sr=16000):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def load_wordlist(csv_path):
    """Return the word list sorted by `order`. The CSV needs a `word` column and ideally an `order` column."""
    df = pd.read_csv(csv_path)
    if 'order' in df.columns:
        df = df.sort_values('order')
    return [str(w).strip() for w in df['word'].tolist()]


def segments_from_textgrid(tg_path, tier_name='speech', label='speech',
                           min_dur=0.05):
    """Read an existing segmentation TextGrid. Segments shorter than 0.4 s are kept:
    a quickly read three-syllable word can easily be shorter than that, and dropping
    them would systematically remove data from the fast condition."""
    from textgrid import TextGrid
    tg = TextGrid.fromFile(tg_path)
    tier = next(t for t in tg.tiers if t.name.lower() == tier_name)
    return [(iv.minTime, iv.maxTime) for iv in tier
            if iv.mark == label and iv.maxTime - iv.minTime >= min_dur]


def segments_from_silero(y, sr=16000, threshold=0.4, min_silence_ms=120,
                         min_speech_ms=80, pad_ms=40):
    """Alternative rough segmentation: Silero VAD. It is insensitive to the onset of
    voiceless fricatives, so scoring and MFA both add padding; the boundaries
    themselves are not used for measurement."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    if sr != 16000:
        raise ValueError('Silero needs 16 kHz audio')
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y), model, sampling_rate=16000,
                               threshold=threshold,
                               min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms,
                               speech_pad_ms=pad_ms, return_seconds=True)
    return [(t['start'], t['end']) for t in ts]


# ---------------------------------------------------------------------------
# Pinyin helpers
# ---------------------------------------------------------------------------
def word_tone3(word):
    """Per-character readings taken at the word level (with context), e.g. 山楂 -> ['shan1', 'zha1']."""
    from pypinyin import pinyin, Style
    return [p[0] for p in pinyin(word, style=Style.TONE3,
                                 neutral_tone_with_five=True, errors='ignore')]


def toneless(syl):
    return re.sub(r'\d', '', syl)


# ---------------------------------------------------------------------------
# CTC scorer
# ---------------------------------------------------------------------------
class CTCScorer:
    """Score (segment, candidate word) pairs with a Chinese wav2vec2-CTC model,
    at the SYLLABLE level.

    The model outputs characters. A character-level score punishes homophones:
    for the non-word 市值盒 the model hears 是只和, and log P(市值盒) is terrible
    even though the audio is a perfect "shi zhi he". So every character's
    probability is pooled (log-sum-exp) into its toneless pinyin syllable, and
    candidates are scored as syllable sequences. Non-words, rare characters
    (痧/楂/衩) and homophones then all score correctly.

    score = [log P(candidate syllables | audio) - log P(best free path)] / n_syll
    ~0 = as good a fit as the model's own best guess; more negative = worse.
    """

    def __init__(self, model_name='jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
                 device=None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from pypinyin import pinyin, Style
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
        self.device = device or 'cpu'
        self.model.to(self.device)
        vocab = self.processor.tokenizer.get_vocab()
        blank = self.processor.tokenizer.pad_token_id
        # class 0 = blank, class 1 = everything non-Chinese, 2.. = syllables
        self.syl2cls, cls_of_tok = {}, {}
        for tok, tid in vocab.items():
            if tid == blank:
                cls_of_tok[tid] = 0
            elif len(tok) == 1 and '\u4e00' <= tok <= '\u9fff':
                syl = toneless(pinyin(tok, style=Style.NORMAL)[0][0])
                cls_of_tok[tid] = self.syl2cls.setdefault(syl, len(self.syl2cls) + 2)
            else:
                cls_of_tok[tid] = 1
        self.n_cls = len(self.syl2cls) + 2
        V = self.model.config.vocab_size
        self.tok2cls = torch.tensor([cls_of_tok.get(t, 1) for t in range(V)])
        self._word_cache = {}

    def map_word(self, word):
        """word -> (list of syllable class ids, list of (syllable, substitute))."""
        if word in self._word_cache:
            return self._word_cache[word]
        from difflib import SequenceMatcher
        ids, replaced = [], []
        for syl in (toneless(x) for x in word_tone3(word)):
            if syl in self.syl2cls:
                ids.append(self.syl2cls[syl])
            else:   # practically never happens; nearest-spelled syllable
                best = max(self.syl2cls, key=lambda q: SequenceMatcher(None, syl, q).ratio())
                ids.append(self.syl2cls[best]); replaced.append((syl, best))
        self._word_cache[word] = (ids, replaced)
        return ids, replaced

    def emissions(self, clip, sr=16000):
        """(T, n_cls) log-probs pooled over characters of the same syllable."""
        torch = self.torch
        min_len = int(0.1 * sr)
        if len(clip) < min_len:
            clip = np.pad(clip, (0, min_len - len(clip)))
        if not np.any(clip):
            clip = clip + 1e-6 * np.random.default_rng(0).standard_normal(len(clip)).astype(np.float32)
        inp = self.processor(clip, sampling_rate=sr, return_tensors='pt').input_values
        with torch.no_grad():
            logits = self.model(inp.to(self.device)).logits[0].float().cpu()
        lp = torch.log_softmax(logits, dim=-1)                       # (T, V)
        T = lp.shape[0]
        out = torch.full((T, self.n_cls), -1e9)
        idx = self.tok2cls.unsqueeze(0).expand(T, -1)
        out = out.scatter_reduce(1, idx, lp, reduce='amax', include_self=True)
        # log-sum-exp per class: amax + log(sum exp(x - amax))
        ex = torch.exp(lp - out.gather(1, idx))
        ssum = torch.zeros((T, self.n_cls)).scatter_add(1, idx, ex)
        return out + torch.log(ssum.clamp_min(1e-30))

    def _ctc_scores(self, lp, greedy, cand_ids):
        torch = self.torch
        T, N = lp.shape[0], len(cand_ids)
        L = max(len(c) for c in cand_ids)
        targets = torch.zeros((N, L), dtype=torch.long)
        tlen = torch.tensor([len(c) for c in cand_ids], dtype=torch.long)
        for n, c in enumerate(cand_ids):
            targets[n, :len(c)] = torch.tensor(c)
        logp = lp.unsqueeze(1).expand(T, N, lp.shape[1]).contiguous()
        loss = torch.nn.functional.ctc_loss(
            logp, targets, torch.full((N,), T, dtype=torch.long), tlen,
            blank=0, reduction='none', zero_infinity=False)
        ll = -loss.numpy()
        ll[~np.isfinite(ll)] = -1e4          # clip too short for this many syllables
        norm = (ll - greedy) / np.maximum(tlen.numpy(), 1)
        return np.maximum(norm, SCORE_FLOOR)

    def transcribe(self, y, sr, seg):
        """Debug helper: what syllables the model hears in one segment."""
        pad = int(SCORE_PAD_S * sr)
        s, e = seg
        lp = self.emissions(y[max(0, int(s * sr) - pad): int(e * sr) + pad], sr)
        inv = {v: k for k, v in self.syl2cls.items()}
        best = lp.argmax(1).tolist()
        out, prev = [], None
        for c in best:
            if c != prev and c >= 2:
                out.append(inv[c])
            prev = c
        return ' '.join(out)

    def score_all(self, y, sr, segs, words, verbose=True):
        """Returns dict with
            uniq   : unique words
            single : (n_seg, n_uniq)  segment = word
            double : (n_seg, n_uniq)  segment = word read twice
            pair   : {i: (n_uniq,)}   segments i + i+1 = word
            mapped : word -> (syllable ids, substitutions)
        """
        dur = len(y) / sr
        beyond = [k for k, (s_, e_) in enumerate(segs) if s_ >= dur - 0.02]
        if beyond:
            print(f'WARNING: {len(beyond)} of {len(segs)} segments start after the audio ends '
                  f'({dur:.1f} s). Is the segmentation TextGrid from this same WAV?')
        uniq = list(dict.fromkeys(words))
        mapped = {w: self.map_word(w) for w in uniq}
        cand_ids = [mapped[w][0] for w in uniq] + [mapped[w][0] * 2 for w in uniq]
        U, n = len(uniq), len(segs)
        single = np.full((n, U), SCORE_FLOOR)
        double = np.full((n, U), SCORE_FLOOR)
        pair = {}
        pad = int(SCORE_PAD_S * sr)

        def clip_of(s, e):
            return y[max(0, int(s * sr) - pad): min(len(y), int(e * sr) + pad)]

        cache = []
        for i, (s, e) in enumerate(segs):
            lp = self.emissions(clip_of(s, e), sr)
            greedy = lp.max(dim=1).values.sum().item()
            cache.append((lp, greedy))             # reused by score_multi()
            sc = self._ctc_scores(lp, greedy, cand_ids)
            single[i], double[i] = sc[:U], sc[U:]
            if i + 1 < n and segs[i + 1][0] - e <= MAX_SPLIT_GAP:
                lp2 = self.emissions(clip_of(s, segs[i + 1][1]), sr)
                pair[i] = self._ctc_scores(lp2, lp2.max(dim=1).values.sum().item(),
                                           cand_ids[:U])
            if verbose and (i + 1) % 100 == 0:
                print(f'  scored {i + 1}/{n} segments')
        bad = {w: r for w, (_, r) in mapped.items() if r}
        if bad and verbose:
            print('Syllables not in model vocabulary (nearest used):', bad)
        return dict(uniq=uniq, single=single, double=double, pair=pair,
                    mapped=mapped, multi={}, _cache=cache)

    def score_multi(self, S, words, path, max_words=6, band=12, verbose=True):
        """Second pass for fast speech: score each segment against runs of
        2..max_words CONSECUTIVE list words near where pass 1 placed it."""
        n = len(S['_cache'])
        anchor = _segment_anchors(path, n, len(words))
        mapped = S['mapped']
        for i in range(n):
            keys, ids = [], []
            lo = max(0, anchor[i] - band)
            hi = min(len(words), anchor[i] + band)
            for j0 in range(lo, hi):
                for k in range(2, max_words + 1):
                    if j0 + k > len(words):
                        break
                    keys.append((i, j0, k))
                    ids.append([c for w in words[j0:j0 + k] for c in mapped[w][0]])
            if not ids:
                continue
            lp, greedy = S['_cache'][i]
            for key, v in zip(keys, self._ctc_scores(lp, greedy, ids)):
                S['multi'][key] = v
            if verbose and (i + 1) % 200 == 0:
                print(f'  multi-scored {i + 1}/{n} segments')
        return S


def _segment_anchors(path, n_seg, n_words):
    """Estimated list position for every segment (interpolated for INSERTs)."""
    anchor = np.full(n_seg, -1)
    for move, si, wj, _ in path:
        for i in si:
            if wj:
                anchor[i] = wj[0]
    known = np.where(anchor >= 0)[0]
    if len(known) == 0:
        return np.linspace(0, n_words - 1, n_seg).astype(int)
    return np.interp(np.arange(n_seg), known, anchor[known]).astype(int)


# ---------------------------------------------------------------------------
# Dynamic-programming alignment
# ---------------------------------------------------------------------------
MOVES = {0: 'MATCH', 1: 'MERGE', 2: 'SPLIT', 3: 'INSERT', 4: 'DELETE', 5: 'MULTI'}


def align(S, words, match_bonus=MATCH_BONUS, ins_pen=INS_PEN,
          del_pen=DEL_PEN, split_pen=SPLIT_PEN):
    """Global alignment of the segment sequence to the word-list sequence. Returns [(move, seg_idxs, word_idxs, score)]."""
    uidx = {w: k for k, w in enumerate(S['uniq'])}
    wid = np.array([uidx[w] for w in words])
    single, double, pair = S['single'], S['double'], S['pair']
    multi = S.get('multi', {})
    # index multi scores by (seg, last_word_idx) -> list of (k, score)
    multi_end = {}
    for (i0, j0, k), v in multi.items():
        multi_end.setdefault((i0, j0 + k - 1), []).append((k, v))
    n, m = single.shape[0], len(words)
    NEG = -1e18
    dp = np.full((n + 1, m + 1), NEG)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    multi_k = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        row_single = single[i - 1] if i >= 1 else None
        row_double = double[i - 1] if i >= 1 else None
        row_pair = pair.get(i - 2) if i >= 2 else None
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, -1
            if i >= 1 and j >= 1:                                   # MATCH
                v = dp[i - 1, j - 1] + match_bonus + row_single[wid[j - 1]]
                if v > best: best, mv = v, 0
            if i >= 1 and j >= 2 and wid[j - 1] == wid[j - 2]:      # MERGE
                v = dp[i - 1, j - 2] + 2 * (match_bonus + row_double[wid[j - 1]])
                if v > best: best, mv = v, 1
            if i >= 1 and multi_end:                                # MULTI
                for k, sc in multi_end.get((i - 1, j - 1), ()):
                    if j - k >= 0:
                        # each word earns the bonus, each word pays the fit
                        v = dp[i - 1, j - k] + k * (match_bonus + sc)
                        if v > best: best, mv, best_k = v, 5, k
                if mv == 5:
                    multi_k[i, j] = best_k
            if row_pair is not None and j >= 1:                     # SPLIT
                v = dp[i - 2, j - 1] + match_bonus + split_pen + row_pair[wid[j - 1]]
                if v > best: best, mv = v, 2
            if i >= 1:                                              # INSERT
                v = dp[i - 1, j] + ins_pen
                if v > best: best, mv = v, 3
            if j >= 1:                                              # DELETE
                v = dp[i, j - 1] + del_pen
                if v > best: best, mv = v, 4
            dp[i, j], bt[i, j] = best, mv
    # backtrace
    path, i, j = [], n, m
    while i > 0 or j > 0:
        mv = bt[i, j]
        if mv == 0:
            path.append(('MATCH', [i - 1], [j - 1], single[i - 1, wid[j - 1]])); i -= 1; j -= 1
        elif mv == 1:
            path.append(('MERGE', [i - 1], [j - 2, j - 1], double[i - 1, wid[j - 1]])); i -= 1; j -= 2
        elif mv == 2:
            path.append(('SPLIT', [i - 2, i - 1], [j - 1], pair[i - 2][wid[j - 1]])); i -= 2; j -= 1
        elif mv == 5:
            k = multi_k[i, j]
            sc = dict(multi_end[(i - 1, j - 1)])[k]
            path.append(('MULTI', [i - 1], list(range(j - k, j)), sc)); i -= 1; j -= k
        elif mv == 3:
            path.append(('INSERT', [i - 1], [], np.nan)); i -= 1
        else:
            path.append(('DELETE', [], [j - 1], np.nan)); j -= 1
    return path[::-1]


def _status(move, score):
    if move in ('DELETE',):
        return 'missing'
    if move == 'INSERT':
        return 'unassigned_segment'
    if score >= OK_THRESH:
        st = 'ok'
    elif score >= CHECK_THRESH:
        st = 'check'
    else:
        st = 'review'
    if move in ('MERGE', 'SPLIT', 'MULTI') and st == 'ok':
        st = 'ok_' + move.lower()
    return st


def alignment_table(path, segs, words, S=None):
    """One row per list position (plus unassigned segments) with rough boundaries, the move, and a confidence."""
    rows, group = [], 0
    for move, si, wj, sc in path:
        start = segs[si[0]][0] if si else np.nan
        end = segs[si[-1]][1] if si else np.nan
        group = group + 1 if (si and wj) else group
        if not wj:
            rows.append(dict(expected_idx=np.nan, word='', move=move,
                             seg_idx=si[0], group=np.nan, start=start, end=end,
                             score=sc, status=_status(move, sc)))
            continue
        for k, j in enumerate(wj):
            rows.append(dict(expected_idx=j, word=words[j], move=move,
                             seg_idx=si[0] if si else np.nan,
                             group=group if si else np.nan,
                             n_words_in_group=len(wj), pos_in_group=k,
                             start=start, end=end, score=sc,
                             status=_status(move, sc)))
    df = pd.DataFrame(rows)
    if S is not None:
        rep = {w: ','.join(f'{a}->{b}' for a, b in r) for w, (_, r) in S['mapped'].items() if r}
        df['homophone_sub'] = df['word'].map(rep).fillna('')
    return df


def summarize(df, n_expected):
    w = df[df['expected_idx'].notna()]
    found = w[w['status'] != 'missing']
    print(f'list positions: {n_expected}')
    print(f'found: {len(found)} ({len(found) / n_expected:.1%})    '
          f'missing: {(w.status == "missing").sum()}    '
          f'unassigned segments: {(df.status == "unassigned_segment").sum()}')
    print(w['status'].value_counts().to_string())
    # Each word is read twice in a row by design: both found and adjacent -> consistency check
    print('\nmove counts:\n' + df['move'].value_counts().to_string())


# ---------------------------------------------------------------------------
# Output: rough TextGrid (for checking)
# ---------------------------------------------------------------------------
def write_rough_textgrid(df, segs, duration, out_path):
    from textgrid import TextGrid, IntervalTier
    tg = TextGrid(minTime=0, maxTime=duration)
    t_seg = IntervalTier('segment', 0, duration)
    t_word = IntervalTier('assigned', 0, duration)
    t_st = IntervalTier('status', 0, duration)
    for s, e in segs:
        t_seg.add(s, e, 'seg')
    seen = set()
    for _, r in df.iterrows():
        if pd.isna(r['start']):
            continue
        key = (r['start'], r['end'])
        if key in seen:
            continue
        seen.add(key)
        same = df[(df['start'] == r['start']) & (df['end'] == r['end'])]
        label = '+'.join(w for w in same['word'] if w) or '(none)'
        t_word.add(r['start'], r['end'], label)
        t_st.add(r['start'], r['end'], f"{r['status']} {r['score']:.1f}"
                 if not pd.isna(r['score']) else r['status'])
    for t in (t_seg, t_word, t_st):
        tg.append(t)
    tg.write(out_path)


# ---------------------------------------------------------------------------
# MFA: one short clip per word + word-level pronunciations (forces the list reading)
# ---------------------------------------------------------------------------
def export_mfa_clips(df, y, sr, corpus_dir, prefix):
    """Export a short wav + .lab for each matched group (1 word, or 2 words read back to back).
    Padding never extends past the midpoint to the neighbouring group, so a neighbour is not pulled in."""
    import soundfile as sf
    os.makedirs(corpus_dir, exist_ok=True)
    g = (df[df['group'].notna() & df['status'].ne('missing')]
         .groupby('group', sort=True)
         .agg(start=('start', 'first'), end=('end', 'first'),
              words=('word', list), idxs=('expected_idx', list))
         .reset_index().sort_values('start').reset_index(drop=True))
    manifest = []
    for k, r in g.iterrows():
        lo = r['start'] - MFA_PAD_S
        hi = r['end'] + MFA_PAD_S
        if k > 0:
            lo = max(lo, (g.loc[k - 1, 'end'] + r['start']) / 2)
        if k + 1 < len(g):
            hi = min(hi, (r['end'] + g.loc[k + 1, 'start']) / 2)
        lo, hi = max(0.0, lo), min(len(y) / sr, hi)
        cid = f"{prefix}_{int(r['group']):05d}"
        sf.write(os.path.join(corpus_dir, cid + '.wav'), y[int(lo * sr):int(hi * sr)], sr)
        with open(os.path.join(corpus_dir, cid + '.lab'), 'w', encoding='utf-8') as f:
            f.write(' '.join(r['words']))      # each word is one token
        manifest.append(dict(clip=cid, offset=lo, words='|'.join(r['words']),
                             expected_idx='|'.join(str(int(x)) for x in r['idxs'])))
    pd.DataFrame(manifest).to_csv(os.path.join(corpus_dir, f'{prefix}_manifest.csv'),
                                  index=False, encoding='utf-8')
    print(f'exported {len(manifest)} clips -> {corpus_dir}')


def _read_mfa_dict(path):
    """Keep every line (heteronyms have several entries). Returns the lines and {character: [phones, ...]}."""
    lines, prons = [], {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            lines.append(line)
            parts = line.split('\t')
            prons.setdefault(parts[0], []).append(parts[-1].strip())
    return lines, prons


def write_mfa_dictionary(words, base_dict, out_dict):
    """Append to the base dictionary one word-level pronunciation per list word, built from
    "context-aware pinyin -> the MFA phones of a homophonous character with a single reading".
    That way (1) there are no `spn` tokens from missing characters, and (2) heteronyms
    (长/刹/什/血) are forced to the list reading instead of letting MFA pick by acoustics,
    which in a lenition study would confuse pronunciation choice with weakening."""
    from pypinyin import pinyin, Style
    lines, prons = _read_mfa_dict(base_dict)
    # syllable -> phones: only use characters with a single reading in both the dictionary and pypinyin
    syl2ph = {}
    for ch, ps in prons.items():
        if len(ch) != 1 or len(set(ps)) != 1:
            continue
        hets = pinyin(ch, style=Style.TONE3, heteronym=True,
                      neutral_tone_with_five=True)[0]
        if len(hets) == 1:
            syl2ph.setdefault(hets[0], ps[0])
    added, problems = [], []
    for w in dict.fromkeys(words):
        sylls = word_tone3(w)
        phones = []
        for k, syl in enumerate(sylls):
            ph = syl2ph.get(syl)
            if ph is None and syl.endswith('5'):
                # neutral tone: not always in the dictionary -> fall back to the base tone and record it
                ch = w[k]
                cit = pinyin(ch, style=Style.TONE3)[0][0]
                ph = syl2ph.get(cit)
                problems.append((w, syl, 'neutral->citation ' + cit if ph else 'not found'))
            elif ph is None:
                problems.append((w, syl, 'not found'))
            phones.append(ph)
        if any(p is None for p in phones):
            continue
        # same format as the entries that already worked
        added.append(f"{w}\t0.99\t0.25\t1.0\t1.0\t{' '.join(phones)}")
    with open(out_dict, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
        for line in added:
            f.write(line + '\n')
    print(f'base dictionary {len(lines)} lines; added {len(added)} word-level pronunciations -> {out_dict}')
    if problems:
        print('syllables that need a manual check:')
        for p in problems:
            print('  ', p)
    return added, problems


def import_mfa_results(mfa_out, corpus_dir, df, duration, out_tg, prefix=None):
    """Stitch the per-clip MFA results back onto the full timeline and write the final TextGrid."""
    from textgrid import TextGrid, IntervalTier
    man_files = [f for f in os.listdir(corpus_dir) if f.endswith('_manifest.csv')]
    if prefix:
        man_files = [f for f in man_files if f.startswith(prefix)]
    man = pd.concat([pd.read_csv(os.path.join(corpus_dir, f)) for f in man_files])
    status = df.dropna(subset=['expected_idx']).set_index('expected_idx')['status'].to_dict()
    words_iv, phones_iv, rows = [], [], []
    for _, r in man.iterrows():
        p = os.path.join(mfa_out, r['clip'] + '.TextGrid')
        idxs = [int(x) for x in str(r['expected_idx']).split('|')]
        if not os.path.exists(p):
            for j in idxs:
                rows.append(dict(expected_idx=j, mfa='no_output'))
            continue
        tg = TextGrid.fromFile(p)
        wt = next(t for t in tg.tiers if 'word' in t.name.lower())
        pt = next(t for t in tg.tiers if 'phone' in t.name.lower())
        ws = [iv for iv in wt if iv.mark.strip()]
        off = r['offset']
        for k, iv in enumerate(ws):
            j = idxs[k] if k < len(idxs) else None
            phs = [p_ for p_ in pt if p_.mark.strip() and p_.minTime >= iv.minTime - 1e-6
                   and p_.maxTime <= iv.maxTime + 1e-6]
            spn = all(p_.mark == 'spn' for p_ in phs) if phs else True
            words_iv.append((iv.minTime + off, iv.maxTime + off, iv.mark))
            for p_ in phs:
                phones_iv.append((p_.minTime + off, p_.maxTime + off, p_.mark))
            rows.append(dict(expected_idx=j, word=iv.mark, start=iv.minTime + off,
                             end=iv.maxTime + off, mfa='spn' if spn else 'ok',
                             align_status=status.get(j, '')))
        if len(ws) != len(idxs):
            rows.append(dict(expected_idx=idxs[0], mfa=f'word_count {len(ws)}!={len(idxs)}'))
    tg = TextGrid(minTime=0, maxTime=duration)
    tw, tp = IntervalTier('words', 0, duration), IntervalTier('phones', 0, duration)
    for tier, ivs in ((tw, words_iv), (tp, phones_iv)):
        last = 0.0
        for s, e, lab in sorted(ivs):
            s = max(s, last)
            if e - s > 1e-4:
                tier.add(s, e, lab)
                last = e
        tg.append(tier)
    tg.write(out_tg)
    res = pd.DataFrame(rows)
    res.to_csv(out_tg.replace('.TextGrid', '_tokens.csv'), index=False, encoding='utf-8')
    print(f'final TextGrid -> {out_tg}')
    print(res['mfa'].value_counts().to_string())
    return res


# ---------------------------------------------------------------------------
# Evaluation: compare with the hand-made TextGrid
# ---------------------------------------------------------------------------
def _label_key(mark):
    """Hand labels may be pinyin ('zha jiang mian') or characters; normalise to toneless pinyin."""
    mark = mark.strip().lower()
    if re.search(r'[\u4e00-\u9fff]', mark):
        return ' '.join(toneless(s) for s in word_tone3(mark))
    return ' '.join(toneless(s) for s in mark.replace('v', 'ü').split())


def evaluate(gold_tg, pred_tg, gold_tier='word', pred_tier='words'):
    from textgrid import TextGrid
    g = TextGrid.fromFile(gold_tg)
    p = TextGrid.fromFile(pred_tg)
    gt = next(t for t in g.tiers if t.name == gold_tier)
    pt = next(t for t in p.tiers if t.name == pred_tier)
    gold = [iv for iv in gt if iv.mark.strip()]
    pred = [iv for iv in pt if iv.mark.strip()]
    rows = []
    for iv in gold:
        best, ov = None, 0.0
        for q in pred:
            o = min(iv.maxTime, q.maxTime) - max(iv.minTime, q.minTime)
            if o > ov:
                best, ov = q, o
        if best is None:
            rows.append(dict(gold=iv.mark, found=False))
            continue
        rows.append(dict(gold=iv.mark, pred=best.mark, found=True,
                         label_ok=_label_key(iv.mark) == _label_key(best.mark),
                         d_on=1000 * (best.minTime - iv.minTime),
                         d_off=1000 * (best.maxTime - iv.maxTime)))
    r = pd.DataFrame(rows)
    f = r[r['found']]
    print(f'gold words {len(r)}; overlapping {len(f)} ({len(f) / len(r):.1%}); '
          f'label correct {f.label_ok.mean():.1%}')
    for col, name in (('d_on', 'onset'), ('d_off', 'offset')):
        a = f[col].abs()
        print(f'{name}: median error {a.median():.0f} ms; <=20 ms {(a <= 20).mean():.1%}; '
              f'<=50 ms {(a <= 50).mean():.1%}; mean bias {f[col].mean():+.0f} ms')
    return r
