# mandarin-wordlist-aligner

Code for annotating long recordings of a read Mandarin word list: finding every word,
getting phone and syllable boundaries with the Montreal Forced Aligner (MFA), and then
measuring whether a retroflex syllable (zh/ch/sh) was *lenited* (weakened into something
approximant-like).

This is research code from my dissertation work on retroflex lenition in Beijing Mandarin.
**The recordings, TextGrids and hand annotations are participant data and are not in this
repo.** What is here is the method, so it will not run end to end without your own
recordings. It is published so the approach can be read, reused and criticized.

## The problem

Each speaker reads a 288-word list in a fixed order, and every word is read twice in a row,
in a slow block and then a fast block. A dissertation-sized dataset is too big to
annotate by hand, so I wanted as much of it as possible done automatically. Off-the-shelf
tools made two kinds of mistakes:

1. **Finding the words.** Energy-based cutting plus free speech recognition mangles a
   word list: it "corrects" non-words into real phrases, drops the first of two
   back-to-back repetitions, and cuts words at fricative onsets.
2. **Boundaries on lenited tokens.** A lenited retroflex is acoustically closer to a
   vowel or approximant than to a fricative, which is what the aligner expects. In my
   first pass the aligner's boundaries were noticeably worse for lenited tokens, and
   how well the aligner fit the audio did **not** predict lenition (AUC about 0.5), so
   alignment quality can't be used as a stand-in for the label.

## What is in here

```
aligner/
  known_list_aligner.py  Score each rough segment against the *known* word list with a Chinese
                         character-level CTC model (wav2vec2), then use dynamic programming to
                         assign segments to list positions. Handles merges (two repetitions with
                         no pause), splits, insertions and missing words.
  hybrid_labels.py       Keep the free-ASR (Whisper) label unless the CTC scoring assigns a list
                         word. On my two hand-annotated speakers this raised label precision
                         from 92.0 to 94.8 % and from 85.0 to 90.6 %.
  context_mfa.py         Repair bad word intervals by aligning a window that also contains the
                         previous and next list words, so MFA decides where the word really is.
                         Includes an end-trim rule based on the recording's noise floor.
  pinned_mfa.py          Per-word MFA that cannot move manually corrected word edges, and
                         reports how far MFA *wanted* to move them (a free uncertainty signal).
  run_speaker.py         The repair pipeline for one speaker, end to end.
  token_features.py      One row per word token: disfluency flags, slow/fast normalization,
                         boundary-confidence features.
lenition/
  lenition_classifier.py Predict a hand-coded "full lenition" label from acoustics only,
                         training on one hand-annotated speaker and testing on the other.
```

## How it fits together

1. Cut the recording into rough segments (any energy- or VAD-based method).
2. `known_list_aligner`: assign segments to list words; export one short clip per word.
3. `context_mfa` / `pinned_mfa` with MFA: get word and phone boundaries, forcing the
   list's pronunciation (so heteronyms like 长, 什, 血 are not chosen by acoustics, which
   would mix up pronunciation choice with lenition).
4. `token_features`: per-token features and flags.
5. `lenition_classifier`: acoustic classifier for lenition.

## Early results (two hand-annotated speakers, October 2026)

- A classifier from acoustics alone, trained on one speaker and tested on the other, reached
  **AUC 0.82 and balanced accuracy 0.72** on medial syllables, and stopped improving around
  160 hand-labelled training tokens.
- Boundary accuracy was lower for lenited than for unlenited medial syllables
  (roughly 41 % vs 70 % of target-syllable boundaries within 30 ms in a first pass), which is the
  reason for the boundary-repair steps.
- Replacing a 12 dB-below-peak end-trim rule with a noise-floor rule raised the share of word
  ends within 50 ms of hand-corrected ends on five other speakers from 0.57 to 0.70.

These come from my working notes on a small number of speakers and a single annotator.
Treat them as preliminary.

## Limitations

- Built for one read word list and one language variety. The word-list scoring depends on
  knowing what was read and in what order.
- Hand annotation is from one coder, so there is no inter-annotator agreement for the
  lenition label yet.
- Lenition is treated as binary in the classifier; the real phenomenon is gradient.
- The scripts assume my folder layout and a conda setup (an environment with
  pandas/transformers/librosa and a separate one for MFA 3.4.3 with the `mandarin_mfa`
  model). Expect to adjust paths.
- Some inline comments and Chinese example words remain in the original language.

## Requirements

See `requirements.txt`. MFA is installed separately, in its own environment, with the
`mandarin_mfa` acoustic model and `mandarin_china_mfa` dictionary.

## License

MIT. See `LICENSE`.
