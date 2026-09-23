# Production audio

`accepted/` currently contains 21 automatic quality candidates / 97.16 seconds.
Two are visually confirmed; 19 rely on a classifier calibrated with 22 visual labels.
Visual review rejected 6 mixed-speaker clips; the remaining 15 still require audio review.

`quarantine/` contains 31 candidates whose Demucs stem margin is below 15 dB.
They must not be consumed automatically.

These are extraction candidates, **not an approved TTS training set**. Review each clip's
speaker, transcript and residual effects in `dataset/manifests/training_review.csv`.
Both WAV directories are local, reproducible caches and are ignored by Git. Their manifests,
review decisions, calibration data and this README remain versionable.
Run `nailong-tts review` to refresh the queue and `nailong-tts audit` to check readiness.
`nailong-tts pack` refuses to write a formal training filelist until the audit passes.

`dataset/manifests/production_manifest.csv` is the source of truth. It records the
speaker basis, utterance IDs, residual stem margin, transcript, and SHA-256 for every
file. Regenerate with:

```bash
nailong calibrate-visual
nailong prepare-production
```
