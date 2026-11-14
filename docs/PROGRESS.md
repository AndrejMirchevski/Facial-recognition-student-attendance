# Progress

## Stage 1 - skeleton, venv, requirements, models - DONE (2026-11-14)

- `.venv` (Python 3.11.9), `requirements.txt` fully pinned; clean install verified in a
  throwaway venv (all wheels, no build tools, no PyTorch/JAX).
- `scripts/download_models.py`: downloads 5 files to `models/`, verifies pinned SHA-256,
  supports fallback sources (tested) and rejects bad hashes (tested).
- `scripts/check_models.py`: loads every model, runs synthetic inputs, and optionally runs the
  full chain on photos given as arguments.
- `setup.bat`: venv + pip + models + check (DB creation to be added in stage 3).
- `models/README.md`: sources, licences, preprocessing of each model.

Measured on the user's laptop (CPU): model load ~0.45 s total; per frame: landmarks ~10 ms,
ArcFace ~20-35 ms, Depth Anything at 266x266 ~70-90 ms, MiniFASNet V2 ~2 ms + V1SE ~37 ms.

### Findings to carry into stage 2
1. **MediaPipe false second face**: on one photo the landmarker returned a second, smaller face
   overlapping the real one. The "more than one face -> reject" rule must ignore detections
   that are much smaller than / overlap the main face, or genuine students get rejected.
2. **MiniFASNet**: verified correct against upstream labelled samples (real 1.000; fakes
   0.065 / 0.002). BUT it gave P(real) ~0.00 on the user's own (real) webcam frame from the
   old prototype, and 0.947 on random noise. It may false-reject on this webcam -> must test
   with the live webcam in stage 2 before trusting it; its threshold needs calibration.
3. ArcFace similarity between two of the user's own old photos: 0.604 (plausible same-person value).
4. Depth Anything: on real faces the nose is clearly closer than the cheeks (signal exists).

## Stage 2 - vision pipeline + webcam test scripts - TODO
## Stage 3 - database + auth + professor account management - TODO
## Stage 4 - enrollment + check-in pages - TODO
## Stage 5 - professor dashboard, attempts log, labelling, CSV, threshold report - TODO
## Stage 6 - tests, README, SYSTEM_DESCRIPTION, DEPLOYMENT - TODO
