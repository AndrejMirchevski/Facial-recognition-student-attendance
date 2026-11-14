# Facial Recognition Student Attendance (v2)

Read this whole file before doing anything. It is the agreed design. If something here
turns out to be impossible or clearly wrong once you try it, say so to the user and propose
an alternative - don't silently change the design.

## What this is

A rebuild, from scratch, of a university student project: a browser-based attendance
system that verifies a student's identity with face recognition plus a liveness
(anti-spoofing) check, so a printed photo or a phone screen cannot be used to check in.

- The research paper this system is built for: `docs/paper.md` (original: `docs/Facial_Recognition_Attendance_Paper.docx`).
  Follow its design (login first, 5 frames, identity + depth/liveness gate, duplicate
  detection, separate student and professor interfaces). Where this file differs from the
  paper, this file wins - e.g. we do NOT use dlib (see Models).
- The old prototype, for reference only: `C:\Users\Andrej\Desktop\FaceAttendanceSystem`
  (Flask + DeepFace Facenet512 + SQLite, vanilla JS frontend). Do not modify it. Do not copy
  its structure blindly; it identifies faces 1:N with no login and no liveness check.

The user is a computer science student at UACS (Skopje) and a co-author of the paper. Explain
decisions in plain language; the user will need to describe the system in the paper.

## Environment

- Windows 10/11, Python 3.11, CPU only (no GPU). Ordinary laptop webcam.
- Create a fresh virtual environment in this folder: `.venv`.
- Prefer packages that install from wheels on Windows with no C++ build tools. Prefer
  ONNX Runtime models over PyTorch/TensorFlow where a good ONNX version exists (smaller,
  faster install, faster on CPU). Verify each package actually installs before relying on it.
- Model weight files go in `models/` and are downloaded by a script (`scripts/download_models.py`),
  not committed by hand. Record each model's source URL and licence in `models/README.md`.
- Runs on localhost now. MUST be easy to put on the internet later (see Security).

## Agreed product decisions

1. **Accounts are created by a professor.** No self sign-up. A setup script creates the first
   professor account (`scripts/create_professor.py`, asks for username + password).
   The professor creates student accounts: student ID, full name, initial password.
2. **Face enrollment happens on the student's first login.** After first login the student is
   forced to the enrollment page and cannot check in until enrolled. Enrollment shows a
   consent checkbox (biometric data, purpose, that they can delete it) and must be ticked.
   Store when consent was given. Student should be asked to change the initial password on
   first login too.
3. **Sessions (lectures/exams) are skipped for now.** Duplicate rule: one successful check-in
   per student per calendar day. Design the data model so a `sessions` table can be added
   later without rewriting (e.g. attendance rows can get a nullable session_id later).
4. **No manual override yet** (professor marking someone present by hand) - planned for later.
5. Student can delete their own face data ("withdraw consent"); this also blocks check-in
   until they re-enroll. Professor can reset a student's enrollment. The professor must
   NEVER see or download face templates/embeddings or raw face images.

## Check-in flow (the core of the paper)

1. Student logs in, clicks Check In.
2. Browser camera captures **5 frames over ~2-3 seconds** while on-screen instructions ask the
   student to slowly turn their head a little left/right. (The head movement is what the
   parallax liveness check needs.) Frames are sent to the backend in one request.
3. Backend, for each frame: detect the face + landmarks. Reject the attempt if any frame has
   no face, or more than one face, or the face is too small/blurry - with a clear retry message.
4. **Identity (1:1 verification, not 1:N search):** compute an embedding per frame, aggregate
   (mean of L2-normalised embeddings, re-normalised), compare to THAT logged-in student's
   enrolled template only. Pass if cosine similarity >= configured threshold.
5. **Liveness - all three must pass** (configurable, thresholds in config):
   a. **Parallax / planarity check (no model).** Using the 2D landmark positions across the
      5 frames: for each frame pair, fit a homography (cv2.findHomography, RANSAC) between the
      landmark sets. A flat photo/screen moving in front of the camera is explained almost
      perfectly by a homography; a real 3D face turning its head is not. Score = median
      reprojection error normalised by face size (e.g. inter-ocular distance). Also require a
      minimum amount of head movement between frames; if the student didn't move enough,
      return "please turn your head slightly and try again" instead of pass/fail.
   b. **Monocular depth check.** Run a monocular depth model on the face crop; measure how far
      the face depth map is from a plane (fit a plane to the depth values over the face mask,
      score = residual relative to depth range; also check the nose region is closer than the
      cheeks). Flat => spoof.
   c. **Texture anti-spoof model** (MiniFASNet-style) on the face crop: real vs print/replay
      probability.
6. Attendance is written ONLY if identity passes AND all liveness checks pass AND the student
   has no successful check-in today. Otherwise nothing is written to attendance.
7. **Every attempt** (pass or fail) is written to an `attempts` table with: student, time,
   similarity, each liveness score, which checks passed/failed, final decision, reason, and
   processing time in ms. Do not store the frames themselves by default.
   This table is how the user fills in Table I of the paper (accuracy, photo/screen
   rejection rate, false rejection rate, average verification time).

Enrollment uses the same pipeline: capture 5 frames with head movement, require liveness to
pass (so nobody enrolls from a photo), require all 5 frames to contain one good face, check
the frames are consistent with each other (same person), then store the aggregated template
(optionally also the 5 individual embeddings).

Make sure enrollment and check-in compute embeddings with exactly the same code path
(same detector, alignment, model, normalisation). That was a real bug in the old prototype.

## Models (agreed)

| Job | Model | Notes |
|---|---|---|
| Face detection + landmarks | **MediaPipe Face Landmarker** (Tasks API, `face_landmarker.task`) | 478 landmarks; used for alignment, quality checks and the parallax check. Apache-2.0. |
| Face recognition | **ArcFace** - InsightFace `buffalo_l` recognition model `w600k_r50.onnx`, run directly with onnxruntime | Do NOT pip install `insightface` if it needs C++ build tools on Windows; instead download the ONNX file and do the standard 5-point similarity-transform alignment to 112x112 yourself (cv2.estimateAffinePartial2D to the standard ArcFace template, using eye/nose/mouth-corner points from MediaPipe). Licence: non-commercial research use - fine for this project, note it in models/README.md. Fallback if this proves unworkable: ArcFace through DeepFace. |
| Depth | **Depth Anything V2 Small** in ONNX form, via onnxruntime | Apache-2.0 (Small only - larger variants are non-commercial). |
| Texture anti-spoof | **MiniFASNet** (Silent-Face-Anti-Spoofing), ONNX if a reliable ONNX export is available | Apache-2.0. If only PyTorch weights are usable, CPU PyTorch is acceptable for this one model. |

Do NOT use dlib. Do NOT use the Haar cascade. Load every model once at startup, not per request.

All thresholds live in `config.py` with comments. Initial thresholds are guesses: tell the
user they must be calibrated with real attempts, and make calibration easy (the attempts log
plus a small script `scripts/threshold_report.py` that prints score distributions from the
attempts table, separated by whether the user labelled the attempt as genuine/photo/screen).
Let the professor (or a test page) label an attempt as genuine / printed photo / screen replay
so the paper's metrics can be computed.

## Architecture

- **Backend:** Flask, application-factory pattern, blueprints: `auth`, `student`, `professor`, `api`.
- **Database:** SQLite through SQLAlchemy (so moving to PostgreSQL later is a config change).
  Tables (suggested): `users` (id, role professor|student, username/student_id, full_name,
  password_hash, must_change_password, created_at), `face_templates` (user_id, template blob,
  model name+version, consent_at, created_at) kept separate from `attendance`
  (id, user_id, date, time, attempt_id), `attempts` (as above).
  Unique constraint on (user_id, date) for successful attendance, enforced in the DB, not just in code.
- **Frontend:** server-rendered Jinja templates + small vanilla JS (no build step, no React).
  Clean, simple design. Pages: login, change password, student home (status, check-in button,
  own attendance history, delete my face data), enrollment, professor dashboard (students list
  with enrolled yes/no, create/disable student, reset password, reset enrollment, attendance by
  date, attempts log with filters and labelling, CSV export of attendance).
- **Folder layout (suggested):** `app/` (package: `__init__.py`, `config.py`, `models.py`,
  `auth/`, `student/`, `professor/`, `api/`, `vision/` [detector.py, align.py, embedder.py,
  liveness_parallax.py, liveness_depth.py, liveness_texture.py, pipeline.py], `templates/`,
  `static/`), `models/`, `scripts/`, `tests/`, `docs/`, `instance/` (db + secret key, gitignored).
- `vision/pipeline.py` exposes ONE function used by both enrollment and check-in, returning a
  result object with all scores and a decision - keep it independent of Flask so it can be tested
  from scripts.

## Security (localhost now, internet later)

- Passwords: werkzeug/argon2 hashing. Never log passwords or embeddings.
- Flask-Login sessions; role checks on every professor route and API.
- CSRF protection on all forms and JSON POSTs (Flask-WTF CSRF or equivalent).
- Login rate limiting / lockout after repeated failures.
- `SECRET_KEY` and all deployment settings from environment / `instance/` config, never hard-coded.
  `SESSION_COOKIE_SECURE`, `SESSION_COOKIE_HTTPONLY`, `SameSite=Lax`; secure cookie turned on
  automatically in production config.
- No hard-coded `127.0.0.1` in the frontend; use relative URLs.
- Note for later: browsers only allow camera access on HTTPS (or localhost), so internet
  deployment needs HTTPS (e.g. waitress behind a reverse proxy with a certificate). Write a short
  `docs/DEPLOYMENT.md` explaining this; don't actually deploy.
- Validate uploads: max request size, max 5 frames, image decode checks, max resolution.
- Server-side time is the source of truth for attendance timestamps.

## Deliverables

- Working app runnable with `start_server.bat` (uses waitress or Flask dev server on localhost).
- `setup.bat`: creates `.venv`, installs `requirements.txt` (pinned versions), downloads models,
  creates the database. Must work on a clean Windows machine with Python 3.11.
- `README.md` for the user: setup, creating the professor, creating a student, enrolling,
  checking in, how to run spoof tests and get the numbers for the paper.
- `docs/SYSTEM_DESCRIPTION.md`: a plain-language description of the final architecture,
  models and checks, written so the user can update the paper's Methodology section
  (the paper currently says dlib - it should say MediaPipe + ArcFace + the three liveness checks).
- Tests in `tests/` (pytest): auth/roles, duplicate rule, attempts logging, pipeline decision
  logic with mocked model outputs, parallax check on synthetic planar vs non-planar landmark sets.

## How to work

- Build in stages and get each one running before the next:
  1. project skeleton, venv, requirements, models download; prove each model loads and runs
     on a test image;
  2. vision pipeline as plain Python + scripts to test it with the webcam;
  3. database + auth + professor account management;
  4. enrollment + check-in pages wired to the pipeline;
  5. professor dashboard, attempts log, labelling, CSV export, threshold report;
  6. tests, README, SYSTEM_DESCRIPTION, DEPLOYMENT docs.
- Keep a short progress list in `docs/PROGRESS.md` so a new session can pick up where the last stopped.
- Ask the user before: adding a big dependency not listed here, changing an agreed decision,
  or deleting anything outside this folder.
- At the end, run the tests and start the server to check it actually works, then give the
  user a step-by-step manual test plan (genuine check-in, printed photo, phone screen,
  duplicate check-in, wrong person logged in, first-login enrollment flow).
