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

## Stage 2 - vision pipeline + webcam test scripts - DONE (2026-11-14)

Closed by the user before full calibration: test rounds are PAUSED, move on to stage 3.
Thresholds stay at their initial values (config.py); parallax_method stays "landmarks"; depth
stays required (option A). Calibration (more labelled genuine / photo / screen attempts, then
`replay_captures.py --suggest`) is deferred - it can be redone later with webcam_test.py or,
once stage 5 exists, from the web app's attempts log.

- `app/config.py`: `VisionConfig` with every threshold, commented (initial guesses).
- `app/vision/`: detector.py (MediaPipe, duplicate-face filter), align.py (ArcFace 5-point,
  shift-inside crops), embedder.py, quality.py (letterbox trim, brightness, sharpness),
  liveness_parallax.py, liveness_depth.py, liveness_texture.py, pipeline.py
  (`run_pipeline(frames, models, cfg, template=None|template)` - one path for enroll + check-in;
  `decide()` holds the pass/fail/retry rules).
- `scripts/webcam_test.py`: live scores + full 5-frame enroll/check-in attempts, labels
  (1 genuine / 2 photo / 3 screen), CSV log and optional saved captures in instance/webcam_test/.
- `scripts/replay_captures.py`: re-score saved captures with the current thresholds.
- Guided capture (`app/vision/guidance.py`, used by webcam_test.py; the web app should follow the
  same rules): straight -> LEFT -> centre -> RIGHT -> centre, each frame taken when the step is
  done (head turn via `Face.yaw_proxy`), with hints (move back / too dark / not so far) and
  per-step timeouts so photos are still captured and judged. Added after the user's first live
  run: all 5 timed attempts failed - face too close (73-88% of frame height) and turning far
  too much (~60-70 deg, face lost). Live texture P(real) on the user's real face in normal room
  light was 0.996, i.e. MiniFASNet works with this camera once the letterbox is removed.
- OPEN for stage 4: guided capture in the browser needs live landmarks there - either MediaPipe
  Tasks for JS (@mediapipe/tasks-vision, served locally) or a light server endpoint. Ask the user.
- `scripts/check_models.py` now uses the app/vision code (no duplicate pre-processing).
- Tests (no models needed): test_parallax.py, test_parallax_flow.py, test_vision_logic.py, test_guidance.py.

### Stage-1 findings - status
1. False second face: handled in `select_main_face` - secondary detections with >=30% of their
   box inside the main face, or <25% of its area, are ignored (counted in the result). Tested.
2. MiniFASNet on the user's webcam: root causes found on the old frame (Iriun phone webcam):
   **black letterbox bars** (P(real) 0.004 -> ~0.5 once trimmed), plus very dark/noisy image
   (brightness 46) and face filling the frame (77-85% of height). Fixed: letterbox trimming,
   crop now shifted-not-clipped like the original, square box matching the original detector;
   quality gates "too dark" / "too close" ask the student to fix the conditions.
   STILL TO DO: live test with the user (genuine / printed photo / phone screen) using
   `webcam_test.py --save-captures`, then calibrate thresholds. If MiniFASNet still rejects a
   real face in good light, report to the user with options BEFORE changing the design.

### Calibration notes so far (synthetic / sample images only)
- Parallax: tilted flat images through MediaPipe score 0.004-0.022 (even at +-35 deg tilt);
  simulated real head turns 0.034-0.065 -> threshold 0.025, min movement 0.04.
- Depth: weakest signal. Sample real face relief 0.021 / nose +0.054; spoofs 0.012-0.029 /
  -0.019..+0.045. Thresholds 0.015 / 0.02 are cautious placeholders.
- Texture: sample real 0.99, sample spoofs 0.20 and 0.02.
- Identity: user's own old photos 0.54-0.80 between each other; other people 0.0-0.17.
- Timing: ~0.65 s per 5-frame attempt (depth ~0.4 s of that).
- Webcam: camera 0 opens at 1280x720 via DirectShow.

### Stage 2 weak-spot fixes (2026-11-14, after the first live test round) - not committed yet
First live data (6 genuine, 4 PC-screen attacks; no printed photos yet): every attack was
rejected, but the margins were thin; landmark parallax: genuine 0.039-0.051 vs one screen 0.033.
- **Second parallax method, optical flow** (`app/vision/liveness_flow.py`): corners in the face
  (goodFeaturesToTrack) tracked with pyramidal Lucas-Kanade + forward-backward check, then the
  same per-pair RANSAC-homography scoring as the landmark method (shared `score_tracks` in
  liveness_parallax.py). Each frame is first warped by the landmark similarity transform so LK only
  follows residual motion - without this, tracking collapsed on the screen attacks (the attacker
  moved the camera ~2x closer between frames; most tracks lost, scores noisy).
- Both methods always computed and logged (PipelineResult: `parallax_landmark_*`,
  `parallax_flow_*`, `parallax_flow_points`, `parallax_method`, `parallax_status`; checks
  `parallax_landmarks` / `parallax_flow` are informational). `config.parallax_method` chooses
  the decider - still **"landmarks"**. New retry reason `tracking_failed` (too few flow points)
  when flow decides.
- New config: parallax_method, parallax_flow_min_movement / parallax_flow_threshold (placeholders
  = landmark values 0.04 / 0.025), parallax_flow_max_corners / fb_max / min_points. No existing
  threshold changed.
- webcam_test.py: both methods on the live panel (rolling 2.5 s), result screen, console,
  attempts.csv (new columns; old files are migrated automatically) and result.json.
- replay_captures.py: per label min/median/max of every liveness score, margin (lowest genuine -
  highest attack) per check and per attack type, `--suggest` (midpoint thresholds + warnings for
  overlap and < 5 attempts per label).
- Tests: tests/test_parallax_flow.py (flat textured image under homographies ~0.002; synthetic
  non-planar scene 0.030-0.044; too few points -> insufficient_points). 45 tests pass.
- Relabelled the earlier unlabelled attempts: the 4 failures as "screen" (user: all failures were
  photos of them on the PC screen), the rest as "genuine".

Results on the existing captures (replay_captures.py, 6 genuine / 3 scored screen):
| check | genuine min..max | screen max | margin |
|---|---|---|---|
| parallax landmarks | 0.0385..0.0511 | 0.0331 | +0.0054 (genuine/attack ratio 1.16) |
| parallax flow | 0.0263..0.0447 | 0.0183 | +0.0080 (ratio 1.44) |
| depth relief | 0.0281..0.0402 | 0.0217 | +0.0064 |
| depth nose | 0.0833..0.1207 | 0.0602 | +0.0231 |
| texture | 0.9219..0.9953 | 0.8316 | +0.0903 |

### Second test round + decisions (2026-11-14)
Round 2 (5 attempts, labelled afterwards from the images): 3 genuine (one in the dark), 2 attacks
with a clean, high-quality photo of a STRANGER on the PC monitor. Totals so far: genuine 14
(8 accepted, 0 rejected, 6 retry: face too close 4, no face 2); attacks 6, all monitor photo
(0 accepted, 4 rejected, 2 retry).

**Finding for the paper - monocular depth is fooled by clean photos:** the stranger photo on the
monitor got depth relief 0.049 and nose 0.123, i.e. MORE "3D" than any genuine attempt
(genuine relief 0.028-0.040, nose 0.083-0.121). Depth Anything infers depth from appearance, so a
sharp, well-lit face photo looks 3D to it. Two alternatives tried offline did not help (larger
crop; face-vs-surroundings depth step: genuine 0.11-0.15, that attack 0.29). Depth only caught
low-quality / strongly angled screen images (e.g. 16:14:27, relief 0.009, which fooled texture
and nearly landmark parallax). **Parallax (geometry across frames) is the stronger "depth" signal**:
it measures real 3D structure instead of guessing it from one image, and a photo can never earn a
parallax pass (zooming/tilting a photo is either explained by a homography or gives
"insufficient movement" -> retry).
- **Decision (user): option A** - depth stays a required check, thresholds unchanged (0.015 /
  0.02; genuine minimums are ~2x and ~4x these, so it costs real students nothing). It is a
  weak extra layer, not the main defence.

Tooling changes:
- webcam_test.py asks after every attempt "1 genuine / 2 photo / 3 screen" (Enter = last label,
  Esc = unlabelled) and for attacks the type: photo -> printed_flat / printed_bent; screen ->
  phone_photo / phone_video / monitor_photo (Enter = last type for that label). Logged as
  `attack_type` in attempts.csv and result.json, and in the capture folder name. Logging happens
  after labelling. The 6 existing screen attempts were tagged monitor_photo (user: PC screen).
- replay_captures.py: outcomes section - genuine split into accepted / REJECTED (false rejection)
  / asked to RETRY with reasons, plus false-rejection rate among decided attempts; attacks per
  attack type (ACCEPTED / rejected by which checks / retry reasons). Distributions and margins
  are per attack type.

### OPEN points (deferred calibration + later stages)
0. **Threshold calibration is STILL OPEN** (all thresholds are initial guesses). Since stage 5 it
   can be done from the web app: label attempts on the professor's Attempts page, then run
   `scripts/threshold_report.py` (same analysis as replay_captures.py --suggest + Table I).
1. Next test round by the user: >= 8 genuine (varied light/distance/glasses), >= 10 printed
   photo, >= 10 screen (PC + phone, photo AND video replay), labelled with 1/2/3.
2. Then `replay_captures.py --suggest`; decide parallax method (flow looked better on this small
   sample, but screen attacks track only 3-32 flow points vs ~100-170 on real faces) and set all
   thresholds. Do NOT change thresholds before that round (user's instruction).
3. One screen attack fooled texture (0.83) and nearly landmark parallax (0.033); only depth
   stopped it - watch this case in the next round. Depth: decided (option A, see above).
4. Face-height limit 0.70: the user sits at 0.65-0.69 with the Iriun phone camera; revisit after
   spoof data (a closer face gives the texture model less context).
5. (decided in stage 4: MediaPipe Tasks for JavaScript, served locally.)
6. Possible later: warn if a new enrollment's face matches another student's template
   (someone enrolling for a friend) - not in the agreed design, ask first.
## Stage 3 - database + auth + professor account management - DONE (2026-11-14)

- **App factory** `app/__init__.py` (`create_app(config_name, overrides)`), blueprints `auth`,
  `student`, `professor`, `api`; `wsgi.py` + `start_server.bat` (waitress on 127.0.0.1:5000).
- **Config** (`app/config.py`): Development / Production / Testing classes. SECRET_KEY from env or
  `instance/secret_key` (auto-created); DATABASE_URL from env (default `instance/attendance.db`,
  PostgreSQL = only this setting); optional `instance/config.py`; APP_ENV selects the class.
  Cookies HttpOnly + SameSite=Lax; Secure in ProductionConfig (or SESSION_COOKIE_SECURE=1).
  TRUST_PROXY_HOPS enables ProxyFix for a reverse proxy later. MAX_CONTENT_LENGTH 12 MB.
- **Models** (`app/models.py`): users, face_templates (separate; professors only ever see
  "enrolled yes/no"), attempts (every PipelineResult score: both parallax methods, depth, texture,
  checks/frames/timings JSON, thresholds in force, + label / attack_type / labelled_by/at / note),
  attendance (UNIQUE(user_id, date) in the DB; local date+time + UTC timestamp + attempt_id;
  session_id can be added later as a nullable FK), login_failures. A test fails if the pipeline
  gains a score the attempts table does not store.
- **Security** (`app/security.py`): argon2id (argon2-cffi), constant-effort login for unknown
  users, password policy (>= 8 chars, not all digits/letters, not = username), lockout after 5
  failures per username or 20 per IP in 15 min (DB table, so it works across waitress threads;
  correct password is refused while locked), role_required decorator, CSRF on all forms and JSON
  (X-CSRFToken header, token in `<meta name="csrf-token">`), session_version in the Flask-Login id
  so password change/reset/disable logs out existing sessions, security headers (strict CSP - no
  inline JS/CSS allowed in templates -, X-Frame-Options, Permissions-Policy camera=self, no-store).
- **Pages**: login, change password (forced on first login: every other page redirects there),
  professor dashboard (create student with given or generated initial password - shown once,
  never stored in readable form -, reset password, disable/enable, reset enrollment), student
  home (placeholder check-in + own attendance history), enrollment PLACEHOLDER (unenrolled
  students are redirected there), error pages.
- **Scripts**: `scripts/create_professor.py` (interactive), `scripts/init_db.py` (run by setup.bat).
- **Tests**: test_auth.py, test_roles.py, test_models.py (+ conftest.py) - 85 tests pass in total.
  Also checked end-to-end against the real waitress server (throwaway DB): CSRF rejection, first
  login -> password change -> enrollment placeholder, 403 for the wrong role, lockout 429.

### Notes / decisions in stage 3
- No new dependencies: rate limiting is the DB-based lockout above (no Flask-Limiter).
- Schema changes: `db.create_all()` only creates missing tables. No migration tool yet
  (Flask-Migrate/Alembic would be a new dependency - ask the user before adding it). Until then,
  a schema change means: delete instance/attendance.db (test data only) or alter it by hand.
- Attendance date = local calendar day in APP_TIMEZONE (default Europe/Skopje), server time.
- Not built yet (later stages): enrollment + check-in + "delete my face data" (stage 4); attendance
  by date, attempts log + labelling, CSV, threshold_report.py (stage 5).

## Stage 4 - enrollment + check-in pages - DONE (2026-11-14)

- **Browser guided capture** with MediaPipe Tasks for JavaScript (`@mediapipe/tasks-vision`
  1.2.0, Apache-2.0; 5 files from the official npm tarball, SHA-256 pinned, downloaded by
  `scripts/download_models.py` into `app/static/vendor/` - git-ignored, served by our server, no
  CDN). The model is our `models/face_landmarker.task`, served at `/student/assets/...` (login).
  - `static/js/guidance.js`: line-by-line port of guidance.py + assess_frame quality checks
    (duplicate-face filter, size, too close, brightness, sharpness with an exact port of
    cv2.warpAffine bilinear alignment, letterbox trim). Thresholds, step names, instructions and
    hints come from config.py via `vision_service.guidance_settings()` (data attribute).
  - `static/js/capture.js`: camera (ideal 1920x1080), IMAGE-mode landmarks (~15/s), instruction,
    hint, dots, timer, head-turn bar; on each step the CURRENT frame is copied at full resolution
    (max width 1920) as JPEG quality 0.95; one multipart upload of the 5 frames + CSRF header.
  - Verified: Node tests (tests/js/guidance.test.mjs, run by tests/test_guidance_js.py) replay
    the Python guidance scenarios; headless Edge on a real frame gives the SAME yaw / IOD /
    face height / brightness (116.26 vs 116.3) / sharpness (962.6 vs 962.9) as Python; full
    end-to-end in headless Edge with a fake camera (y4m from a genuine capture) + real models:
    guidance advanced through all 5 steps, enroll PASS, check-in PASS, attendance written, ~0.74 s.
- **CSP change (narrow)**: only the enroll and check-in pages get `script-src 'self'
  'wasm-unsafe-eval'` (allows compiling WebAssembly, NOT JS eval; files still only from 'self').
  Verified: without it MediaPipe fails with a CSP CompileError. Also `connect-src 'self'` stated
  explicitly. Found: tasks-vision tries to POST usage logs to odml.pa.googleapis.com - blocked by
  the CSP (no data to Google). `.mjs` / `.wasm` MIME types registered in create_app.
- **Server** (`app/attendance.py`, `app/frames.py`, `app/vision_service.py`, `api` blueprint):
  - `POST /api/enroll` (frames + consent=yes) and `POST /api/checkin` (frames), students only,
    CSRF required. Uploads validated: exactly 5 files, <= 4 MB each, JPEG/PNG by magic bytes,
    size read from the header BEFORE decoding (240..2560 px, <= 4.1 MP - decompression-bomb
    safe), decode check, all frames same size.
  - Models loaded once at startup (VISION_PRELOAD), max 2 analyses in parallel, max 10 analysed
    attempts per student per 5 min (429).
  - Enrollment: consent required (400 before analysis), liveness must pass, template + 5 frame
    embeddings stored with consent_at (server time) and enroll_attempt_id.
  - Check-in: 1:1 against the student's own template; attendance only if pass AND none today;
    savepoint + UNIQUE(user_id, date) handle simultaneous requests; a verified second check-in
    is logged with attempts.duplicate = True and a clear message.
  - Every analysed attempt (pass/fail/retry) logged with all scores + thresholds snapshot.
    Requests refused before analysis (no consent, not enrolled, bad upload, rate limit) are not
    attempts.
- **Student pages**: home (today's status, check-in button, history, "Delete my face data" =
  withdraw consent: deletes the template, keeps attendance + attempt scores, sends the student to
  enrollment), enrollment page with consent text + checkbox, check-in page (redirects home if
  already checked in today), clear pass / fail / retry result messages.
- **Professor form fixes**: all spaces removed from student IDs, "/" allowed, errors shown under
  the right field with the typed values kept (400), initial password is a plain text field with
  autocomplete=off (a form without a password field is not autofilled with the professor's login).
- **Tests**: tests/test_capture.py (26, pipeline mocked by a fake camera: consent, enrollment,
  liveness required, check-in, duplicate rule incl. DB race, wrong person, 1:1 template, retry,
  deletion + re-enrollment, upload validation incl. decompression bomb, rate limit, access rules,
  CSRF, CSP only on capture pages, professor form fixes) + JS port tests. 113 tests pass.
- No database schema change in stage 4 (existing instance/attendance.db keeps working).

## Stage 5 - professor dashboard, attempts log, labelling, CSV, threshold report - DONE (2026-11-15)

- **Professor navigation**: Students | Attendance | Attempts (top bar; own row on phones).
- **Attendance by date** (`/professor/attendance?date=`, `app/professor/attendance_views.py`): pick a
  day (date field, previous/next day, today); who checked in and at what time (server time,
  APP_TIMEZONE) with a link to the check-in attempt, plus the active students who did not.
  **Attendance CSV**: one day (`?date=`) or a range (`?date_from=&date_to=`, max ~3 years, swapped
  range accepted, bad dates -> 400): date, time, student_id, full_name, attempt_id.
- **Attempts log** (`/professor/attempts`, `app/professor/attempts_views.py`; filters and CSV rows in
  `app/attempt_log.py`): filters date from/to (local days, converted to UTC bounds), student, mode,
  decision, reason, label (any / not labelled / labelled / each label / each attack type); invalid
  filter values are ignored with a message. 100 per page, newest first. Columns: similarity,
  consistency, landmark parallax score + movement, flow parallax score + movement + points, depth
  relief + nose, texture, server ms, label + note. Current thresholds in the column headers, the
  deciding parallax method marked *, a failed check in red.
- **Attempt detail** (`/professor/attempts/<id>`): decision + the message the student saw, whether
  attendance was written, every score next to the threshold IN FORCE AT THAT TIME
  (attempts.thresholds), per-frame measurements, timings (pipeline, whole request, browser capture,
  end-to-end).
- **Labelling** (POST `/professor/attempts/label`): one value = label + attack type ("genuine",
  "photo", "photo/printed_flat", "photo/printed_bent", "screen", "screen/phone_photo",
  "screen/phone_video", "screen/monitor_photo"; the attack type may stay unspecified) - the same
  types as webcam_test.py. Bulk: checkboxes + select-all (static/js/attempts.js), optional note
  (empty = keep existing notes), "Remove the label" (asks for confirmation). Detail page: form
  prefilled with the current label + note, saved as shown. Re-saving the same label keeps
  labelled_by/at; removing a label clears them. Note max 1000 chars. Redirect back only to the
  attempts log (no open redirect). CSRF required. Helpers in models.py: label_value /
  parse_label_value / label_choices / label_text.
- **Attempts CSV** (`/professor/attempts.csv`, same filters, no paging): all score columns, each
  check pass/fail, processing / server / browser / end-to-end ms + per-step timings, model, the main
  thresholds in force, label, attack type, note, labelled by/at. `?anonymise=1`: student ID + name
  (and labelled_by) replaced by a participant code `P-xxxxxxxx` = HMAC(secret key, user id) - stable
  across exports, not reversible without the server's secret key.
- **CSV safety** (`app/csv_export.py`): UTF-8 with BOM (Cyrillic names in Excel); text cells starting
  with = + - @ get a leading ' (no formula injection); numbers unchanged.
- **No biometric data**: these views never read face_templates except "exists?" (dashboard). Tested
  two ways: the template's bytes/values never appear in any professor page or CSV, and an SQL
  listener shows no professor request ever SELECTs face_templates.template / frame_embeddings.
- **End-to-end time for Table I** (no schema change; stored in attempts.timings_ms): `server_ms`
  (request start, before reading the frames -> analysis done) and `client_capture_ms` (browser:
  camera ready -> 5 frames encoded; capture.js sends `capture_ms`, kept only if 0 < x <= 120000).
  End-to-end = capture + server; network upload time not included. Attempts made before stage 5
  have neither -> "n/a" in the report.
- **Shared analysis** `app/calibration.py` (pure Python): `Record` (from a PipelineResult or an
  attempts row), outcomes, distributions (+ identity similarity, information only), margins,
  suggestions, Table I, threshold-change detection. `replay_captures.py` now uses it - its output on
  the 20 saved captures was verified identical, except the added similarity rows.
- **`scripts/threshold_report.py`**: reads the attempts table (SQLite opened READ-ONLY); options
  --from / --to / --mode / --student / --list / --db. Prints outcomes per label + attack type, score
  distributions, margins + suggested thresholds (always), a NOTE when attempts were made with other
  thresholds than config.py has now, and **Table I**, each value with n and a 95% Wilson interval,
  plus the definitions to use in the paper:
  - identification accuracy = genuine CHECK-INS with identity passed / genuine check-ins where
    identity was evaluated (quality-stopped attempts have no identity score);
  - attack rejection rate = attacks NOT accepted (fail or retry) / all attacks of that kind (= 1 - APCER);
  - false rejection rate = genuine "fail" / genuine pass + fail; genuine retry rate reported separately;
  - verification time = check-ins that ran the full pipeline (not stopped by a quality gate),
    server and end-to-end;
  - participants = distinct students with at least one labelled attempt.
- Also: 400 error page; template filters `localtime` and `num`; dashboard footnote removed.
- **Tests**: tests/test_professor_pages.py (48: roles, attendance page + CSV + bad ranges, CSV formula
  escaping, attempts log scores, every filter incl. the local-day boundary, paging, detail, bulk /
  single labelling, invalid labels, removing, notes, CSRF, open redirect, attempts CSV + anonymised,
  no biometric data in output or SQL); tests/test_threshold_report.py (14: margins, suggestions,
  overlap, Wilson CI, Table I numbers, Record conversion, threshold changes, report text + filters,
  script end-to-end on a DB file incl. a check that the file is unchanged); browser capture-time
  tests in test_capture.py. **180 tests pass.**
- Checked with the real waitress server on a COPY of instance/attendance.db (the real file was not
  touched): every new page 200, both CSVs, bulk labelling via POST, labelling without CSRF -> 400,
  threshold_report on the labelled copy; screenshots of the pages (desktop; at 390 px no sideways
  scrolling).
- No database schema change in stage 5 (the existing instance/attendance.db keeps working).

### Open after stage 5
- **"Wrong person" (impostor) attempts have no label of their own.** Labels say WHAT was in front of
  the camera (genuine / photo / screen), not WHO. A friend checking in on someone else's account is
  a live face, but must NOT be labelled "genuine" (it would count as a false rejection). For now:
  leave such attempts unlabelled and add a note. Proposal (ask the user - it changes the agreed
  label set): a fourth label "impostor" (real, different person) -> the report could also give the
  impostor acceptance rate and a data-based identity threshold.
## Stage 6 - tests, README, SYSTEM_DESCRIPTION, DEPLOYMENT - TODO
