// Guided capture rules for the browser - a line-by-line port of app/vision/guidance.py and of the
// quality checks in app/vision/pipeline.py (assess_frame). All thresholds, step names, instructions
// and hint texts are passed in from the server (config.py -> vision_service.guidance_settings()),
// so the browser follows exactly the same rules as the Python code.
//
// This module only decides WHEN to take the 5 pictures. It is not a security check: the server
// re-checks everything in run_pipeline.
//
// Pure functions/classes, no DOM access: also runs under Node.js for tests (tests/js/).

export const IRIS_LEFT = 468;
export const IRIS_RIGHT = 473;
export const NOSE_TIP = 1;
export const MOUTH_LEFT = 61;
export const MOUTH_RIGHT = 291;
export const NUM_MESH_POINTS = 468;

// Standard ArcFace 112x112 template (same as app/vision/align.py)
export const ARCFACE_TEMPLATE = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                 [41.5493, 92.3655], [70.7299, 92.2041]];
export const FIVE_POINTS = [IRIS_LEFT, IRIS_RIGHT, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT];

// ---------------------------------------------------------------------------------------- Face
export class Face {
  // landmarks: array of [x, y] in pixels (478 points, MediaPipe order)
  constructor(landmarks) {
    this.landmarks = landmarks;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < NUM_MESH_POINTS; i++) {
      const [x, y] = landmarks[i];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    this.bbox = [x0, y0, x1, y1];
  }
  get width() { return this.bbox[2] - this.bbox[0]; }
  get height() { return this.bbox[3] - this.bbox[1]; }
  get area() { return this.width * this.height; }
  get interocular() {
    const [ax, ay] = this.landmarks[IRIS_LEFT], [bx, by] = this.landmarks[IRIS_RIGHT];
    return Math.hypot(bx - ax, by - ay);
  }
  // Same as Face.yaw_proxy: nose offset from the eye midpoint along the eye line, in IOD units.
  // + = nose towards the image's right = the student's LEFT (raw, un-mirrored image).
  get yawProxy() {
    const [lx, ly] = this.landmarks[IRIS_LEFT], [rx, ry] = this.landmarks[IRIS_RIGHT];
    const ax = rx - lx, ay = ry - ly;
    const iod = Math.hypot(ax, ay);
    if (iod < 1e-6) return 0;
    const [nx, ny] = this.landmarks[NOSE_TIP];
    const dx = nx - (lx + rx) / 2, dy = ny - (ly + ry) / 2;
    return (dx * ax + dy * ay) / iod / iod;
  }
}

// MediaPipe JS result (normalised coordinates) -> list of Face in pixels
export function facesFromResult(result, width, height) {
  return (result.faceLandmarks || []).map(
    (lms) => new Face(lms.map((p) => [p.x * width, p.y * height])));
}

function overlapFraction(inner, outer) {
  const [ax0, ay0, ax1, ay1] = inner.bbox, [bx0, by0, bx1, by1] = outer.bbox;
  const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0));
  const ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  return (iw * ih) / Math.max(inner.area, 1e-6);
}

// Port of detector.select_main_face
export function selectMainFace(faces, duplicateOverlap, minAreaRatio) {
  if (!faces.length) return { main: null, others: [], ignored: [] };
  const ordered = [...faces].sort((a, b) => b.area - a.area);
  const main = ordered[0], others = [], ignored = [];
  for (const f of ordered.slice(1)) {
    if (overlapFraction(f, main) >= duplicateOverlap || f.area < minAreaRatio * main.area) ignored.push(f);
    else others.push(f);
  }
  return { main, others, ignored };
}

// ------------------------------------------------------------------------ similarity residuals
// Port of liveness_parallax.similarity_residuals (best shift+rotate+scale, per-point error)
export function similarityResiduals(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  // a = vdot(s, d) / vdot(s, s) with complex numbers s = (x - mx) + i(y - my)
  let re = 0, im = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const a = src[i][0] - sx, b = src[i][1] - sy, c = dst[i][0] - dx, d = dst[i][1] - dy;
    re += a * c + b * d;          // conj(s) * d
    im += a * d - b * c;
    ss += a * a + b * b;
  }
  ss = Math.max(ss, 1e-12);
  const ar = re / ss, ai = im / ss;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = src[i][0] - sx, b = src[i][1] - sy, c = dst[i][0] - dx, d = dst[i][1] - dy;
    out[i] = Math.hypot(c - (ar * a - ai * b), d - (ar * b + ai * a));
  }
  return out;
}

export function median(values) {
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// ----------------------------------------------------------------------- frame quality (port)
// Similarity transform (least squares) mapping the 5 face points onto the ArcFace template.
// Returns [a, b, tx, ty]: x' = a*x - b*y + tx, y' = b*x + a*y + ty.
export function arcfaceTransform(face) {
  const src = FIVE_POINTS.map((i) => face.landmarks[i]);
  const dst = ARCFACE_TEMPLATE;
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let re = 0, im = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const a = src[i][0] - sx, b = src[i][1] - sy, c = dst[i][0] - dx, d = dst[i][1] - dy;
    re += a * c + b * d; im += a * d - b * c; ss += a * a + b * b;
  }
  const A = re / Math.max(ss, 1e-12), B = im / Math.max(ss, 1e-12);
  return [A, B, dx - (A * sx - B * sy), dy - (B * sx + A * sy)];
}

// Source-image box (integer, inclusive-exclusive) that the 112x112 aligned face is sampled from.
export function sourceBox(M, frameW, frameH, size = 112) {
  const [a, b, tx, ty] = M, det = a * a + b * b;
  const inv = (u, v) => [(a * (u - tx) + b * (v - ty)) / det, (-b * (u - tx) + a * (v - ty)) / det];
  const pts = [inv(0, 0), inv(size, 0), inv(0, size), inv(size, size)];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 1), y0 = Math.max(0, Math.floor(Math.min(...ys)) - 1);
  const x1 = Math.min(frameW, Math.ceil(Math.max(...xs)) + 2), y1 = Math.min(frameH, Math.ceil(Math.max(...ys)) + 2);
  return [x0, y0, Math.max(x0 + 1, x1), Math.max(y0 + 1, y1)];
}

// Port of cv2.warpAffine(..., (112, 112), INTER_LINEAR, borderValue=0) used by align_arcface:
// every output pixel is sampled bilinearly at the inverse-mapped position (no extra smoothing, unlike
// a canvas drawImage, which blurs when shrinking and would make faces look less sharp than on the
// server). `rgba` is the source box (w x h) whose top-left corner is (ox, oy) in the frame.
export function warpToArcface(rgba, w, h, ox, oy, M, size = 112) {
  const [a, b, tx, ty] = M, det = a * a + b * b;
  const out = new Uint8ClampedArray(size * size * 4);
  const px = (x, y, c) => (x < 0 || y < 0 || x >= w || y >= h) ? 0 : rgba[4 * (y * w + x) + c];
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      const x = (a * (u - tx) + b * (v - ty)) / det - ox, y = (-b * (u - tx) + a * (v - ty)) / det - oy;
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      const o = 4 * (v * size + u);
      for (let c = 0; c < 3; c++) {
        out[o + c] = Math.round((px(x0, y0, c) * (1 - fx) + px(x0 + 1, y0, c) * fx) * (1 - fy) +
                                (px(x0, y0 + 1, c) * (1 - fx) + px(x0 + 1, y0 + 1, c) * fx) * fy);
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

// Brightness and sharpness of a 112x112 aligned face given as RGBA pixel data, exactly like
// quality.brightness (mean grey of rows 24..99, cols 24..87) and quality.sharpness (variance of the
// 3x3 Laplacian, border reflect-101) on cv2's grey conversion.
export function faceQuality(rgba, size = 112) {
  const gray = new Float64Array(size * size);
  for (let i = 0; i < size * size; i++) {
    gray[i] = Math.round(0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]);
  }
  let bsum = 0, bn = 0;
  for (let y = 24; y < 100; y++) for (let x = 24; x < 88; x++) { bsum += gray[y * size + x]; bn++; }
  const refl = (v) => (v < 0 ? -v : v >= size ? 2 * size - 2 - v : v);
  let sum = 0, sum2 = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const lap = gray[refl(y - 1) * size + x] + gray[refl(y + 1) * size + x] +
                  gray[y * size + refl(x - 1)] + gray[y * size + refl(x + 1)] - 4 * gray[y * size + x];
      sum += lap; sum2 += lap * lap;
    }
  }
  const n = size * size, mean = sum / n;
  return { brightness: bsum / bn, sharpness: sum2 / n - mean * mean };
}

// Port of pipeline.assess_frame's decision part. `quality` = {brightness, sharpness} of the main
// face (or null if not computed), frameHeight = height after letterbox trimming.
export function frameProblem(selection, frameHeight, quality, cfg) {
  if (!selection.main) return "no_face";
  if (selection.others.length) return "multiple_faces";
  const face = selection.main;
  if (face.interocular < cfg.min_interocular_px) return "face_too_small";
  if (face.height / frameHeight > cfg.max_face_height_frac) return "face_too_close";
  if (quality && quality.brightness < cfg.min_face_brightness) return "too_dark";
  if (quality && quality.sharpness < cfg.min_sharpness) return "too_blurry";
  return null;
}

// Port of quality.trim_black_borders on a grey image (Uint8 array, width w, height h).
// Returns {top, bottom, left, right} in the given image's pixels.
export function letterbox(gray, w, h, level = 16, maxTrimFrac = 0.3, minRun = 4) {
  const pct99 = (vals) => { const s = [...vals].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(0.99 * (s.length - 1) + 0.5))]; };
  const rowB = [], colB = [];
  for (let y = 0; y < h; y++) rowB.push(pct99(gray.subarray(y * w, (y + 1) * w)));
  for (let x = 0; x < w; x++) { const col = []; for (let y = 0; y < h; y++) col.push(gray[y * w + x]); colB.push(pct99(col)); }
  const run = (vals, limit) => { let n = 0; while (n < limit && vals[n] < level) n++; return n >= minRun ? n : 0; };
  return { top: run(rowB, Math.floor(h * maxTrimFrac)), bottom: run([...rowB].reverse(), Math.floor(h * maxTrimFrac)),
           left: run(colB, Math.floor(w * maxTrimFrac)), right: run([...colB].reverse(), Math.floor(w * maxTrimFrac)) };
}

// ------------------------------------------------------------------------------ GuidedCapture
// Port of guidance.GuidedCapture. Times are in seconds.
export class GuidedCapture {
  constructor(cfg) {
    if (cfg.num_frames !== cfg.steps.length) throw new Error(`guided capture is defined for ${cfg.steps.length} frames`);
    this.cfg = cfg;
    this.step = 0;
    this._stepStarted = null;
    this._okSince = null;
    this._baseYaw = 0;
    this._baseLandmarks = null;
    this._firstSide = 0;
  }

  update(now, face, problem) {
    const cfg = this.cfg, STEPS = cfg.steps, HINTS = cfg.hints;
    if (this.step >= STEPS.length) return this._state({ instruction: "Done", done: true });
    if (this._stepStarted === null) this._stepStarted = now;
    let name = STEPS[this.step];
    if ((name === "left" || name === "right") && this._firstSide !== 0) {
      // second turn: the side opposite to the first one (+ = student's left)
      name = this._firstSide > 0 ? "right" : "left";
    }
    const timeout = this.step === 0 ? cfg.guide_first_timeout_s : cfg.guide_step_timeout_s;
    const elapsed = now - this._stepStarted;
    const st = this._state({ instruction: cfg.instructions[name], timeLeft: Math.max(0, timeout - elapsed) });
    if (name === "left" || name === "right") { st.direction = name; st.target = cfg.guide_turn_target; }
    if (face && this.step > 0) st.turn = face.yawProxy - this._baseYaw;

    // ---- step 0: wait for good conditions, facing the camera, held briefly
    if (this.step === 0) {
      const ok = problem === null && face !== null && Math.abs(face.yawProxy) <= cfg.guide_straight_max;
      if (problem !== null) st.hint = HINTS[problem] ?? null;
      else if (!ok) st.hint = HINTS.not_straight;
      if (ok) {
        if (this._okSince === null) this._okSince = now;
        if (now - this._okSince >= cfg.guide_hold_s) return this._take(st, now, face);
      } else {
        this._okSince = null;
      }
      if (elapsed >= timeout) {
        if (face === null) { st.aborted = true; st.done = true; return st; }   // nobody there at all
        return this._take(st, now, face);
      }
      return st;
    }

    // ---- later steps
    if (face === null || problem === "no_face" || problem === "multiple_faces") {
      st.hint = (face === null && name !== "centre") ? HINTS.lost_face : (HINTS[problem || "no_face"] ?? null);
      return elapsed >= timeout ? this._take(st, now, face) : st;
    }
    if (problem !== null) st.hint = HINTS[problem] ?? null;

    let done;
    if (name === "centre") {
      done = Math.abs(st.turn) <= cfg.guide_centre_tolerance;
    } else {
      const reached = this._firstSide === 0
        ? Math.abs(st.turn) >= cfg.guide_turn_target               // first turn: either side counts
        : -this._firstSide * st.turn >= cfg.guide_turn_target;      // second turn: the other side
      done = (reached && problem === null) || this._moved(face) >= cfg.guide_alt_movement;
      if (!done && st.hint === null && Math.abs(st.turn) > 0.03) st.hint = HINTS.turn_more;
      if (done && this._firstSide === 0) this._firstSide = Math.sign(st.turn) || 1;
    }
    if (done || elapsed >= timeout) return this._take(st, now, face);
    return st;
  }

  _moved(face) {
    if (this._baseLandmarks === null) return 0;
    const res = similarityResiduals(this._baseLandmarks.slice(0, NUM_MESH_POINTS),
                                    face.landmarks.slice(0, NUM_MESH_POINTS));
    return median(res) / Math.max(face.interocular, 1e-6);
  }

  _take(st, now, face) {
    if (this.step === 0 && face !== null) {
      this._baseYaw = face.yawProxy;
      this._baseLandmarks = face.landmarks.map((p) => [p[0], p[1]]);
    }
    st.capture = true;
    this.step += 1;
    this._stepStarted = now;
    this._okSince = null;
    st.done = this.step >= this.cfg.steps.length;
    return st;
  }

  _state(extra) {
    return { step: this.step, instruction: "", hint: null, capture: false, done: false, aborted: false,
             turn: 0, target: 0, direction: null, timeLeft: 0, ...extra };
  }
}
