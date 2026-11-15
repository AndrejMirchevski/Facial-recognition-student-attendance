// Guided 5-frame capture for the enrollment and check-in pages.
//
// 1. Loads MediaPipe Face Landmarker (WebAssembly, served by our own server) and the camera.
// 2. Every preview frame: landmarks -> the same quality checks and step rules as the Python code
//    (guidance.js) -> on-screen instruction, hint and head-turn bar.
// 3. When a step is done, the CURRENT camera frame is copied at full resolution (JPEG, high quality).
// 4. After 5 frames: one upload to the server, which re-checks everything and decides.
//    Nothing computed here is sent or trusted - only the 5 pictures.
import { GuidedCapture, arcfaceTransform, faceQuality, facesFromResult, frameProblem, letterbox,
         selectMainFace, sourceBox, warpToArcface } from "./guidance.js";

const root = document.getElementById("capture");
const S = JSON.parse(root.dataset.settings);
const G = S.guide;
const $ = (sel) => root.querySelector(sel);
const ui = {
  stage: $(".cap-stage"), video: $("video"), overlay: $("canvas.cap-overlay"),
  start: $(".cap-start"), retry: $(".cap-retry"), cancel: $(".cap-cancel"),
  status: $(".cap-status"), instruction: $(".cap-instruction"), hint: $(".cap-hint"),
  dots: $(".cap-dots"), timer: $(".cap-timer"), bar: $(".cap-bar"), marker: $(".cap-marker"),
  target: $(".cap-target"), band: $(".cap-band"), result: $(".cap-result"),
};
const consent = document.getElementById("consent");          // enrollment page only
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
const MIRRORED = true;                                        // preview shown like a mirror
const BAR_RANGE = 0.30;                                       // head-turn bar: +-0.30 yaw units

let landmarker = null, stream = null, running = false, guide = null, shots = [];
let lb = { top: 0, bottom: 0, left: 0, right: 0 }, lastLetterbox = -1e9, lastDetect = 0;
let captureStart = 0;   // performance.now() when the guided capture started (camera ready)
const small = document.createElement("canvas");                // face region (for the quality check)
const smallCtx = small.getContext("2d", { willReadFrequently: true });
const lbCanvas = document.createElement("canvas");             // downscaled frame (letterbox)
const lbCtx = lbCanvas.getContext("2d", { willReadFrequently: true });

// ------------------------------------------------------------------------------------ helpers
function setStatus(text, kind = "") {
  ui.status.textContent = text;
  ui.status.className = "cap-status " + kind;
  ui.status.hidden = !text;
}

function showResult(kind, title, message, actions = []) {
  ui.result.hidden = false;
  ui.result.className = "cap-result " + kind;
  ui.result.replaceChildren();
  const h = document.createElement("h2"); h.textContent = title;
  const p = document.createElement("p"); p.textContent = message;
  ui.result.append(h, p);
  const row = document.createElement("div"); row.className = "cap-actions";
  for (const a of actions) row.append(a);
  ui.result.append(row);
}

function button(text, cls, onClick) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "btn " + cls; b.textContent = text;
  b.addEventListener("click", onClick);
  return b;
}

function link(text, href, cls = "btn primary") {
  const a = document.createElement("a"); a.href = href; a.className = cls; a.textContent = text;
  return a;
}

function stopCamera() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  ui.video.srcObject = null;
  ui.stage.classList.remove("live");
}

function cameraErrorMessage(err) {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    return "The camera only works on a secure (https://) connection or on localhost.";
  }
  switch (err && err.name) {
    case "NotAllowedError": return "Camera access was blocked. Allow the camera for this site in the browser's address bar, then try again.";
    case "NotFoundError": return "No camera was found. Connect a camera and try again.";
    case "NotReadableError": return "The camera is being used by another program. Close it and try again.";
    default: return "The camera could not be started" + (err && err.message ? ` (${err.message}).` : ".");
  }
}

// --------------------------------------------------------------------------- model + camera
async function loadLandmarker() {
  if (landmarker) return landmarker;
  setStatus("Loading the face detector…");
  const vision = await import(S.bundleUrl);
  const fileset = await vision.FilesetResolver.forVisionTasks(S.wasmPath);
  landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: S.modelUrl, delegate: "CPU" },
    runningMode: "IMAGE",                 // every frame on its own, like the server (no tracking)
    numFaces: G.max_faces_detected,
  });
  return landmarker;
}

async function startCamera() {
  setStatus("Starting the camera…");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  ui.video.srcObject = stream;
  await ui.video.play();
  ui.overlay.width = ui.video.videoWidth;
  ui.overlay.height = ui.video.videoHeight;
  ui.stage.classList.add("live");
}

// ---------------------------------------------------------------------------- per-frame logic
function updateLetterbox(now) {
  if (now - lastLetterbox < 1.5) return;           // bars do not change - check now and then
  lastLetterbox = now;
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  const sw = 320, sh = Math.max(1, Math.round((h * sw) / w));
  lbCanvas.width = sw; lbCanvas.height = sh;
  lbCtx.drawImage(ui.video, 0, 0, sw, sh);
  const px = lbCtx.getImageData(0, 0, sw, sh).data;
  const gray = new Uint8Array(sw * sh);
  for (let i = 0; i < sw * sh; i++) gray[i] = Math.round(0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]);
  const f = w / sw;
  const r = letterbox(gray, sw, sh, 16, 0.3, Math.max(1, Math.round(4 / f)));
  lb = { top: r.top * f, bottom: r.bottom * f, left: r.left * f, right: r.right * f };
}

function measureQuality(face) {
  // copy only the face region at full resolution, then align it exactly like align_arcface
  const M = arcfaceTransform(face);
  const [x0, y0, x1, y1] = sourceBox(M, ui.video.videoWidth, ui.video.videoHeight);
  const w = x1 - x0, h = y1 - y0;
  if (small.width !== w || small.height !== h) { small.width = w; small.height = h; }
  smallCtx.drawImage(ui.video, x0, y0, w, h, 0, 0, w, h);
  return faceQuality(warpToArcface(smallCtx.getImageData(0, 0, w, h).data, w, h, x0, y0, M));
}

function grabFrame() {
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  const scale = Math.min(1, S.maxWidth / w);
  const c = document.createElement("canvas");
  c.width = Math.round(w * scale); c.height = Math.round(h * scale);
  c.getContext("2d").drawImage(ui.video, 0, 0, c.width, c.height);
  return new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("could not encode picture"))), "image/jpeg", S.jpegQuality));
}

function drawOverlay(sel, problem) {
  const ctx = ui.overlay.getContext("2d");
  ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  ctx.lineWidth = Math.max(2, ui.overlay.width / 400);
  for (const f of [...sel.others, ...(sel.main ? [sel.main] : [])]) {
    const [x0, y0, x1, y1] = f.bbox;
    ctx.strokeStyle = f === sel.main && !problem ? "#2fbf71" : "#e5484d";
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function updateUi(st) {
  ui.instruction.textContent = st.instruction;
  ui.hint.textContent = st.hint || "";
  ui.hint.hidden = !st.hint;
  [...ui.dots.children].forEach((d, i) => d.classList.toggle("done", i < guide.step));
  ui.timer.textContent = st.step > 0 ? `${Math.ceil(st.timeLeft)} s` : "";
  // arrow on the side the student should turn to, as they see themselves (mirrored preview)
  ui.instruction.dataset.arrow = st.direction ? ((st.direction === "left") === MIRRORED ? "left" : "right") : "";
  // head-turn bar: + turn = student's left = drawn on the left in a mirrored preview
  ui.bar.hidden = st.step === 0;
  const sgn = MIRRORED ? -1 : 1;
  const pos = (v) => 50 + sgn * (Math.max(-BAR_RANGE, Math.min(BAR_RANGE, v)) / BAR_RANGE) * 45;
  ui.marker.style.left = `${pos(st.turn)}%`;
  if (st.direction) {
    ui.target.hidden = false; ui.band.hidden = true;
    ui.target.style.left = `${pos((st.direction === "left" ? 1 : -1) * st.target)}%`;
  } else {
    ui.target.hidden = true; ui.band.hidden = false;
    const half = (G.guide_centre_tolerance / BAR_RANGE) * 45;
    ui.band.style.left = `${50 - half}%`; ui.band.style.width = `${2 * half}%`;
  }
}

function flash() {
  ui.stage.classList.remove("flash");
  void ui.stage.offsetWidth;                         // restart the CSS animation
  ui.stage.classList.add("flash");
}

function tick() {
  if (!running) return;
  const nowMs = performance.now();
  if (nowMs - lastDetect < 60 || ui.video.readyState < 2) {   // ~15 analyses per second at most
    requestAnimationFrame(tick); return;
  }
  lastDetect = nowMs;
  const now = nowMs / 1000;
  updateLetterbox(now);
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  const sel = selectMainFace(facesFromResult(landmarker.detect(ui.video), w, h),
                             G.duplicate_overlap, G.secondary_face_min_area_ratio);
  const quality = sel.main && !sel.others.length ? measureQuality(sel.main) : null;
  const problem = frameProblem(sel, h - lb.top - lb.bottom, quality, G);
  const st = guide.update(now, sel.main, problem);
  drawOverlay(sel, problem);
  updateUi(st);
  if (st.capture) { shots.push(grabFrame()); flash(); }
  if (st.aborted) { finishAborted(); return; }
  if (st.done) { finishCaptured(); return; }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------------------ the flow
async function begin() {
  if (consent && !consent.checked) { setStatus("Please tick the consent box first.", "error"); return; }
  ui.start.hidden = true; ui.retry.hidden = true; ui.result.hidden = true;
  try {
    await loadLandmarker();
  } catch (err) {
    console.error(err);
    setStatus("The face detector could not be loaded. Reload the page and try again.", "error");
    ui.retry.hidden = false; return;
  }
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    stopCamera();
    setStatus(cameraErrorMessage(err), "error");
    ui.retry.hidden = false; return;
  }
  setStatus("");
  guide = new GuidedCapture(G);
  shots = [];
  ui.cancel.hidden = false;
  captureStart = performance.now();
  running = true;
  requestAnimationFrame(tick);
}

function finishAborted() {
  stopCamera();
  ui.cancel.hidden = true;
  showResult("retry", "No face found", G.hints.no_face + ". Make sure your face is visible and well lit, then try again.",
             [button("Try again", "primary", begin)]);
}

async function finishCaptured() {
  stopCamera();
  ui.cancel.hidden = true;
  setStatus("Checking… this takes a few seconds.", "busy");
  let blobs;
  try {
    blobs = await Promise.all(shots);
  } catch (err) {
    setStatus("The pictures could not be prepared. Please try again.", "error");
    ui.retry.hidden = false; return;
  }
  const form = new FormData();
  blobs.forEach((b, i) => form.append("frames", b, `frame_${i}.jpg`));
  if (S.mode === "enroll") form.append("consent", consent && consent.checked ? "yes" : "no");
  // capture duration (camera ready -> 5 frames encoded), for the end-to-end time in the attempts log
  form.append("capture_ms", String(Math.round(performance.now() - captureStart)));
  let resp, data = {};
  try {
    resp = await fetch(S.uploadUrl, { method: "POST", body: form, credentials: "same-origin",
                                      headers: { "X-CSRFToken": csrfToken, "Accept": "application/json" } });
    data = await resp.json().catch(() => ({}));
  } catch (err) {
    setStatus("Could not reach the server. Check your connection and try again.", "error");
    ui.retry.hidden = false; return;
  }
  setStatus("");
  showServerResult(resp, data);
}

function showServerResult(resp, data) {
  const again = button("Try again", "primary", begin);
  const home = link(S.mode === "enroll" ? "Go to my page" : "Back to my page", S.doneUrl, "btn");
  if (!resp.ok) {
    if (resp.status === 401) { showResult("fail", "Logged out", "Your session ended. Please log in again.", [link("Log in", "/login")]); return; }
    if (data.error === "already_enrolled" || data.error === "not_enrolled") {
      showResult("retry", "Nothing to do", data.message || "", [link("Go to my page", S.doneUrl)]); return;
    }
    showResult("fail", "Something went wrong", data.message || `The server answered with an error (${resp.status}).`, [again, home]);
    return;
  }
  if (data.decision === "pass") {
    if (S.mode === "enroll") showResult("pass", "Enrollment complete", data.message, [link("Go to my page", S.doneUrl)]);
    else showResult(data.duplicate ? "retry" : "pass", data.duplicate ? "Already checked in" : "You are checked in",
                    data.message, [link("Back to my page", S.doneUrl)]);
  } else if (data.decision === "retry") {
    showResult("retry", "Please try again", data.message, [again, home]);
  } else {
    showResult("fail", S.mode === "enroll" ? "Enrollment failed" : "Check-in failed", data.message, [again, home]);
  }
}

// ------------------------------------------------------------------------------ wiring
ui.start.addEventListener("click", begin);
ui.retry.addEventListener("click", begin);
ui.cancel.addEventListener("click", () => {
  stopCamera(); ui.cancel.hidden = true; ui.start.hidden = false;
  setStatus("Cancelled.");
});
if (consent) {
  const sync = () => { ui.start.disabled = !consent.checked; };
  consent.addEventListener("change", sync); sync();
}
for (let i = 0; i < G.num_frames; i++) ui.dots.append(document.createElement("span"));
window.addEventListener("pagehide", stopCamera);
