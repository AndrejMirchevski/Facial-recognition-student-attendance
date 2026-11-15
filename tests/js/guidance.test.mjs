// Node.js tests for app/static/js/guidance.js - the same scenarios as tests/test_guidance.py.
// Run through pytest (tests/test_guidance_js.py), which passes the real settings from config.py
// in the GUIDE_CONFIG environment variable (a JSON file path).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { Face, GuidedCapture, IRIS_LEFT, IRIS_RIGHT, NOSE_TIP, faceQuality, frameProblem, selectMainFace,
         similarityResiduals } from "../../app/static/js/guidance.js";

const CFG = JSON.parse(readFileSync(process.env.GUIDE_CONFIG, "utf8"));

function rng(seed) {            // small deterministic PRNG
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function faceWithYaw(yaw, iod = 100) {
  const r = rng(1);
  const pts = Array.from({ length: 478 }, () => [320 + (r() * 240 - 120), 320 + (r() * 300 - 150)]);
  pts[IRIS_LEFT] = [320 - iod / 2, 300];
  pts[IRIS_RIGHT] = [320 + iod / 2, 300];
  pts[NOSE_TIP] = [320 + yaw * iod, 350];
  return new Face(pts);
}

function feed(g, t, yaw, seconds, problem = null, dt = 0.05) {
  const caps = [];
  const end = t + seconds;
  while (t < end) {
    const st = g.update(t, yaw === null ? null : faceWithYaw(yaw), problem);
    if (st.capture) caps.push(st);
    if (st.done) break;
    t += dt;
  }
  return [t, caps];
}

test("yaw proxy sign and scale", () => {
  assert.ok(Math.abs(faceWithYaw(0).yawProxy) < 1e-9);
  assert.ok(Math.abs(faceWithYaw(0.15).yawProxy - 0.15) < 1e-9);
});

test("real head follows the steps", () => {
  const g = new GuidedCapture(CFG);
  let t, caps;
  [t, caps] = feed(g, 0, 0, 1.0); assert.equal(caps.length, 1); assert.equal(g.step, 1);
  [t, caps] = feed(g, t, 0.05, 1.0); assert.equal(caps.length, 0);
  [t, caps] = feed(g, t, 0.15, 0.2); assert.equal(caps.length, 1); assert.equal(g.step, 2);
  [t, caps] = feed(g, t, 0.0, 0.2); assert.equal(caps.length, 1); assert.equal(g.step, 3);
  [t, caps] = feed(g, t, 0.15, 1.0); assert.equal(caps.length, 0);           // same side again: no
  [t, caps] = feed(g, t, -0.15, 0.2); assert.equal(caps.length, 1);
  [t, caps] = feed(g, t, 0.0, 0.2); assert.equal(caps.length, 1); assert.ok(caps[0].done);
});

test("turning right first then asks for left", () => {
  const g = new GuidedCapture(CFG);
  let [t] = feed(g, 0, 0, 1.0);
  let caps;
  [t, caps] = feed(g, t, -0.15, 0.2); assert.equal(caps.length, 1);
  [t] = feed(g, t, 0.0, 0.2);
  const st = g.update(t, faceWithYaw(0), null);
  assert.ok(st.instruction.includes("LEFT")); assert.equal(st.direction, "left");
});

test("photo that cannot turn is still captured after timeouts", () => {
  const g = new GuidedCapture(CFG);
  const [t] = feed(g, 0, 0, 60);
  assert.equal(g.step, 5);
  assert.ok(t < 1 + 2 * CFG.guide_step_timeout_s + 2);
});

test("too close gives hint and waits", () => {
  const g = new GuidedCapture(CFG);
  const st = g.update(0, faceWithYaw(0), "face_too_close");
  assert.equal(st.hint, CFG.hints.face_too_close); assert.equal(st.capture, false);
  const [, caps] = feed(g, 0, 0, 5, "face_too_close");
  assert.equal(caps.length, 0);
});

test("nobody in front of the camera aborts", () => {
  const g = new GuidedCapture(CFG);
  const [t] = feed(g, 0, null, CFG.guide_first_timeout_s + 1, "no_face");
  assert.ok(g.update(t, null, "no_face").done);
});

test("duplicate face detection is ignored, real second face is not", () => {
  const box = (x0, y0, x1, y1) => {
    const r = rng(3);
    const pts = Array.from({ length: 478 }, () => [x0 + r() * (x1 - x0), y0 + r() * (y1 - y0)]);
    pts[0] = [x0, y0]; pts[1] = [x1, y0]; pts[2] = [x0, y1]; pts[3] = [x1, y1];
    return new Face(pts);
  };
  const main = box(100, 100, 300, 350);
  let s = selectMainFace([box(150, 160, 250, 280), main], CFG.duplicate_overlap, CFG.secondary_face_min_area_ratio);
  assert.equal(s.main, main); assert.equal(s.others.length, 0); assert.equal(s.ignored.length, 1);
  s = selectMainFace([box(50, 100, 250, 350), box(350, 120, 530, 340)], CFG.duplicate_overlap, CFG.secondary_face_min_area_ratio);
  assert.equal(s.others.length, 1);
  assert.equal(frameProblem(s, 720, null, CFG), "multiple_faces");
});

test("similarity residuals are zero for a pure similarity transform", () => {
  const src = [[0, 0], [10, 0], [0, 10], [7, 3]];
  const dst = src.map(([x, y]) => [2 * (x * Math.cos(0.3) - y * Math.sin(0.3)) + 5, 2 * (x * Math.sin(0.3) + y * Math.cos(0.3)) - 1]);
  assert.ok(Math.max(...similarityResiduals(src, dst)) < 1e-9);
});

test("face quality matches the Python definitions", () => {
  const size = 112, rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) { const v = (i % 2) * 200; rgba.set([v, v, v, 255], 4 * i); }
  const q = faceQuality(rgba);
  assert.ok(Math.abs(q.brightness - 100) < 1e-9);
  assert.ok(q.sharpness > 1000);
  const flat = new Uint8ClampedArray(size * size * 4).fill(120);
  assert.equal(faceQuality(flat).sharpness, 0);
});
