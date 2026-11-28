"use strict";
const test = require("node:test");
const assert = require("node:assert");
const P = require("../js/playback.js");
const H = require("../js/heatmap.js");

const frames = [0, 0.25, 0.5, 0.75, 1.0].map((t) => ({ t, people: [{ id: 1, x: t * 4, y: 0 }, { id: 2, x: 0, y: t }] }));
const tracks = P.buildTracks(frames);

test("dot position is interpolated between samples", () => {
  assert.deepStrictEqual(P.posAt(tracks.get(1), 0.375), [1.5, 0]);
});

test("a short detection gap is bridged, a long absence hides the dot", () => {
  const tr = [[0, 0, 0], [0.75, 3, 0], [5, 9, 9]];
  assert.deepStrictEqual(P.posAt(tr, 0.375), [1.5, 0]);
  assert.strictEqual(P.posAt(tr, 3), null);
  assert.strictEqual(P.posAt(tr, -1), null);
});

test("trail only reaches the current time", () => {
  assert.deepStrictEqual(P.trailUpTo(tracks.get(1), 0.375), [[0, 0], [1, 0], [1.5, 0]]);
});

test("everyone is shown until an abnormal behaviour starts, then only that person with a trail", () => {
  const events = [{ id: 2, behavior: "touching", start: 0.5 }];
  const before = P.sceneAt(tracks, events, 0.25);
  assert.strictEqual(before.filtered, false);
  assert.deepStrictEqual(before.people.map((p) => p.id), [1, 2]);
  assert.strictEqual(before.trails.length, 0);
  const after = P.sceneAt(tracks, events, 0.75);
  assert.strictEqual(after.filtered, true);
  assert.deepStrictEqual(after.people, []);
  assert.deepStrictEqual(after.trails.map((r) => r.id), [2]);
  assert.deepStrictEqual(after.trails[0].pts.at(-1), [0, 0.75]);
});

test("the most severe behaviour wins and later offenders are added", () => {
  const events = [{ id: 2, behavior: "stealing", start: 0.9 }, { id: 2, behavior: "touching", start: 0.1 }, { id: 1, behavior: "holding", start: 0.6 }];
  assert.strictEqual(P.flaggedAt(events, 0.5).get(2).behavior, "touching");
  assert.strictEqual(P.flaggedAt(events, 1).get(2).behavior, "stealing");
  assert.strictEqual(P.flaggedAt(events, 1).size, 2);
});

test("framesUpTo keeps samples up to the current time", () => {
  assert.strictEqual(P.framesUpTo(frames, 0.5).length, 3);
  assert.strictEqual(P.framesUpTo(frames, -1).length, 0);
});

test("a fixed colour scale keeps partial heat on the full-run scale", () => {
  const run = (list, opts) => H.computeHeatmap(list, [-1, -1, 5, 2], { cell: 0.25, sigma: 0.5, fps: 4, filterStatic: false, ...(opts || {}) });
  const full = run(frames);
  const caps = Object.fromEntries(Object.entries(full.modes).map(([k, m]) => [k, m.cap]));
  const part = run(frames.slice(0, 2), { caps });
  assert.strictEqual(part.modes.presence.cap, full.modes.presence.cap);
  assert.ok(Math.max(...part.modes.presence.norm) < 1);
});

test("an anomaly is only 'cần xác nhận' until the real contact starts", () => {
  const raw = { decision: "ANOMALY", start_time_s: 0, end_time_s: 10, alarm_spans: [{ start_time_s: 7, end_time_s: 10 }] };
  assert.strictEqual(P.alarmOnset(raw, { alarm_start_time_s: 7 }), 7);
  assert.strictEqual(P.alarmOnset(raw), 7);
  assert.strictEqual(P.alarmOnset({ ...raw, decision: "CANDIDATE" }), null);
  assert.strictEqual(P.alarmOnset(raw, { alarm_start_time_s: 0 }), null);
  const events = [{ id: 1, start: 0, end: 10, onset: 7, decision: "ANOMALY", behavior: "touching", conf: 0.9 }];
  assert.deepStrictEqual(P.activeEventsAt(events, 1).map((e) => [e.decision, e.behavior]), [["CANDIDATE", "uncertain"]]);
  assert.deepStrictEqual(P.activeEventsAt(events, 8).map((e) => [e.decision, e.behavior]), [["ANOMALY", "touching"]]);
  assert.strictEqual(events[0].decision, "ANOMALY");
});

