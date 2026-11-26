"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const EFHeat = require("../js/heatmap.js");

test("Fix 1a: [0,1] normalized map uses coordinate-aware peakRadius instead of scanning whole map", () => {
  const framesDict = {};
  const keys = [];
  // Two well-separated moving visitors: #1 oscillates near (0.2, 0.2), #2 oscillates near (0.8, 0.8)
  for (let i = 0; i < 40; i++) {
    const k = i * 5; // 0.2s per key at 25fps => 8.0s total
    keys.push(k);
    const dx = (i % 6) * 0.006; // max disp ~0.018 > staticRadius(0.015), stays within 0.05 of center
    framesDict[k] = [
      [1, 0.2 + dx, 0.2],
      [2, 0.8 + dx, 0.8],
    ];
  }
  // Make cluster 1 slightly heavier by adding visitor #3 near (0.2, 0.2)
  for (let i = 0; i < 20; i++) {
    const k = i * 5;
    framesDict[k].push([3, 0.2 - (i % 6) * 0.006, 0.2]);
  }

  const H = EFHeat.computeFrameHeat(framesDict, keys, 25);
  assert.ok(H.peak, "expected a peak");
  assert.ok(Math.hypot(H.peak.x - 0.2, H.peak.y - 0.2) < 0.05, "peak should be near (0.2, 0.2)");
  // Total presence across whole map is ~8s + 8s + 4s = 20s; cluster 1 alone is ~12s.
  // Legacy rr = round(1 / (1/64)) = 64 summed all 20s!
  assert.ok(H.peakSec > 10 && H.peakSec < 14, `expected peakSec ~12s, got ${H.peakSec}`);
  assert.equal(H.peakVisitors, 2, "only the 2 visitors near (0.2, 0.2) should be counted");
});

test("Fix 1b: peak aggregation uses Euclidean circle, excluding square corners outside radius", () => {
  // Single peak at (5, 5) with radius = 1.0m, plus a visitor strictly in the square corner (5.85, 5.85)
  // dist = hypot(0.85, 0.85) = 1.202m > 1.0m, but |dx|=0.85<=1 and |dy|=0.85<=1.
  const frames = [];
  for (let i = 0; i < 20; i++) {
    const t = i * 0.25;
    const people = [
      { id: 1, x: 5.0 + (i < 4 ? i * 0.15 : 0), y: 5.0 }, // moves 0.45m then stays at (5,5)
      { id: 2, x: 5.0 - (i < 4 ? i * 0.15 : 0), y: 5.0 },
    ];
    if (i < 10) {
      // Corner visitor moving around (5.88, 5.88) -> always outside R=1.0m from (5,5)
      people.push({ id: 3, x: 5.88 + i * 0.06, y: 5.88 + i * 0.06 });
    }
    frames.push({ t, people });
  }

  const H = EFHeat.computeHeatmap(frames, [0, 0, 10, 10], {
    cell: 0.25,
    sigma: 0.5,
    peakRadius: 1.0,
    isMetric: true,
    filterStatic: false,
  });
  assert.ok(H.peak, "expected peak");
  assert.ok(Math.hypot(H.peak.x - 5.0, H.peak.y - 5.0) < 0.35);
  assert.equal(H.peakVisitors, 2, "corner visitor at dist > 1.0m must not be counted in disk");
  // 2 visitors * 20 frames * 0.25s = 10s (minus tiny edge splat if any), must not include #3's 2.5s
  assert.ok(H.peakSec <= 10.2, `expected peakSec <= 10.2s, got ${H.peakSec}`);
});

test("Fix 2: percentile clipping + log tone mapping keeps walkways visible alongside heavy hotspot", () => {
  const frames = [];
  // 1 visitor walks in (0..2s) and stands at (2, 2) for 120s (total 488 frames at 4fps)
  // 2 visitors walk smoothly along a corridor y=8 from x=1 to x=9
  for (let i = 0; i < 488; i++) {
    const t = i * 0.25;
    const people = [];
    if (i < 8) {
      people.push({ id: 1, x: 0.5 + (i / 8) * 1.5, y: 2.0 });
    } else {
      people.push({ id: 1, x: 2.0, y: 2.0 });
    }
    if (i >= 10 && i < 42) {
      const u = (i - 10) / 32;
      people.push({ id: 2, x: 1.0 + u * 8.0, y: 8.0 });
    }
    if (i >= 50 && i < 82) {
      const u = (i - 50) / 32;
      people.push({ id: 3, x: 9.0 - u * 8.0, y: 8.0 });
    }
    frames.push({ t, people });
  }

  const H = EFHeat.computeHeatmap(frames, [0, 0, 10, 10], {
    cell: 0.25,
    sigma: 0.6,
    peakRadius: 1.0,
    isMetric: true,
  });

  const M = H.modes.presence;
  // Inspect corridor midpoint (5.0, 8.0)
  const ci = Math.floor(5.0 / H.cell);
  const cj = Math.floor(8.0 / H.cell);
  const idx = cj * H.nx + ci;
  const linearVal = M.grid[idx] / M.max;
  const mappedVal = M.norm[idx];

  assert.ok(linearVal < 0.04, `test setup: linear value (${linearVal.toFixed(4)}) should have been washed out (<0.04)`);
  assert.ok(mappedVal >= 0.04, `mapped corridor intensity (${mappedVal.toFixed(4)}) must be visible (>=0.04)`);
});

test("Fix 3: per-frame timestamp deltas handle strided keys, duplicate timestamps, and long gaps", () => {
  // Strided keys at 25fps: every 5th frame = 0.2s/sample, 20 samples = 4.0s
  const framesDict = {};
  const keys = [];
  for (let i = 0; i < 20; i++) {
    const k = i * 5;
    keys.push(k);
    framesDict[k] = [[1, 0.2 + i * 0.02, 0.5]];
  }
  const H = EFHeat.computeFrameHeat(framesDict, keys, 25);
  assert.ok(
    Math.abs(H.analytics.totalPresenceSec - 4.0) < 0.05,
    `expected ~4.0s totalPresenceSec for strided keys, got ${H.analytics.totalPresenceSec}`
  );

  // Irregular timestamps with duplicate and huge 60s gap
  const { dts, durationSec } = EFHeat.computeFrameDeltas([0, 0.5, 0.5, 1.0, 61.0, 61.5], 0.5);
  assert.equal(dts[0], 0.5);
  assert.equal(dts[1], 0, "duplicate timestamp must produce dt = 0");
  assert.equal(dts[2], 0.5);
  assert.ok(dts[3] <= 2.0, `60s gap must be clamped to maxGap (<=2.0s), got ${dts[3]}`);
  assert.equal(dts[4], 0.5);
  assert.equal(dts[5], 0.5);
  assert.ok(durationSec <= 4.1, `clamped duration should be <= 4.1s, got ${durationSec}`);
});

test("Fix 4: filters static ghost #29 while keeping real visitors and separating presence/visitors/dwell", () => {
  const frames = [];
  // Total 30s at 2fps (60 frames)
  for (let i = 0; i < 60; i++) {
    const t = i * 0.5;
    const people = [];

    // Static ghost #29 (painting on West wall at x=0.5, y=5.0) with tiny jitter < 0.1m for whole 30s
    people.push({
      id: 29,
      x: 0.5 + Math.sin(i) * 0.04,
      y: 5.0 + Math.cos(i) * 0.04,
      cams: ["C1"],
    });

    // Real visitor #1: walks from (1, 1) to Exhibit B (8, 8) in 5s, then stands at (8, 8) for 20s
    if (i < 50) {
      if (i < 10) {
        const u = i / 10;
        people.push({ id: 1, x: 1.0 + u * 7.0, y: 1.0 + u * 7.0, cams: ["C1", "C2"] });
      } else {
        people.push({ id: 1, x: 8.0, y: 8.0, cams: ["C2"] });
      }
    }

    // 6 real visitors (#10..#15) walking briskly past Zone A (5, 2) at ~1.5 m/s (no dwell!)
    for (let v = 0; v < 6; v++) {
      const start = v * 6;
      const local = i - start;
      if (local >= 0 && local < 6) {
        people.push({ id: 10 + v, x: 3.5 + local * 0.75, y: 2.0, cams: ["C1"] });
      }
    }

    frames.push({ t, people });
  }

  const H = EFHeat.computeSiteHeat({
    bounds: [0, 0, 10, 10],
    fps: 2,
    frames,
  });

  // Ghost #29 MUST be filtered out; real visitor #1 and #10..#15 MUST remain
  assert.deepEqual(H.analytics.filteredStaticIds, [29]);
  assert.equal(H.analytics.rawVisitors, 8);
  assert.equal(H.analytics.totalVisitors, 7);

  // Presence peak: Exhibit B (8, 8) where visitor #1 spent 20s
  assert.ok(H.modes.presence.peak);
  assert.ok(Math.hypot(H.modes.presence.peak.x - 8.0, H.modes.presence.peak.y - 8.0) < 1.2);
  assert.equal(H.modes.presence.meta.peakLabel, "Tập trung nhất");

  // Visitors peak: Zone A (~5, 2) where 6 distinct visitors walked through
  assert.ok(H.modes.visitors.peak);
  assert.ok(Math.hypot(H.modes.visitors.peak.x - 5.2, H.modes.visitors.peak.y - 2.0) < 1.5);
  assert.equal(H.modes.visitors.peak.uniqueVisitors, 6);
  assert.equal(H.modes.visitors.meta.peakLabel, "Đông khách nhất");

  // Dwell peak: Exhibit B (8, 8) where visitor #1 stopped for ~20s
  assert.ok(H.modes.dwell.peak);
  assert.ok(Math.hypot(H.modes.dwell.peak.x - 8.0, H.modes.dwell.peak.y - 8.0) < 1.2);
  assert.ok(H.modes.dwell.peak.dwellSec >= 17, `expected dwellSec >= 17s, got ${H.modes.dwell.peak.dwellSec}`);
  assert.equal(H.modes.dwell.meta.peakLabel, "Dừng lâu nhất");
});

test("Edge cases: empty, single-frame, degenerate bounds, NaN coords, and all-ghost frames", () => {
  // Empty frames
  const e1 = EFHeat.computeHeatmap([], [0, 0, 10, 10]);
  assert.equal(e1.peak, null);
  assert.equal(e1.people, 0);

  // Degenerate bounds
  const e2 = EFHeat.computeHeatmap([{ t: 0, people: [{ id: 1, x: 5, y: 5 }] }], [5, 5, 5, 5]);
  assert.equal(e2.peak, null);

  // Single frame with NaN + out-of-bounds + 1 valid point
  const e3 = EFHeat.computeHeatmap(
    [
      {
        t: 0,
        people: [
          { id: 1, x: NaN, y: 2 },
          { id: 2, x: 999, y: 999 },
          { id: 3, x: 4, y: 4 },
        ],
      },
    ],
    [0, 0, 10, 10],
    { fps: 4, isMetric: true }
  );
  assert.ok(e3.peak, "valid single-frame point should produce a peak");
  assert.ok(Number.isFinite(e3.peak.x) && Number.isFinite(e3.peak.y));
  assert.equal(e3.people, 1, "out-of-bounds and NaN tracks must not be counted in people");
  assert.equal(e3.analytics.totalVisitors, 1);
  assert.ok(Math.abs(e3.analytics.avgVisitSec - 0.25) < 1e-6, "avgVisitSec must only divide by in-bounds visitors");

  // All-ghost frames -> empty heatmap but filteredStaticIds recorded
  const ghostFrames = Array.from({ length: 30 }, (_, i) => ({
    t: i * 0.5,
    people: [{ id: 29, x: 2.0, y: 2.0 }],
  }));
  const e4 = EFHeat.computeSiteHeat({ bounds: [0, 0, 10, 10], fps: 2, frames: ghostFrames });
  assert.equal(e4.people, 0);
  assert.equal(e4.peak, null);
  assert.deepEqual(e4.analytics.filteredStaticIds, [29]);
});

test("Performance: 1000 frames x 15 visitors computes in < 250ms", () => {
  const frames = [];
  for (let k = 0; k < 1000; k++) {
    const t = k * 0.25;
    const people = [];
    for (let id = 1; id <= 15; id++) {
      people.push({
        id,
        x: 2 + ((id * 1.7 + k * 0.03) % 16),
        y: 2 + ((id * 2.3 + Math.sin(k * 0.05 + id) * 3 + 10) % 12),
      });
    }
    frames.push({ t, people });
  }
  const t0 = performance.now();
  const H = EFHeat.computeSiteHeat({ bounds: [0, 0, 20, 16], fps: 4, frames });
  const elapsed = performance.now() - t0;
  assert.ok(H.peak);
  assert.equal(H.people, 15);
  assert.ok(elapsed < 250, `expected < 250ms, took ${elapsed.toFixed(1)}ms`);
});

test("Regression: boundary bilinear splat preserves 100% mass along walls and corners", () => {
  const wallFrames = Array.from({ length: 40 }, (_, i) => ({
    t: i * 0.25,
    people: [{ id: 1, x: 0.0, y: 5.0 }],
  }));
  const Hw = EFHeat.computeHeatmap(wallFrames, [0, 0, 10, 10], {
    cell: 0.25,
    sigma: 0.5,
    peakRadius: 1.0,
    filterStatic: false,
  });
  assert.ok(Math.abs(Hw.peakSec - 10.0) < 0.05, `expected wall peakSec ~10.0s, got ${Hw.peakSec}`);

  const cornerFrames = Array.from({ length: 40 }, (_, i) => ({
    t: i * 0.25,
    people: [{ id: 1, x: 0.0, y: 0.0 }],
  }));
  const Hc = EFHeat.computeHeatmap(cornerFrames, [0, 0, 10, 10], {
    cell: 0.25,
    sigma: 0.5,
    peakRadius: 1.0,
    filterStatic: false,
  });
  assert.ok(Math.abs(Hc.peakSec - 10.0) < 0.05, `expected corner peakSec ~10.0s, got ${Hc.peakSec}`);
});

test("Regression: visitors mode ranks 3 straight-walking visitors above 1 pacing visitor and fills strided segments", () => {
  const frames = [];
  for (let i = 0; i < 60; i++) {
    const t = i * 0.5;
    const people = [];
    // 1 visitor (#1) pacing inside a 2m x 2m box around (8, 8)
    const gx = (i % 7) * 0.3 - 0.9;
    const gy = (Math.floor(i / 7) % 7) * 0.3 - 0.9;
    people.push({ id: 1, x: 8.0 + gx, y: 8.0 + gy });

    // 3 distinct visitors (#2, #3, #4) walking straight through doorway (2, 2) with 1.0m steps (strided)
    for (let v = 0; v < 3; v++) {
      const local = i - v * 10;
      if (local >= 0 && local < 5) {
        people.push({ id: 2 + v, x: 0.5 + local * 1.0, y: 2.0 });
      }
    }
    frames.push({ t, people });
  }
  const H = EFHeat.computeSiteHeat({ bounds: [0, 0, 10, 10], fps: 2, frames });
  assert.ok(H.modes.visitors.peak, "expected visitors peak");
  assert.ok(
    Math.hypot(H.modes.visitors.peak.x - 2.5, H.modes.visitors.peak.y - 2.0) < 1.8,
    `visitors peak should be on doorway y=2, got (${H.modes.visitors.peak.x}, ${H.modes.visitors.peak.y})`
  );
  assert.equal(H.modes.visitors.peak.uniqueVisitors, 3, "doorway with 3 visitors must beat 1 pacing visitor");
});

test("Regression: unsorted frames are sorted before delta computation and short-clip visitors are not falsely filtered", () => {
  // Unsorted frames
  const unsorted = [
    { t: 1.0, people: [{ id: 1, x: 3, y: 3 }] },
    { t: 0.0, people: [{ id: 1, x: 2, y: 3 }] },
    { t: 0.5, people: [{ id: 1, x: 2.5, y: 3 }] },
  ];
  const Hu = EFHeat.computeHeatmap(unsorted, [0, 0, 10, 10], { isMetric: true, filterStatic: false });
  assert.ok(Math.abs(Hu.analytics.totalPresenceSec - 1.5) < 0.05, `expected ~1.5s, got ${Hu.analytics.totalPresenceSec}`);

  // Short 5s clip where visitor #1 stands viewing an artwork for 3s (coverage = 0.60)
  const shortClip = Array.from({ length: 20 }, (_, i) => ({
    t: i * 0.25,
    people: i < 12 ? [{ id: 1, x: 5.0, y: 5.0 }] : [],
  }));
  const Hs = EFHeat.computeSiteHeat({ bounds: [0, 0, 10, 10], fps: 4, frames: shortClip });
  assert.equal(Hs.people, 1, "visitor viewing artwork for 3s in a 5s clip must not be deleted as static ghost");
  assert.deepEqual(Hs.analytics.filteredStaticIds, []);
});
