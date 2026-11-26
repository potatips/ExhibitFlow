"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const EFHeat = require("../js/heatmap.js");
const EFAnalytics = require("../js/museum_analytics.js");

/**
 * Helper: build frames array from per-visitor waypoint paths.
 * waypoints: Array<[t, x, y]>
 */
function buildFramesFromTracks(trackSpecs, dt = 0.5, duration = 60) {
  const frames = [];
  const n = Math.floor(duration / dt) + 1;
  for (let k = 0; k < n; k++) {
    const t = +(k * dt).toFixed(4);
    const people = [];
    for (const spec of trackSpecs) {
      const wp = spec.pts;
      if (t < wp[0][0] - 1e-6 || t > wp[wp.length - 1][0] + 1e-6) continue;
      let i = 1;
      while (i < wp.length - 1 && wp[i][0] < t) i++;
      const a = wp[i - 1];
      const b = wp[i];
      const u = b[0] === a[0] ? 0 : Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0])));
      people.push({
        id: spec.id,
        x: a[1] + (b[1] - a[1]) * u,
        y: a[2] + (b[2] - a[2]) * u,
      });
    }
    frames.push({ t, people });
  }
  return frames;
}

test("Feature 1: Attraction Power vs Holding Power classifies all 4 quadrants accurately", () => {
  const bounds = [0, 0, 20, 15];
  const exhibits = [
    { id: "E1", name: "Ngôi sao", x: 4, y: 4 },
    { id: "E2", name: "Lướt nhanh", x: 10, y: 4 },
    { id: "E3", name: "Viên ngọc ẩn", x: 16, y: 4 },
    { id: "E4", name: "Vùng lạnh", x: 10, y: 12 },
  ];

  // 4 visitors passing by E1, E2, E3, E4:
  // - E1 (star): all 4 stop for 14s
  // - E2 (skim): all 4 stop for 5s (>=3s, <10s)
  // - E3 (hidden_gem): only visitor 1 stops for 15s; visitors 2,3,4 walk straight past
  // - E4 (cold): all 4 walk past quickly without stopping
  const tracks = [
    {
      id: 1,
      pts: [
        [0, 1, 4], [2, 4, 4], [16, 4, 4], // stop 14s at E1
        [20, 10, 4], [25, 10, 4],         // stop 5s at E2
        [29, 16, 4], [44, 16, 4],         // stop 15s at E3
        [50, 10, 12], [54, 15, 12],       // pass E4 without stopping
      ],
    },
    {
      id: 2,
      pts: [
        [0, 1, 4.3], [2, 4, 4.3], [16, 4, 4.3], // stop 14s at E1
        [20, 10, 4.3], [25, 10, 4.3],           // stop 5s at E2
        [30, 16, 4.3], [33, 19, 4.3],           // walk past E3
        [40, 10, 12.2], [44, 15, 12.2],         // walk past E4
      ],
    },
    {
      id: 3,
      pts: [
        [0, 1, 3.7], [2, 4, 3.7], [16, 4, 3.7], // stop 14s at E1
        [20, 10, 3.7], [25, 10, 3.7],           // stop 5s at E2
        [30, 16, 3.7], [33, 19, 3.7],           // walk past E3
        [40, 10, 11.8], [44, 15, 11.8],         // walk past E4
      ],
    },
    {
      id: 4,
      pts: [
        [0, 1, 4.1], [2, 4, 4.1], [16, 4, 4.1], // stop 14s at E1
        [20, 10, 4.1], [25, 10, 4.1],           // stop 5s at E2
        [30, 16, 4.1], [33, 19, 4.1],           // walk past E3
        [40, 10, 12.1], [44, 15, 12.1],         // walk past E4
      ],
    },
  ];

  const frames = buildFramesFromTracks(tracks, 0.5, 56);
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, exhibits });
  const byId = Object.fromEntries(res.attraction.items.map((it) => [it.id, it]));

  assert.equal(res.attraction.source, "exhibits");
  assert.equal(byId.E1.quadrant, "star");
  assert.equal(byId.E1.passers, 4);
  assert.equal(byId.E1.stoppers, 4);
  assert.equal(byId.E1.attractionPower, 1);
  assert.ok(byId.E1.holdingPowerSec >= 12);

  assert.equal(byId.E2.quadrant, "skim");
  assert.equal(byId.E2.stoppers, 4);
  assert.ok(byId.E2.holdingPowerSec >= 3 && byId.E2.holdingPowerSec < 10);

  assert.equal(byId.E3.quadrant, "hidden_gem");
  assert.equal(byId.E3.passers, 4);
  assert.equal(byId.E3.stoppers, 1);
  assert.equal(byId.E3.attractionPower, 0.25);
  assert.ok(byId.E3.holdingPowerSec >= 12);

  assert.equal(byId.E4.quadrant, "cold");
  assert.equal(byId.E4.stoppers, 0);
  assert.equal(byId.E4.attractionPower, 0);

  assert.deepEqual(res.attraction.quadrantCounts, { star: 1, skim: 1, hidden_gem: 1, cold: 1 });
});

test("Feature 1b: auto-discovers zones from hotspots when exhibits are not provided", () => {
  const bounds = [0, 0, 20, 15];
  const tracks = [
    { id: 1, pts: [[0, 1, 2], [4, 5, 5], [16, 5, 5], [22, 15, 10], [32, 15, 10], [38, 19, 12]] },
    { id: 2, pts: [[0, 2, 2], [4, 5.2, 5.1], [15, 5.2, 5.1], [22, 15.1, 9.9], [31, 15.1, 9.9], [38, 19, 11]] },
  ];
  const frames = buildFramesFromTracks(tracks, 0.5, 40);
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds });

  assert.equal(res.attraction.source, "auto_zones");
  assert.ok(res.attraction.items.length >= 2, "Should auto-discover at least 2 dwell zones");
});

test("Feature 2: Bottleneck Map detects corridor congestion and excludes exhibit viewing zones", () => {
  const bounds = [0, 0, 20, 15];
  const exhibits = [{ id: "E1", name: "Tượng trung tâm", x: 4, y: 4 }];

  // 4 visitors walk fast, gather at exhibit E1 (4,4), then get stuck together in doorway (12, 8)
  const tracks = [1, 2, 3, 4].map((id) => {
    const jitter = (id - 2.5) * 0.18;
    return {
      id,
      pts: [
        [0, 1, 1 + jitter],
        [3, 4 + jitter, 4 + jitter],      // fast walk to exhibit
        [11, 4 + jitter, 4 + jitter],     // dwell at exhibit (should be excluded!)
        [16, 12 + jitter, 8 + jitter],    // fast walk to doorway
        [26, 12.2 + jitter, 8 + jitter],  // crawl/stuck at doorway (bottleneck!)
        [32, 19, 8 + jitter],             // fast walk out
      ],
    };
  });

  const frames = buildFramesFromTracks(tracks, 0.5, 34);
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, exhibits });

  assert.equal(res.bottleneck.awayFromExhibits, true);
  assert.ok(res.bottleneck.hotspots.length >= 1, "Should detect doorway bottleneck");

  const top = res.bottleneck.hotspots[0];
  const distToDoor = Math.hypot(top.x - 12, top.y - 8);
  const distToExhibit = Math.hypot(top.x - 4, top.y - 4);
  assert.ok(distToDoor < 2.2, `Top bottleneck should be near doorway (12,8), got (${top.x.toFixed(1)},${top.y.toFixed(1)})`);
  assert.ok(distToExhibit > 2.0, "Exhibit zone should be excluded from corridor bottlenecks");
  assert.ok(top.peakConcurrent >= 3, `Expected peakConcurrent >= 3, got ${top.peakConcurrent}`);
  assert.ok(top.speedDropPct >= 0.45, `Expected speedDropPct >= 0.45, got ${top.speedDropPct}`);
});

test("Feature 4: Flow Map vectors, Zone Transitions, Skip Rate, and Museum Fatigue detection", () => {
  const bounds = [0, 0, 24, 12];
  const exhibits = [
    { id: "A", name: "Khu A", x: 4, y: 6 },
    { id: "B", name: "Khu B", x: 12, y: 6 },
    { id: "C", name: "Khu C", x: 20, y: 6 },
  ];

  // Visitors move strictly left-to-right (+x), dwell heavily in Q1-Q2 (at A), then rush through B & C in Q3-Q4 (museum fatigue!)
  const fatiguedTracks = [1, 2, 3].map((id) => {
    const dy = (id - 2) * 0.3;
    return {
      id,
      pts: [
        [0, 2, 6 + dy],
        [2, 4, 6 + dy],
        [18, 4, 6 + dy],   // long dwell in Q1-Q2 (0..18s)
        [24, 12, 6 + dy],  // speed up past B
        [30, 20, 6 + dy],  // rush past C
        [34, 23.5, 6 + dy],
      ],
    };
  });

  const frames = buildFramesFromTracks(fatiguedTracks, 0.5, 35);
  const shortOk = { flow: { minFatigueTrackSec: 8 } }; // dữ liệu giả dài 35 s, mặc định thật là 60 s
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, exhibits }, shortOk);

  // Flow vectors should predominantly point right (+x)
  assert.ok(res.flow.vectors.length >= 3, "Should produce dominant flow vectors");
  const avgUx = res.flow.vectors.reduce((s, v) => s + v.ux, 0) / res.flow.vectors.length;
  assert.ok(avgUx > 0.7, `Expected dominant flow along +x, got avgUx=${avgUx.toFixed(2)}`);

  // Zone transitions A -> B and B -> C
  const transKeys = res.flow.transitions.map((t) => `${t.fromId}->${t.toId}`);
  assert.ok(transKeys.includes("A->B"), "Should detect A->B transition");
  assert.ok(transKeys.includes("B->C"), "Should detect B->C transition");

  // Skip rate: A stopped by all (0% skip), B & C skipped by all (100% skip)
  const skipById = Object.fromEntries(res.skipRate.items.map((it) => [it.id, it]));
  assert.equal(skipById.A.passerSkipRate, 0);
  assert.equal(skipById.B.passerSkipRate, 1);
  assert.equal(skipById.C.passerSkipRate, 1);

  // Museum fatigue should be detected
  assert.equal(res.fatigue.eligibleVisitors, 3);
  assert.equal(res.fatigue.detected, true);
  assert.ok(res.fatigue.stopDropPct >= 0.5, `Expected stopDropPct >= 0.5, got ${res.fatigue.stopDropPct}`);
  assert.ok(res.fatigue.speedIncreasePct >= 0.5, `Expected speedIncreasePct >= 0.5, got ${res.fatigue.speedIncreasePct}`);
  assert.ok(res.fatigue.onsetQuartile !== null && res.fatigue.onsetQuartile >= 2);

  // Control: steady walkers with constant speed and no stops should NOT trigger fatigue
  const steadyTracks = [1, 2].map((id) => ({
    id,
    pts: [[0, 1, 4 + id], [20, 23, 4 + id]],
  }));
  const steadyFrames = buildFramesFromTracks(steadyTracks, 0.5, 20);
  const steadyRes = EFAnalytics.computeMuseumAnalytics({ frames: steadyFrames, bounds, exhibits }, shortOk);
  assert.equal(steadyRes.fatigue.detected, false);
});

test("Feature 5: Space Utilization & Dead-Zone Map respects camera FOV and clusters dead zones", () => {
  const bounds = [0, 0, 20, 10];
  // Camera observes [2..18, 1..9], leaving margins outside FOV
  const cams = [
    {
      id: "C1",
      center: [10, 1],
      fov: [[2, 1], [18, 1], [18, 9], [2, 9]],
    },
  ];
  // Visitors only walk in the left portion x in [3..7]
  const tracks = [
    { id: 1, pts: [[0, 3, 2], [10, 3, 8], [20, 6, 8], [30, 6, 2]] },
    { id: 2, pts: [[0, 4, 8], [15, 5, 2], [30, 7, 5]] },
  ];
  const frames = buildFramesFromTracks(tracks, 0.5, 30);
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, cams });

  assert.equal(res.space.usedCameraFov, true);
  assert.equal(res.space.activeCells + res.space.deadCells, res.space.observedCells);
  assert.ok(res.space.observedCells < res.space.nx * res.space.ny, "Out-of-FOV cells must be excluded from observed area");
  assert.ok(res.space.activeAreaRatio > 0.1 && res.space.activeAreaRatio < 0.7, `Unexpected activeAreaRatio: ${res.space.activeAreaRatio}`);
  assert.ok(res.space.deadAreaRatio > 0.3, `Expected deadAreaRatio > 0.3, got ${res.space.deadAreaRatio}`);
  assert.ok(res.space.deadZoneClusters.length >= 1, "Should detect at least one dead-zone cluster");
  assert.ok(res.space.deadZoneClusters[0].cx > 10, "Largest dead-zone cluster should be in the unvisited right half");
});

test("Feature 7: Verón & Levasseur Visitor Archetypes classifies Ant, Butterfly, Fish, Grasshopper", () => {
  const bounds = [0, 0, 20, 15];
  const exhibits = [
    { id: 1, name: "E1", x: 4, y: 4 },
    { id: 2, name: "E2", x: 16, y: 4 },
    { id: 3, name: "E3", x: 16, y: 12 },
    { id: 4, name: "E4", x: 4, y: 12 },
  ];

  const tracks = [
    // Ant (#1): covers all 4 exhibits across entire room, stops long at each (high coverage, high stopRatio)
    {
      id: 1,
      pts: [
        [0, 4, 4], [10, 4, 4],
        [15, 16, 4], [25, 16, 4],
        [30, 16, 12], [40, 16, 12],
        [45, 4, 12], [55, 4, 12],
      ],
    },
    // Butterfly (#2): covers all 4 exhibits across entire room, pauses briefly (high coverage, moderate stopRatio ~0.28)
    {
      id: 2,
      pts: [
        [0, 2, 2], [4, 4, 4], [6.5, 4, 4],
        [14, 16, 4], [16.5, 16, 4],
        [22, 16, 12], [24.5, 16, 12],
        [32, 4, 12], [34.5, 4, 12],
        [40, 18, 2],
      ],
    },
    // Fish (#3): glides quickly across the gallery without stopping (low stopRatio, fast speed)
    {
      id: 3,
      pts: [[0, 1, 7.5], [10, 19, 7.5]],
    },
    // Grasshopper (#4): goes straight to E1 in corner, dwells long, leaves (low coverage, high stopRatio)
    {
      id: 4,
      pts: [[0, 1, 1], [3, 4, 4], [18, 4, 4], [21, 1, 1]],
    },
  ];

  const frames = buildFramesFromTracks(tracks, 0.5, 56);
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, exhibits });
  const byId = Object.fromEntries(res.archetypes.visitors.map((v) => [v.id, v]));

  assert.equal(byId[1].archetype, "ant", `Visitor 1 expected ant, got ${byId[1].archetype} (cov=${byId[1].coverage.toFixed(2)}, stop=${byId[1].stopRatio.toFixed(2)})`);
  assert.equal(byId[2].archetype, "butterfly", `Visitor 2 expected butterfly, got ${byId[2].archetype} (cov=${byId[2].coverage.toFixed(2)}, stop=${byId[2].stopRatio.toFixed(2)})`);
  assert.equal(byId[3].archetype, "fish", `Visitor 3 expected fish, got ${byId[3].archetype} (cov=${byId[3].coverage.toFixed(2)}, stop=${byId[3].stopRatio.toFixed(2)})`);
  assert.equal(byId[4].archetype, "grasshopper", `Visitor 4 expected grasshopper, got ${byId[4].archetype} (cov=${byId[4].coverage.toFixed(2)}, stop=${byId[4].stopRatio.toFixed(2)})`);

  assert.deepEqual(res.archetypes.counts, { ant: 1, butterfly: 1, fish: 1, grasshopper: 1 });
  const shareSum = Object.values(res.archetypes.shares).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(shareSum - 1) < 1e-9);
});

test("Edge cases: empty frames, single frame, normalized [0,1] coords, and negative bounds", () => {
  // 1. Empty frames
  const emptyRes = EFAnalytics.computeMuseumAnalytics({ frames: [], bounds: [0, 0, 10, 10] });
  assert.equal(emptyRes.meta.totalVisitors, 0);
  assert.equal(emptyRes.attraction.items.length, 0);
  assert.equal(emptyRes.bottleneck.summary.count, 0);
  assert.equal(emptyRes.flow.vectors.length, 0);
  assert.equal(emptyRes.fatigue.detected, false);
  assert.equal(emptyRes.archetypes.total, 0);
  assert.equal(emptyRes.archetypes.dominant, null);

  // 2. Normalized [0, 1] single-camera map with fx/fy exhibits
  const framesDict = {};
  const keys = [];
  for (let f = 0; f <= 200; f += 10) {
    keys.push(f);
    const u = f / 200;
    framesDict[f] = [
      [1, 0.1 + 0.8 * u, 0.5],
      [2, 0.25, f < 100 ? 0.3 : 0.3 + 0.003 * (f - 100)],
    ];
  }
  const heatNorm = EFHeat.computeFrameHeat(framesDict, keys, 25);
  const normRes = EFAnalytics.computeMuseumAnalytics({
    heatmap: heatNorm,
    exhibits: [{ id: 1, name: "Tranh", fx: 0.25, fy: 0.3 }],
  });
  assert.equal(normRes.meta.isMetric, false);
  assert.equal(normRes.attraction.items.length, 1);
  assert.ok(normRes.space.activeAreaRatio > 0 && normRes.space.activeAreaRatio <= 1);
  assert.equal(normRes.archetypes.total, 2);

  // 3. Negative metric bounds [-10, -10, 10, 10]
  const negTracks = [{ id: 1, pts: [[0, -8, -8], [10, 8, 8]] }];
  const negFrames = buildFramesFromTracks(negTracks, 0.5, 10);
  const negRes = EFAnalytics.computeMuseumAnalytics({ frames: negFrames, bounds: [-10, -10, 10, 10] });
  assert.equal(negRes.archetypes.total, 1);
  assert.ok(Number.isFinite(negRes.space.activeAreaRatio));
  assert.ok(negRes.flow.vectors.length > 0);
});

test("Performance: full analytics (1000 frames x 15 visitors) completes in < 350ms", () => {
  const bounds = [0, 0, 30, 20];
  const exhibits = [
    { id: 1, name: "E1", x: 6, y: 6 },
    { id: 2, name: "E2", x: 15, y: 6 },
    { id: 3, name: "E3", x: 24, y: 6 },
    { id: 4, name: "E4", x: 15, y: 15 },
  ];
  const frames = [];
  for (let k = 0; k < 1000; k++) {
    const t = k * 0.25;
    const people = [];
    for (let v = 1; v <= 15; v++) {
      const phase = (t * 0.12 + v * 0.7) % (2 * Math.PI);
      const x = 15 + 11 * Math.cos(phase + v * 0.2);
      const y = 10 + 7 * Math.sin(phase * 0.8 + v * 0.3);
      people.push({ id: v, x, y });
    }
    frames.push({ t, people });
  }

  const t0 = performance.now();
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds, exhibits });
  const elapsed = performance.now() - t0;

  assert.equal(res.meta.totalVisitors, 15);
  assert.ok(elapsed < 350, `Expected < 350ms, took ${elapsed.toFixed(1)}ms`);
});

test("Regression (Bottleneck): solo standing + 1-frame fast passerby does not trigger false bottleneck", () => {
  const frames = [];
  for (let k = 0; k < 40; k++) {
    const t = k * 0.25;
    const people = [{ id: 1, x: k >= 35 ? 5 + (k - 34) * 0.4 : 5 + (k % 2) * 0.02, y: 5 }];
    if (k >= 9 && k <= 14) {
      people.push({ id: 2, x: 4.65 + (k - 9) * 0.35, y: 5 });
    }
    frames.push({ t, people });
  }
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds: [0, 0, 10, 10] });
  assert.equal(res.bottleneck.hotspots.length, 0);
});

test("Regression (Flow): opposing 2-way traffic filters out zero/low-alignment phantom vectors", () => {
  const frames = [];
  for (let k = 0; k < 20; k++) {
    const t = k * 0.25;
    frames.push({
      t,
      people: [
        { id: 1, x: 1 + k * 0.35, y: 5 },
        { id: 2, x: 8 - k * 0.3, y: 5 },
      ],
    });
  }
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds: [0, 0, 10, 10] });
  for (const v of res.flow.vectors) {
    assert.ok(v.alignment >= 0.25, `Expected alignment >= 0.25, got ${v.alignment}`);
    assert.ok(Math.hypot(v.ux, v.uy) > 0.99, `Expected unit vector, got (${v.ux}, ${v.uy})`);
    assert.ok(Math.abs(v.angle - Math.atan2(v.uy, v.ux)) < 1e-6);
  }
});

test("Regression (Fatigue): mid-visit Q2 room transition followed by deep Q3-Q4 dwell is NOT fatigue", () => {
  const frames = [];
  for (let k = 0; k < 80; k++) {
    const t = k * 0.25;
    let x = 2;
    if (k < 20) x = 2 + (k % 2) * 0.01;
    else if (k < 40) x = 2 + (k - 20) * 0.35;
    else x = 9 + (k % 2) * 0.01;
    frames.push({ t, people: [{ id: 1, x, y: 5 }] });
  }
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds: [0, 0, 12, 10] });
  assert.equal(res.fatigue.detected, false);
  assert.equal(res.fatigue.onsetQuartile, null);
});

test("Fatigue is not judged for visits shorter than the 60 s default", () => {
  const frames = [];
  for (let i = 0; i <= 40; i++) frames.push({ t: i / 4, people: [{ id: 1, x: 1 + i * 0.1, y: 2 }] });
  const res = EFAnalytics.computeMuseumAnalytics({ frames, bounds: [0, 0, 10, 10] });
  assert.equal(res.fatigue.eligibleVisitors, 0);
  assert.equal(res.fatigue.detected, false);
});

test("A visitor sitting still for part of a short video stays in the heatmap", () => {
  // người ngồi băng ghế: chỉ rung nhẹ ~0.2 m, có mặt 65% video 10 s
  const frames = [];
  for (let i = 0; i < 40; i++) {
    const people = [{ id: 1, x: 1 + i * 0.1, y: 2 }];
    if (i >= 6 && i < 32) people.push({ id: 2, x: 5 + 0.2 * ((i % 5) / 4), y: 5 });
    frames.push({ t: i / 4, people });
  }
  const heat = EFHeat.computeHeatmap(frames, [0, 0, 10, 10], { fps: 4 });
  assert.deepEqual(heat.analytics.filteredStaticIds, []);
  assert.equal(heat.people, 2);
});
