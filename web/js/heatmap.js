"use strict";
/* ============================================================================
 * ExhibitFlow Heatmap & Museum Spatial Analytics Engine
 * Universal module: window.EFHeat (browser) and module.exports (Node.js)
 *
 * Fixes all 4 legacy heatmap flaws:
 *  1. Explicit coordinate-aware circular disk aggregation (Euclidean radius,
 *     not hardcoded 1-unit square scanning the whole [0, 1] map).
 *  2. Robust percentile clipping (p98) + log-compressed perceptual tone mapping
 *     so stationary hotspots never wash out walkways and moving visitors.
 *  3. Per-frame timestamp deltas (dt_k) from actual frame times / strided keys
 *     with gap clamping instead of fixed 1/fps.
 *  4. Multi-metric grids (presence, unique visitors, dwell/stopping zones) +
 *     static ghost-track filtering + accurate mode-specific peak labeling +
 *     museum analytics summary.
 * ============================================================================ */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root && typeof root === "object") {
    root.EFHeat = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const HEATRAMP = [
    [0.0, [20, 40, 120]],
    [0.25, [30, 140, 200]],
    [0.5, [60, 220, 140]],
    [0.7, [255, 210, 60]],
    [0.85, [255, 122, 26]],
    [1.0, [255, 60, 60]],
  ];

  const MODE_META = {
    presence: { label: "Mật độ", peakLabel: "Tập trung nhất", unit: "s" },
    visitors: { label: "Lượt khách", peakLabel: "Đông khách nhất", unit: "khách" },
    dwell: { label: "Dừng chân", peakLabel: "Dừng lâu nhất", unit: "s dừng" },
  };

  function heatColor(v) {
    const x = Math.max(0, Math.min(1, v));
    for (let i = 1; i < HEATRAMP.length; i++) {
      const [a, ca] = HEATRAMP[i - 1];
      const [b, cb] = HEATRAMP[i];
      if (x <= b) {
        const u = (x - a) / (b - a || 1);
        return [
          Math.round(ca[0] + (cb[0] - ca[0]) * u),
          Math.round(ca[1] + (cb[1] - ca[1]) * u),
          Math.round(ca[2] + (cb[2] - ca[2]) * u),
        ];
      }
    }
    return HEATRAMP[HEATRAMP.length - 1][1].slice();
  }

  function percentile(values, p) {
    if (!values || !values.length) return 0;
    const arr = Float64Array.from(values).sort();
    const idx = Math.max(0, Math.min(arr.length - 1, p * (arr.length - 1)));
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return arr[lo];
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo);
  }

  /**
   * Compute per-frame durations (seconds) from actual frame timestamps.
   * Handles strided keys, irregular sampling, duplicate timestamps, and long gaps.
   */
  function computeFrameDeltas(times, fallbackDt) {
    const fb = fallbackDt > 0 && Number.isFinite(fallbackDt) ? fallbackDt : 0.25;
    const n = times ? times.length : 0;
    if (n === 0) return { dts: [], medianDt: fb, maxGap: Math.max(fb * 3, 2.0), durationSec: 0 };
    if (n === 1) return { dts: [fb], medianDt: fb, maxGap: Math.max(fb * 3, 2.0), durationSec: fb };

    const raw = new Float64Array(n - 1);
    const pos = [];
    for (let i = 0; i < n - 1; i++) {
      const d = times[i + 1] - times[i];
      const clamped = Number.isFinite(d) && d > 0 ? d : 0;
      raw[i] = clamped;
      if (clamped > 1e-6) pos.push(clamped);
    }

    const medianDt = pos.length ? percentile(pos, 0.5) : fb;
    const maxGap = Math.max(medianDt * 3, 2.0);
    const dts = new Float64Array(n);
    let durationSec = 0;
    for (let i = 0; i < n - 1; i++) {
      dts[i] = Math.min(raw[i], maxGap);
      durationSec += dts[i];
    }
    dts[n - 1] = Math.min(medianDt, maxGap);
    durationSec += dts[n - 1];

    return { dts, medianDt, maxGap, durationSec };
  }

  /**
   * Separable 2D Gaussian blur with boundary-weight renormalization so
   * intensity near museum walls is not artificially dimmed.
   */
  function gaussianBlur(raw, nx, ny, sigmaCells) {
    const n = nx * ny;
    const s = Math.max(0.8, sigmaCells);
    const r = Math.ceil(s * 2.5);
    const kLen = 2 * r + 1;
    const kernel = new Float32Array(kLen);
    let ks = 0;
    for (let i = -r; i <= r; i++) {
      const v = Math.exp((-i * i) / (2 * s * s));
      kernel[i + r] = v;
      ks += v;
    }
    for (let i = 0; i < kLen; i++) kernel[i] /= ks;

    const tmp = new Float32Array(n);
    const out = new Float32Array(n);

    for (let j = 0; j < ny; j++) {
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        let a = 0;
        let w = 0;
        const q0 = Math.max(-r, -i);
        const q1 = Math.min(r, nx - 1 - i);
        for (let q = q0; q <= q1; q++) {
          const kv = kernel[q + r];
          a += raw[row + i + q] * kv;
          w += kv;
        }
        tmp[row + i] = w > 0 ? a / w : 0;
      }
    }

    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        let a = 0;
        let w = 0;
        const q0 = Math.max(-r, -j);
        const q1 = Math.min(r, ny - 1 - j);
        for (let q = q0; q <= q1; q++) {
          const kv = kernel[q + r];
          a += tmp[(j + q) * nx + i] * kv;
          w += kv;
        }
        out[j * nx + i] = w > 0 ? a / w : 0;
      }
    }
    return out;
  }

  /**
   * Bilinear splat into a 2D grid. Clamps fractional coordinates to grid centers
   * so 100% of mass is preserved even for visitors standing right along walls/corners.
   */
  function splatBilinear(grid, nx, ny, x0, y0, cell, x, y, weight) {
    if (weight <= 0) return;
    const fx = Math.max(0, Math.min(nx - 1, (x - x0) / cell - 0.5));
    const fy = Math.max(0, Math.min(ny - 1, (y - y0) / cell - 0.5));
    const i0 = Math.min(nx - 1, Math.floor(fx));
    const j0 = Math.min(ny - 1, Math.floor(fy));
    const i1 = Math.min(nx - 1, i0 + 1);
    const j1 = Math.min(ny - 1, j0 + 1);
    const tx = i1 === i0 ? 0 : fx - i0;
    const ty = j1 === j0 ? 0 : fy - j0;

    const r0 = j0 * nx;
    grid[r0 + i0] += weight * (1 - tx) * (1 - ty);
    if (i1 !== i0) grid[r0 + i1] += weight * tx * (1 - ty);
    if (j1 !== j0) {
      const r1 = j1 * nx;
      grid[r1 + i0] += weight * (1 - tx) * ty;
      if (i1 !== i0) grid[r1 + i1] += weight * tx * ty;
    }
  }

  /**
   * Precompute integer (di, dj) offsets inside a Euclidean disk of radiusCells.
   */
  function buildDiskOffsets(radiusCells) {
    const r = Math.max(0, Math.floor(radiusCells));
    const r2 = radiusCells * radiusCells;
    const offsets = [];
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        if (di * di + dj * dj <= r2 + 1e-9) {
          offsets.push(di, dj);
        }
      }
    }
    if (!offsets.length) offsets.push(0, 0);
    return offsets;
  }

  function stampVisitorCell(ci, cj, nx, ny, stamp, centerStamp, visitStamp, visitorsRaw, diskOffsets) {
    const cIdx = cj * nx + ci;
    if (centerStamp[cIdx] === stamp) return;
    centerStamp[cIdx] = stamp;
    for (let d = 0; d < diskOffsets.length; d += 2) {
      const ni = ci + diskOffsets[d];
      const nj = cj + diskOffsets[d + 1];
      if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
      const idx = nj * nx + ni;
      if (visitStamp[idx] !== stamp) {
        visitStamp[idx] = stamp;
        visitorsRaw[idx] += 1;
      }
    }
  }

  /**
   * Bresenham line stamping between two grid cells so strided fast walkers
   * never leave dotted gaps in the unique-visitors grid.
   */
  function stampVisitorSegment(i0, j0, i1, j1, nx, ny, stamp, centerStamp, visitStamp, visitorsRaw, diskOffsets) {
    let x0 = i0;
    let y0 = j0;
    const dx = Math.abs(i1 - i0);
    const dy = Math.abs(j1 - j0);
    const sx = i0 < i1 ? 1 : -1;
    const sy = j0 < j1 ? 1 : -1;
    let err = dx - dy;
    for (;;) {
      stampVisitorCell(x0, y0, nx, ny, stamp, centerStamp, visitStamp, visitorsRaw, diskOffsets);
      if (x0 === i1 && y0 === j1) break;
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }
  }

  /**
   * Percentile clipping + log-compressed perceptual normalization.
   */
  function normalizeGrid(grid, opts) {
    const clipP = opts && opts.clipPercentile != null ? opts.clipPercentile : 0.98;
    const logAlpha = opts && opts.logAlpha != null ? opts.logAlpha : 8;
    const minVisible = opts && opts.minVisible != null ? opts.minVisible : 0.04;
    const n = grid.length;
    const norm = new Float32Array(n);

    let maxVal = 0;
    const pos = [];
    for (let i = 0; i < n; i++) {
      const v = grid[i];
      if (v > 1e-9) {
        pos.push(v);
        if (v > maxVal) maxVal = v;
      }
    }
    if (maxVal <= 0) {
      return { norm, max: 0, cap: 0, activeCells: 0 };
    }

    const pCap = pos.length >= 10 ? percentile(pos, clipP) : maxVal;
    // opts.fixedCap: giữ nguyên thang màu của cả video khi vẽ heatmap tích luỹ theo thời gian
    const cap = opts && opts.fixedCap > 0 ? opts.fixedCap : pCap > 1e-9 ? pCap : maxVal;
    const denom = logAlpha > 0 ? Math.log1p(logAlpha) : 1;
    let activeCells = 0;

    for (let i = 0; i < n; i++) {
      const v = grid[i];
      if (v <= 1e-9) continue;
      const u = Math.min(1, v / cap);
      const nv = logAlpha > 0 ? Math.log1p(logAlpha * u) / denom : u;
      norm[i] = nv;
      if (nv >= minVisible) activeCells++;
    }

    return { norm, max: maxVal, cap, activeCells };
  }

  /**
   * Analyze per-track trajectories, instantaneous speed, dwell state, and
   * detect static ghost tracks (e.g., paintings/mannequins false-detected
   * continuously at one spot).
   */
  function analyzeTracks(frames, dts, maxGap, durationSec, opts, bounds) {
    const o = opts || {};
    const isMetric = o.isMetric !== false;
    const staticRadius = o.staticRadius != null ? o.staticRadius : isMetric ? 0.12 : 0.006;
    const minStaticDuration = o.minStaticDuration != null ? o.minStaticDuration : 8.0;
    const minStaticCoverage = o.minStaticCoverage != null ? o.minStaticCoverage : 0.9;
    const minStaticSamples = o.minStaticSamples != null ? o.minStaticSamples : 5;
    const dwellSpeed = o.dwellSpeed != null ? o.dwellSpeed : isMetric ? 0.35 : 0.015;
    const filterStatic = o.filterStatic !== false;
    const hasBounds = Array.isArray(bounds) && bounds.length === 4;
    const bx0 = hasBounds ? bounds[0] : -Infinity;
    const by0 = hasBounds ? bounds[1] : -Infinity;
    const bx1 = hasBounds ? bounds[2] : Infinity;
    const by1 = hasBounds ? bounds[3] : Infinity;

    const byId = new Map();
    for (let k = 0; k < frames.length; k++) {
      const f = frames[k];
      const dt = dts[k] || 0;
      const t = f.t;
      const people = f.people || [];
      for (let m = 0; m < people.length; m++) {
        const p = people[m];
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
        if (p.x < bx0 || p.x > bx1 || p.y < by0 || p.y > by1) continue;
        const id = p.id != null ? p.id : `anon_${k}_${m}`;
        let tr = byId.get(id);
        if (!tr) {
          tr = { id, samples: [], cams: new Set() };
          byId.set(id, tr);
        }
        tr.samples.push({ t, x: p.x, y: p.y, dt, frameIdx: k, speed: 0, isDwell: false });
        if (Array.isArray(p.cams)) {
          for (let c = 0; c < p.cams.length; c++) tr.cams.add(p.cams[c]);
        }
      }
    }

    const validTracks = [];
    const filteredStaticIds = [];

    for (const tr of byId.values()) {
      const S = tr.samples;
      const n = S.length;
      let sx = 0;
      let sy = 0;
      let totalTime = 0;
      for (let i = 0; i < n; i++) {
        sx += S[i].x;
        sy += S[i].y;
        totalTime += S[i].dt;
      }
      const cx = sx / n;
      const cy = sy / n;

      let maxDisp = 0;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(S[i].x - cx, S[i].y - cy);
        if (d > maxDisp) maxDisp = d;
      }

      // Estimate per-sample speed using adjacent valid steps
      let pathLen = 0;
      let dwellTime = 0;
      for (let i = 0; i < n; i++) {
        let spd = 0;
        let hasStep = false;
        if (i > 0) {
          const dtPrev = S[i].t - S[i - 1].t;
          if (dtPrev > 1e-6 && dtPrev <= maxGap) {
            const dist = Math.hypot(S[i].x - S[i - 1].x, S[i].y - S[i - 1].y);
            pathLen += dist;
            spd = dist / dtPrev;
            hasStep = true;
          }
        }
        if (!hasStep && i + 1 < n) {
          const dtNext = S[i + 1].t - S[i].t;
          if (dtNext > 1e-6 && dtNext <= maxGap) {
            const dist = Math.hypot(S[i + 1].x - S[i].x, S[i + 1].y - S[i].y);
            spd = dist / dtNext;
            hasStep = true;
          }
        }
        S[i].speed = spd;
        S[i].isDwell = spd <= dwellSpeed;
        if (S[i].isDwell) dwellTime += S[i].dt;
      }

      const coverage = durationSec > 0 ? totalTime / durationSec : 0;
      const isStaticGhost =
        n >= minStaticSamples &&
        maxDisp < staticRadius &&
        totalTime >= minStaticDuration &&
        coverage >= minStaticCoverage;

      tr.cx = cx;
      tr.cy = cy;
      tr.maxDisp = maxDisp;
      tr.totalTime = totalTime;
      tr.dwellTime = dwellTime;
      tr.pathLen = pathLen;
      tr.avgSpeed = totalTime > 0 ? pathLen / totalTime : 0;
      tr.isStaticGhost = isStaticGhost;

      if (filterStatic && isStaticGhost) {
        filteredStaticIds.push(tr.id);
      } else {
        validTracks.push(tr);
      }
    }

    filteredStaticIds.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    return { validTracks, rawVisitorCount: byId.size, filteredStaticIds };
  }

  /**
   * Render a normalized [0, 1] grid to an HTMLCanvasElement (browser only).
   */
  function renderHeatCanvas(norm, nx, ny, yUp, minVisible) {
    if (typeof document === "undefined") return null;
    const thresh = minVisible != null ? minVisible : 0.04;
    const cv = document.createElement("canvas");
    cv.width = nx;
    cv.height = ny;
    const cx = cv.getContext("2d");
    const im = cx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const row = yUp ? ny - 1 - j : j;
      for (let i = 0; i < nx; i++) {
        const v = norm[j * nx + i];
        if (v >= thresh) {
          const c = heatColor(v);
          const o = (row * nx + i) * 4;
          im.data[o] = c[0];
          im.data[o + 1] = c[1];
          im.data[o + 2] = c[2];
          im.data[o + 3] = Math.round(255 * Math.min(0.9, 0.25 + v * 0.75));
        }
      }
    }
    cx.putImageData(im, 0, 0);
    return cv;
  }

  /**
   * Measure circular disk stats around (cx, cy) with exact Euclidean radius.
   */
  function measureDisk(cx, cy, radius, x0, y0, cell, nx, ny, presenceRaw, dwellRaw, validTracks) {
    const r2 = radius * radius;
    const pi = Math.floor((cx - x0) / cell);
    const pj = Math.floor((cy - y0) / cell);
    const rr = Math.max(1, Math.ceil(radius / cell));
    const iMin = Math.max(0, pi - rr);
    const iMax = Math.min(nx - 1, pi + rr);
    const jMin = Math.max(0, pj - rr);
    const jMax = Math.min(ny - 1, pj + rr);

    let presenceSec = 0;
    let dwellSec = 0;
    for (let j = jMin; j <= jMax; j++) {
      const dy = y0 + (j + 0.5) * cell - cy;
      const dy2 = dy * dy;
      if (dy2 > r2) continue;
      const row = j * nx;
      for (let i = iMin; i <= iMax; i++) {
        const dx = x0 + (i + 0.5) * cell - cx;
        if (dx * dx + dy2 <= r2) {
          presenceSec += presenceRaw[row + i];
          dwellSec += dwellRaw[row + i];
        }
      }
    }

    let uniqueVisitors = 0;
    for (let t = 0; t < validTracks.length; t++) {
      const S = validTracks[t].samples;
      for (let k = 0; k < S.length; k++) {
        const dx = S[k].x - cx;
        const dy = S[k].y - cy;
        if (dx * dx + dy * dy <= r2) {
          uniqueVisitors++;
          break;
        }
      }
    }

    return {
      presenceSec,
      dwellSec,
      uniqueVisitors,
      avgPresenceSec: uniqueVisitors > 0 ? presenceSec / uniqueVisitors : 0,
      avgDwellSec: uniqueVisitors > 0 ? dwellSec / uniqueVisitors : 0,
    };
  }

  /**
   * Non-maximum suppression to find up to topK distinct hotspots separated by minSep.
   */
  function findHotspots(grid, nx, ny, x0, y0, cell, peakRadius, minSep, topK, presenceRaw, dwellRaw, validTracks) {
    const candidates = [];
    for (let j = 0; j < ny; j++) {
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        const v = grid[row + i];
        if (v > 1e-9) candidates.push({ i, j, v });
      }
    }
    if (!candidates.length) return [];
    candidates.sort((a, b) => b.v - a.v);

    const maxScore = candidates[0].v;
    const minScore = maxScore * 0.15;
    const minSep2 = minSep * minSep;
    const picked = [];

    for (let c = 0; c < candidates.length && picked.length < topK; c++) {
      const cand = candidates[c];
      if (cand.v < minScore) break;
      const x = x0 + (cand.i + 0.5) * cell;
      const y = y0 + (cand.j + 0.5) * cell;
      let tooClose = false;
      for (let p = 0; p < picked.length; p++) {
        const dx = x - picked[p].x;
        const dy = y - picked[p].y;
        if (dx * dx + dy * dy < minSep2) {
          tooClose = true;
          break;
        }
      }
      if (tooClose) continue;
      const stats = measureDisk(x, y, peakRadius, x0, y0, cell, nx, ny, presenceRaw, dwellRaw, validTracks);
      picked.push({
        rank: picked.length + 1,
        x,
        y,
        score: cand.v,
        ...stats,
      });
    }
    return picked;
  }

  /**
   * Main heatmap computation entry point.
   *
   * @param {Array<{t: number, people: Array<{id: any, x: number, y: number, cams?: string[]}>}>} frames
   * @param {[number, number, number, number]} bounds [x0, y0, x1, y1]
   * @param {object} opts
   */
  function computeHeatmap(frames, bounds, opts) {
    const o = opts || {};
    const [x0, y0, x1, y1] = bounds || [0, 0, 1, 1];
    const width = x1 - x0;
    const height = y1 - y0;
    const cell = o.cell > 0 ? o.cell : Math.max(0.25, Math.min(1, Math.max(width, height) / 160));
    const sigma = o.sigma > 0 ? o.sigma : cell * 2;
    const peakRadius = o.peakRadius > 0 ? o.peakRadius : Math.max(cell * 2, sigma * 1.2);
    const yUp = o.yUp !== undefined ? !!o.yUp : true;
    const topK = o.topK > 0 ? o.topK : 3;
    const fallbackDt = o.fallbackDt > 0 ? o.fallbackDt : 1 / (o.fps || 4);

    const isMetric = o.isMetric !== false;
    const safeBounds = [x0, y0, x1, y1];

    if (width <= 0 || height <= 0 || !Array.isArray(frames) || !frames.length) {
      return emptyResult(1, 1, cell, yUp, peakRadius, safeBounds, isMetric);
    }

    const nx = Math.max(1, Math.ceil(width / cell));
    const ny = Math.max(1, Math.ceil(height / cell));

    // Ensure non-decreasing timestamps without copying when already sorted
    let orderedFrames = frames;
    let needsSort = false;
    let prevT = -Infinity;
    for (let k = 0; k < frames.length; k++) {
      const tk = frames[k] && Number.isFinite(frames[k].t) ? frames[k].t : k * fallbackDt;
      if (tk < prevT) {
        needsSort = true;
        break;
      }
      prevT = tk;
    }
    if (needsSort) {
      orderedFrames = frames
        .map((f, idx) => ({ f, t: f && Number.isFinite(f.t) ? f.t : idx * fallbackDt, idx }))
        .sort((a, b) => a.t - b.t || a.idx - b.idx)
        .map((item) => item.f);
    }

    const times = orderedFrames.map((f, idx) => (f && Number.isFinite(f.t) ? f.t : idx * fallbackDt));
    const { dts, medianDt, maxGap, durationSec } = computeFrameDeltas(times, fallbackDt);

    const { validTracks, rawVisitorCount, filteredStaticIds } = analyzeTracks(
      orderedFrames,
      dts,
      maxGap,
      durationSec,
      o,
      bounds
    );

    if (!validTracks.length) {
      const res = emptyResult(nx, ny, cell, yUp, peakRadius, safeBounds, isMetric);
      res.analytics.rawVisitors = rawVisitorCount;
      res.analytics.filteredStaticIds = filteredStaticIds;
      res.analytics.durationSec = durationSec;
      return res;
    }

    const nCells = nx * ny;
    const presenceRaw = new Float32Array(nCells);
    const dwellRaw = new Float32Array(nCells);
    const visitorsRaw = new Float32Array(nCells);
    const visitStamp = new Uint32Array(nCells);
    const centerStamp = new Uint32Array(nCells);
    const frameConcurrent = new Uint32Array(orderedFrames.length);
    const visitRadius = Math.min(peakRadius, Math.max(cell, sigma * 0.65));
    const diskOffsets = buildDiskOffsets(visitRadius / cell);

    let totalPresenceSec = 0;
    let totalDwellSec = 0;

    for (let tIdx = 0; tIdx < validTracks.length; tIdx++) {
      const tr = validTracks[tIdx];
      const stamp = tIdx + 1;
      const S = tr.samples;
      let prevCi = -1;
      let prevCj = -1;
      let prevSampleT = -Infinity;
      for (let k = 0; k < S.length; k++) {
        const s = S[k];
        if (s.x < x0 || s.x > x1 || s.y < y0 || s.y > y1) continue;
        frameConcurrent[s.frameIdx]++;
        splatBilinear(presenceRaw, nx, ny, x0, y0, cell, s.x, s.y, s.dt);
        totalPresenceSec += s.dt;
        if (s.isDwell) {
          splatBilinear(dwellRaw, nx, ny, x0, y0, cell, s.x, s.y, s.dt);
          totalDwellSec += s.dt;
        }
        const ci = Math.min(nx - 1, Math.max(0, Math.floor((s.x - x0) / cell)));
        const cj = Math.min(ny - 1, Math.max(0, Math.floor((s.y - y0) / cell)));
        if (prevCi >= 0 && s.t - prevSampleT <= maxGap) {
          stampVisitorSegment(prevCi, prevCj, ci, cj, nx, ny, stamp, centerStamp, visitStamp, visitorsRaw, diskOffsets);
        } else {
          stampVisitorCell(ci, cj, nx, ny, stamp, centerStamp, visitStamp, visitorsRaw, diskOffsets);
        }
        prevCi = ci;
        prevCj = cj;
        prevSampleT = s.t;
      }
    }

    let peakConcurrent = 0;
    for (let k = 0; k < frameConcurrent.length; k++) {
      if (frameConcurrent[k] > peakConcurrent) peakConcurrent = frameConcurrent[k];
    }

    const sigmaCells = sigma / cell;
    const minSep = Math.max(peakRadius * 2, sigma * 2.5);
    const rawMap = { presence: presenceRaw, visitors: visitorsRaw, dwell: dwellRaw };
    const modes = {};

    for (const modeName of ["presence", "visitors", "dwell"]) {
      // visitorsRaw already has disk spatial support (<=1 per visitor per cell), so use light anti-alias blur
      const modeSigma = modeName === "visitors" ? Math.max(0.8, sigmaCells * 0.45) : sigmaCells;
      const smoothed = gaussianBlur(rawMap[modeName], nx, ny, modeSigma);
      const { norm, max, cap, activeCells } = normalizeGrid(smoothed, o.caps ? { ...o, fixedCap: o.caps[modeName] } : o);
      const hotspots = findHotspots(
        smoothed,
        nx,
        ny,
        x0,
        y0,
        cell,
        peakRadius,
        minSep,
        topK,
        presenceRaw,
        dwellRaw,
        validTracks
      );
      const peak = hotspots[0] || null;
      const canvas = max > 0 ? renderHeatCanvas(norm, nx, ny, yUp, o.minVisible) : null;
      modes[modeName] = {
        mode: modeName,
        meta: MODE_META[modeName],
        grid: smoothed,
        norm,
        max,
        cap,
        activeCells,
        hotspots,
        peak,
        peakSec: peak ? peak.presenceSec : 0,
        peakDwellSec: peak ? peak.dwellSec : 0,
        peakVisitors: peak ? peak.uniqueVisitors : 0,
        canvas,
      };
    }

    const defMode = modes[o.mode && modes[o.mode] ? o.mode : "presence"];
    const totalVisitors = validTracks.length;
    const analytics = {
      totalVisitors,
      rawVisitors: rawVisitorCount,
      filteredStaticIds,
      durationSec,
      medianDt,
      totalPresenceSec,
      totalDwellSec,
      avgVisitSec: totalVisitors > 0 ? totalPresenceSec / totalVisitors : 0,
      avgDwellSec: totalVisitors > 0 ? totalDwellSec / totalVisitors : 0,
      stopRatio: totalPresenceSec > 0 ? totalDwellSec / totalPresenceSec : 0,
      peakConcurrent,
      activeAreaRatio: nCells > 0 ? modes.presence.activeCells / nCells : 0,
      hotspots: defMode.hotspots,
    };

    return {
      nx,
      ny,
      cell,
      yUp,
      isMetric,
      bounds: safeBounds,
      peakRadius,
      maxGap,
      people: totalVisitors,
      tracks: validTracks,
      rawGrids: rawMap,
      frameConcurrent,
      modes,
      max: defMode.max,
      cap: defMode.cap,
      peak: defMode.peak,
      peakSec: defMode.peakSec,
      peakDwellSec: defMode.peakDwellSec,
      peakVisitors: defMode.peakVisitors,
      hotspots: defMode.hotspots,
      canvas: defMode.canvas,
      analytics,
    };
  }

  function emptyResult(nx, ny, cell, yUp, peakRadius, bounds, isMetric) {
    const emptyMode = (modeName) => ({
      mode: modeName,
      meta: MODE_META[modeName],
      grid: new Float32Array(nx * ny),
      norm: new Float32Array(nx * ny),
      max: 0,
      cap: 0,
      activeCells: 0,
      hotspots: [],
      peak: null,
      peakSec: 0,
      peakDwellSec: 0,
      peakVisitors: 0,
      canvas: null,
    });
    const modes = {
      presence: emptyMode("presence"),
      visitors: emptyMode("visitors"),
      dwell: emptyMode("dwell"),
    };
    const nCells = nx * ny;
    return {
      nx,
      ny,
      cell,
      yUp,
      isMetric: isMetric !== false,
      bounds: bounds || [0, 0, 1, 1],
      peakRadius,
      maxGap: 2.0,
      people: 0,
      tracks: [],
      rawGrids: {
        presence: new Float32Array(nCells),
        visitors: new Float32Array(nCells),
        dwell: new Float32Array(nCells),
      },
      frameConcurrent: new Uint32Array(0),
      modes,
      max: 0,
      cap: 0,
      peak: null,
      peakSec: 0,
      peakDwellSec: 0,
      peakVisitors: 0,
      hotspots: [],
      canvas: null,
      analytics: {
        totalVisitors: 0,
        rawVisitors: 0,
        filteredStaticIds: [],
        durationSec: 0,
        medianDt: 0,
        totalPresenceSec: 0,
        totalDwellSec: 0,
        avgVisitSec: 0,
        avgDwellSec: 0,
        stopRatio: 0,
        peakConcurrent: 0,
        activeAreaRatio: 0,
        hotspots: [],
      },
    };
  }

  /**
   * Convenience wrapper for calibrated multi-camera site data (metric coordinates).
   */
  function computeSiteHeat(siteData, opts) {
    const [x0, y0, x1, y1] = siteData.bounds;
    const cell = Math.max(0.25, Math.min(1.0, Math.max(x1 - x0, y1 - y0) / 160));
    return computeHeatmap(siteData.frames || [], siteData.bounds, {
      cell,
      sigma: 0.9,
      peakRadius: 1.0,
      yUp: true,
      isMetric: true,
      fps: siteData.fps || 4,
      ...(opts || {}),
    });
  }

  /**
   * Convenience wrapper for single-camera normalized [0, 1] dashboard map frames.
   * MAP.frames[key] = [[id, x, y], ...], MAP.keys = [frameIndex, ...].
   */
  function computeFrameHeat(framesDict, keys, fps, opts) {
    const safeFps = fps > 0 ? fps : 25;
    const sortedKeys = (keys || []).slice().sort((a, b) => a - b);
    const frames = sortedKeys.map((k) => ({
      t: k / safeFps,
      people: (framesDict[k] || []).map((q) => ({ id: q[0], x: q[1], y: q[2] })),
    }));
    return computeHeatmap(frames, [0, 0, 1, 1], {
      cell: 1 / 64,
      sigma: 0.03,
      peakRadius: 0.05,
      yUp: false,
      isMetric: false,
      fps: safeFps,
      ...(opts || {}),
    });
  }

  return {
    HEATRAMP,
    MODE_META,
    heatColor,
    percentile,
    computeFrameDeltas,
    gaussianBlur,
    normalizeGrid,
    analyzeTracks,
    computeHeatmap,
    computeSiteHeat,
    computeFrameHeat,
  };
});
