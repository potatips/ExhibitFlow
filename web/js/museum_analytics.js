"use strict";
/* ============================================================================
 * ExhibitFlow Museum Spatial & Behavioral Analytics Engine
 * Universal module: window.EFAnalytics (browser) and module.exports (Node.js)
 *
 * Implements 5 privacy-preserving museum analytics modules (no face recognition):
 *  1. Attraction Power vs. Holding Power Matrix (4 quadrants)
 *  2. Congestion Bottleneck Map (high concurrent density + speed drop in corridors)
 *  4. Dominant Movement Flow Map, Skip Rate & Museum Fatigue Curve
 *  5. Space Utilization & Dead-Zone Map (inside observed camera FOV / floor)
 *  7. Anonymous Visitor Archetypes (Verón & Levasseur: Ant, Butterfly, Fish, Grasshopper)
 * ============================================================================ */
(function (root, factory) {
  let heatLib = null;
  if (typeof module === "object" && module.exports) {
    try {
      heatLib = require("./heatmap.js");
    } catch (_) {
      heatLib = null;
    }
  } else if (root && root.EFHeat) {
    heatLib = root.EFHeat;
  }
  const api = factory(heatLib);
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root && typeof root === "object") {
    root.EFAnalytics = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (EFHeat) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const QUADRANT_META = {
    star: {
      key: "star",
      label: "Ngôi sao",
      short: "Thu hút & giữ chân tốt",
      color: "#3DDC97",
      advice: "Hiện vật chủ lực — cân nhắc giãn luồng nếu quá đông.",
    },
    skim: {
      key: "skim",
      label: "Lướt nhanh",
      short: "Nhiều người dừng nhưng xem ngắn",
      color: "#FFB020",
      advice: "Bổ sung thuyết minh, ánh sáng điểm nhấn hoặc câu chuyện tương tác.",
    },
    hidden_gem: {
      key: "hidden_gem",
      label: "Viên ngọc ẩn",
      short: "Ít người dừng nhưng xem rất kỹ",
      color: "#5B9BEA",
      advice: "Tăng biển chỉ dẫn, ánh sáng dẫn hướng hoặc điều chỉnh vị trí đặt.",
    },
    cold: {
      key: "cold",
      label: "Vùng lạnh",
      short: "Ít thu hút & dừng ngắn",
      color: "#94A0AC",
      advice: "Xem xét thay đổi cách trưng bày hoặc hoán đổi vị trí.",
    },
  };

  const ARCHETYPE_META = {
    ant: {
      key: "ant",
      label: "Kiến (Ant)",
      short: "Xem kỹ toàn phòng",
      color: "#3DDC97",
      desc: "Đi phủ rộng không gian, dừng lâu và di chuyển chậm rãi.",
    },
    butterfly: {
      key: "butterfly",
      label: "Bướm (Butterfly)",
      short: "Lướt rộng, dừng chọn lọc",
      color: "#FFB020",
      desc: "Di chuyển rộng khắp phòng, dừng ngắn tại nhiều hiện vật.",
    },
    fish: {
      key: "fish",
      label: "Cá (Fish)",
      short: "Đi xuyên nhanh",
      color: "#5B9BEA",
      desc: "Lướt qua trung tâm hành lang nhanh, rất ít khi dừng lại.",
    },
    grasshopper: {
      key: "grasshopper",
      label: "Châu chấu (Grasshopper)",
      short: "Nhắm thẳng vài điểm",
      color: "#B388FF",
      desc: "Đi thẳng tới một số hiện vật cụ thể, bỏ qua phần lớn khu vực khác.",
    },
  };

  function pointInPoly(px, py, poly) {
    if (!Array.isArray(poly) || poly.length < 3) return false;
    let ins = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i][0];
      const b = poly[i][1];
      const c = poly[j][0];
      const d = poly[j][1];
      if ((b > py) !== (d > py) && px < ((c - a) * (py - b)) / (d - b) + a) {
        ins = !ins;
      }
    }
    return ins;
  }

  function percentile(values, p) {
    if (!values || !values.length) return 0;
    const arr = Float64Array.from(values).sort();
    const idx = clamp(p * (arr.length - 1), 0, arr.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return arr[lo];
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo);
  }

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

  function normalizeExhibits(rawExhibits, heatmap, cams) {
    const out = [];
    if (Array.isArray(rawExhibits)) {
      for (let i = 0; i < rawExhibits.length; i++) {
        const a = rawExhibits[i];
        if (!a) continue;
        const x = Number.isFinite(a.x) ? a.x : a.fx;
        const y = Number.isFinite(a.y) ? a.y : a.fy;
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        out.push({
          id: a.id != null ? a.id : i + 1,
          name: a.name || a.label || `Hiện vật ${a.id != null ? a.id : i + 1}`,
          x,
          y,
          radius: Number.isFinite(a.radius) && a.radius > 0 ? a.radius : null,
        });
      }
    }
    if (out.length) {
      return { exhibits: out, source: "exhibits" };
    }

    // Fallback: auto-discover zones from dwell/presence hotspots when exhibit positions are unknown
    const auto = [];
    const dwellHs = heatmap && heatmap.modes && heatmap.modes.dwell ? heatmap.modes.dwell.hotspots || [] : [];
    const presHs = heatmap && heatmap.modes && heatmap.modes.presence ? heatmap.modes.presence.hotspots || [] : [];
    const pool = dwellHs.length ? dwellHs : presHs;
    const isMetric = heatmap ? heatmap.isMetric !== false : true;
    const minSep = isMetric ? 2.5 : 0.14;
    const minSep2 = minSep * minSep;

    const addPool = (list) => {
      for (let i = 0; i < list.length && auto.length < 6; i++) {
        const h = list[i];
        if (!Number.isFinite(h.x) || !Number.isFinite(h.y)) continue;
        let close = false;
        for (let j = 0; j < auto.length; j++) {
          const dx = h.x - auto[j].x;
          const dy = h.y - auto[j].y;
          if (dx * dx + dy * dy < minSep2) {
            close = true;
            break;
          }
        }
        if (close) continue;
        let camTag = "";
        if (Array.isArray(cams) && cams.length) {
          const hit = cams.filter((cm) => pointInPoly(h.x, h.y, cm.fov)).map((cm) => cm.id);
          if (hit.length) camTag = ` (${hit[0]})`;
        }
        auto.push({
          id: `Z${auto.length + 1}`,
          name: `Vùng #${auto.length + 1}${camTag}`,
          x: h.x,
          y: h.y,
          radius: null,
        });
      }
    };
    addPool(pool);
    if (auto.length < 3 && presHs !== pool) addPool(presHs);
    return { exhibits: auto, source: auto.length ? "auto_zones" : "none" };
  }

  /* ==========================================================================
   * FEATURE 1: Attraction Power vs. Holding Power Matrix
   * ========================================================================== */
  function computeAttractionHolding(tracks, exhibits, opts) {
    const o = opts || {};
    const isMetric = o.isMetric !== false;
    const defaultPassRadius = o.passRadius > 0 ? o.passRadius : isMetric ? 2.5 : 0.12;
    const minStopSec = o.minStopSec > 0 ? o.minStopSec : 3.0;
    const attractionThreshold = o.attractionThreshold != null ? o.attractionThreshold : 0.35;
    const holdingThresholdSec = o.holdingThresholdSec != null ? o.holdingThresholdSec : 10.0;
    const validTracks = Array.isArray(tracks) ? tracks : [];
    const list = Array.isArray(exhibits) ? exhibits : [];

    const quadrantCounts = { star: 0, skim: 0, hidden_gem: 0, cold: 0 };
    const items = [];
    let attrSum = 0;
    let holdSum = 0;
    let holdCount = 0;

    for (let eIdx = 0; eIdx < list.length; eIdx++) {
      const ex = list[eIdx];
      const rad = ex.radius > 0 ? ex.radius : defaultPassRadius;
      const r2 = rad * rad;

      let passers = 0;
      let stoppers = 0;
      let stopperDwellSum = 0;
      let totalDwellSec = 0;
      let totalPresenceSec = 0;
      const passerIds = [];
      const stopperIds = [];

      for (let tIdx = 0; tIdx < validTracks.length; tIdx++) {
        const tr = validTracks[tIdx];
        const S = tr.samples || [];
        let passed = false;
        let trDwell = 0;
        let trPres = 0;

        for (let k = 0; k < S.length; k++) {
          const s = S[k];
          const dx = s.x - ex.x;
          const dy = s.y - ex.y;
          if (dx * dx + dy * dy <= r2) {
            passed = true;
            trPres += s.dt;
            if (s.isDwell) trDwell += s.dt;
          }
        }

        if (passed) {
          passers++;
          passerIds.push(tr.id);
          totalPresenceSec += trPres;
          totalDwellSec += trDwell;
          if (trDwell >= minStopSec) {
            stoppers++;
            stopperIds.push(tr.id);
            stopperDwellSum += trDwell;
          }
        }
      }

      const attractionPower = passers > 0 ? stoppers / passers : 0;
      const holdingPowerSec = stoppers > 0 ? stopperDwellSum / stoppers : 0;
      const passerSkipRate = passers > 0 ? (passers - stoppers) / passers : 0;
      const overallSkipRate = validTracks.length > 0 ? (validTracks.length - stoppers) / validTracks.length : 0;

      let quadrant = "cold";
      if (attractionPower >= attractionThreshold && holdingPowerSec >= holdingThresholdSec) {
        quadrant = "star";
      } else if (attractionPower >= attractionThreshold && holdingPowerSec < holdingThresholdSec) {
        quadrant = "skim";
      } else if (attractionPower < attractionThreshold && holdingPowerSec >= holdingThresholdSec) {
        quadrant = "hidden_gem";
      }
      quadrantCounts[quadrant]++;
      if (passers > 0) attrSum += attractionPower;
      if (stoppers > 0) {
        holdSum += holdingPowerSec;
        holdCount++;
      }

      items.push({
        id: ex.id,
        name: ex.name,
        x: ex.x,
        y: ex.y,
        radius: rad,
        passers,
        stoppers,
        attractionPower,
        holdingPowerSec,
        totalDwellSec,
        totalPresenceSec,
        passerSkipRate,
        overallSkipRate,
        quadrant,
        quadrantMeta: QUADRANT_META[quadrant],
        passerIds,
        stopperIds,
      });
    }

    const withPassers = items.filter((it) => it.passers > 0).length;
    return {
      source: o.source || (list.length ? "exhibits" : "none"),
      passRadius: defaultPassRadius,
      minStopSec,
      thresholds: { attractionPower: attractionThreshold, holdingPowerSec: holdingThresholdSec },
      items,
      quadrantCounts,
      quadrantMeta: QUADRANT_META,
      summary: {
        exhibitCount: items.length,
        avgAttractionPower: withPassers > 0 ? attrSum / withPassers : 0,
        avgHoldingPowerSec: holdCount > 0 ? holdSum / holdCount : 0,
      },
    };
  }

  /* ==========================================================================
   * FEATURE 2: Congestion Bottleneck Map (High Concurrency + Speed Drop)
   * ========================================================================== */
  function renderBottleneckCanvas(norm, nx, ny, yUp, minVisible) {
    if (typeof document === "undefined") return null;
    const thresh = minVisible != null ? minVisible : 0.08;
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
          const o = (row * nx + i) * 4;
          // Amber -> Crimson -> Magenta congestion palette
          im.data[o] = Math.round(235 + 20 * v);
          im.data[o + 1] = Math.round(145 * (1 - v) + 25 * v);
          im.data[o + 2] = Math.round(30 * (1 - v) + 110 * v);
          im.data[o + 3] = Math.round(255 * Math.min(0.88, 0.28 + v * 0.62));
        }
      }
    }
    cx.putImageData(im, 0, 0);
    return cv;
  }

  function computeBottlenecks(heatmap, exhibits, opts) {
    const o = opts || {};
    const nx = heatmap ? heatmap.nx : 1;
    const ny = heatmap ? heatmap.ny : 1;
    const cell = heatmap ? heatmap.cell : 1;
    const yUp = heatmap ? !!heatmap.yUp : true;
    const isMetric = heatmap ? heatmap.isMetric !== false : o.isMetric !== false;
    const [x0, y0] = heatmap && heatmap.bounds ? heatmap.bounds : [0, 0, 1, 1];
    const tracks = heatmap && Array.isArray(heatmap.tracks) ? heatmap.tracks : [];
    const nCells = nx * ny;

    const minConcurrent = o.minConcurrent != null ? o.minConcurrent : 2;
    const slowFactor = o.slowFactor != null ? o.slowFactor : 0.55;
    const minPresenceSec = o.minPresenceSec != null ? o.minPresenceSec : 1.0;
    const localRadius = o.localRadius > 0 ? o.localRadius : isMetric ? 1.5 : 0.07;
    const exhibitExclusionRadius =
      o.exhibitExclusionRadius != null ? o.exhibitExclusionRadius : isMetric ? 1.5 : 0.08;
    const topK = o.topK > 0 ? o.topK : 5;

    const empty = {
      nx,
      ny,
      cell,
      yUp,
      grid: new Float32Array(nCells),
      norm: new Float32Array(nCells),
      peakConcurrentGrid: new Uint16Array(nCells),
      avgSpeedGrid: new Float32Array(nCells),
      hotspots: [],
      canvas: null,
      awayFromExhibits: Array.isArray(exhibits) && exhibits.length > 0,
      summary: {
        count: 0,
        maxSeverity: 0,
        baselineSpeed: 0,
        peakLocalConcurrent: 0,
      },
    };
    if (!tracks.length) return empty;

    // 1. Estimate baseline walking speed from non-dwell moving samples
    const movingSpeeds = [];
    const allSpeeds = [];
    const byFrame = new Map();
    for (let t = 0; t < tracks.length; t++) {
      const tr = tracks[t];
      const S = tr.samples || [];
      for (let k = 0; k < S.length; k++) {
        const s = S[k];
        if (s.speed > 1e-6) {
          allSpeeds.push(s.speed);
          if (!s.isDwell) movingSpeeds.push(s.speed);
        }
        let arr = byFrame.get(s.frameIdx);
        if (!arr) {
          arr = [];
          byFrame.set(s.frameIdx, arr);
        }
        arr.push({ s, tid: tr.id != null ? tr.id : t });
      }
    }
    const defaultWalk = isMetric ? 0.9 : 0.04;
    const baselineSpeed =
      movingSpeeds.length >= 5
        ? percentile(movingSpeeds, 0.5)
        : allSpeeds.length >= 5
          ? Math.max(percentile(allSpeeds, 0.75), defaultWalk * 0.5)
          : defaultWalk;

    // 2. Per-frame local concurrency + concurrent-slowdown speed & duration accumulation
    const disk = buildDiskOffsets(localRadius / cell);
    const slowSpeedThresh = baselineSpeed * slowFactor;
    const frameCount = new Uint16Array(nCells);
    const frameSlowCount = new Uint16Array(nCells);
    const frameSpeedSum = new Float32Array(nCells);
    const frameDtSum = new Float32Array(nCells);
    const peakConcurrentGrid = new Uint16Array(nCells);
    const overallSpeedSum = new Float32Array(nCells);
    const overallSpeedWeight = new Float32Array(nCells);
    const congSpeedSum = new Float32Array(nCells);
    const congDur = new Float32Array(nCells);
    const touched = [];
    const seenInFrame = new Set();

    for (const entries of byFrame.values()) {
      touched.length = 0;
      seenInFrame.clear();
      for (let m = 0; m < entries.length; m++) {
        const { s, tid } = entries[m];
        if (seenInFrame.has(tid)) continue;
        seenInFrame.add(tid);
        const isSlow = s.speed < slowSpeedThresh ? 1 : 0;
        const ci = clamp(Math.floor((s.x - x0) / cell), 0, nx - 1);
        const cj = clamp(Math.floor((s.y - y0) / cell), 0, ny - 1);
        for (let d = 0; d < disk.length; d += 2) {
          const ni = ci + disk[d];
          const nj = cj + disk[d + 1];
          if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
          const idx = nj * nx + ni;
          if (frameCount[idx] === 0) touched.push(idx);
          frameCount[idx] += 1;
          frameSlowCount[idx] += isSlow;
          frameSpeedSum[idx] += s.speed;
          frameDtSum[idx] += s.dt;
          overallSpeedSum[idx] += s.speed * s.dt;
          overallSpeedWeight[idx] += s.dt;
        }
      }
      for (let u = 0; u < touched.length; u++) {
        const idx = touched[u];
        const cnt = frameCount[idx];
        if (cnt > peakConcurrentGrid[idx]) {
          peakConcurrentGrid[idx] = cnt;
        }
        if (cnt >= minConcurrent && frameSlowCount[idx] >= minConcurrent) {
          const fAvgSpd = frameSpeedSum[idx] / cnt;
          const fDt = frameDtSum[idx] / cnt;
          congSpeedSum[idx] += fAvgSpd * fDt;
          congDur[idx] += fDt;
        }
        frameCount[idx] = 0;
        frameSlowCount[idx] = 0;
        frameSpeedSum[idx] = 0;
        frameDtSum[idx] = 0;
      }
    }

    // 3. Build exhibit exclusion mask (only when explicit exhibits are provided)
    const exList = Array.isArray(exhibits) ? exhibits : [];
    const exclR2 = exhibitExclusionRadius * exhibitExclusionRadius;
    const isNearExhibit = (wx, wy) => {
      for (let e = 0; e < exList.length; e++) {
        const dx = wx - exList[e].x;
        const dy = wy - exList[e].y;
        if (dx * dx + dy * dy <= exclR2) return true;
      }
      return false;
    };

    const avgSpeedGrid = new Float32Array(nCells);
    const congAvgSpeedGrid = new Float32Array(nCells);
    const rawScore = new Float32Array(nCells);
    let peakLocalConcurrent = 0;
    let maxScore = 0;

    for (let j = 0; j < ny; j++) {
      const wy = y0 + (j + 0.5) * cell;
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        const idx = row + i;
        const pc = peakConcurrentGrid[idx];
        if (pc > peakLocalConcurrent) peakLocalConcurrent = pc;
        const w = overallSpeedWeight[idx];
        avgSpeedGrid[idx] = w > 0 ? overallSpeedSum[idx] / w : 0;

        const cd = congDur[idx];
        const congSpd = cd > 0 ? congSpeedSum[idx] / cd : 0;
        congAvgSpeedGrid[idx] = congSpd;

        if (pc < minConcurrent || cd < minPresenceSec) continue;
        if (exList.length && isNearExhibit(x0 + (i + 0.5) * cell, wy)) continue;

        const speedRatio = baselineSpeed > 1e-6 ? congSpd / baselineSpeed : 1;
        if (speedRatio >= slowFactor) continue;

        const dropFrac = clamp(1 - speedRatio, 0, 1);
        const sc = (pc - minConcurrent + 1) * dropFrac * Math.log1p(cd);
        rawScore[idx] = sc;
        if (sc > maxScore) maxScore = sc;
      }
    }

    const smoothed =
      maxScore > 0 && EFHeat && EFHeat.gaussianBlur
        ? EFHeat.gaussianBlur(rawScore, nx, ny, Math.max(0.8, (localRadius * 0.5) / cell))
        : rawScore;

    let smoothMax = 0;
    for (let i = 0; i < nCells; i++) {
      if (smoothed[i] > smoothMax) smoothMax = smoothed[i];
    }
    const norm = new Float32Array(nCells);
    if (smoothMax > 0) {
      for (let i = 0; i < nCells; i++) {
        norm[i] = smoothed[i] / smoothMax;
      }
    }

    // 4. Non-maximum suppression for topK bottlenecks
    const candidates = [];
    for (let j = 0; j < ny; j++) {
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        const v = rawScore[row + i];
        if (v > 0) candidates.push({ i, j, v, idx: row + i });
      }
    }
    candidates.sort((a, b) => b.v - a.v);

    const minSep = Math.max(localRadius * 1.8, cell * 3);
    const minSep2 = minSep * minSep;
    const hotspots = [];

    for (let c = 0; c < candidates.length && hotspots.length < topK; c++) {
      const cand = candidates[c];
      if (maxScore > 0 && cand.v < maxScore * 0.2) break;
      const wx = x0 + (cand.i + 0.5) * cell;
      const wy = y0 + (cand.j + 0.5) * cell;
      let tooClose = false;
      for (let h = 0; h < hotspots.length; h++) {
        const dx = wx - hotspots[h].x;
        const dy = wy - hotspots[h].y;
        if (dx * dx + dy * dy < minSep2) {
          tooClose = true;
          break;
        }
      }
      if (tooClose) continue;
      const avgSpd = congAvgSpeedGrid[cand.idx];
      const speedDropPct = baselineSpeed > 1e-6 ? clamp(1 - avgSpd / baselineSpeed, 0, 1) : 0;
      hotspots.push({
        rank: hotspots.length + 1,
        x: wx,
        y: wy,
        severity: maxScore > 0 ? clamp(cand.v / maxScore, 0, 1) : 0,
        score: cand.v,
        peakConcurrent: peakConcurrentGrid[cand.idx],
        avgSpeed: avgSpd,
        speedDropPct,
        durationSec: congDur[cand.idx],
      });
    }

    const canvas = smoothMax > 0 ? renderBottleneckCanvas(norm, nx, ny, yUp, o.minVisible) : null;

    return {
      nx,
      ny,
      cell,
      yUp,
      grid: smoothed,
      norm,
      peakConcurrentGrid,
      avgSpeedGrid,
      hotspots,
      canvas,
      awayFromExhibits: exList.length > 0,
      summary: {
        count: hotspots.length,
        maxSeverity: hotspots.length ? hotspots[0].severity : 0,
        baselineSpeed,
        peakLocalConcurrent,
      },
    };
  }

  /* ==========================================================================
   * FEATURE 4: Dominant Flow Map, Skip Rate & Museum Fatigue
   * ========================================================================== */
  function computeFlowAndFatigue(tracks, heatmap, attractionResult, opts) {
    const o = opts || {};
    const validTracks = Array.isArray(tracks) ? tracks : [];
    const bounds = (heatmap && heatmap.bounds) || o.bounds || [0, 0, 1, 1];
    const [x0, y0, x1, y1] = bounds;
    const width = Math.max(1e-6, x1 - x0);
    const height = Math.max(1e-6, y1 - y0);
    const maxGap = (heatmap && heatmap.maxGap) || o.maxGap || 2.0;
    const isMetric = heatmap ? heatmap.isMetric !== false : o.isMetric !== false;

    // 4a. Dominant Movement Flow Vectors on a legible coarse grid
    const maxDim = o.flowMaxDim > 0 ? o.flowMaxDim : 10;
    const aspect = width / height;
    const cols = aspect >= 1 ? maxDim : Math.max(4, Math.round(maxDim * aspect));
    const rows = aspect >= 1 ? Math.max(4, Math.round(maxDim / aspect)) : maxDim;
    const nFlow = cols * rows;
    const cellW = width / cols;
    const cellH = height / rows;

    const sumVx = new Float64Array(nFlow);
    const sumVy = new Float64Array(nFlow);
    const sumUx = new Float64Array(nFlow);
    const sumUy = new Float64Array(nFlow);
    const stepCount = new Uint32Array(nFlow);
    const visStamp = new Uint32Array(nFlow);
    const visCount = new Uint32Array(nFlow);
    const minStepDist = isMetric ? 0.05 : 0.002;

    for (let t = 0; t < validTracks.length; t++) {
      const S = validTracks[t].samples || [];
      const stamp = t + 1;
      for (let k = 1; k < S.length; k++) {
        const a = S[k - 1];
        const b = S[k];
        const dt = b.t - a.t;
        if (dt <= 1e-6 || dt > maxGap) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.hypot(dx, dy);
        if (dist < minStepDist) continue;
        const fi = clamp(Math.floor(((a.x + b.x) * 0.5 - x0) / cellW), 0, cols - 1);
        const fj = clamp(Math.floor(((a.y + b.y) * 0.5 - y0) / cellH), 0, rows - 1);
        const idx = fj * cols + fi;
        sumVx[idx] += dx / dt;
        sumVy[idx] += dy / dt;
        sumUx[idx] += dx / dist;
        sumUy[idx] += dy / dist;
        stepCount[idx] += 1;
        if (visStamp[idx] !== stamp) {
          visStamp[idx] = stamp;
          visCount[idx] += 1;
        }
      }
    }

    const minAlignment = o.minAlignment != null ? o.minAlignment : 0.25;
    const vectors = [];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const idx = j * cols + i;
        const cnt = stepCount[idx];
        if (cnt < 2) continue;
        const vx = sumVx[idx] / cnt;
        const vy = sumVy[idx] / cnt;
        const speed = Math.hypot(vx, vy);
        const uMag = Math.hypot(sumUx[idx], sumUy[idx]);
        const alignment = uMag / cnt;
        if (speed <= 1e-6 || uMag <= 1e-6 || alignment < minAlignment) continue;
        const ux = sumUx[idx] / uMag;
        const uy = sumUy[idx] / uMag;
        vectors.push({
          i,
          j,
          x: x0 + (i + 0.5) * cellW,
          y: y0 + (j + 0.5) * cellH,
          vx,
          vy,
          ux,
          uy,
          angle: Math.atan2(uy, ux),
          speed,
          alignment,
          steps: cnt,
          visitors: visCount[idx],
        });
      }
    }

    // 4b. Zone-to-Zone Transitions
    const exItems = (attractionResult && attractionResult.items) || [];
    const transMap = new Map();
    if (exItems.length >= 2) {
      for (let t = 0; t < validTracks.length; t++) {
        const S = validTracks[t].samples || [];
        const seq = [];
        for (let k = 0; k < S.length; k++) {
          const s = S[k];
          let nearest = null;
          let bestD2 = Infinity;
          for (let e = 0; e < exItems.length; e++) {
            const ex = exItems[e];
            const dx = s.x - ex.x;
            const dy = s.y - ex.y;
            const d2 = dx * dx + dy * dy;
            if (d2 <= ex.radius * ex.radius && d2 < bestD2) {
              bestD2 = d2;
              nearest = ex;
            }
          }
          if (nearest && (!seq.length || seq[seq.length - 1].id !== nearest.id)) {
            seq.push(nearest);
          }
        }
        for (let q = 1; q < seq.length; q++) {
          const from = seq[q - 1];
          const to = seq[q];
          const key = `${from.id}->${to.id}`;
          let rec = transMap.get(key);
          if (!rec) {
            rec = {
              fromId: from.id,
              fromName: from.name,
              fromX: from.x,
              fromY: from.y,
              toId: to.id,
              toName: to.name,
              toX: to.x,
              toY: to.y,
              count: 0,
            };
            transMap.set(key, rec);
          }
          rec.count++;
        }
      }
    }
    const transitions = Array.from(transMap.values()).sort((a, b) => b.count - a.count);
    const totalTransitions = transitions.reduce((acc, tr) => acc + tr.count, 0);
    for (let i = 0; i < transitions.length; i++) {
      transitions[i].share = totalTransitions > 0 ? transitions[i].count / totalTransitions : 0;
    }

    // 4c. Skip Rate summary (derived from attractionResult)
    const skipItems = exItems.map((it) => ({
      id: it.id,
      name: it.name,
      passers: it.passers,
      stoppers: it.stoppers,
      passerSkipRate: it.passerSkipRate,
      overallSkipRate: it.overallSkipRate,
    }));
    const withPassers = skipItems.filter((it) => it.passers > 0);
    const avgPasserSkipRate = withPassers.length
      ? withPassers.reduce((a, b) => a + b.passerSkipRate, 0) / withPassers.length
      : 0;
    const avgOverallSkipRate = skipItems.length
      ? skipItems.reduce((a, b) => a + b.overallSkipRate, 0) / skipItems.length
      : 0;

    // 4d. Museum Fatigue Curve (Q1..Q4 visit progression)
    const minFatigueTrackSec = o.minFatigueTrackSec > 0 ? o.minFatigueTrackSec : 60.0;
    const qSpeedSum = [0, 0, 0, 0];
    const qTimeSum = [0, 0, 0, 0];
    const qDwellSum = [0, 0, 0, 0];
    const qSamples = [0, 0, 0, 0];
    const qElapsedSum = [0, 0, 0, 0];
    let eligibleVisitors = 0;
    let totalEligibleDur = 0;

    for (let t = 0; t < validTracks.length; t++) {
      const tr = validTracks[t];
      const S = tr.samples || [];
      if (S.length < 4 || tr.totalTime < minFatigueTrackSec) continue;
      const tStart = S[0].t;
      const tSpan = S[S.length - 1].t - tStart;
      if (tSpan <= 1e-6) continue;
      eligibleVisitors++;
      totalEligibleDur += tr.totalTime;

      for (let k = 0; k < S.length; k++) {
        const s = S[k];
        const prog = clamp((s.t - tStart) / tSpan, 0, 0.999999);
        const q = Math.floor(prog * 4);
        qSpeedSum[q] += s.speed * s.dt;
        qTimeSum[q] += s.dt;
        if (s.isDwell) qDwellSum[q] += s.dt;
        qSamples[q] += 1;
        qElapsedSum[q] += (s.t - tStart) * s.dt;
      }
    }

    const quartiles = [0, 1, 2, 3].map((q) => {
      const tSum = qTimeSum[q];
      return {
        quartile: q + 1,
        label: `Q${q + 1} (${q * 25}–${(q + 1) * 25}%)`,
        avgSpeed: tSum > 0 ? qSpeedSum[q] / tSum : 0,
        stopRatio: tSum > 0 ? qDwellSum[q] / tSum : 0,
        avgElapsedSec: tSum > 0 ? qElapsedSum[q] / tSum : 0,
        sampleCount: qSamples[q],
      };
    });

    const q1 = quartiles[0];
    const q4 = quartiles[3];
    const speedIncreasePct =
      eligibleVisitors > 0
        ? q1.avgSpeed > 1e-6
          ? (q4.avgSpeed - q1.avgSpeed) / q1.avgSpeed
          : q4.avgSpeed > 1e-6
            ? 1
            : 0
        : 0;
    const stopDropPct =
      eligibleVisitors > 0 && q1.stopRatio > 1e-6 ? (q1.stopRatio - q4.stopRatio) / q1.stopRatio : 0;
    const fatigueScore =
      eligibleVisitors > 0
        ? clamp(0.5 * clamp(speedIncreasePct, 0, 1) + 0.5 * clamp(stopDropPct, 0, 1), 0, 1)
        : 0;

    const quartileFatigued = (cur, thr) => {
      const spdJump =
        q1.avgSpeed > 1e-6
          ? (cur.avgSpeed - q1.avgSpeed) / q1.avgSpeed
          : cur.avgSpeed > 1e-6
            ? 1
            : 0;
      const stpDrop = q1.stopRatio > 1e-6 ? (q1.stopRatio - cur.stopRatio) / q1.stopRatio : 0;
      return stpDrop >= thr || spdJump >= thr;
    };

    const detected =
      eligibleVisitors > 0 &&
      (fatigueScore >= 0.25 || stopDropPct >= 0.25 || speedIncreasePct >= 0.25);

    let onsetQuartile = null;
    if (detected) {
      for (let q = 1; q < 4; q++) {
        if (!quartileFatigued(quartiles[q], 0.25)) continue;
        // Ensure fatigue persists through Q4 rather than being a transient mid-visit corridor walk
        let sustained = true;
        for (let r = q + 1; r < 4; r++) {
          if (!quartileFatigued(quartiles[r], 0.2)) {
            sustained = false;
            break;
          }
        }
        if (sustained) {
          onsetQuartile = q + 1;
          break;
        }
      }
      if (onsetQuartile === null) onsetQuartile = 4;
    }
    const onsetProgressPct = onsetQuartile != null ? (onsetQuartile - 1) * 25 : null;
    const onsetAvgTimeSec = onsetQuartile != null ? quartiles[onsetQuartile - 1].avgElapsedSec : null;

    return {
      flow: {
        cols,
        rows,
        cellW,
        cellH,
        vectors,
        transitions,
        totalTransitions,
      },
      skipRate: {
        items: skipItems,
        avgPasserSkipRate,
        avgOverallSkipRate,
      },
      fatigue: {
        eligibleVisitors,
        avgEligibleVisitSec: eligibleVisitors > 0 ? totalEligibleDur / eligibleVisitors : 0,
        minTrackSec: minFatigueTrackSec,
        quartiles,
        speedIncreasePct,
        stopDropPct,
        fatigueScore,
        onsetQuartile,
        onsetProgressPct,
        onsetAvgTimeSec,
        detected,
      },
    };
  }

  /* ==========================================================================
   * FEATURE 5: Space Utilization & Dead-Zone Map
   * ========================================================================== */
  function renderDeadZoneCanvas(deadMask, observedMask, nx, ny, yUp) {
    if (typeof document === "undefined") return null;
    const cv = document.createElement("canvas");
    cv.width = nx;
    cv.height = ny;
    const cx = cv.getContext("2d");
    const im = cx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const row = yUp ? ny - 1 - j : j;
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const o = (row * nx + i) * 4;
        if (deadMask[idx]) {
          // Subtle diagonal hatch in cool slate-blue/cyan for dead zones
          const hatch = (i + j) % 3 === 0 ? 1 : 0;
          im.data[o] = hatch ? 100 : 56;
          im.data[o + 1] = hatch ? 160 : 95;
          im.data[o + 2] = hatch ? 225 : 150;
          im.data[o + 3] = hatch ? 155 : 95;
        } else if (observedMask[idx]) {
          // Light emerald tint for active utilized area
          im.data[o] = 61;
          im.data[o + 1] = 220;
          im.data[o + 2] = 151;
          im.data[o + 3] = 55;
        }
      }
    }
    cx.putImageData(im, 0, 0);
    return cv;
  }

  function computeSpaceUtilization(heatmap, cams, opts) {
    const o = opts || {};
    const nx = heatmap ? heatmap.nx : 1;
    const ny = heatmap ? heatmap.ny : 1;
    const cell = heatmap ? heatmap.cell : 1;
    const yUp = heatmap ? !!heatmap.yUp : true;
    const isMetric = heatmap ? heatmap.isMetric !== false : true;
    const [x0, y0] = heatmap && heatmap.bounds ? heatmap.bounds : [0, 0, 1, 1];
    const nCells = nx * ny;
    const visitorsRaw =
      heatmap && heatmap.rawGrids && heatmap.rawGrids.visitors
        ? heatmap.rawGrids.visitors
        : new Float32Array(nCells);
    const presenceRaw =
      heatmap && heatmap.rawGrids && heatmap.rawGrids.presence
        ? heatmap.rawGrids.presence
        : new Float32Array(nCells);

    const validCams = Array.isArray(cams)
      ? cams.filter((cm) => cm && Array.isArray(cm.fov) && cm.fov.length >= 3)
      : [];

    const observedMask = new Uint8Array(nCells);
    let observedCells = 0;

    if (validCams.length) {
      for (let j = 0; j < ny; j++) {
        const wy = y0 + (j + 0.5) * cell;
        const row = j * nx;
        for (let i = 0; i < nx; i++) {
          const wx = x0 + (i + 0.5) * cell;
          for (let c = 0; c < validCams.length; c++) {
            if (pointInPoly(wx, wy, validCams[c].fov)) {
              observedMask[row + i] = 1;
              observedCells++;
              break;
            }
          }
        }
      }
    }

    // Fallback to full bounds if no camera FOVs or FOVs didn't overlap grid
    const usedCameraFov = observedCells > 0;
    if (!usedCameraFov) {
      observedMask.fill(1);
      observedCells = nCells;
    }

    const deadMask = new Uint8Array(nCells);
    let activeCells = 0;
    let deadCells = 0;
    let underutilizedCells = 0;
    let maxVis = 0;
    for (let i = 0; i < nCells; i++) {
      if (observedMask[i] && visitorsRaw[i] > maxVis) maxVis = visitorsRaw[i];
    }
    const underThresh = Math.max(1, maxVis * 0.15);

    for (let i = 0; i < nCells; i++) {
      if (!observedMask[i]) continue;
      if (visitorsRaw[i] > 0 || presenceRaw[i] > 1e-9) {
        activeCells++;
        if (visitorsRaw[i] <= underThresh) underutilizedCells++;
      } else {
        deadCells++;
        deadMask[i] = 1;
      }
    }

    // Connected-component BFS to find top dead-zone clusters
    const visited = new Uint8Array(nCells);
    const clusters = [];
    const queue = new Int32Array(nCells);
    const minClusterCells = o.minClusterCells > 0 ? o.minClusterCells : Math.max(3, Math.floor(observedCells * 0.01));

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const start = j * nx + i;
        if (!deadMask[start] || visited[start]) continue;
        let head = 0;
        let tail = 0;
        queue[tail++] = start;
        visited[start] = 1;

        let count = 0;
        let sumX = 0;
        let sumY = 0;
        let minI = i;
        let maxI = i;
        let minJ = j;
        let maxJ = j;

        while (head < tail) {
          const cur = queue[head++];
          const ci = cur % nx;
          const cj = (cur - ci) / nx;
          count++;
          sumX += x0 + (ci + 0.5) * cell;
          sumY += y0 + (cj + 0.5) * cell;
          if (ci < minI) minI = ci;
          if (ci > maxI) maxI = ci;
          if (cj < minJ) minJ = cj;
          if (cj > maxJ) maxJ = cj;

          if (ci > 0 && deadMask[cur - 1] && !visited[cur - 1]) {
            visited[cur - 1] = 1;
            queue[tail++] = cur - 1;
          }
          if (ci + 1 < nx && deadMask[cur + 1] && !visited[cur + 1]) {
            visited[cur + 1] = 1;
            queue[tail++] = cur + 1;
          }
          if (cj > 0 && deadMask[cur - nx] && !visited[cur - nx]) {
            visited[cur - nx] = 1;
            queue[tail++] = cur - nx;
          }
          if (cj + 1 < ny && deadMask[cur + nx] && !visited[cur + nx]) {
            visited[cur + nx] = 1;
            queue[tail++] = cur + nx;
          }
        }

        if (count >= minClusterCells) {
          clusters.push({
            cx: sumX / count,
            cy: sumY / count,
            cellCount: count,
            areaSqM: isMetric ? count * cell * cell : null,
            areaPctOfObserved: observedCells > 0 ? count / observedCells : 0,
            bbox: [
              x0 + minI * cell,
              y0 + minJ * cell,
              x0 + (maxI + 1) * cell,
              y0 + (maxJ + 1) * cell,
            ],
          });
        }
      }
    }

    clusters.sort((a, b) => b.cellCount - a.cellCount);
    const topClusters = clusters.slice(0, o.topClusters > 0 ? o.topClusters : 5);
    const activeAreaRatio = observedCells > 0 ? activeCells / observedCells : 0;
    const deadAreaRatio = observedCells > 0 ? deadCells / observedCells : 0;
    const canvas = deadCells > 0 || activeCells > 0 ? renderDeadZoneCanvas(deadMask, observedMask, nx, ny, yUp) : null;

    return {
      nx,
      ny,
      cell,
      yUp,
      isMetric,
      usedCameraFov,
      observedCells,
      activeCells,
      deadCells,
      underutilizedCells,
      activeAreaRatio,
      deadAreaRatio,
      observedAreaSqM: isMetric ? observedCells * cell * cell : null,
      deadAreaSqM: isMetric ? deadCells * cell * cell : null,
      observedMask,
      deadMask,
      deadZoneClusters: topClusters,
      canvas,
    };
  }

  /* ==========================================================================
   * FEATURE 7: Anonymous Visitor Archetypes (Verón & Levasseur)
   * ========================================================================== */
  function classifyVisitorArchetypes(tracks, bounds, exhibits, opts) {
    const o = opts || {};
    const validTracks = Array.isArray(tracks) ? tracks : [];
    const [x0, y0, x1, y1] = Array.isArray(bounds) && bounds.length === 4 ? bounds : [0, 0, 1, 1];
    const roomW = Math.max(1e-6, x1 - x0);
    const roomH = Math.max(1e-6, y1 - y0);
    const roomDiag = Math.hypot(roomW, roomH);
    const isMetric = o.isMetric !== false;
    const exList = Array.isArray(exhibits) ? exhibits : [];
    const zoneRadius = o.zoneRadius > 0 ? o.zoneRadius : isMetric ? 2.5 : 0.14;
    const zoneR2 = zoneRadius * zoneRadius;

    // Classification thresholds (Verón & Levasseur behavioral axes)
    const highCoverage = o.highCoverage != null ? o.highCoverage : 0.45;
    const highStopRatio = o.highStopRatio != null ? o.highStopRatio : 0.45;
    const lowStopRatio = o.lowStopRatio != null ? o.lowStopRatio : 0.18;
    const fastSpeed = o.fastSpeed != null ? o.fastSpeed : isMetric ? 1.15 : 0.055;

    const counts = { ant: 0, butterfly: 0, fish: 0, grasshopper: 0 };
    const visitors = [];

    for (let t = 0; t < validTracks.length; t++) {
      const tr = validTracks[t];
      const S = tr.samples || [];
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      const zonesHit = new Set();

      for (let k = 0; k < S.length; k++) {
        const s = S[k];
        if (s.x < minX) minX = s.x;
        if (s.x > maxX) maxX = s.x;
        if (s.y < minY) minY = s.y;
        if (s.y > maxY) maxY = s.y;
        for (let e = 0; e < exList.length; e++) {
          const dx = s.x - exList[e].x;
          const dy = s.y - exList[e].y;
          if (dx * dx + dy * dy <= zoneR2) {
            zonesHit.add(exList[e].id);
          }
        }
      }

      const spanDiag = S.length ? Math.hypot(maxX - minX, maxY - minY) : 0;
      const linearCoverage = clamp(spanDiag / roomDiag, 0, 1);
      const zoneCoverage = exList.length >= 2 ? zonesHit.size / exList.length : linearCoverage;
      const coverage = exList.length >= 2 ? 0.5 * linearCoverage + 0.5 * zoneCoverage : linearCoverage;
      const stopRatio = tr.totalTime > 0 ? clamp(tr.dwellTime / tr.totalTime, 0, 1) : 0;
      const avgSpeed = tr.avgSpeed || 0;

      let archetype;
      if (stopRatio < lowStopRatio || (avgSpeed >= fastSpeed * 1.25 && stopRatio < lowStopRatio + 0.04)) {
        archetype = "fish";
      } else if (coverage >= highCoverage) {
        archetype = stopRatio >= highStopRatio ? "ant" : "butterfly";
      } else {
        archetype = "grasshopper";
      }

      counts[archetype]++;
      visitors.push({
        id: tr.id,
        archetype,
        label: ARCHETYPE_META[archetype].label,
        color: ARCHETYPE_META[archetype].color,
        avgSpeed,
        stopRatio,
        coverage,
        linearCoverage,
        zoneCoverage,
        totalTime: tr.totalTime || 0,
        dwellTime: tr.dwellTime || 0,
        pathLen: tr.pathLen || 0,
      });
    }

    const total = visitors.length;
    const shares = {
      ant: total > 0 ? counts.ant / total : 0,
      butterfly: total > 0 ? counts.butterfly / total : 0,
      fish: total > 0 ? counts.fish / total : 0,
      grasshopper: total > 0 ? counts.grasshopper / total : 0,
    };

    let dominant = null;
    let bestCount = 0;
    for (const key of ["ant", "butterfly", "fish", "grasshopper"]) {
      if (counts[key] > bestCount) {
        bestCount = counts[key];
        dominant = key;
      }
    }

    return {
      total,
      counts,
      shares,
      dominant,
      dominantMeta: dominant ? ARCHETYPE_META[dominant] : null,
      meta: ARCHETYPE_META,
      visitors,
    };
  }

  /* ==========================================================================
   * Unified Orchestrator
   * ========================================================================== */
  function computeMuseumAnalytics(input, opts) {
    const src = input || {};
    const o = opts || {};

    let heatmap = src.heatmap || null;
    if (!heatmap && EFHeat && Array.isArray(src.frames)) {
      heatmap = EFHeat.computeHeatmap(src.frames, src.bounds || [0, 0, 1, 1], o.heatmapOpts || o);
    }

    const isMetric =
      o.isMetric !== undefined ? !!o.isMetric : heatmap ? heatmap.isMetric !== false : true;
    const bounds = (heatmap && heatmap.bounds) || src.bounds || [0, 0, 1, 1];
    const tracks = (heatmap && heatmap.tracks) || src.tracks || [];
    const cams = src.cams || [];

    const normEx = normalizeExhibits(src.exhibits, heatmap, cams);
    const attraction = computeAttractionHolding(tracks, normEx.exhibits, {
      ...(o.attraction || {}),
      isMetric,
      source: normEx.source,
    });
    const bottleneck = computeBottlenecks(
      heatmap,
      normEx.source === "exhibits" ? normEx.exhibits : [],
      { ...(o.bottleneck || {}), isMetric }
    );
    const { flow, skipRate, fatigue } = computeFlowAndFatigue(tracks, heatmap, attraction, {
      ...(o.flow || {}),
      bounds,
      isMetric,
    });
    const space = computeSpaceUtilization(heatmap, cams, o.space || {});
    const archetypes = classifyVisitorArchetypes(tracks, bounds, normEx.exhibits, {
      ...(o.archetypes || {}),
      isMetric,
    });

    return {
      meta: {
        isMetric,
        bounds,
        totalVisitors: tracks.length,
        exhibitSource: normEx.source,
        durationSec: heatmap && heatmap.analytics ? heatmap.analytics.durationSec : 0,
      },
      attraction,
      bottleneck,
      flow,
      skipRate,
      fatigue,
      space,
      archetypes,
    };
  }

  return {
    QUADRANT_META,
    ARCHETYPE_META,
    pointInPoly,
    normalizeExhibits,
    computeAttractionHolding,
    computeBottlenecks,
    computeFlowAndFatigue,
    computeSpaceUtilization,
    classifyVisitorArchetypes,
    computeMuseumAnalytics,
  };
});
