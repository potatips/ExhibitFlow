"use strict";
/* Thống kê dừng chân theo hiện vật, để quản lý bảo tàng sắp xếp trưng bày. Chỉ là thống kê: không liên quan cảnh báo hành vi.
 * Một khách "dừng" khi đi chậm hơn STILL (m/s) trong phạm vi REACH (m) trước hiện vật; tính cho hiện vật GẦN NHẤT.
 * Ước tính: hệ thống không biết hướng nhìn. Module thuần: window.EFExhibits (trình duyệt) và module.exports (Node.js). */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && typeof root === "object") root.EFExhibits = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const REACH = 1.5, STILL = 0.35, MIN_STOP = 1.0;

  /* frames = [{t, people:[{id,x,y}]}] (mét), exhibits = [{id, label, pedestal, x, y}] */
  function stats(frames, exhibits, opts) {
    const o = { reach: REACH, still: STILL, minStop: MIN_STOP, ...(opts || {}) };
    const tracks = new Map();
    for (const f of frames || []) for (const p of f.people || []) {
      if (!tracks.has(p.id)) tracks.set(p.id, []);
      tracks.get(p.id).push([f.t, p.x, p.y]);
    }
    const stay = new Map((exhibits || []).map((e) => [e.id, new Map()]));     // hiện vật -> (khách -> giây)
    for (const [pid, q] of tracks) {
      for (let k = 1; k < q.length; k++) {
        const dt = q[k][0] - q[k - 1][0];
        if (!(dt > 0) || dt > 1) continue;
        const a = q[Math.max(0, k - 2)], b = q[Math.min(q.length - 1, k + 2)];
        if (Math.hypot(b[1] - a[1], b[2] - a[2]) / Math.max(1e-6, b[0] - a[0]) >= o.still) continue;
        let best = null, dist = o.reach;
        for (const e of exhibits || []) {
          const d = Math.hypot(q[k][1] - e.x, q[k][2] - e.y);
          if (d <= dist) { dist = d; best = e.id; }
        }
        if (best !== null) stay.get(best).set(pid, (stay.get(best).get(pid) || 0) + dt);
      }
    }
    const visitors = tracks.size;
    const rows = (exhibits || []).map((e) => {
      const who = [...stay.get(e.id)].filter(([, s]) => s >= o.minStop);
      const total = who.reduce((m, [, s]) => m + s, 0);
      return {
        id: e.id, label: e.label || "", pedestal: !!e.pedestal,
        visitors: who.map(([p]) => p).sort((a, b) => a - b),
        share: visitors ? who.length / visitors : 0,
        totalSec: total, avgSec: who.length ? total / who.length : 0,
      };
    }).sort((a, b) => b.totalSec - a.totalSec || a.id - b.id);
    return { visitors, rows };
  }

  function kind(row) {
    if (row.pedestal || /pedestal|display|case|plinth/i.test(row.label)) return "Bục trưng bày";
    return /paint|picture|frame/i.test(row.label) ? "Tranh treo tường" : row.label || "Hiện vật";
  }

  function level(row, rows) {
    if (!row.totalSec) return "Chưa ai dừng";
    return rows.length && row.totalSec >= rows[0].totalSec ? "Được chú ý nhất" : "Có khách dừng";
  }

  return { REACH, STILL, MIN_STOP, stats, kind, level };
});
