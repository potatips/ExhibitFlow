"use strict";
/* Vị trí người trên map THEO THỜI GIAN video: nội suy giữa hai mẫu, vệt đi tới thời điểm đang xem, và quy tắc hiển thị
 * "chưa có hành vi bất thường thì hiện mọi người; đã phát hiện thì chỉ hiện người bất thường kèm vệt đi".
 * Module thuần (không đụng DOM): window.EFPlay (trình duyệt) và module.exports (Node.js). */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && typeof root === "object") root.EFPlay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const GAP = 1.0;   // hai mẫu cách nhau không quá GAP giây thì nội suy (che được vài mẫu phát hiện hụt, chấm không nhấp nháy)
  const HOLD = 0.4;  // hết mẫu: giữ chấm thêm HOLD giây rồi ẩn
  const RANK = { touching: 1, holding: 2, stealing: 3 };

  /* frames = [{t, people:[{id,x,y}]}] -> Map id -> [[t,x,y], ...] theo thời gian */
  function buildTracks(frames) {
    const tracks = new Map();
    for (const f of frames || []) {
      for (const p of f.people || []) {
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
        let tr = tracks.get(p.id);
        if (!tr) tracks.set(p.id, (tr = []));
        tr.push([f.t, p.x, p.y]);
      }
    }
    for (const tr of tracks.values()) tr.sort((a, b) => a[0] - b[0]);
    return tracks;
  }

  function lastIndex(tr, t) {          // chỉ số mẫu cuối cùng có thời điểm <= t, -1 nếu chưa xuất hiện
    let lo = 0, hi = tr.length - 1;
    if (!tr.length || tr[0][0] > t + 1e-9) return -1;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (tr[m][0] <= t + 1e-9) lo = m; else hi = m - 1;
    }
    return lo;
  }

  function posAt(tr, t) {
    const i = lastIndex(tr, t);
    if (i < 0) return null;
    const a = tr[i], b = tr[i + 1];
    if (b && b[0] - a[0] <= GAP) {
      const u = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0] || 1)));
      return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
    }
    return t - a[0] <= HOLD ? [a[1], a[2]] : null;
  }

  /* vệt đi tới thời điểm t: các mẫu đã qua + vị trí nội suy hiện tại */
  function trailUpTo(tr, t) {
    const i = lastIndex(tr, t);
    if (i < 0) return [];
    const pts = tr.slice(0, i + 1).map((q) => [q[1], q[2]]);
    const now = posAt(tr, t);
    if (now) pts.push(now);
    return pts;
  }

  /* events = [{id, behavior, start}] -> Map id người -> hành vi nặng nhất đã bắt đầu tính tới t */
  function flaggedAt(events, t) {
    const out = new Map();
    for (const e of events || []) {
      if (e.id == null || e.start > t + 1e-9) continue;
      const rank = RANK[e.behavior] || 3;
      const cur = out.get(e.id);
      if (!cur || rank > cur.rank) out.set(e.id, { behavior: e.behavior, rank });
    }
    return out;
  }

  /* Cảnh trên map tại thời điểm t. flagged rỗng: people = mọi người đang có mặt. Ngược lại chỉ người bất thường,
   * mỗi người kèm vệt đi (người đã rời đi vẫn giữ vệt, chấm nằm ở vị trí cuối). */
  function sceneAt(tracks, events, t) {
    const flagged = flaggedAt(events, t);
    if (!flagged.size) {
      const people = [];
      for (const [id, tr] of tracks) {
        const p = posAt(tr, t);
        if (p) people.push({ id, x: p[0], y: p[1] });
      }
      return { filtered: false, people, trails: [] };
    }
    const trails = [];
    for (const [id, f] of flagged) {
      const pts = trailUpTo(tracks.get(id) || [], t);
      if (pts.length) trails.push({ id, behavior: f.behavior, pts });
    }
    return { filtered: true, people: [], trails };
  }

  function framesUpTo(frames, t) {
    let lo = 0, hi = (frames || []).length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (frames[m].t <= t + 1e-9) lo = m + 1; else hi = m;
    }
    return (frames || []).slice(0, lo);
  }

  /* Sự kiện bất thường của engine bắt đầu khi người ĐỨNG GẦN, còn lúc chạm thật (báo động) muộn hơn (alarm_start).
   * Trả mốc báo động nếu nó muộn hơn đầu sự kiện, ngược lại null. */
  function alarmOnset(event, alert) {
    if (event.decision !== "ANOMALY") return null;
    const span = (event.alarm_spans || [])[0];
    const s = alert && alert.alarm_start_time_s != null ? alert.alarm_start_time_s : span ? span.start_time_s : null;
    return s != null && s > event.start_time_s + 0.2 ? s : null;
  }

  /* sự kiện đang diễn ra tại t; trước mốc báo động thì sự kiện bất thường chỉ là "cần xác nhận" */
  function activeEventsAt(events, t) {
    return events.filter((e) => t >= e.start && t <= e.end)
      .map((e) => (e.onset != null && t < e.onset ? { ...e, decision: "CANDIDATE", behavior: "uncertain", conf: null } : e));
  }

  /* Số hiện vật được đánh RIÊNG theo từng camera, nên khi có nhiều camera luôn ghi kèm camera (cam = "CAM 1"). */
  const hasId = (id) => id !== null && id !== undefined && id !== "?";
  const camTail = (cam) => (cam ? ` (${cam})` : "");

  /* "Chạm hiện vật" -> "Chạm hiện vật #7 (CAM 1)"; "Cầm / nhấc" -> "Cầm / nhấc hiện vật #7" */
  function actionText(label, id, cam) {
    if (!hasId(id)) return label;
    return `${label}${/hiện vật$/i.test(label) ? "" : " hiện vật"} #${id}${camTail(cam)}`;
  }

  /* "Hiện vật #7 (CAM 1) · tên loại" (bỏ tên nếu tên chỉ là "Hiện vật 7") */
  function artText(id, name, cam) {
    if (!hasId(id)) return name || "";
    const extra = name && !/^hiện vật\s*#?\d*$/i.test(name) ? ` · ${name}` : "";
    return `Hiện vật #${id}${camTail(cam)}${extra}`;
  }

  return { GAP, HOLD, buildTracks, posAt, trailUpTo, flaggedAt, sceneAt, framesUpTo, alarmOnset, activeEventsAt, actionText, artText };
});
