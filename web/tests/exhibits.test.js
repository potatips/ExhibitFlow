"use strict";
const test = require("node:test");
const assert = require("node:assert");
const X = require("../js/exhibits.js");

const exhibits = [{ id: 7, label: "picture frame painting", x: 0, y: 5 }, { id: 4, label: "painting", x: 4, y: 5 }, { id: 9, label: "vase", pedestal: true, x: 20, y: 0 }];
const times = Array.from({ length: 21 }, (_, i) => i * 0.25);          // 5 giây, 4 mẫu/giây
// khách 1 đứng trước #7 suốt; khách 2 đi ngang trước #7 (nhanh); khách 3 đứng cách xa mọi hiện vật
const frames = times.map((t) => ({ t, people: [{ id: 1, x: 0.2, y: 4.2 }, { id: 2, x: -4 + 1.6 * t, y: 4.5 }, { id: 3, x: -8, y: -8 }] }));

test("only people standing close to an exhibit count as stopping there", () => {
  const { visitors, rows } = X.stats(frames, exhibits);
  assert.strictEqual(visitors, 3);
  assert.deepStrictEqual(rows.map((r) => r.id), [7, 4, 9]);
  assert.deepStrictEqual(rows[0].visitors, [1]);
  assert.ok(Math.abs(rows[0].totalSec - 5) < 1e-6 && Math.abs(rows[0].share - 1 / 3) < 1e-9);
  assert.strictEqual(rows[1].totalSec, 0);
});

test("very short pauses are ignored and rows are described for a curator", () => {
  const short = frames.slice(0, 4);
  assert.strictEqual(X.stats(short, exhibits).rows.every((r) => !r.visitors.length), true);
  const { rows } = X.stats(frames, exhibits);
  assert.deepStrictEqual(rows.map((r) => X.level(r, rows)), ["Được chú ý nhất", "Chưa ai dừng", "Chưa ai dừng"]);
  assert.deepStrictEqual(rows.map(X.kind), ["Tranh treo tường", "Tranh treo tường", "Bục trưng bày"]);
  assert.strictEqual(X.kind({ label: "pedestal", pedestal: false }), "Bục trưng bày");
});
