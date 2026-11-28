"use strict";
const test = require("node:test");
const assert = require("node:assert");
const E = require("../js/errors.js");

test("server messages written for the user are kept", () => {
  assert.strictEqual(E.friendly(E.httpError(400, "Thiếu hiệu chuẩn cho: a.mp4")), "Thiếu hiệu chuẩn cho: a.mp4");
  assert.strictEqual(E.friendly(E.userError("Không gán được video")), "Không gán được video");
});

test("unexpected failures become a retry message", () => {
  for (const err of [E.httpError(500, "Internal Server Error"), E.httpError(404, "Không có job này"), E.httpError(422, [{ loc: 1 }]),
    new TypeError("Failed to fetch"), new Error("RuntimeError: Input type (torch.cuda.FloatTensor) should be the same"),
    new Error("RuntimeError: map2d lỗi: File \"x.py\" | thiếu hiệu chuẩn"), undefined]) {
    assert.strictEqual(E.friendly(err), E.RETRY, String(err && err.message));
  }
});

test("background job errors the user can fix are shown without the class name", () => {
  assert.strictEqual(E.friendly(new Error("RuntimeError: Video 640x480 khác tỉ lệ khung của hiệu chuẩn 1280x720")),
    "Video 640x480 khác tỉ lệ khung của hiệu chuẩn 1280x720");
});
