"use strict";
/* Lỗi người dùng tự xử lý được (sai file, sai hiệu chuẩn, server đang bận) thì giữ lời nhắn; mọi lỗi ngoài dự kiến
 * (server hỏng, mất mạng, lỗi lập trình, traceback) đổi thành "xin thử lại". Chi tiết kỹ thuật vẫn ghi ở console.
 * Module thuần: window.EFErr (trình duyệt) và module.exports (Node.js). */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && typeof root === "object") root.EFErr = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const RETRY = "Có lỗi xảy ra, xin thử lại.";
  const USER_STATUS = [400, 409, 413, 501];
  // lỗi của job chạy nền mà người dùng sửa được: hiệu chuẩn, video, server bận
  const KNOWN = /khác tỉ lệ khung|hiệu chuẩn|không nhìn thấy phần sàn|Không mở được video|đang nạp|đang chạy|đang phân tích/i;
  const TECHNICAL = /Traceback|map2d lỗi|File "|\n/;

  /* lỗi từ một response HTTP không thành công; detail = trường `detail` server trả */
  function httpError(status, detail) {
    const e = new Error(typeof detail === "string" && detail ? detail : "HTTP " + status);
    e.user = USER_STATUS.includes(status) && typeof detail === "string" && !!detail;
    return e;
  }

  function userError(message) {
    const e = new Error(message);
    e.user = true;
    return e;
  }

  function friendly(err) {
    const raw = String((err && err.message) || err || "");
    const msg = raw.replace(/^[A-Za-z]*(Error|Exception):\s*/, "");
    if (err && err.user) return msg;
    if (KNOWN.test(msg) && !TECHNICAL.test(msg) && msg.length <= 300) return msg;
    return RETRY;
  }

  return { RETRY, httpError, userError, friendly };
});
