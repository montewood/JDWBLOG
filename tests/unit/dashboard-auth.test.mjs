import assert from "node:assert/strict";
import test from "node:test";
import {
  DASHBOARD_ACCESS_COOKIE,
  getKstDateKey,
  hashSessionId,
  parseCookieHeader,
  signLaunchToken,
  verifyLaunchToken,
} from "../../netlify/shared/dashboard-auth.mjs";

const secret = "test-signing-secret-with-sufficient-entropy";
const now = Date.UTC(2026, 6, 19, 1, 0, 0);

test("launch tokens verify before expiry", async () => {
  const sessionHash = await hashSessionId("session_1234567890", secret);
  const token = await signLaunchToken({
    sessionHash,
    secret,
    now,
    ttlSeconds: 60,
  });
  const payload = await verifyLaunchToken(token, secret, now + 30_000);

  assert.equal(payload?.sid, sessionHash);
  assert.equal(payload?.exp, Math.floor(now / 1000) + 60);
});

test("launch tokens reject expiry and tampering", async () => {
  const sessionHash = await hashSessionId("session_1234567890", secret);
  const token = await signLaunchToken({
    sessionHash,
    secret,
    now,
    ttlSeconds: 60,
  });

  assert.equal(await verifyLaunchToken(token, secret, now + 61_000), null);
  assert.equal(await verifyLaunchToken(`${token}x`, secret, now), null);
  assert.equal(await verifyLaunchToken(token, `${secret}-wrong`, now), null);
});

test("KST quota key changes at Korean midnight", () => {
  assert.equal(getKstDateKey(new Date("2026-07-18T14:59:59Z")), "20260718");
  assert.equal(getKstDateKey(new Date("2026-07-18T15:00:00Z")), "20260719");
});

test("functional access cookie can be parsed", () => {
  const cookies = parseCookieHeader(
    `theme=dark; ${DASHBOARD_ACCESS_COOKIE}=payload.signature`,
  );
  assert.equal(cookies[DASHBOARD_ACCESS_COOKIE], "payload.signature");
});
