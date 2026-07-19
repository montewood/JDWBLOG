import assert from "node:assert/strict";
import test from "node:test";
import {
  DASHBOARD_ACCESS_COOKIE,
  signLaunchToken,
} from "../../netlify/shared/dashboard-auth.mjs";

const secret = "test-signing-secret-with-sufficient-entropy";
globalThis.Netlify = {
  env: {
    get: (name) => name === "DASHBOARD_SIGNING_SECRET" ? secret : undefined,
  },
};

const { default: dashboardGate } = await import(
  "../../netlify/edge-functions/dashboard-gate.ts"
);

const context = {
  next: async () => new Response("allowed", { status: 200 }),
};

test("valid query token becomes a short-lived functional cookie", async () => {
  const token = await signLaunchToken({
    sessionHash: "hashed-session",
    secret,
  });
  const response = await dashboardGate(new Request(
    `https://www.jdwblog.com/apps/profile-widget/?launch_token=${encodeURIComponent(token)}`,
  ), context);

  assert.equal(response.status, 302);
  assert.equal(
    response.headers.get("location"),
    "https://www.jdwblog.com/apps/profile-widget/",
  );
  const cookie = response.headers.get("set-cookie");
  assert.match(cookie, new RegExp(`^${DASHBOARD_ACCESS_COOKIE}=`));
  assert.match(cookie, /HttpOnly/);
  assert.match(cookie, /SameSite=Strict/);
});

test("valid access cookie allows the static app response", async () => {
  const token = await signLaunchToken({
    sessionHash: "hashed-session",
    secret,
  });
  const response = await dashboardGate(new Request(
    "https://www.jdwblog.com/apps/profile-widget/",
    {
      headers: {
        Cookie: `${DASHBOARD_ACCESS_COOKIE}=${encodeURIComponent(token)}`,
      },
    },
  ), context);

  assert.equal(response.status, 200);
  assert.equal(await response.text(), "allowed");
});

test("missing or invalid access is denied", async () => {
  const missing = await dashboardGate(new Request(
    "https://www.jdwblog.com/apps/profile-widget/",
  ), context);
  const invalid = await dashboardGate(new Request(
    "https://www.jdwblog.com/apps/profile-widget/?launch_token=invalid",
  ), context);

  assert.equal(missing.status, 403);
  assert.equal(invalid.status, 403);
});
