import assert from "node:assert/strict";
import test from "node:test";

const environment = {
  UPSTASH_REDIS_REST_URL: "https://redis.test",
  UPSTASH_REDIS_REST_TOKEN: "test-token",
  DASHBOARD_SIGNING_SECRET: "test-signing-secret-with-sufficient-entropy",
  DASHBOARD_DAILY_LIMIT: "20",
};

Object.assign(process.env, environment);

const { default: dashboardLaunch } = await import(
  "../../netlify/functions/dashboard-launch.ts"
);

const sessionId = "session_1234567890";

const installRedisMock = (evalResult) => {
  const commands = [];
  globalThis.fetch = async (_url, options) => {
    const pipeline = JSON.parse(options.body);
    commands.push(...pipeline);
    return new Response(JSON.stringify(
      pipeline.map(([command]) =>
        command === "eval"
          ? { result: evalResult }
          : { result: null }
      ),
    ), {
      headers: { "Content-Type": "application/json" },
    });
  };
  return commands;
};

const launchRequest = () =>
  new Request("https://www.jdwblog.com/api/dashboard-launch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Origin: "https://www.jdwblog.com",
    },
    body: JSON.stringify({ sessionId }),
  });

test("first session launch consumes one daily slot", async () => {
  const commands = installRedisMock([1, 1]);
  const response = await dashboardLaunch(launchRequest());
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.allowed, true);
  assert.equal(body.incremented, true);
  assert.equal(body.remaining, 19);
  assert.match(body.appUrl, /^\/apps\/profile-widget\/\?launch_token=/);
  assert.equal(commands.some(([command]) => command === "eval"), true);
});

test("known session receives access without another increment", async () => {
  installRedisMock([7, 0]);
  const response = await dashboardLaunch(launchRequest());
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.allowed, true);
  assert.equal(body.incremented, false);
  assert.equal(body.count, 7);
});

test("new session is denied after the daily limit", async () => {
  installRedisMock([20, -1]);
  const response = await dashboardLaunch(launchRequest());
  const body = await response.json();

  assert.equal(response.status, 429);
  assert.equal(body.allowed, false);
  assert.equal(body.remaining, 0);
});

test("cross-origin launch request is rejected before Redis", async () => {
  const response = await dashboardLaunch(new Request(
    "https://www.jdwblog.com/api/dashboard-launch",
    {
      method: "POST",
      headers: { Origin: "https://attacker.example" },
    },
  ));
  assert.equal(response.status, 403);
});

test("draft deploys use a quota namespace separate from production", async () => {
  const commands = installRedisMock([0, 0]);
  const response = await dashboardLaunch(new Request(
    "https://deploy-id--jdwblog.netlify.app/api/dashboard-launch",
  ));
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.context, "preview");
  assert.equal(
    commands.some((command) =>
      command.some((value) =>
        String(value).includes("dashboard:v1:preview:launches:")
      )
    ),
    true,
  );
});
