import type { Config, Context } from "@netlify/functions";
import { Redis } from "@upstash/redis";
import {
  DEFAULT_DAILY_LIMIT,
  TOKEN_TTL_SECONDS,
  getKstDateKey,
  hashSessionId,
  signLaunchToken,
} from "../shared/dashboard-auth.mjs";

const COUNTER_TTL_SECONDS = 48 * 60 * 60;
const SESSION_ID_PATTERN = /^[a-zA-Z0-9_-]{16,128}$/;
const INCREMENT_SCRIPT = `
local seen = redis.call("EXISTS", KEYS[2])
local count = tonumber(redis.call("GET", KEYS[1]) or "0")
if seen == 1 then
  return { count, 0 }
end
if count >= tonumber(ARGV[1]) then
  return { count, -1 }
end
count = redis.call("INCR", KEYS[1])
redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
redis.call("SET", KEYS[2], "1", "EX", tonumber(ARGV[2]))
return { count, 1 }
`;

const responseHeaders = {
  "Cache-Control": "no-store",
  "Content-Type": "application/json; charset=utf-8",
  "X-Content-Type-Options": "nosniff",
};

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: responseHeaders,
  });

const getEnvironmentValue = (name: string) =>
  process.env[name] ||
  (typeof Netlify !== "undefined" ? Netlify.env.get(name) : undefined);

const getConfiguration = () => {
  const redisUrl = getEnvironmentValue("UPSTASH_REDIS_REST_URL");
  const redisToken = getEnvironmentValue("UPSTASH_REDIS_REST_TOKEN");
  const signingSecret = getEnvironmentValue("DASHBOARD_SIGNING_SECRET");
  const configuredLimit = Number(
    getEnvironmentValue("DASHBOARD_DAILY_LIMIT") || DEFAULT_DAILY_LIMIT,
  );

  const missing = [
    !redisUrl && "UPSTASH_REDIS_REST_URL",
    !redisToken && "UPSTASH_REDIS_REST_TOKEN",
    !signingSecret && "DASHBOARD_SIGNING_SECRET",
  ].filter(Boolean);
  if (missing.length) return { missing };
  const limit = Number.isInteger(configuredLimit) && configuredLimit > 0
    ? configuredLimit
    : DEFAULT_DAILY_LIMIT;
  return { redisUrl, redisToken, signingSecret, limit };
};

const isSameOriginRequest = (request: Request) => {
  const origin = request.headers.get("Origin");
  return !origin || origin === new URL(request.url).origin;
};

export default async (request: Request, _context: Context) => {
  if (!isSameOriginRequest(request)) {
    return jsonResponse({ error: "Cross-origin launch requests are not allowed." }, 403);
  }

  const configuration = getConfiguration();
  if ("missing" in configuration) {
    return jsonResponse({
      error: "Dashboard quota is not configured.",
      missing: configuration.missing,
    }, 503);
  }

  const { redisUrl, redisToken, signingSecret, limit } = configuration;
  const redis = new Redis({ url: redisUrl, token: redisToken });
  const dateKey = getKstDateKey();
  const hostname = new URL(request.url).hostname;
  const productionHosts = new Set([
    "www.jdwblog.com",
    "jdwblog.com",
    "jdwblog.netlify.app",
  ]);
  const quotaNamespace = productionHosts.has(hostname) ? "production" : "preview";
  const keyPrefix = `dashboard:v1:${quotaNamespace}`;
  const countKey = `${keyPrefix}:launches:${dateKey}`;
  const currentCount = Number(await redis.get<number>(countKey) || 0);

  if (request.method === "GET") {
    const sessionId = new URL(request.url).searchParams.get("session_id") || "";
    let sessionSeen = false;
    if (SESSION_ID_PATTERN.test(sessionId)) {
      const sessionHash = await hashSessionId(sessionId, signingSecret);
      sessionSeen = Boolean(await redis.exists(
        `${keyPrefix}:sessions:${dateKey}:${sessionHash}`,
      ));
    }
    return jsonResponse({
      count: currentCount,
      limit,
      remaining: Math.max(limit - currentCount, 0),
      available: sessionSeen || currentCount < limit,
      sessionSeen,
      date: dateKey,
      context: quotaNamespace,
    });
  }

  if (request.method !== "POST") {
    return jsonResponse({ error: "Method not allowed." }, 405);
  }

  let sessionId = "";
  try {
    const body = await request.json() as { sessionId?: unknown };
    sessionId = typeof body.sessionId === "string" ? body.sessionId : "";
  } catch {
    return jsonResponse({ error: "Invalid JSON body." }, 400);
  }

  if (!SESSION_ID_PATTERN.test(sessionId)) {
    return jsonResponse({ error: "Invalid dashboard session." }, 400);
  }

  const sessionHash = await hashSessionId(sessionId, signingSecret);
  const sessionKey = `${keyPrefix}:sessions:${dateKey}:${sessionHash}`;
  const result = await redis.eval(
    INCREMENT_SCRIPT,
    [countKey, sessionKey],
    [limit, COUNTER_TTL_SECONDS],
  ) as [number | string, number | string];
  const [count, incrementState] = result.map(Number);
  const allowed = incrementState >= 0;

  if (!allowed) {
    return jsonResponse({
      allowed: false,
      count,
      limit,
      remaining: 0,
      date: dateKey,
      context: quotaNamespace,
    }, 429);
  }

  const token = await signLaunchToken({
    sessionHash,
    secret: signingSecret,
  });
  const expiresAt = Date.now() + TOKEN_TTL_SECONDS * 1000;

  return jsonResponse({
    allowed: true,
    incremented: incrementState === 1,
    count,
    limit,
    remaining: Math.max(limit - count, 0),
    date: dateKey,
    context: quotaNamespace,
    expiresAt,
    appUrl: `/apps/profile-widget/?launch_token=${encodeURIComponent(token)}`,
  });
};

export const config = {
  path: "/api/dashboard-launch",
  rateLimit: {
    windowLimit: 30,
    windowSize: 60,
    aggregateBy: ["domain", "ip"],
  },
} satisfies Config;
