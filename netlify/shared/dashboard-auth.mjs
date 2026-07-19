const encoder = new TextEncoder();
const decoder = new TextDecoder();

export const DASHBOARD_ACCESS_COOKIE = "jdw_dashboard_access";
export const DEFAULT_DAILY_LIMIT = 20;
export const TOKEN_TTL_SECONDS = 30 * 60;

const encodeBase64Url = (value) => {
  const bytes = typeof value === "string" ? encoder.encode(value) : value;
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
};

const decodeBase64Url = (value) => {
  const padded = value.replaceAll("-", "+").replaceAll("_", "/")
    .padEnd(Math.ceil(value.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
};

const importSigningKey = (secret) =>
  crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );

export const getKstDateKey = (date = new Date()) => {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return `${values.year}${values.month}${values.day}`;
};

export const hashSessionId = async (sessionId, secret) => {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    encoder.encode(`${secret}:${sessionId}`),
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
};

export const signLaunchToken = async ({
  sessionHash,
  secret,
  now = Date.now(),
  ttlSeconds = TOKEN_TTL_SECONDS,
}) => {
  const payload = encodeBase64Url(JSON.stringify({
    sid: sessionHash,
    exp: Math.floor(now / 1000) + ttlSeconds,
  }));
  const key = await importSigningKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(payload));
  return `${payload}.${encodeBase64Url(new Uint8Array(signature))}`;
};

export const verifyLaunchToken = async (token, secret, now = Date.now()) => {
  if (!token || !secret) return null;
  const [payload, signature, extra] = token.split(".");
  if (!payload || !signature || extra) return null;

  try {
    const key = await importSigningKey(secret);
    const isValid = await crypto.subtle.verify(
      "HMAC",
      key,
      decodeBase64Url(signature),
      encoder.encode(payload),
    );
    if (!isValid) return null;

    const parsed = JSON.parse(decoder.decode(decodeBase64Url(payload)));
    if (
      typeof parsed.sid !== "string" ||
      typeof parsed.exp !== "number" ||
      parsed.exp <= Math.floor(now / 1000)
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
};

export const parseCookieHeader = (cookieHeader = "") =>
  Object.fromEntries(
    cookieHeader
      .split(";")
      .map((entry) => entry.trim())
      .filter(Boolean)
      .map((entry) => {
        const separator = entry.indexOf("=");
        if (separator === -1) return [entry, ""];
        return [
          entry.slice(0, separator),
          decodeURIComponent(entry.slice(separator + 1)),
        ];
      }),
  );
