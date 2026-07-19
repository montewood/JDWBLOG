import type { Config, Context } from "@netlify/edge-functions";
import {
  DASHBOARD_ACCESS_COOKIE,
  TOKEN_TTL_SECONDS,
  parseCookieHeader,
  verifyLaunchToken,
} from "../shared/dashboard-auth.mjs";

const deniedPage = (message: string, status = 403) =>
  new Response(
    `<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>대시보드 실행 제한</title>
  </head>
  <body>
    <main>
      <h1>대시보드를 열 수 없습니다.</h1>
      <p>${message}</p>
      <p><a href="/">홈으로 돌아가기</a></p>
    </main>
  </body>
</html>`,
    {
      status,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": "text/html; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
      },
    },
  );

export default async (request: Request, context: Context) => {
  const signingSecret = Netlify.env.get("DASHBOARD_SIGNING_SECRET");
  if (!signingSecret) {
    return deniedPage("실행 제한 설정을 확인해 주세요.", 503);
  }

  const url = new URL(request.url);
  const launchToken = url.searchParams.get("launch_token");

  if (launchToken) {
    const payload = await verifyLaunchToken(launchToken, signingSecret);
    if (!payload) return deniedPage("실행 권한이 만료되었거나 유효하지 않습니다.");

    url.searchParams.delete("launch_token");
    return new Response(null, {
      status: 302,
      headers: {
        "Cache-Control": "no-store",
        "Location": url.toString(),
        "Set-Cookie":
          `${DASHBOARD_ACCESS_COOKIE}=${encodeURIComponent(launchToken)}; ` +
          `Max-Age=${TOKEN_TTL_SECONDS}; Path=/apps/profile-widget/; ` +
          "HttpOnly; Secure; SameSite=Strict",
      },
    });
  }

  const cookies = parseCookieHeader(request.headers.get("Cookie") || "");
  const cookieToken = cookies[DASHBOARD_ACCESS_COOKIE];
  const payload = await verifyLaunchToken(cookieToken, signingSecret);
  if (!payload) {
    return deniedPage("홈 화면의 대시보드 실행 버튼을 이용해 주세요.");
  }

  return context.next();
};

export const config = {
  path: [
    "/apps/profile-widget/",
    "/apps/profile-widget/index.html",
    "/apps/profile-widget/edit/",
    "/apps/profile-widget/edit/index.html",
  ],
} satisfies Config;
