# Deployment and rollback

## Branches

- `legacy/wowchemy-5.1`: immutable Wowchemy/Hugo 0.82 baseline
- `migration/hugoblox`: migration and Deploy Preview branch
- `main`: production branch after acceptance

The annotated tag `legacy-wowchemy-5.1` identifies the final legacy commit.

## Required build versions

The generated HugoBlox starter pins the supported toolchain:

- Hugo Extended 0.162.0
- Go 1.21.5
- Node.js 22
- pnpm 10.14.0

Netlify reads these values from `netlify.toml`. Do not update Hugo independently
of the HugoBlox module versions in `go.mod`.

## Pre-production checks

Run:

```bash
pnpm install --frozen-lockfile
pnpm run check:unit
pnpm run check:netlify
scripts/validate-site.sh
pnpm run pagefind
scripts/check-quarto-drift.sh
```

On the Netlify Deploy Preview, verify:

1. Homepage profile, recent posts, projects, dark mode, search, and mobile layout.
2. Existing `/post/` URLs, RSS, sitemap, canonical, and Open Graph tags.
3. The homepage visitor dashboard requests only ECharts and the configured daily
   analytics files; it must not request `/apps/profile-widget/`, `R.bin.wasm`,
   or webR packages before user interaction.
4. The default 90-day dashboard reports its successful, missing, legacy, and
   failed dates without failing the whole chart. Backfilled schema v2 files must
   be labelled as daily unique Active Users.
5. Selecting **대시보드 실행** opens the dialog and only then loads the
   Shinylive app under `/apps/profile-widget/`. Closing the dialog must remove
   its iframe and return focus to the launch button.
6. A first session decrements the KST daily quota once. Reopening in the same
   tab session does not decrement it, and a new session is rejected at 20/20.
7. Direct access to `/apps/profile-widget/` without a valid token or access
   cookie returns 403.
8. Browser console has no service-worker, COOP, COEP, or missing-asset errors.
9. Plotly/Leaflet content remains functional.
10. Google Tag Manager fires once with container `GTM-P8STCFM`, and the footer
    privacy link resolves.

Run the focused browser checks with:

```bash
pnpm exec playwright test -g \
  "homepage defers Shinylive|home has no serious accessibility|mobile layouts"
```

Record Netlify bandwidth by URL after promotion. Daily analytics are delivered
by GitHub Raw and do not consume Netlify bandwidth. ECharts is part of the
fingerprinted site JavaScript, while the approximately 68.4 MB Shinylive bundle
is transferred only after an explicit launch.

## Dashboard quota configuration

Create an Upstash Redis database and add the following Netlify environment
variables with Functions scope:

```text
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN
DASHBOARD_SIGNING_SECRET
DASHBOARD_DAILY_LIMIT=20
```

Generate `DASHBOARD_SIGNING_SECRET` from at least 32 random bytes. Do not put
real values in `netlify.toml` or commit a local `.env`; use `.env.example` only
as a key reference.

The `/api/dashboard-launch` Function uses an atomic Redis Lua script to count a
new KST-day session once. It keeps only a secret-salted session hash and expires
counter/session keys within 48 hours. A successful request returns a 30-minute
HMAC token. The `dashboard-gate` Edge Function exchanges that token for the
HttpOnly `jdw_dashboard_access` cookie and protects the app entry documents.
Shinylive runtime assets remain static so its service worker can control the app
scope; the quota is an application-launch control rather than a DRM boundary.

After setting or rotating environment variables, trigger a new deploy. Netlify
Functions cannot read secrets declared only in the build section of
`netlify.toml`.

## Analytics data migration

The separate `montewood/gh-action` working tree contains the schema v2 collector
change. Before production promotion:

1. Review and push its `assets/ga_script.R` and `.github/workflows/daily_ga.yaml`.
2. Run the workflow manually with `start_date` set to 90 days ago and
   `end_date` set to yesterday.
3. Verify each backfilled file has `schemaVersion: 2`, `date`, `activeUsers`,
   and `totalUsers`.
4. Confirm the homepage no longer reports `구형 집계` days.

## Production promotion

After preview acceptance, set `main` as the GitHub default branch and Netlify
production branch. Keep `legacy/wowchemy-5.1` and its tag indefinitely.

## Rollback

If production validation fails:

1. Change the Netlify production branch to `legacy/wowchemy-5.1`.
2. Trigger a clear-cache deploy.
3. Verify the homepage and a known legacy URL such as `/post/regex/`.
4. Keep the failed HugoBlox commit on `main`; fix it through
   `migration/hugoblox` and repeat preview validation.

Do not force-push or delete the legacy branch during rollback.
