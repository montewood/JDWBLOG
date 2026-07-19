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
scripts/validate-site.sh
pnpm run pagefind
scripts/check-quarto-drift.sh
```

On the Netlify Deploy Preview, verify:

1. Homepage profile, recent posts, projects, dark mode, search, and mobile layout.
2. Existing `/post/` URLs, RSS, sitemap, canonical, and Open Graph tags.
3. Plotly/Leaflet content and the Shinylive app under `/apps/profile-widget/`.
4. Browser console has no service-worker, COOP, COEP, or missing-asset errors.
5. Google Tag Manager fires once with container `GTM-P8STCFM`.

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
