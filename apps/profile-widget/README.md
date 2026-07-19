# Profile Widget source module

This directory is the source boundary for the optional Shinylive application at
`/apps/profile-widget/`.

The homepage does not embed this app during initial rendering. The
`jdw-about` block renders a lightweight ECharts dashboard first and creates the
Shinylive iframe only after the visitor selects **대시보드 실행** and receives
a valid daily-quota launch token.
Closing the dialog removes the iframe and releases the webR execution context.

Both dashboards read the same daily analytics source:

```text
https://raw.githubusercontent.com/montewood/gh-action/
refs/heads/main/output/GA-YYYYMMDD.json
```

Schema v2 files contain the GA4 daily unique `activeUsers` value directly.
Historical files are JSON encoded twice and contain minute-level observations;
both implementations retain a fallback parser and label those dates as legacy
until the 90-day backfill is complete.

To update the app:

1. Keep all app source and app-specific data in this directory.
2. Run `Rscript scripts/export-shinylive.R`.
3. The exporter removes the old generated bundle before writing
   `static/apps/profile-widget/`.
4. Test the app directly and through the `jdw-about` dialog.
5. Run the Playwright lazy-load and accessibility checks.

The R app intentionally depends only on `shiny`. Its small browser helper
fetches and parses the daily JSON concurrently, then sends columnar values to R;
filtering and plotting use base R. The regenerated bundle is approximately
68.4 MB, down from the preserved 101.5 MB bundle.

The HugoBlox block contains no Shiny code. It owns the iframe contract: URL,
accessible title, height, loading behavior, sandbox policy, and lifecycle. The
Netlify launch Function and Edge gate own the global quota and app-entry access.
