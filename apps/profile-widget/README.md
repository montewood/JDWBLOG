# Profile Widget source module

This directory is the source boundary for the Shinylive application embedded at
`/apps/profile-widget/`.

The historical repository did not track the source directory referenced by its
theme-local export script. The original `app.R` was recovered from the exported
Shinylive 0.9.1 `app.json` manifest and is now versioned here. The preserved
bundle remains under `static/apps/profile-widget/` until it is regenerated and
verified against a current Shinylive release.

To update the app:

1. Keep all app source and app-specific data in this directory.
2. Run `Rscript scripts/export-shinylive.R`.
3. Review the generated files under `static/apps/profile-widget/`.
4. Test the app directly and through the HugoBlox `embedded-app` block.

The HugoBlox block contains no Shiny code. It only owns the iframe contract:
URL, accessible title, height, loading behavior, and sandbox policy.
