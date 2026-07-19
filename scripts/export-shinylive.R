#!/usr/bin/env Rscript

source_dir <- normalizePath(
  file.path("apps", "profile-widget"),
  mustWork = TRUE
)
app_file <- file.path(source_dir, "app.R")
output_dir <- file.path("static", "apps", "profile-widget")

if (!file.exists(app_file)) {
  stop(
    "Missing apps/profile-widget/app.R. Recover the original app source ",
    "before replacing the preserved static bundle.",
    call. = FALSE
  )
}

if (!requireNamespace("shinylive", quietly = TRUE)) {
  stop(
    "Install the R package 'shinylive' before exporting the app.",
    call. = FALSE
  )
}

if (dir.exists(output_dir)) {
  message("Removing the previous generated Shinylive bundle: ", output_dir)
  unlink(output_dir, recursive = TRUE, force = TRUE)
}

dir.create(dirname(output_dir), recursive = TRUE, showWarnings = FALSE)

shinylive::export(
  appdir = source_dir,
  destdir = output_dir,
  wasm_packages = TRUE,
  package_cache = FALSE
)
