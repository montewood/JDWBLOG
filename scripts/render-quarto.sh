#!/usr/bin/env bash
set -euo pipefail

if ! command -v quarto >/dev/null 2>&1; then
  echo "quarto is required to render .qmd sources" >&2
  exit 1
fi

mapfile -d '' sources < <(find content -type f -name '*.qmd' -print0 | sort -z)

if ((${#sources[@]} == 0)); then
  echo "No Quarto sources found."
  exit 0
fi

for source in "${sources[@]}"; do
  echo "Rendering ${source}"
  quarto render "${source}"
done
