#!/usr/bin/env bash
set -euo pipefail

mapfile -d '' sources < <(find content -type f -name '*.qmd' -print0 | sort -z)
if ((${#sources[@]} == 0)); then
  echo "No Quarto sources found."
  exit 0
fi

backup_dir="$(mktemp -d)"
outputs=()

restore_outputs() {
  for output in "${outputs[@]}"; do
    cp "${backup_dir}/${output}" "${output}"
  done
  rm -rf "${backup_dir}"
}
trap restore_outputs EXIT

for source in "${sources[@]}"; do
  output="${source%.qmd}.md"
  if [[ ! -f "${output}" ]]; then
    echo "Missing committed Quarto output: ${output}" >&2
    exit 1
  fi
  outputs+=("${output}")
  mkdir -p "${backup_dir}/$(dirname "${output}")"
  cp "${output}" "${backup_dir}/${output}"
done

scripts/render-quarto.sh

for output in "${outputs[@]}"; do
  scripts/compare-quarto-output.py "${backup_dir}/${output}" "${output}" || {
    echo "Quarto output drift detected: ${output}" >&2
    exit 1
  }
done

echo "Quarto source and generated Markdown are in sync."
