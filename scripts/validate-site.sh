#!/usr/bin/env bash
set -euo pipefail

required_commands=(hugo)
for command_name in "${required_commands[@]}"; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing required command: ${command_name}" >&2
    exit 1
  }
done

missing_pages=0
while IFS= read -r -d '' post_dir; do
  if ! compgen -G "${post_dir}/index.md" >/dev/null &&
    ! compgen -G "${post_dir}/index.markdown" >/dev/null &&
    ! compgen -G "${post_dir}/index.*.markdown" >/dev/null; then
    echo "Missing publishable page bundle: ${post_dir}" >&2
    missing_pages=1
  fi
done < <(find content/post -mindepth 1 -maxdepth 1 -type d -print0)
((missing_pages == 0))

test -f static/apps/profile-widget/index.html
test -f static/apps/profile-widget/app.json
test -f apps/profile-widget/app.R
test -f hugo-blox/blox/community/embedded-app/block.html

hugo --gc --minify --buildFuture

required_outputs=(
  public/index.html
  public/post/index.html
  public/post/index.xml
  public/post/regex/index.html
  public/post/regexwithstringr/index.html
  public/post/rsthemes/index.html
  public/post/parallel-with-future/index.html
  public/post/rstudio-1-rstudio-server/index.html
  public/apps/profile-widget/index.html
  public/privacy/index.html
  public/sitemap.xml
  public/robots.txt
)

for output in "${required_outputs[@]}"; do
  test -f "${output}" || {
    echo "Missing build output: ${output}" >&2
    exit 1
  }
done

if grep -R -n -E 'href="https://example.com|content="https://example.com' public; then
  echo "Example-domain URL leaked into production output." >&2
  exit 1
fi

scripts/check-internal-links.py public

echo "HugoBlox build and required URL checks passed."
