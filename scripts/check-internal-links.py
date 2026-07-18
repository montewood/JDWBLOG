#!/usr/bin/env python3
"""Validate local links and assets in a generated Hugo site."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attrs:
            if value and name in {"href", "src"}:
                self.references.append(value)


def resolve_reference(site_root: Path, page: Path, reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme in {"mailto", "tel", "data", "javascript"}:
        return None
    if parsed.netloc and parsed.netloc not in {"www.jdwblog.com", "jdwblog.com"}:
        return None

    url_path = unquote(parsed.path)
    if not url_path:
        return None

    if url_path.startswith("/"):
        candidate = site_root / url_path.lstrip("/")
    else:
        candidate = page.parent / url_path

    if url_path.endswith("/"):
        candidate /= "index.html"
    elif not candidate.suffix:
        candidate = candidate / "index.html"

    return candidate


def main() -> int:
    site_root = Path(sys.argv[1] if len(sys.argv) > 1 else "public").resolve()
    failures: list[str] = []

    for page in site_root.rglob("*.html"):
        parser = LinkParser()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for reference in parser.references:
            target = resolve_reference(site_root, page, reference)
            if target is not None and not target.exists():
                failures.append(
                    f"{page.relative_to(site_root)} -> {reference}"
                )

    if failures:
        print("Broken internal references:", file=sys.stderr)
        print("\n".join(failures[:100]), file=sys.stderr)
        return 1

    print("All generated internal links and assets resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
