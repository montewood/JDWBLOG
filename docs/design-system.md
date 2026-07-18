# JDW design system

The JDW design system is a project-owned compatibility layer between the
legacy Wowchemy identity and HugoBlox schema 2. It deliberately does not load
Bootstrap or modify HugoBlox module files.

## Ownership layers

| Layer | Path | Responsibility |
| --- | --- | --- |
| Theme pack | `data/themes/jdw.yaml` | Light/dark brand and surface colors |
| Font pack | `data/fonts/jdw.yaml` | Korean-first font roles, weights, sizes |
| Global skin | `assets/css/custom.css` | Shell, typography, prose, TOC, shared views |
| Block data | `data/blocks/*.yaml` | Reusable homepage widget-like configuration |
| Blocks | `hugo-blox/blox/community/*` | Project-owned section markup and local CSS |
| Views | `layouts/_partials/views/jdw-*` | Post and project collection presentation |
| Author data | `data/authors/jdw.yaml` | Single source of profile content |

## Token contract

Use HugoBlox variables for framework-aware styling:

- `--hb-font-heading`, `--hb-font-body`, `--hb-font-code`, `--hb-font-nav`
- `--hb-font-size-base`, `--hb-font-leading-body`
- `--hb-color-background`, `--hb-color-foreground`

Use project variables for JDW-specific semantics:

- `--jdw-primary`, `--jdw-primary-active`
- `--jdw-section`, `--jdw-section-alt`
- `--jdw-text`, `--jdw-nav-text`
- `--jdw-toc-hover`, `--jdw-toc-active`
- `--jdw-article`, `--jdw-docs`, `--jdw-content`

Do not hardcode these values inside content files.

## Reusable section data

The homepage references files under `data/blocks/`:

```yaml
sections:
  - ref: about
  - ref: recent-posts
  - ref: projects
  - ref: study-notes
```

This is the schema-2 equivalent of managing separate Wowchemy widget files.
Inline section values can override a referenced block without changing its
shared defaults.

## Custom block contracts

### `jdw-about`

```yaml
block: jdw-about
content:
  username: jdw
  app_url: /apps/profile-widget/
  app_title: Accessible iframe title
  app_height: 500
```

Profile content is resolved from `data/authors/<username>.yaml`. The block does
not contain Shiny code. It only embeds the independently exported static app.

### `jdw-study-index`

```yaml
block: jdw-study-index
content:
  title: Study Notes
  text: Introductory copy
  section: courses
```

The block discovers child sections dynamically, so adding a new course series
does not require editing the block template.

### `embedded-app`

This remains available for standalone static applications. Prefer
`jdw-about` only when the app belongs to the profile composition.

## Collection view contracts

`jdw-post-list` is a compact horizontal list for posts. It expects normal Hugo
page bundles, an optional featured image, `summary`, `date`, and
`authors: [jdw]`.

`jdw-project-grid` is a responsive 3/2/1-column project grid. It uses the first
entry in `links` as the card destination and falls back to the project page.

Both views are independent names rather than overrides of upstream `card` or
`date-title-summary`, reducing upgrade risk.

## Content conventions

- Authoring source: `index.qmd`
- Hugo input: generated `index.md`
- Author identity: `authors: [jdw]`
- Featured media: page-bundle `featured.*`
- External project destination:

```yaml
links:
  - type: site
    url: https://example.com/
```

## Upgrade rules

1. Keep `go.mod`, `hugoblox.yaml`, Hugo, Node, Go, and pnpm pins aligned.
2. Upgrade HugoBlox on an isolated branch.
3. Never copy a whole upstream block or CSS bundle into the project.
4. Check whether project partial APIs still accept `wcPage`, `wcBlock`,
   `functions/get_author_profile`, and `functions/get_featured_image`.
5. Run build, link, Quarto, accessibility, and screenshot tests.
6. Review light/dark screenshots before promotion.

## Visual acceptance matrix

The required routes are `/`, `/post/`, `/post/regex/`, `/project/`,
`/courses/`, and `/apps/profile-widget/`.

Each route is checked at 390×844, 768×1024, and 1440×1200 in light and dark
mode. See `tests/visual/` for the executable specification.
