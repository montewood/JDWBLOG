# JDW legacy design baseline

This document records the intentional design decisions from
`legacy/wowchemy-5.1`. It is the source of truth for the Tailwind migration;
the Bootstrap/Academic implementation itself is not.

## Active legacy composition

The production homepage used three active widgets in this order:

1. About (`content/home/about.md`)
2. Recent posts (`content/home/posts.md`, citation-style list, 3 items)
3. Projects (`content/home/projects.md`, card-style portfolio)

Post pages used a docs-style article layout with a desktop table of contents.
Study Notes used the same wide documentation treatment.

## Design tokens

| Concept | Legacy value | Modern decision |
| --- | --- | --- |
| Primary | `#1565c0` | Canonical JDW blue |
| Interactive/active | `#2962ff` | Reserved for active navigation |
| Secondary | `#42a5f5` | Supporting blue |
| Navigation background | `#ffffff` | Light header surface |
| Navigation text | `#34495e` | Slate navigation foreground |
| Navigation title | `#2b2b2b` | Strong heading foreground |
| Odd homepage section | `#ffffff` | Base surface |
| Even homepage section | `#f7f7f7` | Alternate surface |
| Dark background | `#23252f` | Base dark surface |
| Dark alternate | `#272935` | Alternate dark surface |
| Dark link | `#bbdefb` | Accessible dark-mode link |
| TOC hover | `#f8c471` | Warm hover accent |
| TOC active | `#e67e22` | Warm active accent |
| Body/heading/navigation | Noto Sans KR | Primary Korean family |
| Supporting Korean family | Nanum Gothic | Fallback |
| Body leading | `1.6` | Retained |
| Desktop base size | approximately `23px` | Reduced fluidly for modern readability |
| Mobile base size | approximately `17.7px` | Retained as lower bound |
| Article body width | `760px` | Reading measure |
| Documentation shell | `1080px` | Article plus TOC |
| Wide content utility | `1000px` | General wide sections |
| Avatar | circle, `270px` | Responsive 200–270px |
| Homepage section spacing | `110px` desktop, `60px` mobile | Retained conceptually |

## Intentional legacy overrides

The 10,796-line legacy `assets/scss/custom.scss` was mostly a pasted
Bootstrap 4/Academic bundle. Only these concepts are intentionally retained:

- larger post author avatar and stronger author biography
- 1080px documentation container
- warm TOC hover and active states
- 1000px reusable content width
- reduced bottom spacing on the About section
- heavy headings

The duplicated Bootstrap utilities, old Roboto/Montserrat rules, broken
`--main-color`, and conflicting framework defaults are explicitly excluded.

## Responsive acceptance targets

The design must be checked at 390px, 768px, and 1440px in light and dark mode.

- 390px: one-column profile, lists, cards, and no horizontal overflow
- 768px: readable two-column transitions without Bootstrap breakpoint quirks
- 1440px: profile details, compact post list, three-column projects, sticky TOC

## Compatibility boundary

Legacy appearance is reproduced through HugoBlox theme/font packs, scoped
project CSS, project-owned blocks, and project-owned collection views.
Bootstrap is not loaded and HugoBlox module files are never edited.
