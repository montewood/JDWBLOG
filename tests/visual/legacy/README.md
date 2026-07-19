# Legacy visual reference

The immutable visual source is the `legacy/wowchemy-5.1` branch and the
production URL history. Legacy screenshots are review references, not test
snapshots, because the old Hugo 0.82 server is not part of modern CI.

Capture these routes at 390px, 768px, and 1440px in both color modes when a
legacy comparison is needed:

- `/`
- `/post/`
- `/post/regex/`
- `/project/`

The executable current snapshots are generated beside
`design-system.spec.ts` with:

```bash
pnpm check:visual:update
```
