# raglib website

Static site for raglib: Home, Docs and an in-browser Playground (React + TypeScript + Vite).
The Playground runs the real `raglib-py` package from PyPI inside the browser via Pyodide, offline.

```bash
npm install
npm run dev       # local preview
npm run build     # static output in dist/
```

## Deploy to Cloudflare

Live at https://raglib.raglib-site.workers.dev (Cloudflare Workers static assets, free plan).

```bash
npx wrangler login            # once
npm run build && npx wrangler deploy
```

`wrangler.jsonc` points Cloudflare at `dist/`. `public/_headers` sets security headers. When releasing a new raglib version, update
`RAGLIB_VERSION` in `src/pyodide.ts` and `VERSION` in `src/data.ts`.
