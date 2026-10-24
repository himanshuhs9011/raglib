# raglib website

Static site for raglib: Home, Docs and an in-browser Playground (React + TypeScript + Vite).
The Playground runs the real `raglib-py` package from PyPI inside the browser via Pyodide, offline.

```bash
npm install
npm run dev       # local preview
npm run build     # static output in dist/
```

## Deploy to Cloudflare Pages

- **Wrangler:** `npx wrangler login` once, then `npm run build && npx wrangler pages deploy dist --project-name raglib`
- **Git integration:** Pages -> Connect to Git -> root directory `site`, build command `npm run build`, output directory `dist`

`public/_headers` sets security headers. When releasing a new raglib version, update
`RAGLIB_VERSION` in `src/pyodide.ts` and `VERSION` in `src/data.ts`.
