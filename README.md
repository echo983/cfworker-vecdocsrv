# cfworker-vecdocsrv

Cloudflare Worker for note vector recall against the `text-docs` Vectorize index.

Intended upstream repository:

- `https://github.com/echo983/cfworker-vecdocsrv`

Current scope:

- `POST /api/v1/text-docs/search`
- `GET /healthz`

Search flow:

1. Embed the incoming query with `@cf/baai/bge-m3`.
2. Query `text-docs` in Vectorize.
3. Filter by `namespaceId` (prefer server-side metadata filter, fallback to in-worker filtering).
4. Rerank candidates with `@cf/baai/bge-reranker-base`.
5. Return inline text, `nbssfid`, and compact metadata.

Optional body hydration:

- If `hydrateBody=true` in the request and `ENABLE_NBSS_HYDRATE=true`, the Worker will fetch full body text from `NBSS_BASE_URL/<fid-hex>` when `nbssfid` exists.

Local commands:

```bash
cd apps/cfworker-vecdocsrv
npm install
npm run cf-typegen
npm run typecheck
npm run dev
```

Example request:

```bash
curl -X POST http://127.0.0.1:8787/api/v1/text-docs/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "西班牙语动词变位",
    "namespaceId": "ns_user_f416f9c53576",
    "limit": 5,
    "vectorLimit": 20
  }'
```
