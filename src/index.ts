const EMBED_MODEL = "@cf/baai/bge-m3";
const RERANK_MODEL = "@cf/baai/bge-reranker-base";

type JsonRecord = Record<string, unknown>;

export interface Env {
  AI: Ai;
  TEXT_DOCS: VectorizeIndex;
  DEFAULT_NAMESPACE_ID?: string;
  DEFAULT_VECTOR_LIMIT?: string;
  DEFAULT_RESULT_LIMIT?: string;
  DEFAULT_MIN_RERANK_SCORE?: string;
  NBSS_BASE_URL?: string;
  ENABLE_NBSS_HYDRATE?: string;
}

interface EmbeddingResponse {
  data: number[][];
}

interface RerankItem {
  id?: number;
  score?: number;
  text?: string;
}

interface RerankResponse {
  response?: RerankItem[];
}

interface SearchRequest {
  query?: string;
  namespaceId?: string;
  vectorLimit?: number;
  limit?: number;
  minRerankScore?: number;
  hydrateBody?: boolean;
}

interface DetailRequest {
  namespaceId?: string;
  hydrateBody?: boolean;
}

interface TextDocMeta {
  createdAt?: string;
  docKind?: string;
  sourceType?: string;
  title?: string;
  sourceId?: string;
}

interface TextDocMetadata {
  text?: string;
  namespaceId?: string;
  nbssfid?: string;
  meta?: TextDocMeta;
}

interface SearchItem {
  id: string;
  score: number | null;
  rerankScore: number | null;
  namespaceId: string;
  text: string;
  bodyText?: string;
  bodySource?: "inline" | "nbss";
  nbssfid: string;
  meta: TextDocMeta;
}

interface DetailItem {
  id: string;
  namespaceId: string;
  text: string;
  bodyText: string;
  bodySource: "inline" | "nbss";
  nbssfid: string;
  meta: TextDocMeta;
}

type VectorMatch = {
  id: string;
  score?: number;
  metadata?: unknown;
};

function json(status: number, payload: JsonRecord): Response {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" }
  });
}

function asPositiveInt(value: unknown, fallback: number, max: number): number {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) {
    return fallback;
  }
  return Math.min(Math.floor(n), max);
}

function asNonNegativeNumber(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return n;
}

function isTruthy(value: string | undefined): boolean {
  return value === "1" || value === "true" || value === "yes";
}

async function readJson<T>(request: Request): Promise<T | null> {
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

async function embedQuery(query: string, env: Env): Promise<number[]> {
  const result = (await env.AI.run(EMBED_MODEL, { text: [query] })) as EmbeddingResponse;
  const vector = result?.data?.[0];
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new Error("embedding_failed");
  }
  return vector;
}

function normalizeMetadata(input: unknown): TextDocMetadata {
  if (!input || typeof input !== "object") {
    return {};
  }
  const record = input as JsonRecord;
  const metaValue = record.meta;
  const meta = metaValue && typeof metaValue === "object" ? (metaValue as TextDocMeta) : {};
  return {
    text: typeof record.text === "string" ? record.text : "",
    namespaceId: typeof record.namespaceId === "string" ? record.namespaceId : "",
    nbssfid: typeof record.nbssfid === "string" ? record.nbssfid : "",
    meta
  };
}

async function queryVectorize(
  queryVector: number[],
  namespaceId: string,
  vectorLimit: number,
  env: Env
): Promise<VectorMatch[]> {
  const result = await env.TEXT_DOCS.query(queryVector, {
    topK: vectorLimit,
    returnMetadata: "all" as const
  });
  return ((result.matches ?? []) as VectorMatch[]).filter((item) => {
    const metadata = normalizeMetadata(item.metadata);
    return metadata.namespaceId === namespaceId;
  });
}

function buildRerankContext(match: VectorMatch): string {
  const metadata = normalizeMetadata(match.metadata);
  const title = metadata.meta?.title?.trim() ?? "";
  const text = metadata.text?.trim() ?? "";
  if (title && text) {
    return `${title}\n\n${text}`;
  }
  return title || text || match.id;
}

async function rerankMatches(query: string, matches: VectorMatch[], limit: number, env: Env): Promise<SearchItem[]> {
  if (matches.length === 0) {
    return [];
  }
  const contexts = matches.map(buildRerankContext);
  const rerank = (await env.AI.run(RERANK_MODEL, {
    query,
    contexts: contexts.map((text) => ({ text }))
  })) as RerankResponse;

  const ranked = (rerank.response ?? [])
    .filter((item) => typeof item.id === "number" && item.id >= 0 && item.id < matches.length)
    .sort((left, right) => (Number(right.score ?? 0) - Number(left.score ?? 0)));

  const output: SearchItem[] = [];
  for (const item of ranked.slice(0, limit)) {
    const match = matches[item.id as number];
    const metadata = normalizeMetadata(match.metadata);
    output.push({
      id: match.id,
      score: typeof match.score === "number" ? match.score : null,
      rerankScore: typeof item.score === "number" ? item.score : null,
      namespaceId: metadata.namespaceId ?? "",
      text: metadata.text ?? "",
      nbssfid: metadata.nbssfid ?? "",
      meta: metadata.meta ?? {}
    });
  }
  return output;
}

async function maybeHydrateBody(items: SearchItem[], hydrateBody: boolean, env: Env): Promise<void> {
  if (!hydrateBody || !isTruthy(env.ENABLE_NBSS_HYDRATE)) {
    for (const item of items) {
      item.bodyText = item.text;
      item.bodySource = "inline";
    }
    return;
  }
  const nbssBase = (env.NBSS_BASE_URL ?? "").trim().replace(/\/+$/, "");
  for (const item of items) {
    if (!item.nbssfid || !nbssBase.startsWith("http")) {
      item.bodyText = item.text;
      item.bodySource = "inline";
      continue;
    }
    const hex = item.nbssfid.replace(/^NBSS:/i, "");
    try {
      const response = await fetch(`${nbssBase}/${hex}`);
      if (!response.ok) {
        throw new Error(`nbss_http_${response.status}`);
      }
      item.bodyText = await response.text();
      item.bodySource = "nbss";
    } catch (error) {
      console.warn("nbss hydrate failed, falling back to inline text", {
        id: item.id,
        nbssfid: item.nbssfid,
        error: error instanceof Error ? error.message : String(error)
      });
      item.bodyText = item.text;
      item.bodySource = "inline";
    }
  }
}

async function buildDetailItem(
  id: string,
  namespaceId: string,
  hydrateBody: boolean,
  env: Env
): Promise<DetailItem | null> {
  const matches = await env.TEXT_DOCS.getByIds([id]);
  const match = matches[0];
  if (!match) return null;
  const metadata = normalizeMetadata(match.metadata);
  if ((metadata.namespaceId ?? "") !== namespaceId) {
    return null;
  }

  const item: SearchItem = {
    id: match.id,
    score: null,
    rerankScore: null,
    namespaceId: metadata.namespaceId ?? "",
    text: metadata.text ?? "",
    nbssfid: metadata.nbssfid ?? "",
    meta: metadata.meta ?? {}
  };
  await maybeHydrateBody([item], hydrateBody, env);

  return {
    id: item.id,
    namespaceId: item.namespaceId,
    text: item.text,
    bodyText: item.bodyText ?? item.text,
    bodySource: item.bodySource ?? "inline",
    nbssfid: item.nbssfid,
    meta: item.meta
  };
}

async function handleSearch(request: Request, env: Env): Promise<Response> {
  const payload = await readJson<SearchRequest>(request);
  if (!payload) {
    return json(400, { ok: false, error: "invalid_json" });
  }

  const query = String(payload.query ?? "").trim();
  if (!query) {
    return json(400, { ok: false, error: "missing_query" });
  }

  const namespaceId = String(payload.namespaceId ?? env.DEFAULT_NAMESPACE_ID ?? "").trim();
  if (!namespaceId) {
    return json(400, { ok: false, error: "missing_namespace_id" });
  }

  const vectorLimit = asPositiveInt(payload.vectorLimit ?? env.DEFAULT_VECTOR_LIMIT, 40, 100);
  const limit = asPositiveInt(payload.limit ?? env.DEFAULT_RESULT_LIMIT, 10, 20);
  const minRerankScore = asNonNegativeNumber(payload.minRerankScore ?? env.DEFAULT_MIN_RERANK_SCORE, 0.15);
  const hydrateBody = Boolean(payload.hydrateBody);

  const queryVector = await embedQuery(query, env);
  const matches = await queryVectorize(queryVector, namespaceId, vectorLimit, env);
  const reranked = await rerankMatches(query, matches, limit, env);
  const filtered = reranked.filter((item) => (item.rerankScore ?? 0) >= minRerankScore);
  await maybeHydrateBody(filtered, hydrateBody, env);

  return json(200, {
    ok: true,
    service: "cfworker-vecdocsrv",
    index: "text-docs",
    namespaceId,
    query,
    vectorLimit,
    vectorHitCount: matches.length,
    resultCount: filtered.length,
    minRerankScore,
    models: {
      embedding: EMBED_MODEL,
      reranker: RERANK_MODEL
    },
    items: filtered
  });
}

async function handleDetail(request: Request, env: Env, id: string): Promise<Response> {
  const payload = request.method === "POST" ? await readJson<DetailRequest>(request) : null;
  const url = new URL(request.url);
  const namespaceId = String(
    payload?.namespaceId ??
    url.searchParams.get("namespaceId") ??
    env.DEFAULT_NAMESPACE_ID ??
    ""
  ).trim();
  if (!namespaceId) {
    return json(400, { ok: false, error: "missing_namespace_id" });
  }

  const hydrateBody = request.method === "POST"
    ? Boolean(payload?.hydrateBody)
    : isTruthy(url.searchParams.get("hydrateBody") ?? undefined);

  const item = await buildDetailItem(id, namespaceId, hydrateBody, env);
  if (!item) {
    return json(404, { ok: false, error: "not_found" });
  }

  return json(200, {
    ok: true,
    service: "cfworker-vecdocsrv",
    index: "text-docs",
    item
  });
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") {
      return json(200, {
        ok: true,
        service: "cfworker-vecdocsrv",
        index: "text-docs",
        defaultNamespaceId: env.DEFAULT_NAMESPACE_ID ?? ""
      });
    }

    if (request.method === "POST" && url.pathname === "/api/v1/text-docs/search") {
      try {
        return await handleSearch(request, env);
      } catch (error) {
        console.error("text-doc search failed", error);
        return json(500, {
          ok: false,
          error: "internal_error",
          detail: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const detailMatch = url.pathname.match(/^\/api\/v1\/text-docs\/([^/]+)$/);
    if (detailMatch && (request.method === "GET" || request.method === "POST")) {
      try {
        return await handleDetail(request, env, decodeURIComponent(detailMatch[1]));
      } catch (error) {
        console.error("text-doc detail failed", error);
        return json(500, {
          ok: false,
          error: "internal_error",
          detail: error instanceof Error ? error.message : String(error)
        });
      }
    }

    return json(404, { ok: false, error: "not_found" });
  }
};
