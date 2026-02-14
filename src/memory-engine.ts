import { withApiRuntimeConfig } from "./llm.js";

type StoreLike = {
  close: () => void;
  getStatus: () => Record<string, unknown>;
  searchFTS: (query: string, limit?: number) => Array<Record<string, unknown>>;
  findDocument: (filename: string, options?: { includeBody?: boolean }) => Record<string, unknown>;
  getDocumentBody: (
    doc: Record<string, unknown> | { filepath: string },
    fromLine?: number,
    maxLines?: number
  ) => string | null;
};

type StoreModuleLike = {
  createStore: (dbPath?: string) => StoreLike;
  hybridQuery: (
    store: StoreLike,
    query: string,
    options?: { collection?: string; limit?: number; minScore?: number }
  ) => Promise<Array<Record<string, unknown>>>;
  extractSnippet: (body: string, query: string, maxLen?: number, chunkPos?: number) => { line: number; snippet: string };
  addLineNumbers: (text: string, startLine?: number) => string;
};

export type MemoryEngineConfig = {
  dbPath?: string;
  modelMode?: "local" | "api" | "hybrid";
  api?: {
    embedding?: {
      baseUrl?: string;
      model?: string;
      provider?: string;
      timeoutMs?: number;
      apiKeyEnv?: string;
    };
    rerank?: {
      baseUrl?: string;
      model?: string;
      provider?: string;
      timeoutMs?: number;
      apiKeyEnv?: string;
    };
    generate?: {
      baseUrl?: string;
      model?: string;
      provider?: string;
      timeoutMs?: number;
      apiKeyEnv?: string;
    };
  };
  sync?: {
    enabled?: boolean;
    embedOnSync?: boolean;
  };
  models?: {
    embedding?: {
      model?: string;
    };
  };
};

export type MemorySearchParams = {
  query: string;
  limit?: number;
  minScore?: number;
  collection?: string;
  lineNumbers?: boolean;
};

export type MemoryGetParams = {
  path: string;
  fromLine?: number;
  maxLines?: number;
  lineNumbers?: boolean;
};

let storeModulePromise: Promise<StoreModuleLike> | null = null;

function loadStoreModule(): Promise<StoreModuleLike> {
  if (!storeModulePromise) {
    storeModulePromise = import("./store.js") as Promise<StoreModuleLike>;
  }
  return storeModulePromise;
}

function withModelRuntime<T>(config: MemoryEngineConfig, fn: () => Promise<T>): Promise<T> {
  return withApiRuntimeConfig(
    {
      mode: config.modelMode,
      embedding: {
        baseUrl: config.api?.embedding?.baseUrl,
        model: config.api?.embedding?.model,
        provider: config.api?.embedding?.provider,
        timeoutMs: config.api?.embedding?.timeoutMs,
        apiKey: config.api?.embedding?.apiKeyEnv ? process.env[config.api?.embedding?.apiKeyEnv] : undefined,
      },
      rerank: {
        baseUrl: config.api?.rerank?.baseUrl,
        model: config.api?.rerank?.model,
        provider: config.api?.rerank?.provider,
        timeoutMs: config.api?.rerank?.timeoutMs,
        apiKey: config.api?.rerank?.apiKeyEnv ? process.env[config.api?.rerank?.apiKeyEnv] : undefined,
      },
      generate: {
        baseUrl: config.api?.generate?.baseUrl,
        model: config.api?.generate?.model,
        provider: config.api?.generate?.provider,
        timeoutMs: config.api?.generate?.timeoutMs,
        apiKey: config.api?.generate?.apiKeyEnv ? process.env[config.api?.generate?.apiKeyEnv] : undefined,
      },
    },
    fn,
  );
}

async function withStore<T>(config: MemoryEngineConfig, fn: (store: StoreLike, mod: StoreModuleLike) => Promise<T>): Promise<T> {
  return withModelRuntime(config, async () => {
    const mod = await loadStoreModule();
    const store = mod.createStore(config.dbPath);
    try {
      return await fn(store, mod);
    } finally {
      store.close();
    }
  });
}

export async function searchMemory(config: MemoryEngineConfig, params: MemorySearchParams): Promise<{ items: unknown[] }> {
  return withStore(config, async (store, mod) => {
    const limit = typeof params.limit === "number" && Number.isFinite(params.limit)
      ? Math.max(1, Math.trunc(params.limit))
      : 10;
    const minScore = typeof params.minScore === "number" && Number.isFinite(params.minScore)
      ? params.minScore
      : 0;

    try {
      const hybrid = await mod.hybridQuery(store, params.query, {
        collection: params.collection,
        limit,
        minScore
      });

      const items = hybrid.map((r) => {
        const bestChunk = typeof r.bestChunk === "string" ? r.bestChunk : "";
        const extracted = mod.extractSnippet(bestChunk, params.query, 320);
        return {
          id: `#${String(r.docid || "")}`,
          path: String(r.displayPath || ""),
          title: String(r.title || ""),
          score: Number(r.score || 0),
          context: (r.context ?? null) as string | null,
          snippet: params.lineNumbers ? mod.addLineNumbers(extracted.snippet, extracted.line) : extracted.snippet
        };
      });

      return { items };
    } catch {
      const fallback = store.searchFTS(params.query, limit)
        .filter((row) => !params.collection || row.collectionName === params.collection)
        .map((row) => {
          const body = typeof row.body === "string" ? row.body : "";
          const extracted = mod.extractSnippet(body, params.query, 320, Number(row.chunkPos || 0));
          return {
            id: `#${String(row.docid || "")}`,
            path: String(row.displayPath || ""),
            title: String(row.title || ""),
            score: Number(row.score || 0),
            context: (row.context ?? null) as string | null,
            snippet: params.lineNumbers ? mod.addLineNumbers(extracted.snippet, extracted.line) : extracted.snippet
          };
        })
        .filter((row) => row.score >= minScore);

      return { items: fallback };
    }
  });
}

export async function getMemory(config: MemoryEngineConfig, params: MemoryGetParams): Promise<{ text: string; details: Record<string, unknown> }> {
  return withStore(config, async (store, mod) => {
    const found = store.findDocument(params.path, { includeBody: false });
    if ((found as { error?: string }).error) {
      const similar = Array.isArray((found as { similarFiles?: unknown[] }).similarFiles)
        ? (found as { similarFiles: unknown[] }).similarFiles
        : [];
      const maybe = similar.length > 0
        ? `\nDid you mean:\n${similar.map((x) => `- ${String(x)}`).join("\n")}`
        : "";
      throw new Error(`Document not found: ${params.path}${maybe}`);
    }

    const textRaw = store.getDocumentBody(found, params.fromLine, params.maxLines) || "";
    const text = params.lineNumbers ? mod.addLineNumbers(textRaw, params.fromLine || 1) : textRaw;

    return {
      text,
      details: {
        id: `#${String((found as { docid?: string }).docid || "")}`,
        path: String((found as { displayPath?: string }).displayPath || ""),
        title: String((found as { title?: string }).title || ""),
        context: ((found as { context?: string | null }).context ?? null)
      }
    };
  });
}

export async function flushMemory(config: MemoryEngineConfig): Promise<{ status: Record<string, unknown> }> {
  const syncEnabled = config.sync?.enabled !== false;
  const shouldEmbed = config.sync?.embedOnSync === true;

  if (syncEnabled) {
    const indexer = await import("./indexer.js");
    const flushed = await indexer.flushIndex(
      { dbPath: config.dbPath },
      {
        runUpdate: true,
        runEmbed: shouldEmbed,
        embedModel: config.models?.embedding?.model,
      }
    );
    return flushed;
  }

  return withStore(config, async (store) => ({ status: store.getStatus() }));
}
