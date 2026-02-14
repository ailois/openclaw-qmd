/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import type {
  Llama,
  LlamaModel,
  LlamaEmbeddingContext,
  Token as LlamaToken,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";

type LlamaRuntime = {
  getLlama: (options: { logLevel: number }) => Promise<Llama>;
  resolveModelFile: (modelUri: string, cacheDir: string) => Promise<string>;
  LlamaChatSession: new (args: { contextSequence: unknown }) => {
    prompt: (prompt: string, options: Record<string, unknown>) => Promise<string | void>;
  };
  LlamaLogLevel: { error: number };
};

let llamaRuntimePromise: Promise<LlamaRuntime> | null = null;

async function getLlamaRuntime(): Promise<LlamaRuntime> {
  if (!llamaRuntimePromise) {
    llamaRuntimePromise = import("node-llama-cpp") as Promise<LlamaRuntime>;
  }
  return await llamaRuntimePromise;
}

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 10 minutes) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;

type ApiCapability = "embedding" | "rerank" | "generate";

type ApiEndpointConfig = {
  baseUrl: string;
  model?: string;
  apiKey?: string;
  provider?: string;
  timeoutMs: number;
};

type ApiRuntimeConfig = {
  mode: "local" | "api" | "hybrid";
  embedding?: ApiEndpointConfig;
  rerank?: ApiEndpointConfig;
  generate?: ApiEndpointConfig;
};

export type ApiRuntimeConfigOverride = {
  mode?: "local" | "api" | "hybrid";
  embedding?: Partial<ApiEndpointConfig>;
  rerank?: Partial<ApiEndpointConfig>;
  generate?: Partial<ApiEndpointConfig>;
};

let runtimeConfigOverride: ApiRuntimeConfig | null = null;

function mergeRuntimeConfig(base: ApiRuntimeConfig, override: ApiRuntimeConfigOverride): ApiRuntimeConfig {
  const mergeEndpoint = (origin?: ApiEndpointConfig, patch?: Partial<ApiEndpointConfig>): ApiEndpointConfig | undefined => {
    if (!origin && !patch) return undefined;
    const merged = {
      baseUrl: patch?.baseUrl ?? origin?.baseUrl ?? "",
      model: patch?.model ?? origin?.model,
      apiKey: patch?.apiKey ?? origin?.apiKey,
      provider: patch?.provider ?? origin?.provider,
      timeoutMs: patch?.timeoutMs ?? origin?.timeoutMs ?? 30000,
    };
    if (!merged.baseUrl) return undefined;
    return merged;
  };

  return {
    mode: override.mode ?? base.mode,
    embedding: mergeEndpoint(base.embedding, override.embedding),
    rerank: mergeEndpoint(base.rerank, override.rerank),
    generate: mergeEndpoint(base.generate, override.generate),
  };
}

export async function withApiRuntimeConfig<T>(override: ApiRuntimeConfigOverride, fn: () => Promise<T>): Promise<T> {
  const previous = runtimeConfigOverride;
  runtimeConfigOverride = mergeRuntimeConfig(loadApiRuntimeConfig(), override);
  try {
    return await fn();
  } finally {
    runtimeConfigOverride = previous;
  }
}

function getEnvMaybe(name: string): string | undefined {
  const value = process.env[name];
  if (!value) return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function resolveApiKey(explicit?: string, keyEnvName?: string): string | undefined {
  if (explicit && explicit.trim()) return explicit.trim();
  if (keyEnvName && keyEnvName.trim()) {
    return getEnvMaybe(keyEnvName.trim());
  }
  return undefined;
}

function normalizeBaseUrl(raw?: string): string | undefined {
  const value = raw?.trim();
  if (!value) return undefined;
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function loadApiRuntimeConfig(): ApiRuntimeConfig {
  const modeRaw = (getEnvMaybe("QMD_MODEL_MODE") || "local").toLowerCase();
  const mode: "local" | "api" | "hybrid" =
    modeRaw === "api" ? "api" : (modeRaw === "hybrid" ? "hybrid" : "local");

  const defaultTimeout = Number.parseInt(getEnvMaybe("QMD_API_TIMEOUT_MS") || "30000", 10);
  const defaultProvider = getEnvMaybe("QMD_API_PROVIDER") || "openai-compatible";

  const makeEndpoint = (cap: "EMBED" | "RERANK" | "GENERATE", fallbackModel?: string): ApiEndpointConfig | undefined => {
    const baseUrl = normalizeBaseUrl(getEnvMaybe(`QMD_API_${cap}_BASE_URL`));
    if (!baseUrl) return undefined;
    const model = getEnvMaybe(`QMD_API_${cap}_MODEL`) || fallbackModel;
    const apiKey = resolveApiKey(
      getEnvMaybe(`QMD_API_${cap}_KEY`),
      getEnvMaybe(`QMD_API_${cap}_KEY_ENV`)
    ) || getEnvMaybe("QMD_API_KEY");
    const provider = getEnvMaybe(`QMD_API_${cap}_PROVIDER`) || defaultProvider;
    const timeoutMs = Number.parseInt(getEnvMaybe(`QMD_API_${cap}_TIMEOUT_MS`) || String(defaultTimeout), 10);
    return {
      baseUrl,
      model,
      apiKey,
      provider,
      timeoutMs: Number.isFinite(timeoutMs) ? timeoutMs : 30000,
    };
  };

  const loaded = {
    mode,
    embedding: makeEndpoint("EMBED"),
    rerank: makeEndpoint("RERANK"),
    generate: makeEndpoint("GENERATE"),
  };
  if (runtimeConfigOverride) {
    return runtimeConfigOverride;
  }
  return loaded;
}

function shouldUseApiCapability(config: ApiRuntimeConfig, capability: ApiCapability): boolean {
  if (config.mode === "local") return false;
  if (config.mode === "api") return true;
  if (capability === "embedding") return !!config.embedding;
  if (capability === "rerank") return !!config.rerank;
  return !!config.generate;
}

async function fetchJsonWithTimeout<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Request timeout after ${timeoutMs}ms`)), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`HTTP ${response.status} ${response.statusText}: ${body}`);
    }
    return await response.json() as T;
  } finally {
    clearTimeout(timer);
  }
}

export type PullResult = {
  model: string;
  path: string;
  sizeBytes: number;
  refreshed: boolean;
};

type HfRef = {
  repo: string;
  file: string;
};

function parseHfUri(model: string): HfRef | null {
  if (!model.startsWith("hf:")) return null;
  const without = model.slice(3);
  const parts = without.split("/");
  if (parts.length < 3) return null;
  const repo = parts.slice(0, 2).join("/");
  const file = parts.slice(2).join("/");
  return { repo, file };
}

async function getRemoteEtag(ref: HfRef): Promise<string | null> {
  const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return null;
    const etag = resp.headers.get("etag");
    return etag || null;
  } catch {
    return null;
  }
}

export async function pullModels(
  models: string[],
  options: { refresh?: boolean; cacheDir?: string } = {}
): Promise<PullResult[]> {
  const runtime = loadApiRuntimeConfig();
  if (runtime.mode === "api") {
    return models.map((model) => ({
      model,
      path: "api://managed",
      sizeBytes: 0,
      refreshed: false,
    }));
  }

  const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const results: PullResult[] = [];
  for (const model of models) {
    let refreshed = false;
    const hfRef = parseHfUri(model);
    const filename = model.split("/").pop();
    const entries = readdirSync(cacheDir, { withFileTypes: true });
    const cached = filename
      ? entries
          .filter((entry) => entry.isFile() && entry.name.includes(filename))
          .map((entry) => join(cacheDir, entry.name))
      : [];

    if (hfRef && filename) {
      const etagPath = join(cacheDir, `${filename}.etag`);
      const remoteEtag = await getRemoteEtag(hfRef);
      const localEtag = existsSync(etagPath)
        ? readFileSync(etagPath, "utf-8").trim()
        : null;
      const shouldRefresh =
        options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;

      if (shouldRefresh) {
        for (const candidate of cached) {
          if (existsSync(candidate)) unlinkSync(candidate);
        }
        if (existsSync(etagPath)) unlinkSync(etagPath);
        refreshed = cached.length > 0;
      }
    } else if (options.refresh && filename) {
      for (const candidate of cached) {
        if (existsSync(candidate)) unlinkSync(candidate);
        refreshed = true;
      }
    }

    const runtime = await getLlamaRuntime();
    const path = await runtime.resolveModelFile(model, cacheDir);
    const sizeBytes = existsSync(path) ? statSync(path).size : 0;
    if (hfRef && filename) {
      const remoteEtag = await getRemoteEtag(hfRef);
      if (remoteEtag) {
        const etagPath = join(cacheDir, `${filename}.etag`);
        writeFileSync(etagPath, remoteEtag + "\n", "utf-8");
      }
    }
    results.push({ model, path, sizeBytes, refreshed });
  }
  return results;
}

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContext: LlamaEmbeddingContext | null = null;
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  // Ensure we don't load the same model/context concurrently (which can allocate duplicate VRAM).
  private embedModelLoadPromise: Promise<LlamaModel> | null = null;
  private embedContextCreatePromise: Promise<LlamaEmbeddingContext> | null = null;
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;

  private getApiRuntimeConfig(): ApiRuntimeConfig {
    return loadApiRuntimeConfig();
  }

  private buildApiHeaders(endpoint: ApiEndpointConfig): Record<string, string> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (endpoint.apiKey) {
      headers.authorization = `Bearer ${endpoint.apiKey}`;
    }
    return headers;
  }

  private async apiEmbedSingle(text: string, endpoint: ApiEndpointConfig): Promise<EmbeddingResult | null> {
    type EmbeddingResp = {
      data?: Array<{ embedding?: number[] }>;
      model?: string;
    };
    const payload: Record<string, unknown> = {
      input: text,
    };
    if (endpoint.model) payload.model = endpoint.model;

    const resp = await fetchJsonWithTimeout<EmbeddingResp>(
      `${endpoint.baseUrl}/embeddings`,
      {
        method: "POST",
        headers: this.buildApiHeaders(endpoint),
        body: JSON.stringify(payload),
      },
      endpoint.timeoutMs,
    );
    const vec = resp.data?.[0]?.embedding;
    if (!Array.isArray(vec)) return null;
    return {
      embedding: vec,
      model: resp.model || endpoint.model || endpoint.baseUrl,
    };
  }

  private async apiEmbedBatch(texts: string[], endpoint: ApiEndpointConfig): Promise<(EmbeddingResult | null)[]> {
    type EmbeddingResp = {
      data?: Array<{ embedding?: number[] }>;
      model?: string;
    };
    const payload: Record<string, unknown> = {
      input: texts,
    };
    if (endpoint.model) payload.model = endpoint.model;

    const resp = await fetchJsonWithTimeout<EmbeddingResp>(
      `${endpoint.baseUrl}/embeddings`,
      {
        method: "POST",
        headers: this.buildApiHeaders(endpoint),
        body: JSON.stringify(payload),
      },
      endpoint.timeoutMs,
    );
    const arr = resp.data || [];
    return texts.map((_, idx) => {
      const vec = arr[idx]?.embedding;
      if (!Array.isArray(vec)) return null;
      return {
        embedding: vec,
        model: resp.model || endpoint.model || endpoint.baseUrl,
      };
    });
  }

  private async apiGenerate(prompt: string, options: GenerateOptions, endpoint: ApiEndpointConfig): Promise<GenerateResult | null> {
    type ChatResp = {
      model?: string;
      choices?: Array<{ message?: { content?: string } }>;
    };
    const payload: Record<string, unknown> = {
      messages: [{ role: "user", content: prompt }],
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 150,
    };
    if (endpoint.model) payload.model = endpoint.model;

    const resp = await fetchJsonWithTimeout<ChatResp>(
      `${endpoint.baseUrl}/chat/completions`,
      {
        method: "POST",
        headers: this.buildApiHeaders(endpoint),
        body: JSON.stringify(payload),
      },
      endpoint.timeoutMs,
    );
    const text = resp.choices?.[0]?.message?.content;
    if (typeof text !== "string") return null;
    return {
      text,
      model: resp.model || endpoint.model || endpoint.baseUrl,
      done: true,
    };
  }

  private async apiExpandQuery(query: string, options: { context?: string; includeLexical?: boolean }, endpoint: ApiEndpointConfig): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const contextText = options.context ? `\nContext: ${options.context}` : "";
    const prompt = [
      "Return query expansions using one item per line in this exact format:",
      "lex: ...",
      "vec: ...",
      "hyde: ...",
      "Only return these lines.",
      `Query: ${query}${contextText}`,
    ].join("\n");

    const generated = await this.apiGenerate(prompt, { maxTokens: 400, temperature: 0.7 }, endpoint);
    const resultText = generated?.text || "";
    const lines = resultText.trim().split("\n");
    const parsed: Queryable[] = lines
      .map((line) => {
        const idx = line.indexOf(":");
        if (idx < 0) return null;
        const type = line.slice(0, idx).trim();
        if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
        const text = line.slice(idx + 1).trim();
        if (!text) return null;
        return { type, text } as Queryable;
      })
      .filter((v): v is Queryable => v !== null);

    if (parsed.length > 0) {
      return includeLexical ? parsed : parsed.filter((q) => q.type !== "lex");
    }

    const fallback: Queryable[] = [{ type: "vec", text: query }];
    if (includeLexical) fallback.unshift({ type: "lex", text: query });
    return fallback;
  }

  private async apiRerank(query: string, documents: RerankDocument[], endpoint: ApiEndpointConfig): Promise<RerankResult> {
    type RerankResp = {
      model?: string;
      results?: Array<{ index?: number; score?: number }>;
      data?: Array<{ index?: number; relevance_score?: number; score?: number }>;
    };

    const payload: Record<string, unknown> = {
      query,
      documents: documents.map((d) => d.text),
      top_n: documents.length,
    };
    if (endpoint.model) payload.model = endpoint.model;

    const resp = await fetchJsonWithTimeout<RerankResp>(
      `${endpoint.baseUrl}/rerank`,
      {
        method: "POST",
        headers: this.buildApiHeaders(endpoint),
        body: JSON.stringify(payload),
      },
      endpoint.timeoutMs,
    );

    const ranked = (resp.results || resp.data || [])
      .map((item) => {
        const index = typeof item.index === "number" ? item.index : -1;
        const rawScore = typeof item.score === "number"
          ? item.score
          : (typeof item.relevance_score === "number" ? item.relevance_score : 0);
        return { index, score: rawScore };
      })
      .filter((item) => item.index >= 0 && item.index < documents.length)
      .sort((a, b) => b.score - a.score);

    const results: RerankDocumentResult[] = ranked.map((item) => ({
      file: documents[item.index]!.file,
      score: item.score,
      index: item.index,
    }));

    return {
      results,
      model: resp.model || endpoint.model || endpoint.baseUrl,
    };
  }


  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
  }

  /**
   * Reset the inactivity timer. Called after each model operation.
   * When timer fires, models are unloaded to free memory (if no active sessions).
   */
  private touchActivity(): void {
    // Clear existing timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Only set timer if we have disposable contexts and timeout is enabled
    if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        // Check if session manager allows unloading
        // canUnloadLLM is defined later in this file - it checks the session manager
        // We use dynamic import pattern to avoid circular dependency issues
        if (typeof canUnloadLLM === 'function' && !canUnloadLLM()) {
          // Active sessions/operations - reschedule timer
          this.touchActivity();
          return;
        }
        this.unloadIdleResources().catch(err => {
          console.error("Error unloading idle resources:", err);
        });
      }, this.inactivityTimeoutMs);
      // Don't keep process alive just for this timer
      this.inactivityTimer.unref();
    }
  }

  /**
   * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
   */
  private hasLoadedContexts(): boolean {
    return !!(this.embedContext || this.rerankContext);
  }

  /**
   * Unload idle resources but keep the instance alive for future use.
   *
   * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
   * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
   */
  async unloadIdleResources(): Promise<void> {
    // Don't unload if already disposed
    if (this.disposed) {
      return;
    }

    // Clear timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Dispose contexts first
    if (this.embedContext) {
      await this.embedContext.dispose();
      this.embedContext = null;
    }
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Optionally dispose models too (opt-in)
    if (this.disposeModelsOnInactivity) {
      if (this.embedModel) {
        await this.embedModel.dispose();
        this.embedModel = null;
      }
      if (this.generateModel) {
        await this.generateModel.dispose();
        this.generateModel = null;
      }
      if (this.rerankModel) {
        await this.rerankModel.dispose();
        this.rerankModel = null;
      }
      // Reset load promises so models can be reloaded later
      this.embedModelLoadPromise = null;
      this.generateModelLoadPromise = null;
      this.rerankModelLoadPromise = null;
    }

    // Note: We keep llama instance alive - it's lightweight
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    if (!this.llama) {
      const runtime = await getLlamaRuntime();
      this.llama = await runtime.getLlama({ logLevel: runtime.LlamaLogLevel.error });
    }
    return this.llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    const runtime = await getLlamaRuntime();
    return await runtime.resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Load embedding model (lazy)
   */
  private async ensureEmbedModel(): Promise<LlamaModel> {
    if (this.embedModel) {
      return this.embedModel;
    }
    if (this.embedModelLoadPromise) {
      return await this.embedModelLoadPromise;
    }

    this.embedModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.embedModelUri);
      const model = await llama.loadModel({ modelPath });
      this.embedModel = model;
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
      return model;
    })();

    try {
      return await this.embedModelLoadPromise;
    } finally {
      // Keep the resolved model cached; clear only the in-flight promise.
      this.embedModelLoadPromise = null;
    }
  }

  /**
   * Load embedding context (lazy). Context can be disposed and recreated without reloading the model.
   * Uses promise guard to prevent concurrent context creation race condition.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    if (!this.embedContext) {
      // If context creation is already in progress, wait for it
      if (this.embedContextCreatePromise) {
        return await this.embedContextCreatePromise;
      }

      // Start context creation and store promise so concurrent calls wait
      this.embedContextCreatePromise = (async () => {
        const model = await this.ensureEmbedModel();
        const context = await model.createEmbeddingContext();
        this.embedContext = context;
        return context;
      })();

      try {
        const context = await this.embedContextCreatePromise;
        this.touchActivity();
        return context;
      } finally {
        this.embedContextCreatePromise = null;
      }
    }
    this.touchActivity();
    return this.embedContext;
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    if (!this.generateModel) {
      if (this.generateModelLoadPromise) {
        return await this.generateModelLoadPromise;
      }

      this.generateModelLoadPromise = (async () => {
        const llama = await this.ensureLlama();
        const modelPath = await this.resolveModel(this.generateModelUri);
        const model = await llama.loadModel({ modelPath });
        this.generateModel = model;
        return model;
      })();

      try {
        await this.generateModelLoadPromise;
      } finally {
        this.generateModelLoadPromise = null;
      }
    }
    this.touchActivity();
    if (!this.generateModel) {
      throw new Error("Generate model not loaded");
    }
    return this.generateModel;
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    if (this.rerankModel) {
      return this.rerankModel;
    }
    if (this.rerankModelLoadPromise) {
      return await this.rerankModelLoadPromise;
    }

    this.rerankModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      const model = await llama.loadModel({ modelPath });
      this.rerankModel = model;
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
      this.rerankModelLoadPromise = null;
    }
  }

  /**
   * Load rerank context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const model = await this.ensureRerankModel();
      this.rerankContext = await model.createRankingContext();
    }
    this.touchActivity();
    return this.rerankContext;
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    await this.ensureEmbedContext();  // Ensure model is loaded
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.tokenize(text);
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    await this.ensureEmbedContext();
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const runtime = this.getApiRuntimeConfig();
    if (shouldUseApiCapability(runtime, "embedding") && runtime.embedding) {
      try {
        return await this.apiEmbedSingle(text, runtime.embedding);
      } catch (error) {
        console.error("Embedding API error:", error);
        if (runtime.mode === "api") return null;
      }
    }

    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    try {
      const context = await this.ensureEmbedContext();
      const embedding = await context.getEmbeddingFor(text);

      return {
        embedding: Array.from(embedding.vector),
        model: this.embedModelUri,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    const runtime = this.getApiRuntimeConfig();
    if (shouldUseApiCapability(runtime, "embedding") && runtime.embedding) {
      try {
        return await this.apiEmbedBatch(texts, runtime.embedding);
      } catch (error) {
        console.error("Batch embedding API error:", error);
        if (runtime.mode === "api") return texts.map(() => null);
      }
    }

    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    if (texts.length === 0) return [];

    try {
      const context = await this.ensureEmbedContext();

      // node-llama-cpp handles batching internally when we make parallel requests
      const embeddings = await Promise.all(
        texts.map(async (text) => {
          try {
            const embedding = await context.getEmbeddingFor(text);
            this.touchActivity();  // Keep-alive during slow batches
            return {
              embedding: Array.from(embedding.vector),
              model: this.embedModelUri,
            };
          } catch (err) {
            console.error("Embedding error for text:", err);
            return null;
          }
        })
      );

      return embeddings;
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const runtime = this.getApiRuntimeConfig();
    if (shouldUseApiCapability(runtime, "generate") && runtime.generate) {
      try {
        return await this.apiGenerate(prompt, options, runtime.generate);
      } catch (error) {
        console.error("Generate API error:", error);
        if (runtime.mode === "api") return null;
      }
    }

    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    // Ensure model is loaded
    await this.ensureGenerateModel();

    // Create fresh context -> sequence -> session for each call
    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const llamaRuntime = await getLlamaRuntime();
    const session = new llamaRuntime.LlamaChatSession({ contextSequence: sequence });

    const maxTokens = options.maxTokens ?? 150;
    // Qwen3 recommends temp=0.7, topP=0.8, topK=20 for non-thinking mode
    // DO NOT use greedy decoding (temp=0) - causes repetition loops
    const temperature = options.temperature ?? 0.7;

    let result = "";
    try {
      await session.prompt(prompt, {
        maxTokens,
        temperature,
        topK: 20,
        topP: 0.8,
        onTextChunk: (text) => {
          result += text;
        },
      });

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
      await context.dispose();
    }
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    const runtime = this.getApiRuntimeConfig();
    if (runtime.mode !== "local") {
      if (runtime.embedding?.model === modelUri || runtime.rerank?.model === modelUri || runtime.generate?.model === modelUri) {
        return { name: modelUri, exists: true, path: "api://managed" };
      }
    }

    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const runtime = this.getApiRuntimeConfig();
    if (shouldUseApiCapability(runtime, "generate") && runtime.generate) {
      try {
        return await this.apiExpandQuery(query, options, runtime.generate);
      } catch (error) {
        console.error("Structured query expansion API failed:", error);
        if (runtime.mode === "api") {
          const fallback: Queryable[] = [{ type: 'vec', text: query }];
          if ((options.includeLexical ?? true)) fallback.unshift({ type: 'lex', text: query });
          return fallback;
        }
      }
    }

    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    const llama = await this.ensureLlama();
    await this.ensureGenerateModel();

    const includeLexical = options.includeLexical ?? true;
    const context = options.context;

    const grammar = await llama.createGrammar({
      grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
    });

    const prompt = `/no_think Expand this search query: ${query}`;

    // Create fresh context for each call
    const genContext = await this.generateModel!.createContext();
    const sequence = genContext.getSequence();
    const llamaRuntime = await getLlamaRuntime();
    const session = new llamaRuntime.LlamaChatSession({ contextSequence: sequence });

    try {
      // Qwen3 recommended settings for non-thinking mode:
      // temp=0.7, topP=0.8, topK=20, presence_penalty for repetition
      // DO NOT use greedy decoding (temp=0) - causes infinite loops
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 600,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repeatPenalty: {
          lastTokens: 64,
          presencePenalty: 0.5,
        },
      });

      const lines = result.trim().split("\n");
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        if (queryTerms.length === 0) return true;
        return queryTerms.some(term => lower.includes(term));
      };

      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      }).filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
      if (filtered.length > 0) return filtered;

      const fallback: Queryable[] = [
        { type: 'hyde', text: `Information about ${query}` },
        { type: 'lex', text: query },
        { type: 'vec', text: query },
      ];
      return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    } finally {
      await genContext.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    const runtime = this.getApiRuntimeConfig();
    if (shouldUseApiCapability(runtime, "rerank") && runtime.rerank) {
      try {
        return await this.apiRerank(query, documents, runtime.rerank);
      } catch (error) {
        console.error("Rerank API error:", error);
        if (runtime.mode === "api") {
          return {
            model: runtime.rerank.model || runtime.rerank.baseUrl,
            results: documents.map((doc, index) => ({ file: doc.file, index, score: 0 })),
          };
        }
      }
    }

    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    const context = await this.ensureRerankContext();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Use the proper ranking API - returns [{document: string, score: number}] sorted by score
    const ranked = await context.rankAndSort(query, texts);

    // Map back to our result format using the text-to-doc map
    const results: RerankDocumentResult[] = ranked.map((item) => {
      const docInfo = textToDoc.get(item.document)!;
      return {
        file: docInfo.file,
        score: item.score,
        index: docInfo.index,
      };
    });

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  async dispose(): Promise<void> {
    // Prevent double-dispose
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    // Clear inactivity timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Disposing llama cascades to models and contexts automatically
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
    // Note: llama.dispose() can hang indefinitely, so we use a timeout
    if (this.llama) {
      const disposePromise = this.llama.dispose();
      const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 1000));
      await Promise.race([disposePromise, timeoutPromise]);
    }

    // Clear references
    this.embedContext = null;
    this.rerankContext = null;
    this.embedModel = null;
    this.generateModel = null;
    this.rerankModel = null;
    this.llama = null;

    // Clear any in-flight load/create promises
    this.embedModelLoadPromise = null;
    this.embedContextCreatePromise = null;
    this.generateModelLoadPromise = null;
    this.rerankModelLoadPromise = null;
  }
}

// =============================================================================
// Session Management Layer
// =============================================================================

/**
 * Manages LLM session lifecycle with reference counting.
 * Coordinates with LlamaCpp idle timeout to prevent disposal during active sessions.
 */
class LLMSessionManager {
  private llm: LlamaCpp;
  private _activeSessionCount = 0;
  private _inFlightOperations = 0;

  constructor(llm: LlamaCpp) {
    this.llm = llm;
  }

  get activeSessionCount(): number {
    return this._activeSessionCount;
  }

  get inFlightOperations(): number {
    return this._inFlightOperations;
  }

  /**
   * Returns true only when both session count and in-flight operations are 0.
   * Used by LlamaCpp to determine if idle unload is safe.
   */
  canUnload(): boolean {
    return this._activeSessionCount === 0 && this._inFlightOperations === 0;
  }

  acquire(): void {
    this._activeSessionCount++;
  }

  release(): void {
    this._activeSessionCount = Math.max(0, this._activeSessionCount - 1);
  }

  operationStart(): void {
    this._inFlightOperations++;
  }

  operationEnd(): void {
    this._inFlightOperations = Math.max(0, this._inFlightOperations - 1);
  }

  getLlamaCpp(): LlamaCpp {
    return this.llm;
  }
}

/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}

/**
 * Scoped LLM session with automatic lifecycle management.
 * Wraps LlamaCpp methods with operation tracking and abort handling.
 */
class LLMSession implements ILLMSession {
  private manager: LLMSessionManager;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private name: string;

  constructor(manager: LLMSessionManager, options: LLMSessionOptions = {}) {
    this.manager = manager;
    this.name = options.name || "unnamed";
    this.abortController = new AbortController();

    // Link external abort signal if provided
    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
        }, { once: true });
      }
    }

    // Set up max duration timer
    const maxDuration = options.maxDuration ?? 10 * 60 * 1000; // Default 10 minutes
    if (maxDuration > 0) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new Error(`Session "${this.name}" exceeded max duration of ${maxDuration}ms`));
      }, maxDuration);
      this.maxDurationTimer.unref(); // Don't keep process alive
    }

    // Acquire session lease
    this.manager.acquire();
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  /**
   * Release the session and decrement ref count.
   * Called automatically by withLLMSession when the callback completes.
   */
  release(): void {
    if (this.released) return;
    this.released = true;

    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }

    this.abortController.abort(new Error("Session released"));
    this.manager.release();
  }

  /**
   * Wrap an operation with tracking and abort checking.
   */
  private async withOperation<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.isValid) {
      throw new SessionReleasedError();
    }

    this.manager.operationStart();
    try {
      // Check abort before starting
      if (this.abortController.signal.aborted) {
        throw new SessionReleasedError(
          this.abortController.signal.reason?.message || "Session aborted"
        );
      }
      return await fn();
    } finally {
      this.manager.operationEnd();
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.withOperation(() => this.manager.getLlamaCpp().embed(text, options));
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().embedBatch(texts));
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().expandQuery(query, options));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.withOperation(() => this.manager.getLlamaCpp().rerank(query, documents, options));
  }
}

// Session manager for the default LlamaCpp instance
let defaultSessionManager: LLMSessionManager | null = null;

/**
 * Get the session manager for the default LlamaCpp instance.
 */
function getSessionManager(): LLMSessionManager {
  const llm = getDefaultLlamaCpp();
  if (!defaultSessionManager || defaultSessionManager.getLlamaCpp() !== llm) {
    defaultSessionManager = new LLMSessionManager(llm);
  }
  return defaultSessionManager;
}

/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const embeddings = await session.embedBatch(texts);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
 * ```
 */
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const manager = getSessionManager();
  const session = new LLMSession(manager, options);

  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM(): boolean {
  if (!defaultSessionManager) return true;
  return defaultSessionManager.canUnload();
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp();
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}
