import { homedir } from "node:os";
import { isAbsolute, resolve } from "node:path";
import { readFileSync } from "node:fs";
import type { MemoryEngineConfig } from "./memory-engine.js";
import type { ApiRuntimeConfigOverride } from "./llm.js";

export type ApiModelConfig = {
  provider?: string;
  timeoutMs?: number;
  baseUrl?: string;
  apiKeyEnv?: string;
  model?: string;
};

export type OpenClawMemoryPluginConfig = {
  index?: {
    dbPath?: string;
    reuseQmdCache?: boolean;
  };
  models?: {
    mode?: "local" | "api" | "hybrid";
    api?: {
      embedding?: ApiModelConfig;
      rerank?: ApiModelConfig;
      generate?: ApiModelConfig;
    };
  };
  sync?: {
    enabled?: boolean;
    intervalSec?: number;
    embedOnSync?: boolean;
  };
};

type OpenClawJsonShape = {
  plugins?: {
    entries?: Record<string, { enabled?: boolean; config?: unknown }>;
  };
};

export function resolveHomePath(input: string): string {
  if (!input.startsWith("~")) return input;
  const home = process.env.HOME || homedir();
  if (input === "~") return home;
  if (input.startsWith("~/")) return `${home}/${input.slice(2)}`;
  return input;
}

export function normalizeMemoryPluginConfig(value: unknown): OpenClawMemoryPluginConfig {
  if (!value || typeof value !== "object") {
    return {};
  }
  return value as OpenClawMemoryPluginConfig;
}

export function pluginConfigToMemoryEngineConfig(config: OpenClawMemoryPluginConfig): MemoryEngineConfig {
  const dbPathRaw = config.index?.dbPath?.trim();
  return {
    dbPath: dbPathRaw ? resolveHomePath(dbPathRaw) : undefined,
    modelMode: config.models?.mode,
    api: {
      embedding: config.models?.api?.embedding,
      rerank: config.models?.api?.rerank,
      generate: config.models?.api?.generate,
    },
    sync: {
      enabled: config.sync?.enabled,
      embedOnSync: config.sync?.embedOnSync,
    },
    models: {
      embedding: {
        model: config.models?.api?.embedding?.model,
      },
    },
  };
}

export function pluginConfigToApiRuntimeOverride(config: OpenClawMemoryPluginConfig): ApiRuntimeConfigOverride {
  return {
    mode: config.models?.mode,
    embedding: {
      baseUrl: config.models?.api?.embedding?.baseUrl,
      model: config.models?.api?.embedding?.model,
      provider: config.models?.api?.embedding?.provider,
      timeoutMs: config.models?.api?.embedding?.timeoutMs,
      apiKey: config.models?.api?.embedding?.apiKeyEnv
        ? process.env[config.models.api.embedding.apiKeyEnv]
        : undefined,
    },
    rerank: {
      baseUrl: config.models?.api?.rerank?.baseUrl,
      model: config.models?.api?.rerank?.model,
      provider: config.models?.api?.rerank?.provider,
      timeoutMs: config.models?.api?.rerank?.timeoutMs,
      apiKey: config.models?.api?.rerank?.apiKeyEnv
        ? process.env[config.models.api.rerank.apiKeyEnv]
        : undefined,
    },
    generate: {
      baseUrl: config.models?.api?.generate?.baseUrl,
      model: config.models?.api?.generate?.model,
      provider: config.models?.api?.generate?.provider,
      timeoutMs: config.models?.api?.generate?.timeoutMs,
      apiKey: config.models?.api?.generate?.apiKeyEnv
        ? process.env[config.models.api.generate.apiKeyEnv]
        : undefined,
    },
  };
}

export function loadMemoryPluginConfigFromOpenClawFile(filePath: string, pluginId: string = "qmd-memory"): OpenClawMemoryPluginConfig {
  const absolute = isAbsolute(filePath) ? filePath : resolve(process.cwd(), filePath);
  const raw = readFileSync(absolute, "utf-8");
  const parsed = JSON.parse(raw) as OpenClawJsonShape;
  const entry = parsed.plugins?.entries?.[pluginId];
  if (!entry || !entry.config) {
    throw new Error(`Plugin config not found for '${pluginId}' in ${absolute}`);
  }
  return normalizeMemoryPluginConfig(entry.config);
}
