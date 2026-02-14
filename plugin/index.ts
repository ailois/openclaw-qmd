import type { OpenClawPluginApi, ToolContext } from "openclaw/plugin-sdk";
import { flushMemory, getMemory, searchMemory, type MemoryEngineConfig } from "../src/memory-engine.js";
import {
  normalizeMemoryPluginConfig,
  pluginConfigToMemoryEngineConfig,
  type OpenClawMemoryPluginConfig,
} from "../src/openclaw-memory-config.js";

type SearchParams = {
  query?: string;
  limit?: number;
  minScore?: number;
  collection?: string;
  lineNumbers?: boolean;
};

type GetParams = {
  path?: string;
  fromLine?: number;
  maxLines?: number;
  lineNumbers?: boolean;
};

function normalizeConfig(value: unknown): OpenClawMemoryPluginConfig {
  return normalizeMemoryPluginConfig(value);
}

function getPluginConfig(api: OpenClawPluginApi, ctx?: ToolContext): OpenClawMemoryPluginConfig {
  if (api.pluginConfig && typeof api.pluginConfig === "object") {
    return normalizeConfig(api.pluginConfig);
  }
  const fromFull = ctx?.config?.plugins?.entries?.[api.id]?.config;
  return normalizeConfig(fromFull);
}

function resolveHomePath(input: string): string {
  if (!input.startsWith("~")) return input;
  const home = process.env.HOME || "";
  if (input === "~") return home;
  if (input.startsWith("~/")) return `${home}/${input.slice(2)}`;
  return input;
}

function toEngineConfig(config: OpenClawMemoryPluginConfig): MemoryEngineConfig {
  const engine = pluginConfigToMemoryEngineConfig(config);
  if (engine.dbPath) {
    return {
      ...engine,
      dbPath: resolveHomePath(engine.dbPath),
    };
  }
  return engine;
}

function summarizeSearch(query: string, count: number): string {
  if (count === 0) return `No memory results for "${query}".`;
  return `Found ${count} memory result${count === 1 ? "" : "s"} for "${query}".`;
}

const qmdMemoryPlugin = {
  id: "qmd-memory",
  name: "QMD Memory",
  kind: "memory",

  register(api: OpenClawPluginApi) {
    const searchSchema = {
      type: "object",
      properties: {
        query: { type: "string", description: "Memory query" },
        limit: { type: "integer", description: "Maximum result count" },
        minScore: { type: "number", description: "Minimum relevance score" },
        collection: { type: "string", description: "Collection filter" },
        lineNumbers: { type: "boolean", description: "Add line numbers to snippets" }
      },
      required: ["query"]
    };

    const getSchema = {
      type: "object",
      properties: {
        path: { type: "string", description: "Document path or #docid" },
        fromLine: { type: "integer", description: "Start line (1-indexed)" },
        maxLines: { type: "integer", description: "Maximum number of lines" },
        lineNumbers: { type: "boolean", description: "Add line numbers" }
      },
      required: ["path"]
    };

    const flushSchema = {
      type: "object",
      properties: {},
      required: []
    };

    api.registerTool(
      (ctx) => {
        const config = getPluginConfig(api, ctx);
        const engineConfig = toEngineConfig(config);

        const searchTool = (name: string, description: string) => ({
          name,
          description,
          parameters: searchSchema,
          async execute(_toolCallId: string, params: unknown) {
            const { query, limit, minScore, collection, lineNumbers } = (params || {}) as SearchParams;
            if (!query || !query.trim()) {
              return {
                content: [{ type: "text", text: "Missing required parameter: query" }],
                details: { error: "missing_query" }
              };
            }

            try {
              const result = await searchMemory(engineConfig, {
                query,
                limit,
                minScore,
                collection,
                lineNumbers
              });

              return {
                content: [{ type: "text", text: summarizeSearch(query, result.items.length) }],
                details: {
                  query,
                  total: result.items.length,
                  items: result.items,
                  models: config.models?.api || null
                }
              };
            } catch (error) {
              return {
                content: [{ type: "text", text: error instanceof Error ? error.message : "memory_search failed" }],
                details: {
                  error: "search_failed",
                  models: config.models?.api || null
                }
              };
            }
          }
        });

        const getTool = (name: string, description: string) => ({
          name,
          description,
          parameters: getSchema,
          async execute(_toolCallId: string, params: unknown) {
            const { path, fromLine, maxLines, lineNumbers } = (params || {}) as GetParams;
            if (!path || !path.trim()) {
              return {
                content: [{ type: "text", text: "Missing required parameter: path" }],
                details: { error: "missing_path" }
              };
            }

            try {
              const result = await getMemory(engineConfig, {
                path: path.trim(),
                fromLine,
                maxLines,
                lineNumbers
              });

              return {
                content: [{ type: "text", text: result.text }],
                details: result.details
              };
            } catch (error) {
              return {
                content: [{ type: "text", text: error instanceof Error ? error.message : `memory_get failed for ${path}` }],
                details: { error: "get_failed", path }
              };
            }
          }
        });

        const flushTool = (name: string, description: string) => ({
          name,
          description,
          parameters: flushSchema,
          async execute(_toolCallId: string) {
            try {
              const result = await flushMemory(engineConfig);
              return {
                content: [{ type: "text", text: "QMD memory flush completed." }],
                details: {
                  status: result.status,
                  sync: config.sync || null
                }
              };
            } catch (error) {
              return {
                content: [{ type: "text", text: error instanceof Error ? error.message : "memory_flush failed" }],
                details: { error: "flush_failed", sync: config.sync || null }
              };
            }
          }
        });

        return [
          searchTool("memory_search", "Search long-term memory using QMD retrieval."),
          getTool("memory_get", "Get memory content by path or docid."),
          flushTool("memory_flush", "Refresh memory status and report index health.")
        ];
      },
      { names: ["memory_search", "memory_get", "memory_flush"] }
    );
  }
};

export default qmdMemoryPlugin;
