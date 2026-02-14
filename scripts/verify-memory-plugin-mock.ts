import { randomUUID } from "node:crypto";
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

type Tool = {
  name: string;
  execute: (toolCallId: string, params: unknown) => Promise<{ content: Array<{ type: string; text: string }>; details?: Record<string, unknown> }>;
};

async function startMockApiServer() {
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/v1/embeddings") {
        const body = await req.json() as { input: string | string[] };
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        return Response.json({
          model: "mock-embed",
          data: inputs.map((v, i) => ({ embedding: [v.length / 100, i + 1, 0.5] })),
        });
      }
      if (url.pathname === "/v1/chat/completions") {
        return Response.json({
          model: "mock-generate",
          choices: [{ message: { content: "lex: memory query\nvec: memory query semantic\nhyde: memory narrative" } }],
        });
      }
      if (url.pathname === "/v1/rerank") {
        const body = await req.json() as { documents?: string[] };
        const docs = body.documents || [];
        return Response.json({
          model: "mock-rerank",
          results: docs.map((_, index) => ({ index, score: docs.length - index })),
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return server;
}

async function setupIndexedDoc(dbPath: string) {
  const storeMod = await import("../src/store.js");
  const store = storeMod.createStore(dbPath);
  try {
    const now = new Date().toISOString();
    const content = "Project memory document with API routing verification.";
    const hash = await storeMod.hashContent(content);
    store.insertContent(hash, content, now);
    store.insertDocument("memory", "notes/memory.md", "Memory Note", hash, now, now);
  } finally {
    store.close();
  }
}

async function makeTools(config: Record<string, unknown>): Promise<Map<string, Tool>> {
  const pluginModule = await import("../plugin/index.ts");
  const plugin = pluginModule.default as {
    id: string;
    register: (api: {
      id: string;
      pluginConfig?: Record<string, unknown>;
      registerTool: (factory: (ctx: unknown) => Tool[], options?: { names?: string[] }) => void;
    }) => void;
  };

  const tools = new Map<string, Tool>();
  plugin.register({
    id: plugin.id,
    pluginConfig: config,
    registerTool(factory) {
      for (const tool of factory({})) {
        tools.set(tool.name, tool);
      }
    },
  });
  return tools;
}

async function runScenario(mode: "api" | "hybrid", serverBaseUrl: string, dbPath: string) {
  const tools = await makeTools({
    index: { dbPath },
    models: {
      mode,
      api: {
        embedding: { baseUrl: serverBaseUrl, model: "mock-embed", timeoutMs: 5000 },
        rerank: { baseUrl: serverBaseUrl, model: "mock-rerank", timeoutMs: 5000 },
        generate: { baseUrl: serverBaseUrl, model: "mock-generate", timeoutMs: 5000 },
      },
    },
    sync: { enabled: false, embedOnSync: false },
  });

  const memorySearch = tools.get("memory_search");
  const memoryGet = tools.get("memory_get");
  const memoryFlush = tools.get("memory_flush");
  if (!memorySearch || !memoryGet || !memoryFlush) {
    throw new Error(`Missing required tools for mode=${mode}`);
  }

  const search = await memorySearch.execute(randomUUID(), { query: "memory", limit: 3 });
  const get = await memoryGet.execute(randomUUID(), { path: "memory.md" });
  const flush = await memoryFlush.execute(randomUUID(), {});

  if (!search.details || Number(search.details.total || 0) < 1) {
    throw new Error(`memory_search returned no items in mode=${mode}`);
  }
  if (!get.content[0]?.text?.includes("Project memory document")) {
    throw new Error(`memory_get unexpected content in mode=${mode}`);
  }
  if (!flush.details || !flush.details.status) {
    throw new Error(`memory_flush missing status in mode=${mode}`);
  }
}

async function main() {
  const tmpRoot = join("/tmp", `qmd-memory-verify-${Date.now()}`);
  const dbPath = join(tmpRoot, "memory.sqlite");
  mkdirSync(tmpRoot, { recursive: true });

  const server = await startMockApiServer();
  const baseUrl = `http://127.0.0.1:${server.port}/v1`;

  try {
    try {
      await setupIndexedDoc(dbPath);
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      if (text.includes("sqlite-vec") || text.includes("node-llama-cpp")) {
        console.log("SKIP: missing optional local runtime deps for full memory verification");
        return;
      }
      throw error;
    }

    await runScenario("api", baseUrl, dbPath);
    await runScenario("hybrid", baseUrl, dbPath);
    console.log("OK: memory_search/memory_get/memory_flush passed in api + hybrid modes");
  } finally {
    server.stop(true);
    rmSync(tmpRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
