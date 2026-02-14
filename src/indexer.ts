import { Glob } from "bun";
import { readFileSync, statSync } from "node:fs";
import {
  DEFAULT_EMBED_MODEL,
  clearCache,
  chunkDocumentByTokens,
  createStore,
  extractTitle,
  formatDocForEmbedding,
  handelize,
  hashContent,
  listCollections,
  type Store,
} from "./store.js";
import { getCollection as getCollectionFromYaml } from "./collections.js";
import { withLLMSession } from "./llm.js";

type Logger = {
  info: (message: string) => void;
  warn: (message: string) => void;
};

export type IndexerConfig = {
  dbPath?: string;
  logger?: Logger;
};

export type UpdateCollectionsResult = {
  collectionsUpdated: number;
  needsEmbedding: number;
};

export type EmbedResult = {
  documents: number;
  chunksEmbedded: number;
  errors: number;
};

export type FlushOptions = {
  runUpdate?: boolean;
  runEmbed?: boolean;
  embedModel?: string;
  forceEmbed?: boolean;
};

const defaultLogger: Logger = {
  info: (message) => console.log(message),
  warn: (message) => console.warn(message),
};

function getLogger(config: IndexerConfig): Logger {
  return config.logger || defaultLogger;
}

function withStore<T>(config: IndexerConfig, fn: (store: Store) => Promise<T>): Promise<T> {
  const store = createStore(config.dbPath);
  return fn(store).finally(() => {
    store.close();
  });
}

async function indexCollection(store: Store, collection: { name: string; pwd: string; glob_pattern: string }, suppressEmbedNotice: boolean, logger: Logger): Promise<void> {
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];
  const glob = new Glob(collection.glob_pattern);
  const files: string[] = [];

  for await (const file of glob.scan({ cwd: collection.pwd, onlyFiles: true, followSymlinks: true })) {
    const parts = file.split("/");
    const shouldSkip = parts.some((part) => part === "node_modules" || part.startsWith(".") || excludeDirs.includes(part));
    if (!shouldSkip) {
      files.push(file);
    }
  }

  if (files.length === 0) {
    logger.info("No files found matching pattern.");
    return;
  }

  let indexed = 0;
  let updated = 0;
  let unchanged = 0;
  let removed = 0;
  const seenPaths = new Set<string>();

  for (const relativeFile of files) {
    const filepath = store.resolveVirtualPath(`qmd://${collection.name}/${relativeFile}`) || `${collection.pwd}/${relativeFile}`;
    const path = handelize(relativeFile);
    seenPaths.add(path);

    const content = readFileSync(filepath, "utf-8");
    if (!content.trim()) {
      continue;
    }

    const hash = await hashContent(content);
    const title = extractTitle(content, relativeFile);
    const existing = store.findActiveDocument(collection.name, path);

    if (existing) {
      if (existing.hash === hash) {
        if (existing.title !== title) {
          store.updateDocumentTitle(existing.id, title, now);
          updated++;
        } else {
          unchanged++;
        }
      } else {
        store.insertContent(hash, content, now);
        const stat = statSync(filepath);
        store.updateDocument(existing.id, title, hash, stat ? new Date(stat.mtime).toISOString() : now);
        updated++;
      }
    } else {
      indexed++;
      store.insertContent(hash, content, now);
      const stat = statSync(filepath);
      store.insertDocument(
        collection.name,
        path,
        title,
        hash,
        stat ? new Date(stat.birthtime).toISOString() : now,
        stat ? new Date(stat.mtime).toISOString() : now,
      );
    }
  }

  const allActive = store.getActiveDocumentPaths(collection.name);
  for (const path of allActive) {
    if (!seenPaths.has(path)) {
      store.deactivateDocument(collection.name, path);
      removed++;
    }
  }

  const orphanedContent = store.cleanupOrphanedContent();
  const needsEmbedding = store.getHashesNeedingEmbedding();

  logger.info(`Indexed: ${indexed} new, ${updated} updated, ${unchanged} unchanged, ${removed} removed`);
  if (orphanedContent > 0) {
    logger.info(`Cleaned up ${orphanedContent} orphaned content hash(es)`);
  }
  if (needsEmbedding > 0 && !suppressEmbedNotice) {
    logger.info(`Run 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }
}

export async function updateCollections(config: IndexerConfig = {}): Promise<UpdateCollectionsResult> {
  const logger = getLogger(config);

  return withStore(config, async (store) => {
    store.clearCache();
    const collections = listCollections(store.db);

    if (collections.length === 0) {
      logger.info("No collections found. Run 'qmd collection add .' to index markdown files.");
      return { collectionsUpdated: 0, needsEmbedding: 0 };
    }

    logger.info(`Updating ${collections.length} collection(s)...`);

    for (let i = 0; i < collections.length; i++) {
      const col = collections[i];
      if (!col) continue;
      logger.info(`[${i + 1}/${collections.length}] ${col.name} (${col.glob_pattern})`);

      const yamlCol = getCollectionFromYaml(col.name);
      if (yamlCol?.update) {
        logger.info(`Running update command: ${yamlCol.update}`);
        const proc = Bun.spawn(["/usr/bin/env", "bash", "-c", yamlCol.update], {
          cwd: col.pwd,
          stdout: "pipe",
          stderr: "pipe",
        });
        const output = (await new Response(proc.stdout).text()).trim();
        const errorOutput = (await new Response(proc.stderr).text()).trim();
        const exitCode = await proc.exited;

        if (output) logger.info(output);
        if (errorOutput) logger.warn(errorOutput);
        if (exitCode !== 0) {
          throw new Error(`Update command failed with exit code ${exitCode}`);
        }
      }

      await indexCollection(store, col, true, logger);
    }

    const needsEmbedding = store.getHashesNeedingEmbedding();
    logger.info("All collections updated.");
    if (needsEmbedding > 0) {
      logger.info(`Run 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
    }

    return { collectionsUpdated: collections.length, needsEmbedding };
  });
}

export async function embedVectors(config: IndexerConfig = {}, options: { model?: string; force?: boolean } = {}): Promise<EmbedResult> {
  const logger = getLogger(config);
  const model = options.model || DEFAULT_EMBED_MODEL;
  const force = options.force === true;

  return withStore(config, async (store) => {
    if (force) {
      logger.info("Force re-indexing: clearing all vectors...");
      store.clearAllEmbeddings();
    }

    const hashesToEmbed = store.getHashesForEmbedding();
    if (hashesToEmbed.length === 0) {
      logger.info("All content hashes already have embeddings.");
      return { documents: 0, chunksEmbedded: 0, errors: 0 };
    }

    const now = new Date().toISOString();
    const allChunks: Array<{ hash: string; title: string; text: string; seq: number; pos: number }> = [];

    for (const item of hashesToEmbed) {
      const title = extractTitle(item.body, item.path);
      const chunks = await chunkDocumentByTokens(item.body);
      for (let seq = 0; seq < chunks.length; seq++) {
        const chunk = chunks[seq];
        if (!chunk) continue;
        allChunks.push({
          hash: item.hash,
          title,
          text: chunk.text,
          seq,
          pos: chunk.pos,
        });
      }
    }

    if (allChunks.length === 0) {
      logger.info("No non-empty documents to embed.");
      return { documents: hashesToEmbed.length, chunksEmbedded: 0, errors: 0 };
    }

    let chunksEmbedded = 0;
    let errors = 0;

    await withLLMSession(async (session) => {
      const first = allChunks[0];
      if (!first) {
        throw new Error("No chunks available to embed");
      }
      const firstText = formatDocForEmbedding(first.text, first.title);
      const firstResult = await session.embed(firstText);
      if (!firstResult) {
        throw new Error("Failed to get embedding dimensions from first chunk");
      }
      store.ensureVecTable(firstResult.embedding.length);

      for (const chunk of allChunks) {
        try {
          const text = formatDocForEmbedding(chunk.text, chunk.title);
          const result = await session.embed(text);
          if (!result) {
            errors++;
            continue;
          }
          store.insertEmbedding(chunk.hash, chunk.seq, chunk.pos, new Float32Array(result.embedding), model, now);
          chunksEmbedded++;
        } catch {
          errors++;
        }
      }
    }, { maxDuration: 30 * 60 * 1000, name: "indexer-embed" });

    logger.info(`Embedded ${chunksEmbedded} chunks from ${hashesToEmbed.length} documents.`);
    if (errors > 0) {
      logger.warn(`${errors} chunks failed to embed.`);
    }

    return { documents: hashesToEmbed.length, chunksEmbedded, errors };
  });
}

export async function flushIndex(config: IndexerConfig = {}, options: FlushOptions = {}): Promise<{ status: Record<string, unknown> }> {
  const runUpdate = options.runUpdate !== false;
  const runEmbed = options.runEmbed === true;

  if (runUpdate) {
    await updateCollections(config);
  }
  if (runEmbed) {
    await embedVectors(config, {
      model: options.embedModel,
      force: options.forceEmbed,
    });
  }

  return withStore(config, async (store) => ({ status: store.getStatus() }));
}
