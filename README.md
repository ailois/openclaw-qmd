# QMD Memory (OpenClaw Edition)

`QMD Memory` is an OpenClaw memory plugin built on top of the original [`qmd`](https://github.com/tobi/qmd).

This repository keeps QMD's hybrid retrieval core (BM25 + vector + rerank) and adds:
- OpenClaw memory plugin integration (`kind: "memory"`)
- `openclaw.json`-driven configuration
- API model mode with separate providers for `embedding` / `rerank` / `generate`

---

## Upstream & Fork Relationship

- Upstream project: [`tobi/qmd`](https://github.com/tobi/qmd)
- This project: extends upstream QMD for OpenClaw memory replacement use-cases
- Design goal: keep QMD CLI usable while enabling OpenClaw-native memory plugin workflow

---

## Features

- **QMD retrieval engine**: BM25 + vector search + reranking pipeline
- **OpenClaw plugin**: `openclaw.plugin.json` with `qmd-memory` (`kind: memory`)
- **Memory tools compatibility**:
  - `memory_search`
  - `memory_get`
  - `memory_flush`
- **Model backend modes**:
  - `local` (original GGUF / node-llama-cpp flow)
  - `api` (OpenAI-compatible APIs)
  - `hybrid` (per-capability API/local fallback)
- **Split model API endpoints**:
  - `models.api.embedding`
  - `models.api.rerank`
  - `models.api.generate`

---

## Requirements

- Bun >= 1.0
- Linux/macOS (Windows should work with Bun + sqlite-vec runtime support)
- OpenClaw runtime (for plugin usage)

If you use `local` model mode, ensure local model/runtime dependencies are available (`node-llama-cpp`, model files, etc.).

---

## Installation

### 1) Clone repository

```bash
git clone <your-repo-url> qmd-memory
cd qmd-memory
```

### 2) Install dependencies

```bash
bun install
```

### 3) (Optional) Link CLI for local usage

```bash
bun link
qmd --help
```

---

## OpenClaw Plugin Setup

Recommended layout:

```text
~/.openclaw/extensions/qmd-memory/
  ├── openclaw.plugin.json
  ├── plugin/
  └── src/
```

You can either:
- clone this repo directly into `~/.openclaw/extensions/qmd-memory`, or
- symlink this repo directory into `~/.openclaw/extensions/qmd-memory`.

---

## openclaw.json Configuration

Use this pattern to replace OpenClaw's memory slot with `qmd-memory`:

```json
{
  "plugins": {
    "slots": {
      "memory": "qmd-memory"
    },
    "entries": {
      "qmd-memory": {
        "enabled": true,
        "config": {
          "index": {
            "name": "memory",
            "dbPath": "~/.openclaw/cache/qmd-memory/memory.sqlite"
          },
          "models": {
            "mode": "api",
            "api": {
              "embedding": {
                "provider": "openai-compatible",
                "timeoutMs": 30000,
                "baseUrl": "https://embed-provider.example.com/v1",
                "apiKeyEnv": "QMD_EMBED_API_KEY",
                "model": "text-embedding-3-small"
              },
              "rerank": {
                "provider": "openai-compatible",
                "timeoutMs": 30000,
                "baseUrl": "https://rerank-provider.example.com/v1",
                "apiKeyEnv": "QMD_RERANK_API_KEY",
                "model": "bge-reranker-v2-m3"
              },
              "generate": {
                "provider": "openai-compatible",
                "timeoutMs": 30000,
                "baseUrl": "https://chat-provider.example.com/v1",
                "apiKeyEnv": "QMD_GENERATE_API_KEY",
                "model": "gpt-4o-mini"
              }
            }
          },
          "sync": {
            "enabled": true,
            "intervalSec": 300,
            "embedOnSync": false
          }
        }
      }
    }
  }
}
```

Set corresponding environment variables before running OpenClaw:

```bash
export QMD_EMBED_API_KEY="..."
export QMD_RERANK_API_KEY="..."
export QMD_GENERATE_API_KEY="..."
```

---

## Verify Configuration

This repo provides a smoke command:

```bash
bun src/qmd.ts memory-verify --config openclaw.json
```

Expected result:
- `memory-verify passed (...)` if configured endpoints are reachable and return compatible responses.

---

## CLI Usage (QMD Core)

```bash
# collection management
qmd collection add ~/notes --name notes
qmd collection list
qmd update

# embedding
qmd embed

# retrieval
qmd search "auth"
qmd vsearch "how to deploy"
qmd query "quarterly planning process"

# document fetch
qmd get "notes/meeting.md"
qmd multi-get "notes/*.md" --json
```

---

## OpenClaw Memory Tooling

When loaded as an OpenClaw memory plugin, these tools are exposed:

- `memory_search`: hybrid memory retrieval
- `memory_get`: fetch memory content by path/docid
- `memory_flush`: refresh/sync status path

---

## Development

```bash
bun install
bun test
bun src/qmd.ts --help
```

Useful checks:

```bash
bun scripts/verify-memory-plugin-mock.ts
bun src/qmd.ts memory-verify --config openclaw.json
```

---

## Project Structure

```text
.
├── src/                    # QMD core runtime, retrieval, llm integration
├── plugin/                 # OpenClaw plugin entrypoint
├── openclaw.plugin.json    # Plugin manifest (kind: memory)
├── openclaw.json           # Example OpenClaw config
├── finetune/               # Upstream finetune/eval tooling
└── qmd                     # CLI wrapper
```

---

## License

MIT
