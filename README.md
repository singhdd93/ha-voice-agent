# HA Voice Agent

A local-first Home Assistant conversation agent that calls Ollama directly via `/api/chat` — no OpenAI API, no cloud.

Built as a lighter, faster replacement for Extended OpenAI Conversation, with full device attribute injection (fan speed, AC temperature, light brightness) in the prompt context.

## Features

- **Direct Ollama API** — calls `/api/chat` natively, no OpenAI shim
- **Attribute-aware context** — fans, climate, lights, media players all include relevant attributes in the prompt
- **Tool calling** — uses `execute_services` tool compatible with nemotron, mistral-nemo, and other models
- **Configurable** — Ollama URL, model, context window, temperature, system prompt all editable via UI

## Installation via HACS

1. In HACS → Integrations → ⋮ → Custom repositories
2. Add: `https://git.villa5.top/damandeep/ha-voice-agent`  Category: Integration
3. Install "HA Voice Agent"
4. Restart Home Assistant
5. Settings → Devices & Services → Add Integration → "HA Voice Agent"

## Configuration

| Setting | Default | Description |
|---|---|---|
| Ollama URL | `http://10.5.6.50:11434` | Your Ollama instance |
| Model | `nemotron-3-nano-ha:latest` | Model with tool calling support |
| Context window | `4096` | Tokens (reduce for faster prefill) |
| Max response tokens | `512` | |
| Temperature | `0.1` | Lower = more deterministic |
| Max tool call rounds | `3` | Prevents infinite loops |
| Vector search | `true` | Enable semantic entity filtering |
| Embed model | `nomic-embed-text-v2-moe:latest` | Embedding model (must be pulled in Ollama) |
| Top K entities | `15` | Max entities from vector search (+ area entities always included) |
| System prompt | (see const.py) | Full Jinja2 template |

## Tested models

| Model | Tool calling | Notes |
|---|---|---|
| `nemotron-3-nano:4b` | ✅ | Recommended — fast, accurate entity IDs |
| `mistral-nemo:12b` | ✅ | Works but strips domain prefix from entity IDs |
| `qwen2.5:7b` | ❌ | go_template tool calling broken in Ollama |
