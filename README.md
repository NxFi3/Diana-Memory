# Diana-Memory

Diana-Memory is a local-first memory system for an LLM-powered Telegram bot. It combines short-term conversational memory with persistent long-term memory, lexical and semantic retrieval, memory scoring, duplicate handling, graph-style associations, and an LLM-based memory policy layer.

The project is a practical memory/RAG prototype rather than a standalone vector database. Its main goal is to decide what information deserves to survive a conversation, keep recent context available cheaply, and retrieve useful long-term information when a later query needs it.

## Architecture

At a high level, Diana separates memory into two layers:

- **STM (short-term memory):** recent conversation events plus a bounded semantic index.
- **LTM (long-term memory):** durable memory items stored in SQLite and indexed for lexical and semantic retrieval.

~~~text
Telegram message
      |
      v
MemoryController
      |
      +--> STM retrieval
      |
      +--> LTM retrieval
      |      +-- FTS5 candidate search
      |      +-- embedding similarity
      |      +-- RRF-style fusion
      |      +-- MMR diversity selection
      |      +-- graph-neighbor expansion
      |      +-- importance / recency / frequency ranking
      |      +-- CrossEncoder reranking
      |
      v
LLM response
      |
      v
Memory policy layer
      |
      +--> extract durable user facts
      +--> compare against existing LTM
      +--> merge near-duplicates when appropriate
      +--> store high-value memories
~~~

## Long-Term Memory

LTM items are stored in SQLite with:

- memory value
- memory type
- embedding
- optional graph connections
- creation time
- last-access time
- access count
- importance
- deletion state

SQLite FTS5 provides lexical candidate retrieval. The current retrieval implementation then computes semantic similarity over the candidate set, applies ranking and diversity logic, expands through stored graph IDs, and finally uses the BGE reranker before returning the top results.

### Saving a memory

When Diana receives a new user message, the memory policy model is asked to identify durable information from the **user message itself**.

The save path includes:

1. LLM-based save decision and rewriting.
2. Embedding of the candidate memory.
3. Similarity search against existing LTM.
4. Very-high similarity duplicate merging.
5. LLM-assisted merging for close candidates.
6. A surprise/usefulness score for deciding whether novel information should be stored.
7. Optional graph association with nearby memories.
8. SQLite + FTS5 persistence.

The current implementation uses a strict semantic duplicate threshold around `0.989` and an LLM merge path above `0.96`.

## Retrieval

Normal LTM retrieval currently follows this pipeline:

~~~text
Query
  |
  +-------------> FTS5 top candidates
  |
  +-------------> FAISS semantic search
                         |
                         v
                    RRF fusion
                         |
                         v
                    MMR selection
                         |
                         v
                 graph expansion
                         |
                         v
              importance/recency/frequency
                         |
                         v
                  CrossEncoder
                         |
                         v
                      top-k
~~~

The lexical and semantic stages are complementary, but the current implementation is lightweight: lexical retrieval narrows candidates and the later semantic stages operate on the resulting set rather than maintaining two completely independent ranking pipelines.

## Short-Term Memory

STM maintains:

- a bounded FAISS `IndexIDMap`
- a dictionary of recent memory records
- a 10-entry conversation buffer
- access counts and timestamps
- a simple forgetting policy when the configured STM capacity is exceeded

STM is persisted with pickle and its FAISS index is saved alongside it.

## Models

The default configuration uses:

| Component | Model |
|---|---|
| LLM | `gemma4:e4b` |
| Embedding | `intfloat/multilingual-e5-base` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

`Engine/Generator.py` loads the configured LLM through Ollama and loads the embedding and reranker models through Sentence Transformers.

## Telegram Bot

`telegrambot.py` connects Diana to Telegram using `python-telegram-bot`.

The bot supports:

- private chats
- group/supergroup replies
- direct mentions
- the `/say` command
- user IDs passed into the memory controller

The bot can also use a local SOCKS5 proxy when the runtime environment requires one.

Set the token through an environment variable:

~~~bash
export TELEGRAM_BOT_TOKEN="your-token"
python telegrambot.py
~~~

Optional bot username:

~~~bash
export BOT_USERNAME="NXFIGOVbot"
~~~

The proxy is currently configured in code for `127.0.0.1:10808`.

## Configuration

The main configuration file is `Config.json`:

~~~json
{
  "LLM": "gemma4:e4b",
  "encoder": "intfloat/multilingual-e5-base",
  "reranker": "BAAI/bge-reranker-v2-m3",
  "LTM_DATABASE_PATH": "Long_Term_Memory.db",
  "STM_SIZE": 100,
  "STM_PATH": "STM_memory.pkl"
}
~~~

## Project Layout

~~~text
Diana-Memory/
├── Config.json
├── telegrambot.py
├── Engine/
│   ├── Generator.py
│   └── prompts.py
├── Memory/
│   ├── DatabaseManager.py
│   ├── Graph_Manager.py
│   ├── LTM_manager.py
│   ├── MemoryItems.py
│   ├── Memorycontroller.py
│   ├── Retrievals.py
│   ├── STM_manager.py
│   └── stmold.py
└── Utils/
    ├── LLM_handler.py
    └── logger.py
~~~

`stmold.py` is retained as a legacy STM implementation; the active controller uses `STM_manager.py`.

## Running Locally

Diana expects Ollama to be available locally and the configured LLM to exist in Ollama.

The Python stack used by the project includes:

- `ollama`
- `sentence-transformers`
- `faiss`
- `numpy`
- `python-telegram-bot`
- `pysocks`

Example:

~~~bash
python telegrambot.py
~~~

The first startup may load the embedding and reranker models, so model initialization can take substantially longer than subsequent requests.

## Design Goals

Diana-Memory was built around a few practical goals:

- Keep short-term conversation bounded.
- Store durable information instead of every message.
- Combine lexical and semantic retrieval.
- Reduce redundant long-term memories.
- Track importance, recency, and frequency.
- Keep related memories connectable through graph IDs.
- Use an LLM for semantic memory decisions instead of treating every message as a document.
- Remain local-first and easy to experiment with.

## Current Limitations

This repository is still a research/engineering prototype. In particular:

- LTM semantic retrieval currently works from a candidate set rather than a fully independent semantic ranking pipeline.
- Graph storage is represented as arrays of related memory IDs rather than a dedicated graph database.
- User isolation is primarily expressed in the prompt/context flow; the LTM schema does not currently enforce a per-user namespace.
- Memory decision parsing relies on LLM-generated JSON plus fallback parsing.
- There is no dedicated dependency lockfile or automated test suite in the repository yet.
- `stmold.py` represents an older STM design and is not part of the main controller path.

These are areas for future iteration; the repository should be treated as a prototype rather than production-ready infrastructure.

## License

MIT License.
