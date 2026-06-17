# llama-cmd - ollama-style model management

This module adds ollama-like subcommands to the unified `llama` binary. They provide a
`pull / run / list / ps / stop / rm / create / show` workflow on top of the existing
infrastructure - there is **no new daemon or binary**: the inference backend is the
router-mode `llama serve` process, model files live in the HuggingFace cache, and model
presets ("Modelfiles") are the INI presets understood by `llama serve --models-preset`.

The commands are compiled into the `llama-cmd` library and registered as subcommands by
`app/llama.cpp`. They are only available as `llama <command>`.

## Commands

| Command | Description |
| --- | --- |
| `llama pull <repo[:tag]>` | Download a model from a HuggingFace repo into the local cache. Quantization defaults to `Q4_K_M` when no tag is given. |
| `llama run <repo[:tag]\|name> [prompt]` | Ensure the daemon is running, (auto)load the model and chat with it. With a `prompt`, prints a single streamed response; without one, opens an interactive REPL. Pulls the model first if it is not cached. |
| `llama list` (alias `ls`) | List models known to the daemon (or, offline, the HF cache) with size and status. |
| `llama ps` | List models currently loaded/loading/sleeping. |
| `llama stop <repo[:tag]\|name>` | Unload a running model from memory. |
| `llama rm <repo[:tag]\|preset>` | Unload (best-effort), then either delete a cached model from disk, or, if the name is a `create`d preset, remove that preset (the underlying weights stay cached). |
| `llama create <name> --from <repo[:tag]> [-k key=value ...]` | Create/update a named preset (the Modelfile equivalent). Keys are `llama serve` arguments (e.g. `n-gpu-layers`, `ctx-size`, `LLAMA_ARG_*`). |
| `llama show <name>` | Print a preset in INI form. |

## The daemon

`run` (and any command that needs a live backend) talks to a router-mode `llama serve`
daemon over HTTP. If none is reachable it is auto-spawned, detached, as:

```
llama serve --host <host> --port <port> --models-preset <cache>/run/models.ini --models-max 4 --models-autoload
```

- Detection: `GET /health` returns `200` **and** `GET /props` reports `"role":"router"`
  (so a single-model `llama serve -m model.gguf` is not mistaken for the daemon).
- POSIX: `fork()` + `setsid()` with stdio redirected to `<cache>/run/server.log`.
- Windows: `CreateProcess` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`.
- State (host/port) is written to `<cache>/run/daemon.json`; `/health` is the source of truth.

Here `<cache>` is the app cache directory (`fs_get_cache_directory()`), which is distinct
from the HuggingFace hub cache used to store the model weights.

## Configuration

The endpoint defaults to `127.0.0.1:8080` (matching `llama serve`). Override with the
`LLAMA_RUN_HOST` / `LLAMA_RUN_PORT` environment variables, or the `--host` / `--port` flags
on any command.

## Examples

```sh
llama pull   ggml-org/gemma-4-E4B-it-GGUF
llama run    ggml-org/gemma-4-E4B-it-GGUF "Say hi"   # single-shot
llama run    ggml-org/gemma-4-E4B-it-GGUF            # interactive REPL
llama list
llama ps
llama stop   ggml-org/gemma-4-E4B-it-GGUF
llama create my-smol --from ggml-org/gemma-4-E4B-it-GGUF -k n-gpu-layers=0
llama show   my-smol
llama rm     ggml-org/gemma-4-E4B-it-GGUF
```
