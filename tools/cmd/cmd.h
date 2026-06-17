#pragma once

//
// Entry points for the ollama-style management subcommands.
//
// These are dispatched in-process by the unified `llama` binary (see app/llama.cpp).
// Each follows the int(int argc, char ** argv) convention shared by all subcommands,
// where argv[0] is the subcommand name and argv[1..] are its arguments.
//
// The commands are thin clients around the existing infrastructure:
//   - model download / cache:  common/download.{h,cpp}, common/hf-cache.{h,cpp}
//   - model presets (Modelfile): common/preset.{h,cpp}
//   - inference backend:        the router-mode `llama serve` daemon (HTTP)
//

int llama_pull  (int argc, char ** argv); // download a model from a HuggingFace repo
int llama_run   (int argc, char ** argv); // run a model (single-shot or interactive) via the daemon
int llama_list  (int argc, char ** argv); // list available models (alias: ls)
int llama_ps    (int argc, char ** argv); // list running models
int llama_stop  (int argc, char ** argv); // unload a running model
int llama_rm    (int argc, char ** argv); // delete a model from the local cache
int llama_create(int argc, char ** argv); // create a named preset (Modelfile equivalent)
int llama_show  (int argc, char ** argv); // show a model preset
