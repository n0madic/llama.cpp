#pragma once

#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

hf_files get_cached_files(const std::string & repo_id = {});

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const hf_file & file);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

// Delete a cached repository (the "models--<owner>--<name>" directory) from the HF cache.
// Returns true if a directory was removed. Performs safety checks to ensure the resolved
// path stays inside the cache directory and matches the "models--*" naming convention.
bool delete_cached_repo(const std::string & repo_id);

// Delete specific cached files of a repo (and the blobs they reference) from the HF cache, e.g.
// to remove a single quant without dropping the rest of the repo. `paths` are snapshot paths as
// returned by the cache enumeration; each must resolve inside the repo's cache directory.
// Returns the number of snapshot files removed.
size_t delete_cached_files(const std::string & repo_id, const std::vector<std::string> & paths);

} // namespace hf_cache
