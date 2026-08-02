#pragma once

#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include "json.hpp"

namespace sdetai {

namespace fs = std::filesystem;

struct ModelConfig {
    fs::path model_path;
    int n_ctx = 4096;
    int n_threads = 0;  // 0 = auto
    int n_gpu_layers = 0;  // 0 = CPU only, -1 = all layers on GPU
    float temperature = 0.1f;
    float top_p = 0.9f;
    int top_k = 40;
    int n_predict = 2048;
    float repeat_penalty = 1.1f;
    int seed = -1;
    bool verbose = false;
    
    // Advanced llama.cpp params
    bool use_mmap = true;
    bool use_mlock = false;
    int n_batch = 512;
    int n_ubatch = 512;
    bool offload_kqv = true;
    bool flash_attn = false;
    std::optional<float> rope_freq_base;
    std::optional<float> rope_freq_scale;
    std::optional<int> rope_scaling;
};

struct AgentConfig {
    fs::path workspace_root;
    fs::path model_path;
    std::string initial_task;
    int max_iterations = 10;
    int max_context_tokens = 32768;
    float temperature = 0.1f;
    bool auto_approve = false;
    bool run_tests = true;
    std::vector<std::string> test_commands;
    std::vector<std::string> build_commands;
    std::vector<std::string> ignore_patterns;
    ModelConfig model;
};

struct ToolResult {
    bool success = false;
    std::string output;
    std::string error;
    int exit_code = 0;
};

struct FileEdit {
    fs::path path;
    std::string old_text;
    std::string new_text;
    bool is_new_file = false;
};

struct AgentAction {
    enum class Type {
        READ_FILE,
        WRITE_FILE,
        EDIT_FILE,
        LIST_FILES,
        GLOB_FILES,
        EXEC_COMMAND,
        RUN_TESTS,
        BUILD_PROJECT,
        THINK,
        FINISH
    } type;

    std::string reasoning;
    // Parameters per action type
    fs::path file_path;
    std::string content;
    std::vector<FileEdit> edits;
    std::string pattern;
    std::string command;
    int timeout_seconds = 30;
};

struct AgentStep {
    AgentAction action;
    ToolResult result;
    std::string observation;
};

struct AgentContext {
    fs::path workspace_root;
    std::vector<AgentStep> history;
    std::string current_task;
    std::vector<fs::path> relevant_files;
    nlohmann::json metadata;
};

struct ModelResponse {
    std::string text;
    std::vector<AgentAction> actions;
    bool is_final = false;
    int tokens_used = 0;
};

} // namespace sdetai

// nlohmann/json serialization for filesystem::path
namespace nlohmann {
    template<>
    struct adl_serializer<std::filesystem::path> {
        static void to_json(json& j, const std::filesystem::path& p) {
            j = p.string();
        }
        static void from_json(const json& j, std::filesystem::path& p) {
            p = j.get<std::string>();
        }
    };
}