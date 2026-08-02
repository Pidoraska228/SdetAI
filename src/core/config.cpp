#include "config.hpp"
#include "filesystem.hpp"
#include "../../json.hpp"
#include <fstream>
#include <thread>

namespace sdetai {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::optional<AgentConfig> load_config(const fs::path& path) {
    try {
        if (!FileSystem::is_file(path)) return std::nullopt;
        
        auto content = FileSystem::read_file(path);
        if (!content) return std::nullopt;
        
        json j = json::parse(*content);
        AgentConfig config;
        
        if (j.contains("workspace_root")) config.workspace_root = j["workspace_root"].get<fs::path>();
        if (j.contains("model_path")) config.model_path = j["model_path"].get<fs::path>();
        if (j.contains("initial_task")) config.initial_task = j["initial_task"].get<std::string>();
        if (j.contains("max_iterations")) config.max_iterations = j["max_iterations"].get<int>();
        if (j.contains("max_context_tokens")) config.max_context_tokens = j["max_context_tokens"].get<int>();
        if (j.contains("temperature")) config.temperature = j["temperature"].get<float>();
        if (j.contains("auto_approve")) config.auto_approve = j["auto_approve"].get<bool>();
        if (j.contains("run_tests")) config.run_tests = j["run_tests"].get<bool>();
        if (j.contains("test_commands")) config.test_commands = j["test_commands"].get<std::vector<std::string>>();
        if (j.contains("build_commands")) config.build_commands = j["build_commands"].get<std::vector<std::string>>();
        if (j.contains("ignore_patterns")) config.ignore_patterns = j["ignore_patterns"].get<std::vector<std::string>>();
        
        // Model config
        if (j.contains("model")) {
            auto& m = j["model"];
            if (m.contains("n_ctx")) config.model.n_ctx = m["n_ctx"].get<int>();
            if (m.contains("n_threads")) config.model.n_threads = m["n_threads"].get<int>();
            if (m.contains("n_gpu_layers")) config.model.n_gpu_layers = m["n_gpu_layers"].get<int>();
            if (m.contains("temperature")) config.model.temperature = m["temperature"].get<float>();
            if (m.contains("top_p")) config.model.top_p = m["top_p"].get<float>();
            if (m.contains("top_k")) config.model.top_k = m["top_k"].get<int>();
            if (m.contains("n_predict")) config.model.n_predict = m["n_predict"].get<int>();
            if (m.contains("repeat_penalty")) config.model.repeat_penalty = m["repeat_penalty"].get<float>();
            if (m.contains("seed")) config.model.seed = m["seed"].get<int>();
            if (m.contains("verbose")) config.model.verbose = m["verbose"].get<bool>();
        }
        
        return config;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

bool save_config(const AgentConfig& config, const fs::path& path) {
    try {
        json j;
        
        j["workspace_root"] = config.workspace_root.string();
        j["model_path"] = config.model_path.string();
        j["initial_task"] = config.initial_task;
        j["max_iterations"] = config.max_iterations;
        j["max_context_tokens"] = config.max_context_tokens;
        j["temperature"] = config.temperature;
        j["auto_approve"] = config.auto_approve;
        j["run_tests"] = config.run_tests;
        j["test_commands"] = config.test_commands;
        j["build_commands"] = config.build_commands;
        j["ignore_patterns"] = config.ignore_patterns;
        
        json m;
        m["n_ctx"] = config.model.n_ctx;
        m["n_threads"] = config.model.n_threads;
        m["n_gpu_layers"] = config.model.n_gpu_layers;
        m["temperature"] = config.model.temperature;
        m["top_p"] = config.model.top_p;
        m["top_k"] = config.model.top_k;
        m["n_predict"] = config.model.n_predict;
        m["repeat_penalty"] = config.model.repeat_penalty;
        m["seed"] = config.model.seed;
        m["verbose"] = config.model.verbose;
        j["model"] = m;
        
        std::string output = j.dump(4);
        return FileSystem::write_file(path, output);
    } catch (const std::exception&) {
        return false;
    }
}

AgentConfig create_default_config(const fs::path& workspace_root) {
    AgentConfig config;
    config.workspace_root = FileSystem::normalize(workspace_root);
    config.max_iterations = 10;
    config.max_context_tokens = 32768;
    config.temperature = 0.1f;
    config.auto_approve = false;
    config.run_tests = true;
    
    // Default test commands (tries common ones)
    config.test_commands = {
        "ctest --output-on-failure",
        "cmake --build build --target test",
        "python -m pytest",
        "npm test",
        "cargo test",
        "go test ./...",
        "mvn test",
        "gradle test"
    };
    
    // Default build commands
    config.build_commands = {
        "cmake --build build",
        "make",
        "ninja",
        "cargo build --release",
        "go build ./...",
        "mvn compile",
        "gradle build"
    };
    
    // Default ignore patterns
    config.ignore_patterns = {
        ".git",
        "build",
        "*.o", "*.obj",
        "*.exe", "*.dll", "*.so", "*.dylib",
        "*.class", "*.jar", "*.war",
        "node_modules",
        "__pycache__",
        "*.pyc",
        "target",
        "dist",
        "*.log"
    };
    
    // Default model config (llama.cpp defaults)
    config.model.n_ctx = 4096;
    config.model.n_threads = std::thread::hardware_concurrency();
    config.model.n_gpu_layers = 0;  // CPU by default
    config.model.temperature = 0.1f;
    config.model.top_p = 0.9f;
    config.model.top_k = 40;
    config.model.n_predict = 2048;
    config.model.repeat_penalty = 1.1f;
    config.model.seed = -1;
    config.model.verbose = false;
    
    return config;
}

} // namespace sdetai