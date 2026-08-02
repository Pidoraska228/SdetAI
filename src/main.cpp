#include "sdetai/llm.hpp"
#include "sdetai/config.hpp"
#include "sdetai/filesystem.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

void print_usage(const char* prog) {
    std::cout << "SdetAI - Lightweight AI Coding Agent\n"
              << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --workspace <path>    Workspace root directory\n"
              << "  --model <path>        Path to GGUF model file\n"
              << "  --config <path>       Config file (JSON)\n"
              << "  --task <text>         Task description\n"
              << "  --task-file <path>    Read task from file\n"
              << "  --max-iter <n>        Max iterations (default: 10)\n"
              << "  --auto-approve        Auto-approve all actions\n"
              << "  --no-tests            Skip running tests\n"
              << "  --verbose             Verbose output\n"
              << "  --help                Show this help\n";
}

struct Args {
    fs::path workspace;
    fs::path model_path;
    fs::path config_path;
    std::string task;
    int max_iterations = 10;
    bool auto_approve = false;
    bool run_tests = true;
    bool verbose = false;
};

std::optional<Args> parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        } else if (arg == "--workspace" && i + 1 < argc) {
            args.workspace = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            args.task = argv[++i];
        } else if (arg == "--task-file" && i + 1 < argc) {
            std::ifstream f(argv[++i]);
            if (f) {
                args.task = std::string((std::istreambuf_iterator<char>(f)), {});
            }
        } else if (arg == "--max-iter" && i + 1 < argc) {
            args.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--auto-approve") {
            args.auto_approve = true;
        } else if (arg == "--no-tests") {
            args.run_tests = false;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return std::nullopt;
        }
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    if (!args) return args.has_value() ? 0 : 1;
    
    // Load config if provided
    sdetai::AgentConfig config;
    if (!args->config_path.empty()) {
        auto loaded = sdetai::load_config(args->config_path);
        if (loaded) config = *loaded;
    } else {
        config = sdetai::create_default_config(args->workspace.empty() ? fs::current_path() : args->workspace);
    }
    
    // Override from command line
    if (!args->workspace.empty()) config.workspace_root = args->workspace;
    if (!args->model_path.empty()) config.model_path = args->model_path;
    if (!args->task.empty()) config.initial_task = args->task;
    config.max_iterations = args->max_iterations;
    config.auto_approve = args->auto_approve;
    config.run_tests = args->run_tests;
    
    // Validate
    if (!sdetai::FileSystem::is_directory(config.workspace_root)) {
        std::cerr << "Workspace not found: " << config.workspace_root << "\n";
        return 1;
    }
    
    if (!sdetai::FileSystem::is_file(config.model_path)) {
        std::cerr << "Model not found: " << config.model_path << "\n";
        std::cerr << "Download a GGUF model (e.g., from HuggingFace) and specify with --model\n";
        return 1;
    }
    
    std::cout << "SdetAI v0.1.0 - Lightweight AI Coding Agent\n";
    std::cout << "Workspace: " << config.workspace_root << "\n";
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Max iterations: " << config.max_iterations << "\n";
    
    if (!config.initial_task.empty()) {
        std::cout << "Task: " << config.initial_task << "\n";
    }
    
    // TODO: Initialize agent and run
    std::cout << "\n[TODO] Agent loop not yet implemented\n";
    std::cout << "Next steps:\n";
    std::cout << "  1. Implement Agent class with Plan/Act/Observe loop\n";
    std::cout << "  2. Add tools: ReadFile, WriteFile, EditFile, RunCommand, SearchCode\n";
    std::cout << "  3. Implement context management (file tree, git status, diagnostics)\n";
    std::cout << "  4. Add test runner integration (ctest, custom commands)\n";
    
    return 0;
}