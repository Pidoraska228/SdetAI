#include "llm_wrapper.hpp"
#include "sparse_dynamic_nn.hpp"
#include "low_rank.hpp"
#include "quantization.hpp"
#include "http_client.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <regex>
#include "../json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// =============================================================================
// Agent: Plan -> Act -> Observe loop with LLM
// =============================================================================

struct AgentConfig {
    fs::path workspace_root;
    fs::path model_path;
    int max_iterations = 10;
    bool auto_approve = false;
    bool verbose = false;
};

struct ToolResult {
    bool success = false;
    std::string output;
    std::string error;
    int exit_code = 0;
};

enum class ActionType {
    READ_FILE, WRITE_FILE, EDIT_FILE, LIST_FILES, GLOB_FILES,
    EXEC_COMMAND, RUN_TESTS, BUILD_PROJECT, THINK, FINISH
};

struct AgentAction {
    ActionType type;
    std::string reasoning;
    fs::path file_path;
    std::string content;
    std::vector<std::pair<std::string, std::string>> edits;  // old, new
    std::string pattern;
    std::string command;
    int timeout_seconds = 30;
};

struct AgentStep {
    AgentAction action;
    ToolResult result;
    std::string observation;
};

class CodingAgent {
public:
    CodingAgent(const AgentConfig& config);
    ~CodingAgent() = default;
    
    bool run_task(const std::string& task);
    
private:
    AgentConfig config_;
    std::unique_ptr<sdetai::LlmWrapper> llm_;
    std::vector<AgentStep> history_;
    std::string current_task_;
    
    // LLM prompt building
    std::string build_system_prompt() const;
    std::string build_user_prompt() const;
    std::vector<sdetai::ChatMessage> build_messages() const;
    
    // Action parsing
    std::optional<AgentAction> parse_action(const std::string& response);
    
    // Tool execution
    ToolResult execute_action(const AgentAction& action);
    ToolResult read_file(const fs::path& path);
    ToolResult write_file(const fs::path& path, const std::string& content);
    ToolResult edit_file(const fs::path& path, const std::vector<std::pair<std::string, std::string>>& edits);
    ToolResult list_files(const fs::path& path);
    ToolResult glob_files(const std::string& pattern);
    ToolResult exec_command(const std::string& cmd, int timeout);
    ToolResult run_tests();
    ToolResult build_project();
    
    // Helpers
    fs::path resolve_path(const fs::path& path) const;
    bool is_safe_path(const fs::path& path) const;
    void print_step(const AgentStep& step);
};

CodingAgent::CodingAgent(const AgentConfig& config) : config_(config) {
    // Initialize LLM
    sdetai::ModelConfig model_config;
    model_config.model_path = config.model_path;
    model_config.n_ctx = 8192;
    model_config.n_threads = std::thread::hardware_concurrency();
    model_config.n_gpu_layers = 0;  // CPU for now
    
    llm_ = std::make_unique<sdetai::LlmWrapper>(model_config);
    
    if (!llm_->is_valid()) {
        std::cerr << "Failed to load model: " << config.model_path << "\n";
    }
}

bool CodingAgent::run_task(const std::string& task) {
    current_task_ = task;
    history_.clear();
    
    std::cout << "\n=== SdetAI Agent Started ===\n";
    std::cout << "Task: " << task << "\n";
    std::cout << "Workspace: " << config_.workspace_root << "\n";
    std::cout << "Model: " << (llm_->is_valid() ? "loaded" : "NOT LOADED") << "\n\n";
    
    if (!llm_->is_valid()) {
        std::cerr << "Cannot run without valid LLM\n";
        return false;
    }
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << config_.max_iterations << " ---\n";
        
        // Build prompt
        auto messages = build_messages();
        
        // Get LLM response
        sdetai::GenerationConfig gen_config;
        gen_config.temperature = 0.1f;
        gen_config.top_p = 0.9f;
        gen_config.n_predict = 2048;
        gen_config.stop_sequences = {"<|eot_id|>"};
        
        std::string response = "";
        bool stream_done = llm_->chat_stream(messages, [&](const std::string& token) {
            response += token;
            std::cout << token << std::flush;
            return true;
        }, gen_config);
        
        std::cout << "\n";
        
        // Parse action
        auto action = parse_action(response);
        if (!action) {
            std::cout << "No valid action parsed, continuing...\n";
            continue;
        }
        
        // Execute
        ToolResult result = execute_action(*action);
        
        // Record step
        AgentStep step{*action, result, ""};
        history_.push_back(step);
        
        print_step(step);
        
        // Check if finished
        if (action->type == ActionType::FINISH) {
            std::cout << "\n=== Task Completed ===\n";
            return true;
        }
        
        // Auto-approve or ask
        if (!config_.auto_approve && action->type != ActionType::THINK) {
            std::cout << "Continue? (y/n): ";
            char c; std::cin >> c;
            if (c != 'y' && c != 'Y') break;
        }
    }
    
    std::cout << "\n=== Max iterations reached ===\n";
    return false;
}

std::string CodingAgent::build_system_prompt() const {
    return R"(You are SdetAI, an autonomous coding agent. You work in a C++/CMake project.

AVAILABLE ACTIONS (respond with exactly ONE action in JSON format):

{
  "type": "READ_FILE|WRITE_FILE|EDIT_FILE|LIST_FILES|GLOB_FILES|EXEC_COMMAND|RUN_TESTS|BUILD_PROJECT|THINK|FINISH",
  "reasoning": "Why you take this action",
  "file_path": "relative/path.ext",      // for file ops
  "content": "file content",              // for WRITE_FILE
  "edits": [["old text", "new text"]],   // for EDIT_FILE
  "pattern": "*.cpp",                     // for GLOB_FILES
  "command": "cmd args",                  // for EXEC_COMMAND
  "timeout_seconds": 30
}

RULES:
- Always reason before acting
- Use relative paths from workspace root
- Prefer small, focused edits
- Run tests after changes
- FINISH when task is complete
- Output ONLY the JSON action, no extra text)";
}

std::string CodingAgent::build_user_prompt() const {
    std::ostringstream oss;
    
    oss << "TASK: " << current_task_ << "\n\n";
    oss << "WORKSPACE: " << config_.workspace_root << "\n\n";
    
    // Show file tree
    oss << "FILE TREE:\n";
    for (auto& p : fs::recursive_directory_iterator(config_.workspace_root)) {
        if (p.is_regular_file()) {
            auto rel = fs::relative(p.path(), config_.workspace_root);
            std::string rel_str = rel.generic_string();
            if (rel_str.find("build") == std::string::npos && 
                rel_str.find(".git") == std::string::npos) {
                oss << "  " << rel_str << "\n";
            }
        }
    }
    
    // Show recent history
    if (!history_.empty()) {
        oss << "\nRECENT HISTORY:\n";
        for (size_t i = std::max(size_t(0), history_.size() - 3); i < history_.size(); ++i) {
            const auto& step = history_[i];
            oss << "  " << int(step.action.type) << ": " << step.action.reasoning << "\n";
            if (!step.result.success) {
                oss << "    ERROR: " << step.result.error << "\n";
            }
        }
    }
    
    return oss.str();
}

std::vector<sdetai::ChatMessage> CodingAgent::build_messages() const {
    std::vector<sdetai::ChatMessage> messages;
    messages.push_back({sdetai::ChatMessage::Role::SYSTEM, build_system_prompt()});
    messages.push_back({sdetai::ChatMessage::Role::USER, build_user_prompt()});
    return messages;
}

std::optional<AgentAction> CodingAgent::parse_action(const std::string& response) {
    try {
        // Find JSON in response
        size_t start = response.find('{');
        size_t end = response.rfind('}');
        if (start == std::string::npos || end == std::string::npos) return std::nullopt;
        
        std::string json_str = response.substr(start, end - start + 1);
        auto j = json::parse(json_str);
        
        AgentAction action;
        std::string type_str = j.value("type", "");
        
        if (type_str == "READ_FILE") action.type = ActionType::READ_FILE;
        else if (type_str == "WRITE_FILE") action.type = ActionType::WRITE_FILE;
        else if (type_str == "EDIT_FILE") action.type = ActionType::EDIT_FILE;
        else if (type_str == "LIST_FILES") action.type = ActionType::LIST_FILES;
        else if (type_str == "GLOB_FILES") action.type = ActionType::GLOB_FILES;
        else if (type_str == "EXEC_COMMAND") action.type = ActionType::EXEC_COMMAND;
        else if (type_str == "RUN_TESTS") action.type = ActionType::RUN_TESTS;
        else if (type_str == "BUILD_PROJECT") action.type = ActionType::BUILD_PROJECT;
        else if (type_str == "THINK") action.type = ActionType::THINK;
        else if (type_str == "FINISH") action.type = ActionType::FINISH;
        else return std::nullopt;
        
        action.reasoning = j.value("reasoning", "");
        action.file_path = j.value("file_path", "");
        action.content = j.value("content", "");
        
        if (j.contains("edits")) {
            for (auto& e : j["edits"]) {
                action.edits.emplace_back(e[0].get<std::string>(), e[1].get<std::string>());
            }
        }
        action.pattern = j.value("pattern", "");
        action.command = j.value("command", "");
        action.timeout_seconds = j.value("timeout_seconds", 30);
        
        return action;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

ToolResult CodingAgent::execute_action(const AgentAction& action) {
    switch (action.type) {
        case ActionType::READ_FILE: return read_file(action.file_path);
        case ActionType::WRITE_FILE: return write_file(action.file_path, action.content);
        case ActionType::EDIT_FILE: return edit_file(action.file_path, action.edits);
        case ActionType::LIST_FILES: return list_files(action.file_path);
        case ActionType::GLOB_FILES: return glob_files(action.pattern);
        case ActionType::EXEC_COMMAND: return exec_command(action.command, action.timeout_seconds);
        case ActionType::RUN_TESTS: return run_tests();
        case ActionType::BUILD_PROJECT: return build_project();
        case ActionType::THINK: 
            return {true, "Thought: " + action.reasoning, "", 0};
        case ActionType::FINISH:
            return {true, "Task finished: " + action.reasoning, "", 0};
    }
    return {false, "", "Unknown action", -1};
}

fs::path CodingAgent::resolve_path(const fs::path& path) const {
    if (path.is_absolute()) return path;
    return config_.workspace_root / path;
}

bool CodingAgent::is_safe_path(const fs::path& path) const {
    auto abs = fs::weakly_canonical(resolve_path(path));
    auto ws = fs::weakly_canonical(config_.workspace_root);
    return abs.string().rfind(ws.string(), 0) == 0;
}

ToolResult CodingAgent::read_file(const fs::path& path) {
    auto full = resolve_path(path);
    if (!is_safe_path(path) || !fs::exists(full) || !fs::is_regular_file(full)) {
        return {false, "", "File not found or unsafe: " + path.string(), -1};
    }
    
    std::ifstream f(full);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    return {true, content, "", 0};
}

ToolResult CodingAgent::write_file(const fs::path& path, const std::string& content) {
    auto full = resolve_path(path);
    if (!is_safe_path(path)) {
        return {false, "", "Unsafe path: " + path.string(), -1};
    }
    
    fs::create_directories(full.parent_path());
    std::ofstream f(full);
    f << content;
    return {true, "Written " + std::to_string(content.size()) + " bytes", "", 0};
}

ToolResult CodingAgent::edit_file(const fs::path& path, const std::vector<std::pair<std::string, std::string>>& edits) {
    auto read_result = read_file(path);
    if (!read_result.success) return read_result;
    
    std::string content = read_result.output;
    for (const auto& [old_text, new_text] : edits) {
        size_t pos = content.find(old_text);
        if (pos == std::string::npos) {
            return {false, "", "Edit not found: " + old_text.substr(0, 50), -1};
        }
        content.replace(pos, old_text.size(), new_text);
    }
    
    return write_file(path, content);
}

ToolResult CodingAgent::list_files(const fs::path& path) {
    auto full = resolve_path(path.empty() ? "." : path);
    if (!is_safe_path(path) || !fs::exists(full) || !fs::is_directory(full)) {
        return {false, "", "Directory not found", -1};
    }
    
    std::string output;
    for (auto& p : fs::directory_iterator(full)) {
        output += (p.is_directory() ? "[D] " : "[F] ") + fs::relative(p.path(), config_.workspace_root).generic_string() + "\n";
    }
    return {true, output, "", 0};
}

ToolResult CodingAgent::glob_files(const std::string& pattern) {
    std::string output;
    std::regex re(pattern);
    
    for (auto& p : fs::recursive_directory_iterator(config_.workspace_root)) {
        if (p.is_regular_file()) {
            auto rel = fs::relative(p.path(), config_.workspace_root);
            std::string rel_str = rel.generic_string();
            if (std::regex_match(rel_str, re)) {
                output += rel_str + "\n";
            }
        }
    }
    return {true, output, "", 0};
}

ToolResult CodingAgent::exec_command(const std::string& cmd, int timeout) {
    // Simple exec - in production use proper process handling
    std::string full_cmd = "cd " + config_.workspace_root.string() + " && " + cmd;
    FILE* pipe = _popen(full_cmd.c_str(), "r");
    if (!pipe) return {false, "", "Failed to execute", -1};
    
    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    int exit_code = _pclose(pipe);
    
    return {exit_code == 0, result, exit_code != 0 ? "Command failed" : "", exit_code};
}

ToolResult CodingAgent::run_tests() {
    // Try common test commands
    std::vector<std::string> cmds = {
        "ctest --output-on-failure",
        "cmake --build build --target test",
        "python -m pytest"
    };
    
    for (const auto& cmd : cmds) {
        auto result = exec_command(cmd, 60);
        if (result.success || !result.output.empty()) {
            return result;
        }
    }
    return {false, "", "No test command worked", -1};
}

ToolResult CodingAgent::build_project() {
    auto result = exec_command("cmake --build build", 120);
    return result;
}

void CodingAgent::print_step(const AgentStep& step) {
    std::cout << "\n[Action] " << int(step.action.type) << ": " << step.action.reasoning << "\n";
    if (!step.result.success) {
        std::cout << "[Error] " << step.result.error << "\n";
    }
    if (!step.result.output.empty() && step.result.output.size() < 500) {
        std::cout << "[Output] " << step.result.output.substr(0, 500) << "\n";
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse args
    fs::path workspace = fs::current_path();
    fs::path model_path;
    std::string task;
    int max_iter = 10;
    bool auto_approve = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--workspace" && i + 1 < argc) workspace = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--task" && i + 1 < argc) task = argv[++i];
        else if (arg == "--max-iter" && i + 1 < argc) max_iter = std::stoi(argv[++i]);
        else if (arg == "--auto-approve") auto_approve = true;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --model <gguf> --task \"...\" [options]\n";
            return 0;
        }
    }
    
    // Default model locations
    if (model_path.empty()) {
        std::vector<fs::path> candidates = {
            workspace / "models" / "model.gguf",
            fs::path(std::getenv("HOME")) / ".cache" / "sdetai" / "model.gguf",
            "C:/models/model.gguf"
        };
        for (auto& p : candidates) {
            if (fs::exists(p)) { model_path = p; break; }
        }
    }
    
    if (model_path.empty() || !fs::exists(model_path)) {
        std::cerr << "Model not found. Use --model <path>\n";
        return 1;
    }
    
    if (task.empty()) {
        std::cerr << "Task required. Use --task \"...\"\n";
        return 1;
    }
    
    AgentConfig config;
    config.workspace_root = fs::weakly_canonical(workspace);
    config.model_path = model_path;
    config.max_iterations = max_iter;
    config.auto_approve = auto_approve;
    
    CodingAgent agent(config);
    bool success = agent.run_task(task);
    
    return success ? 0 : 1;
}