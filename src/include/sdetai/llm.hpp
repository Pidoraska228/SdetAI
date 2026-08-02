#pragma once

#include "sdetai/config.hpp"
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>

struct llama_context;
struct llama_model;
struct llama_sampler;

namespace sdetai {

class LlamaWrapper {
public:
    explicit LlamaWrapper(const ModelConfig& config);
    ~LlamaWrapper();
    
    // Non-copyable, movable
    LlamaWrapper(const LlamaWrapper&) = delete;
    LlamaWrapper& operator=(const LlamaWrapper&) = delete;
    LlamaWrapper(LlamaWrapper&&) noexcept;
    LlamaWrapper& operator=(LlamaWrapper&&) noexcept;
    
    bool is_valid() const { return ctx_ != nullptr; }
    
    // Generate completion from prompt
    std::optional<std::string> generate(
        const std::string& prompt,
        int n_predict = -1,
        float temperature = -1.0f,
        float top_p = -1.0f,
        int top_k = -1,
        float repeat_penalty = -1.0f
    );
    
    // Streaming generation with callback
    using TokenCallback = std::function<bool(const std::string& token)>;
    bool generate_stream(
        const std::string& prompt,
        TokenCallback callback,
        int n_predict = -1,
        float temperature = -1.0f,
        float top_p = -1.0f,
        int top_k = -1,
        float repeat_penalty = -1.0f
    );
    
    // Tokenize text
    std::vector<int> tokenize(const std::string& text, bool add_bos = true) const;
    
    // Detokenize tokens
    std::string detokenize(const std::vector<int>& tokens) const;
    
    // Get model info
    int n_ctx() const { return config_.n_ctx; }
    int n_vocab() const;
    int n_embd() const;
    
    // Reset context (clear KV cache)
    void reset();
    
    // Get current context usage
    int used_tokens() const;
    int remaining_tokens() const;

private:
    ModelConfig config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    std::mutex mutex_;
    
    void init_sampler();
    std::string sample_token(float temperature, float top_p, int top_k, float repeat_penalty);
};

class PromptBuilder {
public:
    struct Message {
        enum class Role { SYSTEM, USER, ASSISTANT, TOOL } role;
        std::string content;
        std::string tool_name;  // For tool messages
    };
    
    PromptBuilder& add_system(const std::string& content);
    PromptBuilder& add_user(const std::string& content);
    PromptBuilder& add_assistant(const std::string& content);
    PromptBuilder& add_tool_result(const std::string& tool_name, const std::string& result);
    
    std::string build() const;
    std::vector<Message> get_messages() const { return messages_; }
    void clear() { messages_.clear(); }
    
private:
    std::vector<Message> messages_;
};

} // namespace sdetai