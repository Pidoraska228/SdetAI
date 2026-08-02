#pragma once
// =============================================================================
// LLM Wrapper using llama.cpp C API
// Provides: completion, streaming, chat templates, tool calling format
// =============================================================================

#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <filesystem>
#include <mutex>

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace sdetai {

struct ModelConfig {
    std::filesystem::path model_path;
    int n_ctx = 4096;
    int n_batch = 512;
    int n_ubatch = 512;
    int n_threads = 0;           // 0 = auto
    int n_gpu_layers = 0;        // 0 = CPU only, -1 = all on GPU
    bool use_mmap = true;
    bool use_mlock = false;
    bool flash_attn = false;
};

struct GenerationConfig {
    int n_predict = 2048;
    float temperature = 0.1f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int seed = -1;
    std::vector<std::string> stop_sequences;
};

struct ChatMessage {
    enum class Role { SYSTEM, USER, ASSISTANT, TOOL } role;
    std::string content;
    std::string tool_name;  // For tool messages
};

class LlmWrapper {
public:
    explicit LlmWrapper(const ModelConfig& config);
    ~LlmWrapper();
    
    LlmWrapper(const LlmWrapper&) = delete;
    LlmWrapper& operator=(const LlmWrapper&) = delete;
    LlmWrapper(LlmWrapper&&) noexcept;
    LlmWrapper& operator=(LlmWrapper&&) noexcept;
    
    bool is_valid() const { return ctx_ != nullptr; }
    
    // Simple completion
    std::optional<std::string> generate(
        const std::string& prompt,
        const GenerationConfig& gen_config = {}
    );
    
    // Streaming with callback (returns false to stop)
    using TokenCallback = std::function<bool(const std::string& token)>;
    bool generate_stream(
        const std::string& prompt,
        TokenCallback callback,
        const GenerationConfig& gen_config = {}
    );
    
    // Chat completion with messages
    std::optional<std::string> chat(
        const std::vector<ChatMessage>& messages,
        const GenerationConfig& gen_config = {}
    );
    
    // Streaming chat
    bool chat_stream(
        const std::vector<ChatMessage>& messages,
        TokenCallback callback,
        const GenerationConfig& gen_config = {}
    );
    
    // Tokenize / detokenize
    std::vector<int> tokenize(const std::string& text, bool add_bos = true) const;
    std::string detokenize(const std::vector<int>& tokens) const;
    
    // Context info
    int n_ctx() const { return config_.n_ctx; }
    int n_vocab() const;
    int n_embd() const;
    int used_tokens() const;
    int remaining_tokens() const;
    
    void reset_context();
    
    // Apply chat template (Llama-3, ChatML, etc.)
    static std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true
    );

private:
    ModelConfig config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    std::mutex mutex_;
    
    void init_sampler(const GenerationConfig& gen_config);
    std::string sample_loop(
        const std::vector<int>& input_tokens,
        const GenerationConfig& gen_config,
        TokenCallback callback = nullptr
    );
};

} // namespace sdetai