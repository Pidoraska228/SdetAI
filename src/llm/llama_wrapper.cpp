#include "sdetai/llm.hpp"
#include "llama.h"
#include <algorithm>
#include <sstream>

namespace sdetai {

LlamaWrapper::LlamaWrapper(const ModelConfig& config)
    : config_(config) {
    
    // Initialize llama.cpp
    llama_backend_init();
    
    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.use_mmap = config.use_mmap;
    model_params.use_mlock = config.use_mlock;
    
    // Load model
    model_ = llama_load_model_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        return;
    }
    
    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_ubatch = config.n_ubatch;
    ctx_params.n_threads = config.n_threads > 0 ? config.n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_params.rope_scaling_type = config.rope_scaling ? LLAMA_ROPE_SCALING_TYPE_LINEAR : LLAMA_ROPE_SCALING_TYPE_NONE;
    ctx_params.rope_freq_base = config.rope_freq_base;
    ctx_params.rope_freq_scale = config.rope_freq_scale;
    ctx_params.offload_kqv = config.offload_kqv;
    ctx_params.flash_attn = config.flash_attn;
    
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (!ctx_) {
        llama_free_model(model_);
        model_ = nullptr;
        return;
    }
    
    init_sampler();
}

LlamaWrapper::~LlamaWrapper() {
    if (sampler_) llama_sampler_free(sampler_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
    llama_backend_free();
}

LlamaWrapper::LlamaWrapper(LlamaWrapper&& other) noexcept
    : config_(std::move(other.config_))
    , model_(other.model_)
    , ctx_(other.ctx_)
    , sampler_(other.sampler_) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.sampler_ = nullptr;
}

LlamaWrapper& LlamaWrapper::operator=(LlamaWrapper&& other) noexcept {
    if (this != &other) {
        if (sampler_) llama_sampler_free(sampler_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_free_model(model_);
        
        config_ = std::move(other.config_);
        model_ = other.model_;
        ctx_ = other.ctx_;
        sampler_ = other.sampler_;
        
        other.model_ = nullptr;
        other.ctx_ = nullptr;
        other.sampler_ = nullptr;
    }
    return *this;
}

void LlamaWrapper::init_sampler() {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(params);
    
    // Add samplers in order
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(config_.top_k > 0 ? config_.top_k : 40));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(config_.top_p > 0 ? config_.top_p : 0.95f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(config_.temperature > 0 ? config_.temperature : 0.8f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(config_.seed));
    
    if (config_.repeat_penalty > 1.0f) {
        llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
            config_.repeat_penalty, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false));
    }
}

std::optional<std::string> LlamaWrapper::generate(
    const std::string& prompt,
    int n_predict,
    float temperature,
    float top_p,
    int top_k,
    float repeat_penalty
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!ctx_ || !model_) return std::nullopt;
    
    // Tokenize prompt
    auto tokens = tokenize(prompt);
    if (tokens.empty()) return std::nullopt;
    
    int n_predict_actual = n_predict > 0 ? n_predict : config_.n_predict > 0 ? config_.n_predict : n_ctx() - tokens.size() - 4;
    
    // Evaluate prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_, batch) != 0) {
        return std::nullopt;
    }
    
    // Generate tokens
    std::ostringstream output;
    int n_cur = tokens.size();
    
    for (int i = 0; i < n_predict_actual; ++i) {
        // Apply temp overrides
        if (temperature > 0) llama_sampler_chain_remove(sampler_, "temp");
        if (temperature > 0) llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temperature), "temp");
        
        if (top_p > 0) llama_sampler_chain_remove(sampler_, "top_p");
        if (top_p > 0) llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p), "top_p");
        
        if (top_k > 0) llama_sampler_chain_remove(sampler_, "top_k");
        if (top_k > 0) llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k), "top_k");
        
        if (repeat_penalty > 1.0f) llama_sampler_chain_remove(sampler_, "penalties");
        if (repeat_penalty > 1.0f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
                repeat_penalty, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false), "penalties");
        }
        
        // Sample next token
        llama_token id = llama_sampler_sample(sampler_, ctx_, -1);
        
        if (llama_vocab_is_eog(llama_model_get_vocab(model_), id)) {
            break;
        }
        
        // Decode token
        char buf[128];
        int n = llama_token_to_piece(model_, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.write(buf, n);
        }
        
        // Feed token back
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }
        
        n_cur++;
    }
    
    return output.str();
}

bool LlamaWrapper::generate_stream(
    const std::string& prompt,
    TokenCallback callback,
    int n_predict,
    float temperature,
    float top_p,
    int top_k,
    float repeat_penalty
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!ctx_ || !model_) return false;
    
    auto tokens = tokenize(prompt);
    if (tokens.empty()) return false;
    
    int n_predict_actual = n_predict > 0 ? n_predict : config_.n_predict > 0 ? config_.n_predict : n_ctx() - tokens.size() - 4;
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_, batch) != 0) return false;
    
    for (int i = 0; i < n_predict_actual; ++i) {
        if (temperature > 0) {
            llama_sampler_chain_remove(sampler_, "temp");
            llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temperature), "temp");
        }
        if (top_p > 0) {
            llama_sampler_chain_remove(sampler_, "top_p");
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p), "top_p");
        }
        if (top_k > 0) {
            llama_sampler_chain_remove(sampler_, "top_k");
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k), "top_k");
        }
        if (repeat_penalty > 1.0f) {
            llama_sampler_chain_remove(sampler_, "penalties");
            llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
                repeat_penalty, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false), "penalties");
        }
        
        llama_token id = llama_sampler_sample(sampler_, ctx_, -1);
        
        if (llama_vocab_is_eog(llama_model_get_vocab(model_), id)) {
            break;
        }
        
        char buf[128];
        int n = llama_token_to_piece(model_, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token(buf, n);
            if (!callback(token)) {
                break;  // Callback returned false -> stop
            }
        }
        
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_, batch) != 0) break;
    }
    
    return true;
}

std::vector<int> LlamaWrapper::tokenize(const std::string& text, bool add_bos) const {
    if (!model_) return {};
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    int n_tokens = text.empty() ? 0 : llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    
    if (n_tokens <= 0) return {};
    
    std::vector<int> tokens(n_tokens);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tokens, add_bos, true);
    os, true);
    return tokens;
}

std::string LlamaWrapper::detokenize(const std::vector<int>& tokens) const {
    if (!model_ || tokens.empty()) return "";
    
    std::string result;
    result.reserve(tokens.size() * 4);
    
    for (int token : tokens) {
        char buf[128];
        int n = llama_token_to_piece(model_, token, buf, sizeof(buf), 0, true);
        if (n > 0) result.append(buf, n);
    }
    return result;
}

int LlamaWrapper::n_vocab() const {
    return model_ ? llama_n_vocab(model_) : 0;
}

int LlamaWrapper::n_embd() const {
    return model_ ? llama_n_embd(model_) : 0;
}

void LlamaWrapper::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ctx_) {
        llama_kv_cache_clear(ctx_);
    }
}

int LlamaWrapper::used_tokens() const {
    return ctx_ ? llama_n_tokens(ctx_) : 0;
}

int LlamaWrapper::remaining_tokens() const {
    return ctx_ ? n_ctx() - llama_n_tokens(ctx_) : 0;
}

PromptBuilder& PromptBuilder::add_system(const std::string& content) {
    messages_.push_back({Message::Role::SYSTEM, content, ""});
    return *this;
}

PromptBuilder& PromptBuilder::add_user(const std::string& content) {
    messages_.push_back({Message::Role::USER, content, ""});
    return *this;
}

PromptBuilder& PromptBuilder::add_assistant(const std::string& content) {
    messages_.push_back({Message::Role::ASSISTANT, content, ""});
    return *this;
}

PromptBuilder& PromptBuilder::add_tool_result(const std::string& tool_name, const std::string& result) {
    messages_.push_back({Message::Role::TOOL, result, tool_name});
    return *this;
}

std::string PromptBuilder::build() const {
    std::ostringstream oss;
    
    for (const auto& msg : messages_) {
        switch (msg.role) {
            case Message::Role::SYSTEM:
                oss << "<|system|>\n" << msg.content << "\n";
                break;
            case Message::Role::USER:
                oss << "<|user|>\n" << msg.content << "\n";
                break;
            case Message::Role::ASSISTANT:
                oss << "<|assistant|>\n" << msg.content << "\n";
                break;
            case Message::Role::TOOL:
                oss << "<|tool|>" << msg.tool_name << "\n" << msg.content << "\n";
                break;
        }
    }
    oss << "<|assistant|>\n";
    
    return oss.str();
}

} // namespace sdetai