#include "llm_wrapper.hpp"
#include "llama.h"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace sdetai {

LlmWrapper::LlmWrapper(const ModelConfig& config)
    : config_(config) {
    
    llama_backend_init();
    
    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.use_mmap = config.use_mmap;
    model_params.use_mlock = config.use_mlock;
    
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
    ctx_params.flash_attn = config.flash_attn;
    ctx_params.offload_kqv = true;
    
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (!ctx_) {
        llama_free_model(model_);
        model_ = nullptr;
        return;
    }
    
    // Default sampler
    init_sampler({});
}

LlmWrapper::~LlmWrapper() {
    if (sampler_) llama_sampler_free(sampler_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
    llama_backend_free();
}

LlmWrapper::LlmWrapper(LlmWrapper&& other) noexcept
    : config_(std::move(other.config_))
    , model_(other.model_)
    , ctx_(other.ctx_)
    , sampler_(other.sampler_) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.sampler_ = nullptr;
}

LlmWrapper& LlmWrapper::operator=(LlmWrapper&& other) noexcept {
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

void LlmWrapper::init_sampler(const GenerationConfig& gen_config) {
    if (sampler_) llama_sampler_free(sampler_);
    
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(params);
    
    float temp = gen_config.temperature > 0 ? gen_config.temperature : 0.8f;
    float top_p = gen_config.top_p > 0 ? gen_config.top_p : 0.95f;
    int top_k = gen_config.top_k > 0 ? gen_config.top_k : 40;
    float repeat_pen = gen_config.repeat_penalty > 1.0f ? gen_config.repeat_penalty : 1.1f;
    int seed = gen_config.seed >= 0 ? gen_config.seed : (int)std::random_device{}();
    
    // Order matters: penalties -> top_k -> top_p -> temp -> dist
    if (repeat_pen > 1.0f) {
        llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
            repeat_pen, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false));
    }
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(seed));
}

std::string LlmWrapper::sample_loop(
    const std::vector<int>& input_tokens,
    const GenerationConfig& gen_config,
    TokenCallback callback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!ctx_ || !model_) return "";
    
    init_sampler(gen_config);
    
    // Evaluate prompt
    llama_batch batch = llama_batch_get_one(input_tokens.data(), input_tokens.size());
    if (llama_decode(ctx_, batch) != 0) {
        return "";
    }
    
    std::ostringstream output;
    int n_predict = gen_config.n_predict > 0 ? gen_config.n_predict : config_.n_ctx - input_tokens.size() - 4;
    int n_cur = input_tokens.size();
    
    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(sampler_, ctx_, -1);
        
        if (llama_vocab_is_eog(llama_model_get_vocab(model_), id)) {
            break;
        }
        
        // Check stop sequences
        char buf[128];
        int n = llama_token_to_piece(model_, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token(buf, n);
            output << token;
            
            // Check stop sequences
            bool should_stop = false;
            for (const auto& stop : gen_config.stop_sequences) {
                std::string recent = output.str();
                if (recent.size() >= stop.size() &&
                    recent.compare(recent.size() - stop.size(), stop.size(), stop) == 0) {
                    should_stop = true;
                    break;
                }
            }
            
            if (callback && !callback(token)) {
                break;
            }
            if (should_stop) break;
        }
        
        // Feed back
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_, batch) != 0) break;
        n_cur++;
    }
    
    return output.str();
}

std::optional<std::string> LlmWrapper::generate(
    const std::string& prompt,
    const GenerationConfig& gen_config
) {
    auto tokens = tokenize(prompt);
    if (tokens.empty()) return std::nullopt;
    
    std::string result = sample_loop(tokens, gen_config);
    return result.empty() ? std::nullopt : std::optional<std::string>(result);
}

bool LlmWrapper::generate_stream(
    const std::string& prompt,
    TokenCallback callback,
    const GenerationConfig& gen_config
) {
    auto tokens = tokenize(prompt);
    if (tokens.empty()) return false;
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ctx_ || !model_) return false;
    
    init_sampler(gen_config);
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_, batch) != 0) return false;
    
    int n_predict = gen_config.n_predict > 0 ? gen_config.n_predict : config_.n_ctx - tokens.size() - 4;
    
    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(sampler_, ctx_, -1);
        
        if (llama_vocab_is_eog(llama_model_get_vocab(model_), id)) {
            return true;
        }
        
        char buf[128];
        int n = llama_token_to_piece(model_, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token(buf, n);
            
            // Check stop sequences
            for (const auto& stop : gen_config.stop_sequences) {
                // Note: need to accumulate for stop check - simplified here
            }
            
            if (!callback(token)) {
                return true;
            }
        }
        
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_, batch) != 0) break;
    }
    
    return true;
}

std::optional<std::string> LlmWrapper::chat(
    const std::vector<ChatMessage>& messages,
    const GenerationConfig& gen_config
) {
    std::string prompt = apply_chat_template(messages, true);
    return generate(prompt, gen_config);
}

bool LlmWrapper::chat_stream(
    const std::vector<ChatMessage>& messages,
    TokenCallback callback,
    const GenerationConfig& gen_config
) {
    std::string prompt = apply_chat_template(messages, true);
    return generate_stream(prompt, callback, gen_config);
}

std::vector<int> LlmWrapper::tokenize(const std::string& text, bool add_bos) const {
    if (!model_) return {};
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    int n_tokens = text.empty() ? 0 : llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    
    if (n_tokens <= 0) return {};
    
    std::vector<int> tokens(n_tokens);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tokens, add_bos, true);
    return tokens;
}

std::string LlmWrapper::detokenize(const std::vector<int>& tokens) const {
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

int LlmWrapper::n_vocab() const {
    return model_ ? llama_n_vocab(model_) : 0;
}

int LlmWrapper::n_embd() const {
    return model_ ? llama_n_embd(model_) : 0;
}

int LlmWrapper::used_tokens() const {
    return ctx_ ? llama_n_tokens(ctx_) : 0;
}

int LlmWrapper::remaining_tokens() const {
    return ctx_ ? n_ctx() - llama_n_tokens(ctx_) : 0;
}

void LlmWrapper::reset_context() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ctx_) llama_kv_cache_clear(ctx_);
}

std::string LlmWrapper::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt
) {
    std::ostringstream oss;
    
    for (const auto& msg : messages) {
        switch (msg.role) {
            case ChatMessage::Role::SYSTEM:
                oss << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    << msg.content << "<|eot_id|>";
                break;
            case ChatMessage::Role::USER:
                oss << "<|start_header_id|>user<|end_header_id|>\n"
                    << msg.content << "<|eot_id|>";
                break;
            case ChatMessage::Role::ASSISTANT:
                oss << "<|start_header_id|>assistant<|end_header_id|>\n"
                    << msg.content << "<|eot_id|>";
                break;
            case ChatMessage::Role::TOOL:
                oss << "<|start_header_id|>tool<|end_header_id|>\n"
                    << msg.tool_name << "\n"
                    << msg.content << "<|eot_id|>";
                break;
        }
    }
    
    if (add_generation_prompt) {
        oss << "<|start_header_id|>assistant<|end_header_id|>\n";
    }
    
    return oss.str();
}

} // namespace sdetai