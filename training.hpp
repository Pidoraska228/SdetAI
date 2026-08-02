#pragma once
// =============================================================================
// Training Pipeline for Sparse Dynamic Neural Network
// Supports: Low-Rank factorization, INT8/INT4 quantization, sparse activation
// =============================================================================

#include <vector>
#include <string>
#include <filesystem>
#include <functional>
#include <random>
#include <cstdint>
#include <optional>

namespace sparse_nn {

struct TrainingConfig {
    // Model architecture
    size_t total_neurons = 1'000'000;
    size_t active_neurons = 100'000;
    size_t state_dim = 4;
    size_t rank = 256;              // Low-rank dimension
    float sparsity = 0.9f;          // 90% sparse connections
    
    // Training hyperparameters
    float learning_rate = 1e-3f;
    float weight_decay = 1e-4f;
    int batch_size = 32;
    int seq_length = 128;           // Unroll steps
    int epochs = 100;
    int warmup_steps = 1000;
    
    // Optimization
    bool use_adam = true;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float grad_clip = 1.0f;
    
    // Quantization-aware training
    bool quant_aware = true;        // Simulate quantization during training
    int quant_start_epoch = 10;     // Start quantization after this epoch
    
    // Checkpointing
    std::string checkpoint_dir = "checkpoints";
    int save_every = 5;             // Save every N epochs
    int eval_every = 1;             // Evaluate every N epochs
    
    // Data
    std::string train_data_path;
    std::string val_data_path;
    int num_workers = 4;
    
    // Logging
    std::string log_dir = "logs";
    int log_every = 100;
};

// Data sample for sequence modeling
struct SequenceSample {
    std::vector<float> input;   // [seq_len * state_dim]
    std::vector<float> target;  // [seq_len * state_dim]
    int seq_len = 0;
};

// Dataset interface
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual SequenceSample get(size_t idx) const = 0;
    virtual std::vector<SequenceSample> get_batch(const std::vector<size_t>& indices) const = 0;
};

// Synthetic dataset for testing (can be replaced with real data)
class SyntheticDataset : public Dataset {
public:
    SyntheticDataset(size_t num_samples, int seq_len, int state_dim, unsigned seed = 42);
    
    size_t size() const override { return samples_.size(); }
    SequenceSample get(size_t idx) const override { return samples_[idx]; }
    std::vector<SequenceSample> get_batch(const std::vector<size_t>& indices) const override;
    
private:
    std::vector<SequenceSample> samples_;
};

// =============================================================================
// Trainer: handles forward/backward, optimization, checkpointing
// =============================================================================
class Trainer {
public:
    explicit Trainer(const TrainingConfig& config);
    ~Trainer();
    
    // Train for one epoch
    float train_epoch(Dataset& train_data);
    
    // Evaluate on validation data
    float evaluate(Dataset& val_data);
    
    // Full training loop
    void train(Dataset& train_data, Dataset& val_data);
    
    // Save/load checkpoint
    void save_checkpoint(int epoch, float loss);
    bool load_checkpoint(const std::string& path);
    
    // Get model for inference
    class SparseDynamicNetwork* get_network() { return network_.get(); }
    
    // Export quantized model for deployment
    void export_quantized_model(const std::string& path);
    
private:
    TrainingConfig config_;
    std::unique_ptr<SparseDynamicNetwork> network_;
    
    // Optimizer state (Adam)
    struct OptimizerState {
        std::vector<float> m;   // First moment
        std::vector<float> v;   // Second moment
        int step = 0;
    };
    OptimizerState opt_state_;
    
    // Training state
    int current_epoch_ = 0;
    int global_step_ = 0;
    float best_val_loss_ = std::numeric_limits<float>::infinity();
    
    // Forward pass with quantization simulation
    float forward_pass(const SequenceSample& sample, bool training);
    
    // Backward pass (compute gradients)
    void backward_pass(const SequenceSample& sample);
    
    // Update parameters
    void optimizer_step();
    
    // Quantization-aware forward
    float forward_quantized(const SequenceSample& sample);
    
    // Logging
    void log_metrics(float loss, float lr, const std::string& phase);
};

} // namespace sparse_nn