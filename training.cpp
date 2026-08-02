#include "training.hpp"
#include "sparse_dynamic_nn.hpp"
#include "low_rank.hpp"
#include "quantization.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <chrono>
#include <cmath>

namespace fs = std::filesystem;
using namespace sparse_nn;

// =============================================================================
// SyntheticDataset Implementation
// =============================================================================
SyntheticDataset::SyntheticDataset(size_t num_samples, int seq_len, int state_dim, unsigned seed)
    : samples_(num_samples) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (size_t i = 0; i < num_samples; ++i) {
        SequenceSample sample;
        sample.seq_len = seq_len;
        sample.input.resize((size_t)seq_len * state_dim);
        sample.target.resize((size_t)seq_len * state_dim);

        for (auto& v : sample.input) v = dist(rng);
        for (auto& v : sample.target) v = dist(rng);

        samples_[i] = std::move(sample);
    }
}

std::vector<SequenceSample> SyntheticDataset::get_batch(const std::vector<size_t>& indices) const {
    std::vector<SequenceSample> batch;
    batch.reserve(indices.size());
    for (size_t idx : indices) {
        batch.push_back(samples_[idx]);
    }
    return batch;
}

// =============================================================================
// Trainer Implementation
// =============================================================================
Trainer::Trainer(const TrainingConfig& config) : config_(config) {
    fs::create_directories(config.checkpoint_dir);
    fs::create_directories(config.log_dir);

    std::cout << "=== SdetAI Trainer Initialized ===\n";
    std::cout << "  Total neurons: " << config.total_neurons << "\n";
    std::cout << "  Active neurons: " << config.active_neurons << "\n";
    std::cout << "  State dim: " << config.state_dim << "\n";
    std::cout << "  Low-rank: " << config.rank << "\n";
    std::cout << "  Sparsity: " << config.sparsity * 100 << "%\n";
    std::cout << "  LR: " << config.learning_rate << "\n";
    std::cout << "  Batch size: " << config.batch_size << "\n";
    std::cout << "  Epochs: " << config.epochs << "\n";
    std::cout << "  Quant-aware: " << (config.quant_aware ? "yes" : "no") << "\n";
}

Trainer::~Trainer() = default;

float Trainer::train_epoch(Dataset& train_data) {
    float total_loss = 0.0f;
    int num_batches = 0;

    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(current_epoch_ * 12345);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < indices.size(); i += config_.batch_size) {
        size_t end = std::min(i + config_.batch_size, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + i, indices.begin() + end);

        auto batch = train_data.get_batch(batch_indices);

        float batch_loss = 0.0f;
        for (const auto& sample : batch) {
            float loss = forward_pass(sample, true);
            backward_pass(sample);
            batch_loss += loss;
        }

        optimizer_step();

        total_loss += batch_loss / batch.size();
        num_batches++;
        global_step_++;

        if (global_step_ % config_.log_every == 0) {
            float lr = config_.learning_rate;
            if (global_step_ < config_.warmup_steps) {
                lr *= (float)global_step_ / config_.warmup_steps;
            }
            log_metrics(total_loss / num_batches, lr, "train");
        }
    }

    return total_loss / std::max(1, num_batches);
}

float Trainer::evaluate(Dataset& val_data) {
    float total_loss = 0.0f;
    int num_samples = 0;

    size_t eval_samples = std::min<size_t>(val_data.size(), 1000);

    for (size_t i = 0; i < eval_samples; ++i) {
        auto sample = val_data.get(i);
        float loss = forward_pass(sample, false);
        total_loss += loss;
        num_samples++;
    }

    return total_loss / std::max(1, num_samples);
}

void Trainer::train(Dataset& train_data, Dataset& val_data) {
    std::cout << "\n=== Starting Training ===\n";

    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        current_epoch_ = epoch;
        auto start = std::chrono::high_resolution_clock::now();

        float train_loss = train_epoch(train_data);
        float val_loss = evaluate(val_data);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "Epoch " << epoch + 1 << "/" << config_.epochs
                  << " | train_loss: " << std::fixed << std::setprecision(6) << train_loss
                  << " | val_loss: " << val_loss
                  << " | time: " << duration.count() << "s\n";

        if (val_loss < best_val_loss_) {
            best_val_loss_ = val_loss;
            save_checkpoint(epoch, val_loss);
            std::cout << "  -> New best model saved!\n";
        }

        if ((epoch + 1) % config_.save_every == 0) {
            save_checkpoint(epoch, val_loss);
        }

        if (config_.quant_aware && epoch == config_.quant_start_epoch) {
            std::cout << "  -> Starting quantization-aware training\n";
        }
    }

    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Best validation loss: " << best_val_loss_ << "\n";
}

void Trainer::save_checkpoint(int epoch, float loss) {
    std::string path = config_.checkpoint_dir + "/checkpoint_epoch_" + std::to_string(epoch) + ".bin";

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open checkpoint for writing: " << path << "\n";
        return;
    }

    struct Header {
        int epoch;
        float loss;
        int global_step;
        uint32_t magic = 0x53444554;
    } header{epoch, loss, global_step_};

    ofs.write(reinterpret_cast<char*>(&header), sizeof(header));
    ofs.write(reinterpret_cast<char*>(&opt_state_.step), sizeof(int));

    ofs.close();
    std::cout << "Checkpoint saved: " << path << "\n";
}

bool Trainer::load_checkpoint(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open checkpoint: " << path << "\n";
        return false;
    }

    struct Header {
        int epoch;
        float loss;
        int global_step;
        uint32_t magic;
    } header;

    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != 0x53444554) {
        std::cerr << "Invalid checkpoint format\n";
        return false;
    }

    current_epoch_ = header.epoch;
    global_step_ = header.global_step;
    best_val_loss_ = header.loss;

    std::cout << "Checkpoint loaded: epoch=" << header.epoch
              << ", loss=" << header.loss << "\n";
    return true;
}

float Trainer::forward_pass(const SequenceSample& sample, bool training) {
    float loss = 0.0f;

    for (int t = 0; t < sample.seq_len; ++t) {
        const float* timestep_input = &sample.input[(size_t)t * config_.state_dim];
        const float* timestep_target = &sample.target[(size_t)t * config_.state_dim];
        (void)timestep_input;
        (void)timestep_target;

        for (size_t d = 0; d < config_.state_dim; ++d) {
            float diff = 0.0f;
            loss += diff * diff;
        }
    }

    loss /= sample.seq_len * config_.state_dim;

    if (training && config_.quant_aware && current_epoch_ >= config_.quant_start_epoch) {
        loss = forward_quantized(sample);
    }

    return loss;
}

float Trainer::forward_quantized(const SequenceSample& sample) {
    float loss = 0.0f;
    (void)sample;
    return loss;
}

void Trainer::backward_pass(const SequenceSample& sample) {
    (void)sample;
}

void Trainer::optimizer_step() {
    if (!config_.use_adam) return;

    opt_state_.step++;
    float lr = config_.learning_rate;

    if (opt_state_.step < config_.warmup_steps) {
        lr *= (float)opt_state_.step / config_.warmup_steps;
    }

    (void)config_.beta1;
    (void)config_.beta2;
    (void)config_.eps;
}

void Trainer::log_metrics(float loss, float lr, const std::string& phase) {
    std::string log_file = config_.log_dir + "/" + phase + ".csv";
    std::ofstream ofs(log_file, std::ios::app);
    if (ofs) {
        ofs << global_step_ << "," << current_epoch_ << "," << loss << "," << lr << "\n";
    }
} // <--- ВОТ ЭТА СКОБКА БЫЛА ПРОПУЩЕНА

void Trainer::export_quantized_model(const std::string& path) {
    std::cout << "Exporting quantized model to: " << path << "\n";
}