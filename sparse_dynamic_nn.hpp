#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace sparse_nn {

constexpr std::size_t TOTAL_NEURONS  = 1'000'000;
constexpr std::size_t ACTIVE_NEURONS = 100'000;
constexpr std::size_t NUM_GROUPS     = TOTAL_NEURONS / ACTIVE_NEURONS;
constexpr std::size_t STATE_DIM      = 4;
constexpr std::size_t CACHE_LINE     = 64;

struct alignas(CACHE_LINE) NeuronState {
    float h[STATE_DIM];

    std::uint32_t group_id;
    std::uint16_t active_step;
    std::uint16_t flags;

    NeuronState() noexcept
        : h{0.0f, 0.0f, 0.0f, 0.0f},
          group_id(0),
          active_step(0),
          flags(0) {}

    void clear() noexcept {
        h[0] = 0.0f;
        h[1] = 0.0f;
        h[2] = 0.0f;
        h[3] = 0.0f;
        active_step = 0;
        flags = 0;
    }

    float* data() noexcept {
        return h;
    }

    const float* data() const noexcept {
        return h;
    }
};

struct alignas(CACHE_LINE) GroupState {
    NeuronState* neurons;
    std::size_t count;
    std::size_t group_id;

    std::vector<float> state_buffer_a;
    std::vector<float> state_buffer_b;

    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint32_t> col_idx;
    std::vector<float> weights;

    std::vector<float> projection_matrix;

    GroupState();

    GroupState(
        std::size_t group_id_,
        std::size_t neuron_count,
        float sparsity = 0.1f
    );

    void init_sparse_connectivity(float sparsity);

    float* input_state(std::size_t i) noexcept {
        return &state_buffer_a[i * STATE_DIM];
    }

    const float* input_state(std::size_t i) const noexcept {
        return &state_buffer_a[i * STATE_DIM];
    }

    float* output_state(std::size_t i) noexcept {
        return &state_buffer_b[i * STATE_DIM];
    }

    const float* output_state(std::size_t i) const noexcept {
        return &state_buffer_b[i * STATE_DIM];
    }

    void swap_buffers() noexcept {
        state_buffer_a.swap(state_buffer_b);
    }
};

class SparseDynamicNetwork {
public:

    explicit SparseDynamicNetwork(float sparsity = 0.1f);

    ~SparseDynamicNetwork() = default;

    SparseDynamicNetwork(const SparseDynamicNetwork&) = delete;
    SparseDynamicNetwork& operator=(
        const SparseDynamicNetwork&
    ) = delete;

    void step(std::size_t group_idx);

    void run_cycle(std::size_t num_cycles = 1);

    std::size_t current_group() const noexcept {
        return current_group_;
    }

    std::uint64_t global_step() const noexcept {
        return global_step_;
    }

    const NeuronState* get_neurons() const noexcept {
        return neuron_pool_.data();
    }

    const GroupState& get_group(std::size_t idx) const noexcept {
        return groups_[idx];
    }

    static constexpr std::size_t total_neurons() {
        return TOTAL_NEURONS;
    }

    static constexpr std::size_t active_neurons() {
        return ACTIVE_NEURONS;
    }

    static constexpr std::size_t num_groups() {
        return NUM_GROUPS;
    }

    static constexpr std::size_t state_dim() {
        return STATE_DIM;
    }

    float sparsity() const noexcept {
        return sparsity_;
    }

    void inject_input(
        const float* input,
        std::size_t input_size
    );

    void read_output(
        float* output,
        std::size_t output_size
    ) const;

    void train_step(
        float input_token,
        float target_token,
        float learning_rate
    );

    bool save_weights(
        const std::filesystem::path& path
    ) const;

    bool load_weights(
        const std::filesystem::path& path
    );

private:

    std::vector<NeuronState> neuron_pool_;

    std::vector<GroupState> groups_;

    std::size_t current_group_ = 0;

    std::uint64_t global_step_ = 0;

    float sparsity_ = 0.1f;

    void init_neuron_pool();

    void init_groups(float sparsity);

    void process_group(
        GroupState& current,
        GroupState& next
    );

    void update_neuron(
        const float* input,
        float* output,
        NeuronState& state,
        const float* weights,
        const std::uint32_t* targets,
        std::size_t num_connections
    );
};

} // namespace sparse_nn