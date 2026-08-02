#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>

namespace sparse_nn {

// =============================================================================
// Configuration constants
// =============================================================================
constexpr size_t TOTAL_NEURONS = 1'000'000;
constexpr size_t ACTIVE_NEURONS = 100'000;        // 10% active at any time
constexpr size_t NUM_GROUPS = TOTAL_NEURONS / ACTIVE_NEURONS;  // 10 groups
constexpr size_t STATE_DIM = 4;                    // State vector per neuron (h, c, etc.)

// Cache line alignment for zero-false-sharing
constexpr size_t CACHE_LINE = 64;

// =============================================================================
// Neuron state - packed for SIMD and cache efficiency
// =============================================================================
struct alignas(CACHE_LINE) NeuronState {
    // Core state (16 bytes = 1 cache line quarter)
    float h[STATE_DIM];    // Hidden state / cell state / etc.
    
    // Optional: quantized weights for this neuron (if per-neuron)
    // float weights[...];  // Keep separate for sparsity
    
    // Metadata (packed into 8 bytes)
    uint32_t group_id;     // Which group this neuron belongs to
    uint16_t active_step;  // Last step this neuron was active
    uint16_t flags;        // Bitflags: active, dirty, refractory, etc.
    
    // Default constructor zero-initializes
    NeuronState() noexcept { clear(); }
    
    void clear() noexcept {
        std::fill(std::begin(h), std::end(h), 0.0f);
        group_id = 0;
        active_step = 0;
        flags = 0;
    }
    
    // SIMD-friendly access
    float* data() noexcept { return h; }
    const float* data() const noexcept { return h; }
};

static_assert(sizeof(NeuronState) == 64, "NeuronState must be exactly one cache line");

// =============================================================================
// Group state - contiguous block of active neurons
// =============================================================================
struct alignas(CACHE_LINE) GroupState {
    // Pointers into global neuron pool (non-owning views)
    NeuronState* neurons = nullptr;
    size_t count = 0;
    size_t group_id = 0;
    
    // Accumulator buffers for state passing (double-buffered)
    // [2][ACTIVE_NEURONS][STATE_DIM] - ping/pong between steps
    std::vector<float> state_buffer_a;  // Input state for this group
    std::vector<float> state_buffer_b;  // Output state to next group
    
    // Sparse connectivity: which neurons connect to which in next group
    // CSR format: row_ptr[i] -> row_ptr[i+1] indices in col_idx
    std::vector<uint32_t> row_ptr;   // Size: count + 1
    std::vector<uint32_t> col_idx;   // Size: nnz (non-zero connections)
    std::vector<float> weights;      // Size: nnz
    
    // NEW: Dense projection matrix for thinking (W_proj * input)
    // Dimension: [STATE_DIM * STATE_DIM]
    std::vector<float> projection_matrix;

    GroupState() = default;
    
    GroupState(size_t group_id_, size_t neuron_count, float sparsity = 0.1f)
        : count(neuron_count), group_id(group_id_) {
        
        // Double buffers: [count * STATE_DIM]
        size_t buf_size = count * STATE_DIM;
        state_buffer_a.resize(buf_size);
        state_buffer_b.resize(buf_size);
        
        // Initialize projection matrix (Identity + small noise)
        projection_matrix.resize(STATE_DIM * STATE_DIM, 0.0f);
        for(size_t i=0; i<STATE_DIM; ++i) projection_matrix[i * STATE_DIM + i] = 1.0f;
        
        std::fill(state_buffer_a.begin(), state_buffer_a.end(), 0.0f);
        std::fill(state_buffer_b.begin(), state_buffer_b.end(), 0.0f);
        
        // Initialize sparse connectivity to next group
        init_sparse_connectivity(sparsity);
    }
    
    void init_sparse_connectivity(float sparsity) {
        // Each neuron connects to sparsity * next_group_size neurons
        // For simplicity: random fixed fanout
        constexpr size_t FANOUT = 16;  // Connections per neuron
        
        row_ptr.resize(count + 1);
        row_ptr[0] = 0;
        
        std::mt19937 rng(group_id * 12345 + 67890);
        std::uniform_int_distribution<uint32_t> dist(0, ACTIVE_NEURONS - 1);
        
        size_t nnz = 0;
        for (size_t i = 0; i < count; ++i) {
            row_ptr[i + 1] = row_ptr[i] + FANOUT;
            nnz += FANOUT;
        }
        
        col_idx.resize(nnz);
        weights.resize(nnz);
        
        std::uniform_real_distribution<float> wdist(-0.1f, 0.1f);
        for (size_t i = 0; i < count; ++i) {
            for (size_t k = 0; k < FANOUT; ++k) {
                col_idx[row_ptr[i] + k] = dist(rng);
                weights[row_ptr[i] + k] = wdist(rng);
            }
        }
    }
    
    // Get input state pointer for neuron i
    float* input_state(size_t i) noexcept {
        return &state_buffer_a[i * STATE_DIM];
    }
    const float* input_state(size_t i) const noexcept {
        return &state_buffer_a[i * STATE_DIM];
    }
    
    // Get output state pointer for neuron i
    float* output_state(size_t i) noexcept {
        return &state_buffer_b[i * STATE_DIM];
    }
    const float* output_state(size_t i) const noexcept {
        return &state_buffer_b[i * STATE_DIM];
    }
    
    // Swap buffers (ping-pong)
    void swap_buffers() noexcept {
        std::swap(state_buffer_a, state_buffer_b);
    }
};

// =============================================================================
// Global network orchestrator
// =============================================================================
class SparseDynamicNetwork {
public:
    SparseDynamicNetwork(float sparsity = 0.1f);
    ~SparseDynamicNetwork() = default;
    
    // Single forward step: process one group, pass state to next
    void step(size_t group_idx);
    
    // Run full cycle through all groups
    void run_cycle(size_t num_cycles = 1);
    
    // Get current active group
    size_t current_group() const noexcept { return current_group_; }
    uint64_t global_step() const noexcept { return global_step_; }
    
    // Access neuron state (read-only)
    const NeuronState* get_neurons() const noexcept { return neuron_pool_.data(); }
    const GroupState& get_group(size_t idx) const noexcept { return groups_[idx]; }
    
    // Getters for demo
    static constexpr size_t total_neurons() { return TOTAL_NEURONS; }
    static constexpr size_t active_neurons() { return ACTIVE_NEURONS; }
    static constexpr size_t num_groups() { return NUM_GROUPS; }
    static constexpr size_t state_dim() { return STATE_DIM; }
    float sparsity() const { return 0.9f; }  // Placeholder
    
    // Inject external input into first group
    void inject_input(const float* input, size_t input_size);
    
    // Read output from last group
    void read_output(float* output, size_t output_size) const;

private:
    // All neurons in one contiguous allocation (cache-friendly)
    std::vector<NeuronState> neuron_pool_;
    
    // Groups reference slices of neuron_pool_
    std::vector<GroupState> groups_;
    
    size_t current_group_ = 0;
    uint64_t global_step_ = 0;
    
    // Initialize neuron pool and group views
    void init_neuron_pool();
    void init_groups(float sparsity);
    
    // Process single group: compute activations, produce output for next group
    void process_group(GroupState& current, GroupState& next);
    
    // Neuron update function (virtual for customization)
    virtual void update_neuron(
        const float* input,     // STATE_DIM inputs from previous group
        float* output,          // STATE_DIM outputs to next group
        NeuronState& state,     // Persistent neuron state
        const float* weights,   // Connection weights
        const uint32_t* targets,// Target neuron indices in next group
        size_t num_connections
    );
};

// =============================================================================
// Inline implementations for hot paths
// =============================================================================
inline void SparseDynamicNetwork::step(size_t group_idx) {
    GroupState& current = groups_[group_idx];
    GroupState& next = groups_[(group_idx + 1) % NUM_GROUPS];
    
    process_group(current, next);
    
    // Swap buffers so next group reads fresh state
    next.swap_buffers();
    
    current_group_ = (group_idx + 1) % NUM_GROUPS;
    ++global_step_;
}

inline void SparseDynamicNetwork::inject_input(const float* input, size_t input_size) {
    GroupState& first = groups_[0];
    size_t copy_size = std::min(input_size, first.count * STATE_DIM);
    std::copy(input, input + copy_size, first.state_buffer_a.begin());
}

inline void SparseDynamicNetwork::read_output(float* output, size_t output_size) const {
    const GroupState& last = groups_[NUM_GROUPS - 1];
    size_t copy_size = std::min(output_size, last.count * STATE_DIM);
    std::copy(last.state_buffer_a.begin(), last.state_buffer_a.begin() + copy_size, output);
}

} // namespace sparse_nn