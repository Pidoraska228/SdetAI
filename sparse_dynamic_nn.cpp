#include "sparse_dynamic_nn.hpp"
#include <immintrin.h>  // AVX2/SSE intrinsics
#include <cmath>
#include <cstring>

namespace sparse_nn {

// =============================================================================
// SIMD-optimized neuron update
// =============================================================================
inline void simd_dense_mul_acc(
    const float* __restrict weights,
    const float* __restrict inputs,
    float* __restrict output,
    size_t rows, size_t cols, size_t nnz
) {
    // CSR SpMV: output[i] += sum_j weights[k] * inputs[col_idx[k]]
    // This is a simplified version - real impl would use proper SpMV
    
    for (size_t i = 0; i < rows; ++i) {
        // Accumulate into output[i * STATE_DIM : (i+1) * STATE_DIM]
        // For now, scalar fallback - replace with AVX2 SpMV kernel
    }
}

// Fast activation: SiLU/Swish approximation
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Vectorized SiLU for 4 floats
inline void silu4(float* __restrict x) {
    #ifdef __AVX2__
    __m256 vx = _mm256_loadu_ps(x);
    __m256 vex = _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), vx));
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 denom = _mm256_add_ps(vone, vex);
    __m256 result = _mm256_div_ps(vx, denom);
    _mm256_storeu_ps(x, result);
    #else
    for (int i = 0; i < 4; ++i) x[i] = silu(x[i]);
    #endif
}

// =============================================================================
// Constructor
// =============================================================================
SparseDynamicNetwork::SparseDynamicNetwork(float sparsity) {
    init_neuron_pool();
    init_groups(sparsity);
}

void SparseDynamicNetwork::init_neuron_pool() {
    neuron_pool_.resize(TOTAL_NEURONS);
    
    // Assign group IDs
    for (size_t g = 0; g < NUM_GROUPS; ++g) {
        size_t start = g * ACTIVE_NEURONS;
        for (size_t i = 0; i < ACTIVE_NEURONS; ++i) {
            neuron_pool_[start + i].group_id = static_cast<uint32_t>(g);
        }
    }
}

void SparseDynamicNetwork::init_groups(float sparsity) {
    groups_.reserve(NUM_GROUPS);
    
    for (size_t g = 0; g < NUM_GROUPS; ++g) {
        size_t start = g * ACTIVE_NEURONS;
        NeuronState* group_neurons = &neuron_pool_[start];
        
        groups_.emplace_back(g, ACTIVE_NEURONS, sparsity);
        groups_.back().neurons = group_neurons;
    }
}

// =============================================================================
// Hot path: process one group
// =============================================================================
void SparseDynamicNetwork::process_group(GroupState& current, GroupState& next) {
    // For each neuron in current group:
    // 1. Read its input state (from previous group's output)
    // 2. Apply neuron update (RNN/LSTM/GRU cell)
    // 3. Write to its persistent state
    // 4. Sparse project to next group's input buffer
    
    const float* __restrict input_buf = current.state_buffer_a.data();
    float* __restrict output_buf = current.state_buffer_b.data();
    float* __restrict next_input_buf = next.state_buffer_a.data();
    
    // Clear next group's input buffer
    std::fill(next_input_buf, next_input_buf + next.count * STATE_DIM, 0.0f);
    
    // Process each neuron
    for (size_t i = 0; i < current.count; ++i) {
        NeuronState& ns = current.neurons[i];
        const float* neuron_input = &input_buf[i * STATE_DIM];
        float* neuron_output = &output_buf[i * STATE_DIM];

        // --- DYNAMIC SPARSITY: Skip inactive neurons ---
        float activation_power = 0.0f;
        for (int d = 0; d < STATE_DIM; ++d) activation_power += std::abs(neuron_input[d]);
        if (activation_power < 0.001f) {
            std::fill(neuron_output, neuron_output + STATE_DIM, 0.0f);
            continue;
        }
        
        // --- THINKING: Projection Matrix multiplication ---
        // Projected = W_proj * input
        std::array<float, STATE_DIM> projected = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < STATE_DIM; ++r) {
            for (int c = 0; c < STATE_DIM; ++c) {
                projected[r] += current.projection_matrix[r * STATE_DIM + c] * neuron_input[c];
            }
        }

        // Accumulate: state = state + projected_input (residual)
        for (int d = 0; d < STATE_DIM; ++d) {
            neuron_output[d] = ns.h[d] + projected[d];
        }
        
        // Non-linearity
        silu4(neuron_output);
        
        // Update persistent state
        for (int d = 0; d < STATE_DIM; ++d) {
            ns.h[d] = neuron_output[d];
        }
        
        ns.active_step = static_cast<uint16_t>(global_step_);
        ns.flags |= 0x1;  // active flag
        
        // ---- SPARSE PROJECTION TO NEXT GROUP ----
        for (uint32_t k = current.row_ptr[i]; k < current.row_ptr[i + 1]; ++k) {
            uint32_t target = current.col_idx[k];
            float weight = current.weights[k];
            float* target_input = &next_input_buf[target * STATE_DIM];
            
            for (int d = 0; d < STATE_DIM; ++d) {
                target_input[d] += neuron_output[d] * weight;
            }
        }
    }
}

void SparseDynamicNetwork::update_neuron(
    const float* input,
    float* output,
    NeuronState& state,
    const float* weights,
    const uint32_t* targets,
    size_t num_connections
) {
    // Virtual - override for custom neuron types (LSTM, GRU, etc.)
    // Default implementation in process_group above
}

void SparseDynamicNetwork::run_cycle(size_t num_cycles) {
    for (size_t c = 0; c < num_cycles; ++c) {
        for (size_t g = 0; g < NUM_GROUPS; ++g) {
            step(g);
        }
    }
}

} // namespace sparse_nn