#include "sparse_dynamic_nn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sparse_nn {

// ============================================================
// Fast helpers
// ============================================================

namespace {

inline float sigmoid_fast(float x) noexcept
{
    // Защита от переполнения exp()
    if (x < -20.0f)
        return 0.0f;

    if (x > 20.0f)
        return 1.0f;

    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_fast(float x) noexcept
{
    return x * sigmoid_fast(x);
}

inline float clamp_float(float x, float lo, float hi) noexcept
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

} // anonymous namespace


// ============================================================
// GroupState
// ============================================================

GroupState::GroupState()
    : neurons(nullptr),
      count(0),
      group_id(0)
{
    projection_matrix.resize(
        STATE_DIM * STATE_DIM,
        0.0f
    );
}


GroupState::GroupState(
    std::size_t group_id_,
    std::size_t neuron_count,
    float sparsity
)
    : neurons(nullptr),
      count(neuron_count),
      group_id(group_id_)
{
    // --------------------------------------------------------
    // State buffers
    // --------------------------------------------------------

    const std::size_t state_size =
        count * STATE_DIM;

    state_buffer_a.resize(
        state_size,
        0.0f
    );

    state_buffer_b.resize(
        state_size,
        0.0f
    );

    // --------------------------------------------------------
    // Projection matrix
    // --------------------------------------------------------

    projection_matrix.resize(
        STATE_DIM * STATE_DIM
    );

    // Small deterministic initialization.
    //
    // Important:
    // We do NOT initialize with huge random values.
    // Huge values can make the network explode immediately.
    // --------------------------------------------------------

    std::mt19937 rng(
        static_cast<std::uint32_t>(
            0x12345678u +
            static_cast<std::uint32_t>(group_id_)
        )
    );

    std::uniform_real_distribution<float> dist(
        -0.05f,
        0.05f
    );

    for (std::size_t i = 0;
         i < projection_matrix.size();
         ++i)
    {
        projection_matrix[i] = dist(rng);
    }

    // Add a small diagonal residual.
    for (std::size_t i = 0;
         i < STATE_DIM;
         ++i)
    {
        projection_matrix[
            i * STATE_DIM + i
        ] += 0.25f;
    }

    // --------------------------------------------------------
    // Sparse connectivity
    // --------------------------------------------------------

    init_sparse_connectivity(sparsity);
}


void GroupState::init_sparse_connectivity(float sparsity)
{
    if (count == 0)
    {
        row_ptr.clear();
        col_idx.clear();
        weights.clear();
        return;
    }

    // --------------------------------------------------------
    // Clamp sparsity to a safe range.
    //
    // Here sparsity means fraction of possible connections.
    // Example:
    //   0.01 = 1%
    //   0.05 = 5%
    //   0.10 = 10%
    //
    // --------------------------------------------------------

    sparsity = clamp_float(
        sparsity,
        0.001f,
        1.0f
    );

    row_ptr.resize(count + 1);

    // --------------------------------------------------------
    // IMPORTANT PERFORMANCE NOTE
    //
    // A group contains 100,000 neurons.
    //
    // We cannot create:
    //
    // 100000 * 100000
    //
    // connections.
    //
    // That would be billions of connections.
    //
    // Instead we create a fixed small number of connections
    // per neuron.
    // --------------------------------------------------------

    constexpr std::size_t BASE_CONNECTIONS = 8;

    std::size_t connections_per_neuron =
        static_cast<std::size_t>(
            static_cast<float>(BASE_CONNECTIONS) *
            (0.5f + sparsity * 5.0f)
        );

    connections_per_neuron =
        std::max<std::size_t>(
            2,
            connections_per_neuron
        );

    connections_per_neuron =
        std::min<std::size_t>(
            32,
            connections_per_neuron
        );

    // --------------------------------------------------------
    // Total connections
    // --------------------------------------------------------

    const std::size_t total_connections =
        count * connections_per_neuron;

    col_idx.resize(
        total_connections
    );

    weights.resize(
        total_connections
    );

    // --------------------------------------------------------
    // CSR row pointers
    // --------------------------------------------------------

    row_ptr[0] = 0;

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        row_ptr[i + 1] =
            row_ptr[i] +
            static_cast<std::uint32_t>(
                connections_per_neuron
            );
    }

    // --------------------------------------------------------
    // Connectivity initialization
    // --------------------------------------------------------

    std::mt19937 rng(
        static_cast<std::uint32_t>(
            0xA53C9E17u +
            static_cast<std::uint32_t>(
                group_id * 7919
            )
        )
    );

    std::uniform_int_distribution<std::uint32_t>
        neuron_dist(
            0,
            static_cast<std::uint32_t>(
                count - 1
            )
        );

    std::uniform_real_distribution<float>
        weight_dist(
            -0.05f,
            0.05f
        );

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        const std::size_t begin =
            row_ptr[i];

        const std::size_t end =
            row_ptr[i + 1];

        for (std::size_t k = begin;
             k < end;
             ++k)
        {
            col_idx[k] =
                neuron_dist(rng);

            weights[k] =
                weight_dist(rng);
        }
    }
}


// ============================================================
// SparseDynamicNetwork constructor
// ============================================================

SparseDynamicNetwork::SparseDynamicNetwork(
    float sparsity
)
    : current_group_(0),
      global_step_(0),
      sparsity_(sparsity)
{
    sparsity_ = clamp_float(
        sparsity_,
        0.001f,
        1.0f
    );

    init_neuron_pool();

    init_groups(sparsity_);
}


// ============================================================
// Initialize neuron pool
// ============================================================

void SparseDynamicNetwork::init_neuron_pool()
{
    neuron_pool_.resize(
        TOTAL_NEURONS
    );

    // --------------------------------------------------------
    // Assign neurons to groups
    // --------------------------------------------------------

    for (std::size_t g = 0;
         g < NUM_GROUPS;
         ++g)
    {
        const std::size_t start =
            g * ACTIVE_NEURONS;

        const std::size_t end =
            std::min(
                start + ACTIVE_NEURONS,
                TOTAL_NEURONS
            );

        for (std::size_t i = start;
             i < end;
             ++i)
        {
            neuron_pool_[i].group_id =
                static_cast<std::uint32_t>(g);

            neuron_pool_[i].active_step = 0;
            neuron_pool_[i].flags = 0;

            neuron_pool_[i].h[0] = 0.0f;
            neuron_pool_[i].h[1] = 0.0f;
            neuron_pool_[i].h[2] = 0.0f;
            neuron_pool_[i].h[3] = 0.0f;
        }
    }
}


// ============================================================
// Initialize groups
// ============================================================

void SparseDynamicNetwork::init_groups(
    float sparsity
)
{
    groups_.clear();

    groups_.reserve(
        NUM_GROUPS
    );

    for (std::size_t g = 0;
         g < NUM_GROUPS;
         ++g)
    {
        const std::size_t start =
            g * ACTIVE_NEURONS;

        NeuronState* group_neurons =
            neuron_pool_.data() + start;

        groups_.emplace_back(
            g,
            ACTIVE_NEURONS,
            sparsity
        );

        groups_.back().neurons =
            group_neurons;
    }
}


// ============================================================
// Process one group
// ============================================================

void SparseDynamicNetwork::process_group(
    GroupState& current,
    GroupState& next
)
{
    const std::size_t count =
        current.count;

    if (count == 0)
        return;

    const float* input =
        current.state_buffer_a.data();

    float* output =
        current.state_buffer_b.data();

    float* next_input =
        next.state_buffer_a.data();

    // --------------------------------------------------------
    // Clear output and next input.
    // --------------------------------------------------------

    std::fill(
        output,
        output + count * STATE_DIM,
        0.0f
    );

    std::fill(
        next_input,
        next_input + next.count * STATE_DIM,
        0.0f
    );

    // --------------------------------------------------------
    // Process neurons
    // --------------------------------------------------------

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        NeuronState& neuron =
            current.neurons[i];

        const float* neuron_input =
            input + i * STATE_DIM;

        float* neuron_output =
            output + i * STATE_DIM;

        // ----------------------------------------------------
        // Check whether neuron has meaningful input.
        // ----------------------------------------------------

        const float input_energy =
            std::abs(neuron_input[0]) +
            std::abs(neuron_input[1]) +
            std::abs(neuron_input[2]) +
            std::abs(neuron_input[3]);

        if (input_energy < 1e-8f)
        {
            neuron_output[0] = 0.0f;
            neuron_output[1] = 0.0f;
            neuron_output[2] = 0.0f;
            neuron_output[3] = 0.0f;

            continue;
        }

        // ----------------------------------------------------
        // Projection
        // ----------------------------------------------------

        float projected[STATE_DIM] = {
            0.0f,
            0.0f,
            0.0f,
            0.0f
        };

        const float* matrix =
            current.projection_matrix.data();

        for (std::size_t r = 0;
             r < STATE_DIM;
             ++r)
        {
            float sum = 0.0f;

            const std::size_t base =
                r * STATE_DIM;

            for (std::size_t c = 0;
                 c < STATE_DIM;
                 ++c)
            {
                sum +=
                    matrix[base + c] *
                    neuron_input[c];
            }

            projected[r] = sum;
        }

        // ----------------------------------------------------
        // Residual state update
        // ----------------------------------------------------

        for (std::size_t d = 0;
             d < STATE_DIM;
             ++d)
        {
            float value =
                neuron.h[d] +
                projected[d];

            // Prevent runaway activations.
            value =
                clamp_float(
                    value,
                    -10.0f,
                    10.0f
                );

            neuron_output[d] =
                silu_fast(value);
        }

        // ----------------------------------------------------
        // Persistent neuron state
        // ----------------------------------------------------

        neuron.h[0] =
            neuron_output[0];

        neuron.h[1] =
            neuron_output[1];

        neuron.h[2] =
            neuron_output[2];

        neuron.h[3] =
            neuron_output[3];

        neuron.active_step =
            static_cast<std::uint16_t>(
                global_step_ & 0xFFFFu
            );

        neuron.flags |= 1u;

        // ----------------------------------------------------
        // Sparse projection into next group
        // ----------------------------------------------------

        if (i + 1 >= current.row_ptr.size())
            continue;

        const std::uint32_t begin =
            current.row_ptr[i];

        const std::uint32_t end =
            current.row_ptr[i + 1];

        for (std::uint32_t k = begin;
             k < end;
             ++k)
        {
            const std::uint32_t target =
                current.col_idx[k];

            if (target >= next.count)
                continue;

            const float weight =
                current.weights[k];

            float* target_state =
                next_input +
                static_cast<std::size_t>(
                    target
                ) * STATE_DIM;

            target_state[0] +=
                neuron_output[0] * weight;

            target_state[1] +=
                neuron_output[1] * weight;

            target_state[2] +=
                neuron_output[2] * weight;

            target_state[3] +=
                neuron_output[3] * weight;
        }
    }

    // --------------------------------------------------------
    // Keep current output available as next input.
    // --------------------------------------------------------

    current.state_buffer_a.swap(
        current.state_buffer_b
    );
}


// ============================================================
// step
// ============================================================

void SparseDynamicNetwork::step(
    std::size_t group_idx
)
{
    if (groups_.empty())
        return;

    if (group_idx >= groups_.size())
        group_idx %= groups_.size();

    const std::size_t next_idx =
        (group_idx + 1) % groups_.size();

    process_group(
        groups_[group_idx],
        groups_[next_idx]
    );

    current_group_ =
        next_idx;

    ++global_step_;
}


// ============================================================
// run_cycle
// ============================================================

void SparseDynamicNetwork::run_cycle(
    std::size_t num_cycles
)
{
    if (groups_.empty())
        return;

    for (std::size_t c = 0;
         c < num_cycles;
         ++c)
    {
        for (std::size_t g = 0;
             g < groups_.size();
             ++g)
        {
            step(g);
        }
    }
}


// ============================================================
// inject_input
// ============================================================

void SparseDynamicNetwork::inject_input(
    const float* input,
    std::size_t input_size
)
{
    if (input == nullptr ||
        input_size == 0 ||
        groups_.empty())
    {
        return;
    }

    GroupState& first =
        groups_[0];

    const std::size_t available =
        first.state_buffer_a.size();

    const std::size_t n =
        std::min(
            input_size,
            available
        );

    // --------------------------------------------------------
    // Clear previous input.
    // --------------------------------------------------------

    std::fill(
        first.state_buffer_a.begin(),
        first.state_buffer_a.end(),
        0.0f
    );

    // --------------------------------------------------------
    // Put input at the beginning of group 0.
    //
    // For token training input is normally one float.
    // --------------------------------------------------------

    for (std::size_t i = 0;
         i < n;
         ++i)
    {
        first.state_buffer_a[i] =
            input[i];
    }
}


// ============================================================
// read_output
// ============================================================

void SparseDynamicNetwork::read_output(
    float* output,
    std::size_t output_size
) const
{
    if (output == nullptr ||
        output_size == 0 ||
        groups_.empty())
    {
        return;
    }

    const GroupState& last =
        groups_.back();

    const std::size_t available =
        last.state_buffer_a.size();

    const std::size_t n =
        std::min(
            output_size,
            available
        );

    if (n > 0)
    {
        std::memcpy(
            output,
            last.state_buffer_a.data(),
            n * sizeof(float)
        );
    }

    // --------------------------------------------------------
    // Clear anything requested beyond available data.
    // --------------------------------------------------------

    if (output_size > n)
    {
        std::fill(
            output + n,
            output + output_size,
            0.0f
        );
    }
}


// ============================================================
// train_step
// ============================================================

void SparseDynamicNetwork::train_step(
    float input_token,
    float target_token,
    float learning_rate
)
{
    if (groups_.empty())
        return;

    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    inject_input(
        &input_token,
        1
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    run_cycle(1);

    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    float predicted = 0.0f;

    read_output(
        &predicted,
        1
    );

    // --------------------------------------------------------
    // Error
    // --------------------------------------------------------

    float error =
        target_token -
        predicted;

    // Prevent extreme updates.
    error =
        clamp_float(
            error,
            -10.0f,
            10.0f
        );

    // --------------------------------------------------------
    // Learning rate safety
    // --------------------------------------------------------

    learning_rate =
        clamp_float(
            learning_rate,
            0.0f,
            0.1f
        );

    if (learning_rate <= 0.0f)
        return;

    // --------------------------------------------------------
    // Simple online update.
    //
    // This is intentionally lightweight.
    // Full backpropagation through 1,000,000 neurons would
    // be much more expensive.
    // --------------------------------------------------------

    const float projection_update =
        learning_rate *
        error *
        0.0001f;

    const float weight_update =
        learning_rate *
        error *
        0.00001f;

    for (auto& group : groups_)
    {
        // ----------------------------------------------------
        // Projection matrix
        // ----------------------------------------------------

        for (float& w :
             group.projection_matrix)
        {
            w += projection_update;

            w = clamp_float(
                w,
                -2.0f,
                2.0f
            );
        }

        // ----------------------------------------------------
        // Sparse weights
        // ----------------------------------------------------

        for (float& w :
             group.weights)
        {
            w += weight_update;

            w = clamp_float(
                w,
                -1.0f,
                1.0f
            );
        }
    }
}


// ============================================================
// save_weights
// ============================================================

bool SparseDynamicNetwork::save_weights(
    const std::filesystem::path& path
) const
{
    std::ofstream ofs(
        path,
        std::ios::binary
    );

    if (!ofs)
        return false;

    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    constexpr std::uint32_t MAGIC =
        0x53444554u; // SDET

    constexpr std::uint32_t VERSION =
        1u;

    ofs.write(
        reinterpret_cast<const char*>(
            &MAGIC
        ),
        sizeof(MAGIC)
    );

    ofs.write(
        reinterpret_cast<const char*>(
            &VERSION
        ),
        sizeof(VERSION)
    );

    // --------------------------------------------------------
    // Network configuration
    // --------------------------------------------------------

    const std::uint32_t num_groups =
        static_cast<std::uint32_t>(
            groups_.size()
        );

    ofs.write(
        reinterpret_cast<const char*>(
            &num_groups
        ),
        sizeof(num_groups)
    );

    // --------------------------------------------------------
    // Groups
    // --------------------------------------------------------

    for (const auto& group :
         groups_)
    {
        const std::uint64_t
            projection_size =
                static_cast<std::uint64_t>(
                    group.projection_matrix.size()
                );

        ofs.write(
            reinterpret_cast<const char*>(
                &projection_size
            ),
            sizeof(projection_size)
        );

        if (projection_size > 0)
        {
            ofs.write(
                reinterpret_cast<const char*>(
                    group.projection_matrix.data()
                ),
                static_cast<std::streamsize>(
                    projection_size *
                    sizeof(float)
                )
            );
        }

        const std::uint64_t
            weights_size =
                static_cast<std::uint64_t>(
                    group.weights.size()
                );

        ofs.write(
            reinterpret_cast<const char*>(
                &weights_size
            ),
            sizeof(weights_size)
        );

        if (weights_size > 0)
        {
            ofs.write(
                reinterpret_cast<const char*>(
                    group.weights.data()
                ),
                static_cast<std::streamsize>(
                    weights_size *
                    sizeof(float)
                )
            );
        }
    }

    return static_cast<bool>(ofs);
}


// ============================================================
// load_weights
// ============================================================

bool SparseDynamicNetwork::load_weights(
    const std::filesystem::path& path
)
{
    std::ifstream ifs(
        path,
        std::ios::binary
    );

    if (!ifs)
        return false;

    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    std::uint32_t magic = 0;
    std::uint32_t version = 0;

    ifs.read(
        reinterpret_cast<char*>(&magic),
        sizeof(magic)
    );

    ifs.read(
        reinterpret_cast<char*>(&version),
        sizeof(version)
    );

    if (!ifs)
        return false;

    if (magic != 0x53444554u)
        return false;

    if (version != 1u)
        return false;

    // --------------------------------------------------------
    // Number of groups
    // --------------------------------------------------------

    std::uint32_t num_groups = 0;

    ifs.read(
        reinterpret_cast<char*>(
            &num_groups
        ),
        sizeof(num_groups)
    );

    if (!ifs)
        return false;

    if (num_groups !=
        static_cast<std::uint32_t>(
            groups_.size()
        ))
    {
        return false;
    }

    // --------------------------------------------------------
    // Load every group
    // --------------------------------------------------------

    for (auto& group :
         groups_)
    {
        // ----------------------------------------------------
        // Projection matrix
        // ----------------------------------------------------

        std::uint64_t
            projection_size = 0;

        ifs.read(
            reinterpret_cast<char*>(
                &projection_size
            ),
            sizeof(projection_size)
        );

        if (!ifs)
            return false;

        // Safety check.
        if (projection_size >
            1024 * 1024)
        {
            return false;
        }

        group.projection_matrix.resize(
            static_cast<std::size_t>(
                projection_size
            )
        );

        if (projection_size > 0)
        {
            ifs.read(
                reinterpret_cast<char*>(
                    group.projection_matrix.data()
                ),
                static_cast<std::streamsize>(
                    projection_size *
                    sizeof(float)
                )
            );

            if (!ifs)
                return false;
        }

        // ----------------------------------------------------
        // Sparse weights
        // ----------------------------------------------------

        std::uint64_t
            weights_size = 0;

        ifs.read(
            reinterpret_cast<char*>(
                &weights_size
            ),
            sizeof(weights_size)
        );

        if (!ifs)
            return false;

        // Safety limit.
        //
        // Prevent corrupted files from attempting to allocate
        // enormous amounts of memory.
        // ----------------------------------------------------

        constexpr std::uint64_t
            MAX_WEIGHTS =
                100000000ULL;

        if (weights_size >
            MAX_WEIGHTS)
        {
            return false;
        }

        group.weights.resize(
            static_cast<std::size_t>(
                weights_size
            )
        );

        if (weights_size > 0)
        {
            ifs.read(
                reinterpret_cast<char*>(
                    group.weights.data()
                ),
                static_cast<std::streamsize>(
                    weights_size *
                    sizeof(float)
                )
            );

            if (!ifs)
                return false;
        }
    }

    return true;
}

} // namespace sparse_nn