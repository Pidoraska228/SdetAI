#pragma once
// =============================================================================
// Low-Rank Factorization for neural network weight matrices
// W (m x n) ≈ A (m x r) × B (r x n),  where r << min(m, n)
// Memory savings: m*n  ->  (m+n)*r  (ratio = (m+n)*r / (m*n) = r*(1/m + 1/n))
// Example: 4096x4096 fp32 = 64 MB  ->  rank-256 = (4096+4096)*256*4 = 8 MB  (8x less)
// =============================================================================

#include <vector>
#include <cstdint>
#include <cstddef>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace sparse_nn {

// -----------------------------------------------------------------------------
// LowRankMatrix: stores W as A (m x r) and B (r x n) in fp16
// -----------------------------------------------------------------------------
struct LowRankMatrix {
    int rows = 0;       // m
    int cols = 0;       // n
    int rank = 0;       // r (low-rank approximation)
    
    // Factor A: shape (rows, rank)  — stored row-major
    std::vector<float> A;   // (rows * rank) floats
    
    // Factor B: shape (rank, cols) — stored row-major
    std::vector<float> B;   // (rank * cols) floats
    
    // Optional bias (per output row)
    std::vector<float> bias;  // size = rows
    
    LowRankMatrix() = default;
    LowRankMatrix(int m, int n, int r) { resize(m, n, r); }
    
    void resize(int m, int n, int r) {
        rows = m; cols = n; rank = r;
        A.resize((size_t)m * r);
        B.resize((size_t)r * n);
        bias.assign(m, 0.0f);
    }
    
    // Memory comparison
    size_t bytes_full() const {
        return (size_t)rows * cols * sizeof(float);
    }
    size_t bytes_lowrank() const {
        return ((size_t)rows * rank + (size_t)rank * cols + (size_t)rows) * sizeof(float);
    }
    float compression_ratio() const {
        return (float)bytes_full() / bytes_lowrank();
    }
    
    // Initialize with random Gaussian factors (scaled by 1/sqrt(rank))
    void init_random(unsigned seed = 42, float scale = 0.02f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, scale / std::sqrt((float)rank));
        
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        // Scale A so output variance ~ scale^2
    }
    
    // y = A * B * x + bias   (matrix-vector product)
    // Optimized: first z = B*x (rank flops per output), then y = A*z (rank flops)
    void matvec(const float* __restrict x, float* __restrict y) const {
        // Step 1: z = B * x, z has size = rank
        std::vector<float> z(rank);
        for (int r = 0; r < rank; ++r) {
            const float* Brow = &B[(size_t)r * cols];
            float acc = 0.0f;
            for (int c = 0; c < cols; ++c) {
                acc += Brow[c] * x[c];
            }
            z[r] = acc;
        }
        
        // Step 2: y = A * z + bias
        for (int m = 0; m < rows; ++m) {
            const float* Arow = &A[(size_t)m * rank];
            float acc = bias[m];
            for (int r = 0; r < rank; ++r) {
                acc += Arow[r] * z[r];
            }
            y[m] = acc;
        }
    }
    
    // In-place matvec with accumulation: y += A * B * x
    void matvec_acc(const float* __restrict x, float* __restrict y) const {
        std::vector<float> z(rank);
        for (int r = 0; r < rank; ++r) {
            const float* Brow = &B[(size_t)r * cols];
            float acc = 0.0f;
            for (int c = 0; c < cols; ++c) {
                acc += Brow[c] * x[c];
            }
            z[r] = acc;
        }
        
        for (int m = 0; m < rows; ++m) {
            const float* Arow = &A[(size_t)m * rank];
            float acc = 0.0f;
            for (int r = 0; r < rank; ++r) {
                acc += Arow[r] * z[r];
            }
            y[m] += acc + bias[m];
        }
    }
};

// -----------------------------------------------------------------------------
// SVD-based low-rank approximation (offline, for pre-trained weights)
// -----------------------------------------------------------------------------
// Uses power iteration to approximate top-k singular vectors (no external deps)
class LowRankApproximator {
public:
    // Given full matrix W (m x n), produce LowRankMatrix with given rank
    // Uses randomized SVD: O(m*n*rank) work, good for large matrices
    static LowRankMatrix approximate(const float* W, int m, int n, int rank, int n_iter = 5);
};

} // namespace sparse_nn