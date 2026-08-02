#include "low_rank.hpp"
#include <iostream>

namespace sparse_nn {

// =============================================================================
// Randomized SVD: W ≈ A * B via power iteration
// Reference: Halko, Martinsson, Tropp 2009
// =============================================================================
LowRankMatrix LowRankApproximator::approximate(
    const float* W, int m, int n, int rank, int n_iter
) {
    // Algorithm:
    // 1. Sample random Omega (n x rank)
    // 2. Y = W * Omega (m x rank)
    // 3. Power iteration: Y = W * (W^T * Y) ... for n_iter steps
    // 4. Orthonormalize Y (QR): Q (m x rank)
    // 5. B = Q^T * W (rank x n)
    // 6. Return A = Q, B = B  (so W ≈ A * B)
    
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // --- Step 1: random projection Omega (n x rank) ---
    std::vector<float> Omega((size_t)n * rank);
    for (auto& v : Omega) v = dist(rng);
    
    // --- Step 2: Y = W * Omega  (m x rank) ---
    std::vector<float> Y((size_t)m * rank, 0.0f);
    for (int i = 0; i < m; ++i) {
        const float* Wi = &W[(size_t)i * n];
        for (int r = 0; r < rank; ++r) {
            float acc = 0.0f;
            const float* om = &Omega[(size_t)r * n];
            for (int j = 0; j < n; ++j) {
                acc += Wi[j] * om[j];
            }
            Y[(size_t)i * rank + r] = acc;
        }
    }
    
    // --- Step 3: Power iteration (improves singular vector accuracy) ---
    // Y_new = W * (W^T * Y)
    std::vector<float> WT_Y((size_t)n * rank);
    std::vector<float> Y_new((size_t)m * rank);
    for (int it = 0; it < n_iter; ++it) {
        // W^T * Y  (n x rank)
        std::fill(WT_Y.begin(), WT_Y.end(), 0.0f);
        for (int i = 0; i < m; ++i) {
            const float* Wi = &W[(size_t)i * n];
            const float* Yi = &Y[(size_t)i * rank];
            for (int j = 0; j < n; ++j) {
                float wij = Wi[j];
                for (int r = 0; r < rank; ++r) {
                    WT_Y[(size_t)j * rank + r] += wij * Yi[r];
                }
            }
        }
        // Y_new = W * WT_Y (m x rank)
        for (int i = 0; i < m; ++i) {
            const float* Wi = &W[(size_t)i * n];
            for (int r = 0; r < rank; ++r) {
                float acc = 0.0f;
                for (int j = 0; j < n; ++j) {
                    acc += Wi[j] * WT_Y[(size_t)j * rank + r];
                }
                Y_new[(size_t)i * rank + r] = acc;
            }
        }
        Y = Y_new;
    }
    
    // --- Step 4: Modified Gram-Schmidt orthonormalization on Y ---
    // Q (m x rank), orthonormalized columns
    std::vector<float>& Q = Y;  // In-place orthonormalization
    for (int r = 0; r < rank; ++r) {
        // Subtract projections of previous columns
        for (int p = 0; p < r; ++p) {
            float dot = 0.0f;
            for (int i = 0; i < m; ++i) {
                dot += Q[(size_t)i * rank + p] * Q[(size_t)i * rank + r];
            }
            for (int
                i = 0; i < m; ++i) {
                Q[(size_t)i * rank + r] -= dot * Q[(size_t)i * rank + p];
            }
        }
        // Normalize
        float norm = 0.0f;
        for (int i = 0; i < m; ++i) {
            float v = Q[(size_t)i * rank + r];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            float inv = 1.0f / norm;
            for (int i = 0; i < m; ++i) {
                Q[(size_t)i * rank + r] *= inv;
            }
        }
    }
    
    // --- Step 5: B = Q^T * W (rank x n) ---
    std::vector<float> B((size_t)rank * n, 0.0f);
    for (int r = 0; r < rank; ++r) {
        for (int i = 0; i < m; ++i) {
            float q = Q[(size_t)i * rank + r];
            const float* Wi = &W[(size_t)i * n];
            for (int j = 0; j < n; ++j) {
                B[(size_t)r * n + j] += q * Wi[j];
            }
        }
    }
    
    // --- Step 6: Construct LowRankMatrix ---
    LowRankMatrix result(m, n, rank);
    // A = Q (m x rank)
    result.A = Q;
    // B = computed above (rank x n)
    result.B = B;
    // bias = 0
    std::fill(result.bias.begin(), result.bias.end(), 0.0f);
    
    return result;
}

} // namespace sparse_nn