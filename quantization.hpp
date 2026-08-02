#pragma once
// =============================================================================
// Quantization: compress float32 -> int8/int4 with per-channel scales
// Memory savings: 4x (INT8) or 8x (INT4) compared to fp32
// Quality loss: minimal with proper per-channel scaling
// =============================================================================

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace sparse_nn {

// -----------------------------------------------------------------------------
// INT8 Quantization: 32-bit float -> 8-bit integer + per-channel scale
// -----------------------------------------------------------------------------
struct Int8Tensor {
    std::vector<int8_t> data;     // Quantized values
    std::vector<float> scales;   // Per-channel scale: fp32 = int8 * scale
    int rows = 0;
    int cols = 0;
    
    Int8Tensor() = default;
    Int8Tensor(int r, int c) { resize(r, c); }
    
    void resize(int r, int c) {
        rows = r; cols = c;
        data.resize((size_t)r * c);
        scales.resize(r);  // Per-row scale (one scale per output channel)
    }
    
    // Quantize: fp32 -> int8
    void quantize_from(const float* src) {
        constexpr float MAX_INT8 = 127.0f;
        
        for (int r = 0; r < rows; ++r) {
            const float* row = src + (size_t)r * cols;
            
            // Find max abs in this row
            float max_abs = 0.0f;
            for (int c = 0; c < cols; ++c) {
                max_abs = std::max(max_abs, std::abs(row[c]));
            }
            
            // Compute scale: scale = max_abs / 127
            float scale = max_abs > 0 ? max_abs / MAX_INT8 : 0.0f;
            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            scales[r] = scale;
            
            // Quantize
            int8_t* dst_row = &data[(size_t)r * cols];
            for (int c = 0; c < cols; ++c) {
                float q = std::round(row[c] * inv_scale);
                q = std::clamp(q, -128.0f, 127.0f);
                dst_row[c] = (int8_t)q;
            }
        }
    }
    
    // Dequantize: int8 -> fp32
    void dequantize_to(float* dst) const {
        for (int r = 0; r < rows; ++r) {
            const int8_t* src_row = &data[(size_t)r * cols];
            float* dst_row = dst + (size_t)r * cols;
            float scale = scales[r];
            for (int c = 0; c < cols; ++c) {
                dst_row[c] = (float)src_row[c] * scale;
            }
        }
    }
    
    // Quantized matvec: y = W * x  (W quantized, x float, y float)
    // Uses integer arithmetic then rescales per row
    void matvec(const float* __restrict x, float* __restrict y) const {
        for (int r = 0; r < rows; ++r) {
            const int8_t* row = &data[(size_t)r * cols];
            int32_t acc = 0;
            for (int c = 0; c < cols; ++c) {
                acc += (int32_t)row[c] * (int32_t)std::round(x[c] * 127.0f / scales[r]);
                // Note: in practice, x would be quantized too, with separate scale
            }
            y[r] = (float)acc * scales[r] / 127.0f;
        }
    }
    
    size_t bytes() const {
        return data.size() * sizeof(int8_t) + scales.size() * sizeof(float);
    }
    size_t bytes_fp32() const {
        return (size_t)rows * cols * sizeof(float);
    }
    float compression_ratio() const {
        return (float)bytes_fp32() / bytes();
    }
};

// -----------------------------------------------------------------------------
// INT4 Quantization: 32-bit float -> 4-bit integer (packed 2 per byte)
// -----------------------------------------------------------------------------
struct Int4Tensor {
    std::vector<uint8_t> data;   // Packed: 2 int4 values per byte
    std::vector<float> scales;   // Per-channel scale
    int rows = 0;
    int cols = 0;
    
    using int4_val = int8_t;  // int4 stored in int8_t
    
    Int4Tensor() = default;
    Int4Tensor(int r, int c) { resize(r, c); }
    
    void resize(int r, int c) {
        rows = r; cols = c;
        data.resize((size_t)r * c / 2 + 1);  // 2 values per byte + padding
        scales.resize(r);
    }
    
    // Pack two int4 values into one byte
    static inline uint8_t pack(int4_val a, int4_val b) {
        return (uint8_t)(((a & 0x0F) << 4) | (b & 0x0F));
    }
    
    // Unpack two int4 values from one byte
    static inline std::pair<int8_t, int8_t> unpack(uint8_t packed) {
        // int4 range: -8..7 (two's complement)
        int8_t a = (packed >> 4) & 0x0F;
        int8_t b = packed & 0x0F;
        if (a & 0x08) a |= 0xF0;  // Sign-extend
        if (b & 0x08) b |= 0xF0;
        return {a, b};
    }
    
    // Quantize: fp32 -> int4
    void quantize_from(const float* src) {
        constexpr float MAX_INT4 = 7.0f;
        
        for (int r = 0; r < rows; ++r) {
            const float* row = src + (size_t)r * cols;
            
            float max_abs = 0.0f;
            for (int c = 0; c < cols; ++c) {
                max_abs = std::max(max_abs, std::abs(row[c]));
            }
            
            float scale = max_abs > 0 ? max_abs / MAX_INT4 : 0.0f;
            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            scales[r] = scale;
            
            for (int c = 0; c < cols; ++c) {
                float q = std::round(row[c] * inv_scale);
                q = std::clamp(q, -8.0f, 7.0f);
                int8_t qi = (int8_t)q;
                
                size_t idx = ((size_t)r * cols + c) / 2;
                bool high = ((r * cols + c) % 2) == 0;
                if (high) {
                    data[idx] = (data[idx] & 0x0F) | ((qi & 0x0F) << 4);
                } else {
                    data[idx] = (data[idx] & 0xF0) | (qi & 0x0F);
                }
            }
        }
    }
    
    // Dequantize: int4 -> fp32
    void dequantize_to(float* dst) const {
        for (int r = 0; r < rows; ++r) {
            float scale = scales[r];
            for (int c = 0; c < cols; ++c) {
                size_t idx = ((size_t)r * cols + c) / 2;
                bool high = (((size_t)r * cols + c) % 2) == 0;
                int8_t qi = high ? (data[idx] >> 4) : (data[idx] & 0x0F);
                if (qi & 0x08) qi |= 0xF0;  // Sign-extend
                dst[(size_t)r * cols + c] = (float)qi * scale;
            }
        }
    }
    
    size_t bytes() const {
        return data.size() + scales.size() * sizeof(float);
    }
    size_t bytes_fp32() const {
        return (size_t)rows * cols * sizeof(float);
    }
    float compression_ratio() const {
        return (float)bytes_fp32() / bytes();
    }
};
} // namespace sparse_nn