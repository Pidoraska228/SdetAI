#include "quantization.hpp"

namespace sparse_nn {

// All Int8Tensor and Int4Tensor implementations are header-only (inline).
// This file exists for future non-inline implementations and for completeness.

// Potential additions:
// - SIMD-accelerated quantize/dequantize (AVX2/AVX-512)
// - Per-channel asymmetric quantization (zero-point)
// - Group quantization (per 32-element group scales)
// - Dynamic quantization (online calibration)

} // namespace sparse_nn