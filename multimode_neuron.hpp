#pragma once
#include <cstdint>
#include <algorithm>

namespace ai {

/**
 * @brief 4-bit-packed neuron state.
 * 4 внутренних 2-битных состояния упакованы в 1 байт (uint8_t).
 */
struct MultiModeNeuron {
    uint8_t packed;   // 4 × 2-bit fields

    constexpr MultiModeNeuron() noexcept : packed(0) {}
    constexpr explicit MultiModeNeuron(uint8_t p) noexcept : packed(p) {}

    // Доступ к отдельным 2-битным полям (индекс 0..3)
    [[nodiscard]] constexpr uint8_t state(size_t idx) const noexcept {
        return (packed >> (idx * 2)) & 0b11u;
    }

    constexpr void set_state(size_t idx, uint8_t value) noexcept {
        value &= 0b11u;
        const uint8_t shift = static_cast<uint8_t>(idx * 2);
        packed = (packed & ~(0b11u << shift)) | (value << shift);
    }

    // Выбор наиболее вероятного (доминирующего) состояния
    [[nodiscard]] size_t dominant_mode() const noexcept {
        size_t best = 0;
        uint8_t bestVal = state(0);
        for (size_t i = 1; i < 4; ++i) {
            uint8_t v = state(i);
            if (v > bestVal) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    constexpr void reset() noexcept { packed = 0; }
};

} // namespace ai