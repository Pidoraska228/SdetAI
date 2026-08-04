#pragma once
#include <vector>
#include <cstdint>
#include "multimode_neuron.hpp"

namespace ai {

/**
 * @brief Слой с разреженной активацией и эстафетной передачей
 */
struct SparseLayer {
    explicit SparseLayer(size_t neuronCount)
        : neurons(neuronCount),
          sleepFlags(neuronCount, false),  // false = активен, true = спит
          stateCurr(neuronCount, 0.0f),
          stateNext(neuronCount, 0.0f)
    {}

    std::vector<MultiModeNeuron> neurons;    // Packed 2-битные состояния
    std::vector<bool>            sleepFlags; // Флаги "сна" (Sparse Activation)
    std::vector<float>           stateCurr;  // Вектор текущего состояния
    std::vector<float>           stateNext;  // Буфер для эстафетной передачи

    [[nodiscard]] bool is_active(size_t idx) const noexcept {
        return !sleepFlags[idx];
    }

    void set_sleep(size_t idx, bool sleep) noexcept {
        if (idx < sleepFlags.size()) {
            sleepFlags[idx] = sleep;
        }
    }

    void reset() noexcept {
        std::fill(neurons.begin(), neurons.end(), MultiModeNeuron{});
        std::fill(sleepFlags.begin(), sleepFlags.end(), false);
        std::fill(stateCurr.begin(), stateCurr.end(), 0.0f);
        std::fill(stateNext.begin(), stateNext.end(), 0.0f);
    }
};

} // namespace ai