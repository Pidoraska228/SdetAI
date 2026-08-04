#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "sparse_layer.hpp"

namespace ai {

/**
 * @brief ИИ-движок с эстафетной передачей и адаптивной сложностью
 */
class AIEngine {
public:
    explicit AIEngine(size_t neuronsPerLayer, size_t maxLayers = 4)
        : maxLayers_(maxLayers) {
        if (maxLayers_ == 0) maxLayers_ = 1;
        layers_.reserve(maxLayers_);
        for (size_t i = 0; i < maxLayers_; ++i) {
            layers_.emplace_back(neuronsPerLayer);
        }
        activeLayers_ = 1; // Начинаем с минимальной сложности
    }

    // Загрузить входной вектор в первый слой
    void set_input(const std::vector<float>& input) {
        if (input.empty()) return;
        size_t count = std::min(input.size(), layers_[0].stateCurr.size());
        std::copy(input.begin(), input.begin() + count, layers_[0].stateCurr.begin());
    }

    // Прямой проход с учетом разреженной активации и адаптивной сложности
    std::vector<float> forward() {
        // 1. Адаптивная сложность: определяем глубину сети по энергии входа
        adapt_compute_depth();

        // 2. Эстафетная передача (Sequential Relay) от слоя к слою
        for (size_t l = 0; l < activeLayers_; ++l) {
            auto& cur = layers_[l];
            auto& nxt = (l + 1 < activeLayers_) ? layers_[l + 1] : cur;

            for (size_t i = 0; i < cur.neurons.size(); ++i) {
                // Разреженная активация: мгновенный пропуск спящих нейронов
                if (!cur.is_active(i)) {
                    continue;
                }

                // Эмуляция обработки мультимодального нейрона:
                // Меняем одно из 2-битных состояний на основе текущей активности
                uint8_t mode = cur.neurons[i].state(0);
                uint8_t next_mode = (mode + 1) & 0b11u;
                cur.neurons[i].set_state(0, next_mode);

                // Эстафетная передача вектора состояния дальше по конвейеру
                nxt.stateNext[i] = cur.stateCurr[i] * 1.05f; // простейшее преобразование
            }

            // Ротация буферов эстафеты
            std::swap(cur.stateCurr, cur.stateNext);
            if (l + 1 < activeLayers_) {
                std::swap(nxt.stateNext, nxt.stateCurr);
            }
        }

        // Возвращаем результат последнего активного слоя
        return layers_[activeLayers_ - 1].stateCurr;
    }

    [[nodiscard]] size_t get_active_layers() const noexcept {
        return activeLayers_;
    }

private:
    // Регулятор адаптивной сложности
    void adapt_compute_depth() {
        float energy = 0.0f;
        for (float v : layers_[0].stateCurr) {
            energy += std::abs(v);
        }

        // Простая эвристика: сложный запрос (высокая энергия) -> больше слоев
        if (energy > 15.0f && activeLayers_ < maxLayers_) {
            activeLayers_ = maxLayers_;
        } else if (energy > 7.0f && activeLayers_ < 2) {
            activeLayers_ = 2;
        } else if (energy <= 7.0f) {
            activeLayers_ = 1; // Простой запрос обрабатываем быстро и дешево
        }
    }

    std::vector<SparseLayer> layers_;
    size_t maxLayers_;
    size_t activeLayers_;
};

} // namespace ai