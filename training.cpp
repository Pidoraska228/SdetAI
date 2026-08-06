#include "training.hpp"
#include "sparse_dynamic_nn.hpp"
#include <iostream>
#include <vector>
#include <cmath>

namespace sdetai {

Trainer::Trainer(sparse_nn::SparseDynamicNetwork& net) : network_(net) {}

void Trainer::train_on_tokens(const std::vector<int32_t>& tokens) {
    if (tokens.size() < 2) {
        std::cout << "Недостаточно токенов для обучения." << std::endl;
        return;
    }

    std::cout << "=== Старт настоящего обучения весов SparseDynamicNetwork ===" << std::endl;
    int epochs = 50;
    float learning_rate = 0.0005f;

    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0.0f;
        size_t count = 0;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            float current_token = static_cast<float>(tokens[i]);
            float target_token = static_cast<float>(tokens[i+1]);

            // Вызываем настоящий метод обучения сети
            network_.train_step(current_token, target_token, learning_rate);

            // Считаем Loss для логов
            float predicted = 0.0f;
            network_.inject_input(&current_token, 1);
            network_.run_cycle(1);
            network_.read_output(&predicted, 1);

            float err = target_token - predicted;
            total_loss += err * err;
            count++;
        }

        float mean_loss = (count > 0) ? (total_loss / count) : 0.0f;
        float perplexity = std::exp(std::min(mean_loss, 10.0f));

        // КАЖДЫЙ РАЗ ПЕЧАТАЕМ ПРОГРЕСС ЭПОХИ В КОНСОЛЬ (чтобы GitHub выводил это в лог)
        std::cout << "[Эпоха " << (e + 1) << "/" << epochs << "] "
                  << "Loss: " << mean_loss
                  << " | Perplexity (PPL): " << perplexity << std::endl;
        std::cout.flush(); // Принудительный сброс буфера
    }

    std::cout << "Обучение весов успешно завершено. Параметры сети реально обновлены." << std::endl;
}

bool Trainer::save_weights(const std::filesystem::path& path) const {
    return network_.save_weights(path);
}

} // namespace sdetai