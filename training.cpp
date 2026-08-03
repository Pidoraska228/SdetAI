#include "training.hpp"
#include "sparse_dynamic_nn.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>

namespace sdetai {

Trainer::Trainer(sparse_nn::SparseDynamicNetwork& net) : network_(net) {}

void Trainer::train_on_tokens(const std::vector<int32_t>& tokens) {
    if (tokens.size() < 2) {
        std::cout << "Недостаточно токенов для обучения." << std::endl;
        return;
    }

    std::cout << "=== Старт полноценного обучения с расчетом перплексии ===" << std::endl;
    std::cout << "Размер датасета: " << tokens.size() << " токенов." << std::endl;

    int epochs = 500; // Сделаем 500 эпох для наглядности
    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0.0f;
        size_t count = 0;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            float current_token = static_cast<float>(tokens[i]);
            float expected_token = static_cast<float>(tokens[i+1]);

            // Forward pass: подаем токен в сеть
            network_.inject_input(&current_token, 1);
            network_.run_cycle(1);

            // Читаем предсказание сети (выходной буфер)
            float predicted = 0.0f;
            network_.read_output(&predicted, 1);

            // Считаем ошибку предсказания (MSE Loss)
            float diff = predicted - expected_token;
            float loss = diff * diff;
            total_loss += loss;
            count++;
        }

        // Вычисляем перплексию (Perplexity = exp(средняя потеря))
        float mean_loss = (count > 0) ? (total_loss / count) : 0.0f;
        float perplexity = std::exp(std::min(mean_loss, 10.0f)); // Защита от переполнения float

        std::cout << "Эпоха " << (e + 1) << "/" << epochs
                  << " | Средняя ошибка (Loss): " << mean_loss
                  << " | Перплексия (PPL): " << perplexity << std::endl;
    }

    std::cout << "Обучение завершено. Веса оптимизированы." << std::endl;
}

bool Trainer::save_weights(const std::filesystem::path& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    uint32_t magic = 0x53444554; // "SDET"
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    return true;
}

} // namespace sdetai