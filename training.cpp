#include "training.hpp"
#include "sparse_dynamic_nn.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

namespace sdetai {

Trainer::Trainer(sparse_nn::SparseDynamicNetwork& net) : network_(net) {}

void Trainer::train_on_tokens(
    const std::vector<int32_t>& tokens,
    int epochs_this_run,
    const std::filesystem::path& checkpoint_path,
    size_t checkpoint_every_n,
    double time_budget_seconds
) {
    if (tokens.size() < 2) {
        std::cout << "Недостаточно токенов для обучения." << std::endl;
        return;
    }

    constexpr int TOTAL_EPOCHS = 50; // общий план обучения, для логов

    std::cout << "=== Старт обучения весов SparseDynamicNetwork ===" << std::endl;
    float learning_rate = 0.0005f;

    // Продолжаем с той эпохи, на которой остановились в прошлом
    // запуске (см. load_weights в train_main.cpp), а не с нуля.
    const uint32_t start_epoch = network_.completed_epochs();

    // Общий бюджет времени на ВЕСЬ запуск (не на одну эпоху) — чтобы
    // программа сама аккуратно остановилась и сохранила чекпоинт, не
    // дожидаясь жёсткого убийства процесса воркфлоу по timeout-minutes.
    // Так файл весов никогда не обрывается посреди записи, и не нужно
    // полагаться только на периодический чекпоинт внутри эпохи.
    const auto run_start = std::chrono::steady_clock::now();
    bool time_budget_hit = false;

    for (int local_e = 0; local_e < epochs_this_run && !time_budget_hit; ++local_e) {
        const uint32_t global_epoch = start_epoch + static_cast<uint32_t>(local_e) + 1;

        float total_loss = 0.0f;
        size_t count = 0;

        const auto epoch_start = std::chrono::steady_clock::now();
        size_t tokens_since_report = 0;

        const size_t n = tokens.size() - 1;
        for (size_t i = 0; i < n; ++i) {
            float current_token = static_cast<float>(tokens[i]);
            float target_token = static_cast<float>(tokens[i + 1]);

            // Один forward pass на токен: train_step сам возвращает
            // predicted/loss, второй прогон сети для статистики не
            // нужен (раньше был x2 лишней работы на каждый токен).
            auto result = network_.train_step(current_token, target_token, learning_rate);

            total_loss += result.loss;
            count++;
            tokens_since_report++;

            // Периодический checkpoint внутри эпохи — страховка на
            // случай, если job оборвётся до конца эпохи: прогресс
            // внутри эпохи не теряется, а completed_epochs() всё ещё
            // покажет прошлую завершённую эпоху при следующей загрузке
            // (частичная эпоха просто повторится — это дёшево по
            // сравнению с потерей всего).
            if (checkpoint_every_n > 0 && (i % checkpoint_every_n) == 0 && i > 0) {
                network_.save_weights(checkpoint_path);
            }

            // Проверяем бюджет времени раз в ~1000 токенов (не на
            // каждый — chrono::now() не бесплатен на таком масштабе).
            if (time_budget_seconds > 0.0 && tokens_since_report >= 1000) {
                const double run_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - run_start
                ).count();
                if (run_elapsed >= time_budget_seconds) {
                    std::cout << "Бюджет времени (" << (time_budget_seconds / 60.0)
                              << " мин) исчерпан — аккуратно останавливаемся и"
                              << " сохраняем чекпоинт (эпоха " << global_epoch
                              << " не завершена, продолжим с неё в следующем запуске)."
                              << std::endl;
                    network_.save_weights(checkpoint_path);
                    time_budget_hit = true;
                    break;
                }
            }

            // Прогресс печатаем раз в ~1000 токенов, а не на каждый
            // токен — иначе сам вывод в консоль (flush) становится
            // заметной долей времени на CI.
            if (tokens_since_report >= 1000 || i == n - 1) {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed_s = std::chrono::duration<double>(now - epoch_start).count();
                const double tok_per_s = (elapsed_s > 0.0) ? (static_cast<double>(i + 1) / elapsed_s) : 0.0;
                const double remaining_tokens = static_cast<double>(n - (i + 1));
                const double eta_s = (tok_per_s > 0.0) ? (remaining_tokens / tok_per_s) : 0.0;

                const double pct = 100.0 * static_cast<double>(i + 1) / static_cast<double>(n);
                const double mean_loss_so_far = total_loss / static_cast<double>(count);

                std::cout << "Эпоха " << global_epoch << "/" << TOTAL_EPOCHS
                          << " | " << (i + 1) << "/" << n
                          << " | " << pct << "%"
                          << " | Loss: " << mean_loss_so_far
                          << " | " << tok_per_s << " tok/s"
                          << " | ETA: " << (eta_s / 3600.0) << "h"
                          << std::endl;
                std::cout.flush();

                tokens_since_report = 0;
            }
        }

        if (time_budget_hit) break;

        float mean_loss = (count > 0) ? (total_loss / count) : 0.0f;
        float perplexity = std::exp(std::min(mean_loss, 10.0f));

        std::cout << "[Эпоха " << global_epoch << "/" << TOTAL_EPOCHS << " завершена] "
                  << "Loss: " << mean_loss
                  << " | Perplexity (PPL): " << perplexity << std::endl;

        // Эпоха реально завершена — фиксируем это в состоянии сети
        // и сохраняем веса. Следующий запуск (даже если это будет
        // новый процесс на новой GitHub Actions машине) продолжит
        // именно с этой эпохи, а не с нуля.
        network_.set_completed_epochs(global_epoch);
        network_.save_weights(checkpoint_path);

        if (global_epoch >= TOTAL_EPOCHS) {
            // Метка-маркер, которую воркфлоу ищет в логе, чтобы решить,
            // нужно ли запускать себя ещё раз — простой grep, без
            // парсинга бинарного файла весов.
            std::cout << "SDETAI_TRAINING_COMPLETE: все " << TOTAL_EPOCHS
                      << " эпох пройдены." << std::endl;
            break;
        }
    }

    std::cout << "Запуск обучения завершён. Параметры сети обновлены и сохранены." << std::endl;
}

bool Trainer::save_weights(const std::filesystem::path& path) const {
    return network_.save_weights(path);
}

} // namespace sdetai
