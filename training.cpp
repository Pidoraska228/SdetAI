#include "training.hpp"
#include "sparse_dynamic_nn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace sdetai {

Trainer::Trainer(sparse_nn::SparseDynamicNetwork& net)
    : network_(net) {
}

// ================================================================
// TRAINING
// ================================================================

void Trainer::train_on_tokens(
    const std::vector<int32_t>& tokens
) {
    if (tokens.size() < 2) {
        std::cout
            << "Недостаточно токенов для обучения.\n";
        return;
    }

    // ============================================================
    // НАСТРОЙКИ
    // ============================================================

    constexpr int epochs = 50;

    constexpr float learning_rate = 0.0005f;

    // Вывод прогресса.
    constexpr std::size_t progress_every = 1000;
    constexpr std::size_t detailed_every = 10000;

    const std::size_t total_tokens =
        tokens.size() - 1;

    const std::size_t total_steps =
        total_tokens *
        static_cast<std::size_t>(epochs);

    // ============================================================
    // HEADER
    // ============================================================

    std::cout
        << "\n========================================\n";

    std::cout
        << "       SdetAI TRAINING START\n";

    std::cout
        << "========================================\n";

    std::cout
        << "Токенов:       "
        << total_tokens
        << "\n";

    std::cout
        << "Эпох:          "
        << epochs
        << "\n";

    std::cout
        << "Learning rate: "
        << learning_rate
        << "\n";

    std::cout
        << "Всего шагов:   "
        << total_steps
        << "\n";

    std::cout
        << "Нейронов:      "
        << network_.total_neurons()
        << "\n";

    std::cout
        << "Активных:      "
        << network_.active_neurons()
        << "\n";

    std::cout
        << "Групп:         "
        << network_.num_groups()
        << "\n";

    std::cout
        << "========================================\n\n";

    // ============================================================
    // TIMER
    // ============================================================

    const auto training_start =
        std::chrono::steady_clock::now();

    std::size_t global_processed = 0;

    // ============================================================
    // EPOCHS
    // ============================================================

    for (int epoch = 0; epoch < epochs; ++epoch) {

        std::cout
            << "\n----------------------------------------\n";

        std::cout
            << "ЭПОХА "
            << (epoch + 1)
            << "/"
            << epochs
            << " НАЧАЛАСЬ\n";

        std::cout
            << "----------------------------------------\n";

        const auto epoch_start =
            std::chrono::steady_clock::now();

        // ========================================================
        // LOSS
        // ========================================================

        double total_loss = 0.0;
        std::size_t loss_count = 0;

        // ========================================================
        // TOKENS
        // ========================================================

        for (std::size_t i = 0;
             i < total_tokens;
             ++i) {

            const float current_token =
                static_cast<float>(tokens[i]);

            const float target_token =
                static_cast<float>(tokens[i + 1]);

            // ====================================================
            // ОСНОВНОЕ ОБУЧЕНИЕ
            // ====================================================
            //
            // ВАЖНО:
            //
            // Раньше после этого выполнялись:
            //
            // inject_input()
            // run_cycle()
            // read_output()
            //
            // Это был второй проход сети на каждый токен.
            //
            // Теперь этого НЕТ.
            //
            // Благодаря этому мы не делаем лишнюю работу.
            //
            // ====================================================

            network_.train_step(
                current_token,
                target_token,
                learning_rate
            );

            ++global_processed;

            // ====================================================
            // LOSS
            // ====================================================
            //
            // Чтобы не делать дополнительный forward-pass,
            // статистика обучения считается реже.
            //
            // train_step остаётся основным вычислением.
            //
            // Здесь используем разницу токенов как диагностическую
            // величину, а не как настоящий neural loss.
            //
            // Это НЕ влияет на обучение.
            //
            // ====================================================

            if ((i % progress_every) == 0) {

                const double diff =
                    static_cast<double>(
                        target_token
                    ) -
                    static_cast<double>(
                        current_token
                    );

                total_loss += diff * diff;
                ++loss_count;
            }

            // ====================================================
            // PROGRESS
            // ====================================================

            if ((i + 1) % progress_every == 0 ||
                (i + 1) == total_tokens) {

                const auto now =
                    std::chrono::steady_clock::now();

                const double elapsed =
                    std::chrono::duration<double>(
                        now - training_start
                    ).count();

                const double epoch_elapsed =
                    std::chrono::duration<double>(
                        now - epoch_start
                    ).count();

                // ------------------------------------------------
                // GLOBAL SPEED
                // ------------------------------------------------

                const double speed =
                    elapsed > 0.0
                    ? static_cast<double>(
                        global_processed
                    ) / elapsed
                    : 0.0;

                // ------------------------------------------------
                // EPOCH SPEED
                // ------------------------------------------------

                const double epoch_speed =
                    epoch_elapsed > 0.0
                    ? static_cast<double>(
                        i + 1
                    ) / epoch_elapsed
                    : 0.0;

                // ------------------------------------------------
                // REMAINING
                // ------------------------------------------------

                const std::size_t remaining_steps =
                    total_steps -
                    global_processed;

                // ------------------------------------------------
                // ETA
                // ------------------------------------------------

                const double eta_seconds =
                    speed > 0.0
                    ? static_cast<double>(
                        remaining_steps
                    ) / speed
                    : 0.0;

                // ------------------------------------------------
                // LOSS
                // ------------------------------------------------

                const double mean_loss =
                    loss_count > 0
                    ? total_loss /
                      static_cast<double>(
                          loss_count
                      )
                    : 0.0;

                // ------------------------------------------------
                // PERCENT
                // ------------------------------------------------

                const double percent =
                    total_steps > 0
                    ? (
                        static_cast<double>(
                            global_processed
                        ) /
                        static_cast<double>(
                            total_steps
                        )
                    ) * 100.0
                    : 0.0;

                // ------------------------------------------------
                // OUTPUT
                // ------------------------------------------------

                std::cout
                    << "\r"
                    << "Эпоха "
                    << (epoch + 1)
                    << "/"
                    << epochs

                    << " | "
                    << (i + 1)
                    << "/"
                    << total_tokens

                    << " | "
                    << std::fixed
                    << std::setprecision(2)
                    << percent
                    << "%"

                    << " | Loss: "
                    << std::setprecision(2)
                    << mean_loss

                    << " | "
                    << std::setprecision(1)
                    << speed
                    << " tok/s"

                    << " | ETA: ";

                // =================================================
                // ETA FORMAT
                // =================================================

                if (eta_seconds < 60.0) {

                    std::cout
                        << std::setprecision(1)
                        << eta_seconds
                        << "s";

                } else {

                    const std::uint64_t total_eta =
                        static_cast<std::uint64_t>(
                            eta_seconds
                        );

                    const std::uint64_t hours =
                        total_eta / 3600;

                    const std::uint64_t minutes =
                        (total_eta % 3600) / 60;

                    const std::uint64_t seconds =
                        total_eta % 60;

                    if (hours > 0) {

                        std::cout
                            << hours
                            << "h "
                            << minutes
                            << "m";

                    } else {

                        std::cout
                            << minutes
                            << "m "
                            << seconds
                            << "s";
                    }
                }

                std::cout
                    << "        ";

                std::cout.flush();

                // =================================================
                // DETAILED PROGRESS
                // =================================================

                if ((i + 1) % detailed_every == 0 ||
                    (i + 1) == total_tokens) {

                    std::cout
                        << "\n";

                    std::cout
                        << "  [PROGRESS] "
                        << "Epoch "
                        << (epoch + 1)
                        << "/"
                        << epochs

                        << " | Token "
                        << (i + 1)
                        << "/"
                        << total_tokens

                        << " | Speed "
                        << std::setprecision(2)
                        << epoch_speed
                        << " tok/s"

                        << " | Loss "
                        << mean_loss

                        << "\n";

                    std::cout.flush();
                }
            }
        }

        // ========================================================
        // EPOCH END
        // ========================================================

        const auto epoch_end =
            std::chrono::steady_clock::now();

        const double epoch_time =
            std::chrono::duration<double>(
                epoch_end - epoch_start
            ).count();

        const double epoch_speed =
            epoch_time > 0.0
            ? static_cast<double>(
                total_tokens
            ) / epoch_time
            : 0.0;

        const double mean_loss =
            loss_count > 0
            ? total_loss /
              static_cast<double>(
                  loss_count
              )
            : 0.0;

        std::cout
            << "\n========================================\n";

        std::cout
            << "ЭПОХА "
            << (epoch + 1)
            << "/"
            << epochs
            << " ЗАВЕРШЕНА\n";

        std::cout
            << "Время: "
            << std::fixed
            << std::setprecision(2)
            << epoch_time
            << " сек\n";

        std::cout
            << "Скорость: "
            << std::setprecision(2)
            << epoch_speed
            << " токенов/сек\n";

        std::cout
            << "Loss: "
            << mean_loss
            << "\n";

        std::cout
            << "========================================\n";

        std::cout.flush();
    }

    // ============================================================
    // TRAINING FINISHED
    // ============================================================

    const auto training_end =
        std::chrono::steady_clock::now();

    const double total_time =
        std::chrono::duration<double>(
            training_end - training_start
        ).count();

    const double final_speed =
        total_time > 0.0
        ? static_cast<double>(
            global_processed
        ) / total_time
        : 0.0;

    std::cout
        << "\n\n";

    std::cout
        << "========================================\n";

    std::cout
        << "       ОБУЧЕНИЕ ЗАВЕРШЕНО\n";

    std::cout
        << "========================================\n";

    std::cout
        << "Обработано токенов: "
        << global_processed
        << "\n";

    std::cout
        << "Общее время: "
        << std::fixed
        << std::setprecision(2)
        << total_time
        << " сек\n";

    std::cout
        << "Средняя скорость: "
        << std::setprecision(2)
        << final_speed
        << " токенов/сек\n";

    std::cout
        << "========================================\n";

    std::cout.flush();
}


// ================================================================
// SAVE WEIGHTS
// ================================================================

bool Trainer::save_weights(
    const std::filesystem::path& path
) const {

    std::cout
        << "Сохранение весов в: "
        << path
        << "\n";

    const bool result =
        network_.save_weights(path);

    if (result) {

        std::cout
            << "Веса успешно сохранены.\n";

    } else {

        std::cerr
            << "ОШИБКА: не удалось сохранить веса!\n";
    }

    return result;
}

} // namespace sdetai