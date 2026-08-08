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
// TRAIN ON TOKENS
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

    // Частота обновления строки прогресса.
    constexpr std::size_t progress_every = 100;

    // Подробный вывод.
    constexpr std::size_t detailed_every = 5000;

    // Prediction теперь делаем редко.
    //
    // Раньше prediction выполнялся КАЖДЫЙ токен:
    //
    // train_step()
    // +
    // inject_input()
    // run_cycle()
    // read_output()
    //
    // Это давало огромную лишнюю нагрузку.
    //
    // Теперь полный дополнительный проход делается
    // только раз в 1000 токенов.
    constexpr std::size_t prediction_every = 1000;

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
        << "       SdetAI FAST CPU TRAINING\n";

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
        << "Prediction:    каждые "
        << prediction_every
        << " токенов\n";

    std::cout
        << "========================================\n\n";

    // ============================================================
    // GLOBAL TIMER
    // ============================================================

    const auto training_start =
        std::chrono::steady_clock::now();

    std::size_t global_processed = 0;

    // ============================================================
    // EPOCHS
    // ============================================================

    for (int epoch = 0;
         epoch < epochs;
         ++epoch) {

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

        // ========================================================
        // LOSS
        // ========================================================

        double total_loss = 0.0;

        std::size_t loss_count = 0;

        // ========================================================
        // EPOCH TIMER
        // ========================================================

        const auto epoch_start =
            std::chrono::steady_clock::now();

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
            // ЭТОТ ВЫЗОВ ПРОИСХОДИТ НА КАЖДОМ ТОКЕНЕ.
            //
            // Мы НЕ уменьшаем количество train_step.
            //

            network_.train_step(
                current_token,
                target_token,
                learning_rate
            );

            ++global_processed;

            // ====================================================
            // ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА
            // ====================================================
            //
            // Prediction больше не выполняется каждый раз.
            //
            // Это было главным тормозом:
            //
            // train_step
            // inject_input
            // run_cycle
            // read_output
            //
            // Теперь дополнительный проход только каждые
            // prediction_every токенов.
            //

            if (
                (i % prediction_every == 0) ||
                (i + 1 == total_tokens)
            ) {

                float predicted = 0.0f;

                network_.inject_input(
                    &current_token,
                    1
                );

                network_.run_cycle(1);

                network_.read_output(
                    &predicted,
                    1
                );

                const float error =
                    target_token - predicted;

                const double loss =
                    static_cast<double>(error) *
                    static_cast<double>(error);

                total_loss += loss;

                ++loss_count;
            }

            // ====================================================
            // PROGRESS
            // ====================================================

            if (
                ((i + 1) % progress_every == 0) ||
                (i + 1 == total_tokens)
            ) {

                const auto now =
                    std::chrono::steady_clock::now();

                // ------------------------------------------------
                // GLOBAL ELAPSED
                // ------------------------------------------------

                const double elapsed =
                    std::chrono::duration<double>(
                        now - training_start
                    ).count();

                // ------------------------------------------------
                // EPOCH ELAPSED
                // ------------------------------------------------

                const double epoch_elapsed =
                    std::chrono::duration<double>(
                        now - epoch_start
                    ).count();

                // ------------------------------------------------
                // SPEED
                // ------------------------------------------------

                const double speed =
                    elapsed > 0.0
                    ? static_cast<double>(
                        global_processed
                      ) / elapsed
                    : 0.0;

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
                // PPL
                // ------------------------------------------------
                //
                // Это диагностическая метрика.
                //
                // Не используется в самом обучении.
                //

                const double perplexity =
                    std::exp(
                        std::min(
                            mean_loss,
                            10.0
                        )
                    );

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
                    << "\r";

                std::cout
                    << "Эпоха "
                    << (epoch + 1)
                    << "/"
                    << epochs;

                std::cout
                    << " | "
                    << (i + 1)
                    << "/"
                    << total_tokens;

                std::cout
                    << " | "
                    << std::fixed
                    << std::setprecision(2)
                    << percent
                    << "%";

                std::cout
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(2)
                    << mean_loss;

                std::cout
                    << " | "
                    << std::fixed
                    << std::setprecision(1)
                    << speed
                    << " tok/s";

                std::cout
                    << " | ETA: ";

                // ------------------------------------------------
                // ETA FORMAT
                // ------------------------------------------------

                if (eta_seconds < 60.0) {

                    std::cout
                        << std::fixed
                        << std::setprecision(1)
                        << eta_seconds
                        << "s";

                } else {

                    const int hours =
                        static_cast<int>(
                            eta_seconds / 3600.0
                        );

                    const int minutes =
                        static_cast<int>(
                            (
                                eta_seconds -
                                static_cast<double>(
                                    hours * 3600
                                )
                            ) / 60.0
                        );

                    const int seconds =
                        static_cast<int>(
                            eta_seconds
                        ) % 60;

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

                if (
                    ((i + 1) % detailed_every == 0) ||
                    (i + 1 == total_tokens)
                ) {

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
                        << std::fixed
                        << std::setprecision(2)
                        << epoch_speed

                        << " tok/s"

                        << " | Loss "
                        << std::fixed
                        << std::setprecision(2)
                        << mean_loss

                        << " | PPL "
                        << std::fixed
                        << std::setprecision(2)
                        << perplexity

                        << "\n";
                }
            }
        }

        // ========================================================
        // END OF EPOCH
        // ========================================================

        const auto epoch_end =
            std::chrono::steady_clock::now();

        const double epoch_time =
            std::chrono::duration<double>(
                epoch_end - epoch_start
            ).count();

        const double mean_loss =
            loss_count > 0
            ? total_loss /
              static_cast<double>(
                  loss_count
              )
            : 0.0;

        const double epoch_speed =
            epoch_time > 0.0
            ? static_cast<double>(
                total_tokens
              ) / epoch_time
            : 0.0;

        // ========================================================
        // EPOCH SUMMARY
        // ========================================================

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
            << std::fixed
            << std::setprecision(2)
            << epoch_speed
            << " токенов/сек\n";

        std::cout
            << "Loss: "
            << std::fixed
            << std::setprecision(2)
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

    // ============================================================
    // FINAL OUTPUT
    // ============================================================

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
        << std::fixed
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