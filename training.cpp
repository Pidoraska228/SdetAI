#include "training.hpp"
#include "sparse_dynamic_nn.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace sdetai {

Trainer::Trainer(sparse_nn::SparseDynamicNetwork& net, TrainerConfig cfg)
    : network_(net), cfg_(cfg) {}

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

    // Защита: id токена за пределами таблицы сети молча превращался бы в
    // <unk> (так обучение "успешно" шло на искажённых данных). Лучше остановиться.
    {
        size_t out_of_range = 0;
        int32_t max_id = 0;
        for (int32_t t : tokens) {
            if (t < 0 || static_cast<size_t>(t) >= network_.vocab_size()) ++out_of_range;
            max_id = std::max(max_id, t);
        }
        if (out_of_range > 0) {
            std::cerr << "ОШИБКА: " << out_of_range << " токенов вне размера словаря сети ("
                      << network_.vocab_size() << ", максимальный id " << max_id
                      << "). Обучение остановлено: размер сети и словаря должны совпадать." << std::endl;
            return;
        }
    }

    constexpr int TOTAL_EPOCHS = 50; // общий план обучения, для логов

    std::cout << "=== Старт обучения весов SparseDynamicNetwork (AdamW, cross-entropy) ===" << std::endl;
    const double baseline = std::log(static_cast<double>(network_.vocab_size()));
    std::cout << "lr=" << cfg_.learning_rate << " wd=" << cfg_.weight_decay
              << " batch=" << cfg_.batch_size << " warmup=" << cfg_.warmup_steps
              << " | словарь " << network_.vocab_size()
              << ", loss случайной модели ln(V)=" << baseline
              << " (PPL=" << network_.vocab_size() << ")" << std::endl;
    const size_t batch = std::max<size_t>(1, cfg_.batch_size);
    uint64_t opt_steps_in_run = 0;
    auto current_lr = [&]() {
        const double w = std::min(1.0, static_cast<double>(opt_steps_in_run + 1) /
                                       static_cast<double>(std::max<size_t>(1, cfg_.warmup_steps)));
        return static_cast<float>(cfg_.learning_rate * w);
    };

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

        // Начало эпохи — чистое рекуррентное состояние сети.
        network_.reset_state();
        double total_loss = 0.0;
        double window_loss = 0.0;
        size_t window_count = 0;
        size_t count = 0;

        const auto epoch_start = std::chrono::steady_clock::now();
        size_t tokens_since_report = 0;

        const size_t n = tokens.size() - 1;
        for (size_t i = 0; i < n; ++i) {
            // Один forward на токен; градиенты копятся, а веса меняет
            // AdamW раз в батч.
            auto result = network_.train_step(tokens[i], tokens[i + 1]);

            total_loss += result.loss;
            window_loss += result.loss;
            window_count++;
            count++;
            tokens_since_report++;

            if (network_.pending_tokens() >= batch) {
                network_.optimizer_step(current_lr(), cfg_.weight_decay);
                ++opt_steps_in_run;
            }

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
                    network_.zero_grad(); // неполный батч отбрасываем
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
                const double recent_loss = window_loss / static_cast<double>(std::max<size_t>(1, window_count));
                window_loss = 0.0;
                window_count = 0;

                std::cout << "Эпоха " << global_epoch << "/" << TOTAL_EPOCHS
                          << " | " << (i + 1) << "/" << n
                          << " | " << pct << "%"
                          << " | Loss(окно): " << recent_loss
                          << " | Loss(эпоха): " << mean_loss_so_far
                          << " | " << tok_per_s << " tok/s"
                          << " | ETA: " << (eta_s / 3600.0) << "h"
                          << std::endl;
                std::cout.flush();

                tokens_since_report = 0;
            }
        }

        if (time_budget_hit) break;

        // Хвост эпохи — неполный батч тоже применяем.
        if (network_.pending_tokens() > 0) {
            network_.optimizer_step(current_lr(), cfg_.weight_decay);
            ++opt_steps_in_run;
        }

        const double mean_loss = (count > 0) ? (total_loss / static_cast<double>(count)) : 0.0;
        const double perplexity = std::exp(mean_loss);

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
