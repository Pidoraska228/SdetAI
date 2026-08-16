#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"
#include "training.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>

int main() {
    std::cout << "=== Запуск цикла обучения SdetAI ===" << std::endl;

    // 1. Чтение датасета
    std::string data_path = "data/data.jsonl";
    if (!std::filesystem::exists(data_path)) {
        if (std::filesystem::exists("../data/data.jsonl")) {
            data_path = "../data/data.jsonl";
        } else {
            std::cerr << "Ошибка: Не найден файл данных data/data.jsonl!" << std::endl;
            return 1;
        }
    }

    std::cout << "Чтение данных из " << data_path << "..." << std::endl;
    std::ifstream file(data_path);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл данных!" << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text_data = buffer.str();
    std::cout << "Загружено символов: " << text_data.size() << std::endl;

    std::filesystem::create_directories("data");

    // 2. Токенизатор: обучаем ТОЛЬКО если словаря ещё нет.
    //
    //    ВАЖНО: раньше токенизатор переобучался заново на каждом
    //    запуске. BPE-словарь зависит от данных и не гарантированно
    //    даёт одинаковые id токенов между запусками (тем более если
    //    датасет со временем растёт). Если id токенов "плывут", а
    //    веса сети загружены из чекпоинта прошлого запуска — сеть
    //    обучалась на других id, и продолжение обучения становится
    //    некорректным (тот же id может значить другой кусок текста).
    //    Поэтому словарь фиксируется один раз и переиспользуется.
    sdetai::BPETokenizer tokenizer;
    const std::string vocab_path = "data/vocab.bin";
    if (std::filesystem::exists(vocab_path) && tokenizer.load(vocab_path)) {
        std::cout << "Словарь загружен из " << vocab_path << " (переобучение пропущено)." << std::endl;
    } else {
        std::cout << "\nШаг 1: Обучение токенизатора (первый запуск)..." << std::endl;
        tokenizer.train(text_data);
        tokenizer.save(vocab_path);
        std::cout << "Словарь сохранён в " << vocab_path << std::endl;
    }

    // 3. Токенизация
    std::cout << "\nШаг 2: Токенизация текста..." << std::endl;
    auto tokens = tokenizer.encode(text_data);
    std::cout << "Получено токенов: " << tokens.size() << std::endl;

    if (tokens.size() > 500000) {
        tokens.resize(500000);
        std::cout << "Для стабильности обучения взяты первые 500000 токенов." << std::endl;
    }

    // 4. Сеть: подгружаем чекпоинт прошлого запуска, если он есть.
    //
    //    ВАЖНО: раньше load_weights() не вызывался нигде в проекте —
    //    каждый запуск создавал новую случайную сеть и весь прогресс
    //    предыдущих запусков полностью терялся, несмотря на то что
    //    data/weights.bin коммитился обратно в репозиторий.
    const std::string weights_path = "data/weights.bin";
    sparse_nn::SparseDynamicNetwork net;
    if (std::filesystem::exists(weights_path) && net.load_weights(weights_path)) {
        std::cout << "\nЗагружен чекпоинт " << weights_path
                  << " — пройдено эпох: " << net.completed_epochs() << std::endl;
    } else {
        std::cout << "\nЧекпоинт не найден — старт с новой случайной сети (эпоха 0)." << std::endl;
    }

    // Сколько эпох проходить максимум за этот запуск — реальным
    // ограничителем теперь служит бюджет времени ниже (программа сама
    // аккуратно остановится и сохранится), а не число эпох. Ставим
    // высокий потолок (99), чтобы, если скорость вдруг окажется
    // достаточной для нескольких эпох за один запуск, обучение не
    // останавливалось искусственно раньше времени.
    int epochs_this_run = 99;
    if (const char* env_epochs = std::getenv("SDETAI_EPOCHS_PER_RUN")) {
        epochs_this_run = std::max(1, std::atoi(env_epochs));
    }

    // Бюджет времени на весь запуск (в минутах) — по истечении
    // программа сама аккуратно остановится (с полным, не обрезанным
    // чекпоинтом), не дожидаясь жёсткого убийства процесса воркфлоу по
    // timeout-minutes. По умолчанию 320 минут (5ч20м) — с запасом
    // относительно timeout-minutes: 350 в workflow, чтобы успели
    // отработать шаги commit/upload после завершения программы.
    double time_budget_seconds = 320.0 * 60.0;
    if (const char* env_budget = std::getenv("SDETAI_TIME_BUDGET_MINUTES")) {
        time_budget_seconds = std::max(1.0, std::atof(env_budget)) * 60.0;
    }

    std::cout << "\nШаг 3: Обучение весов сети (Trainer), бюджет времени "
              << (time_budget_seconds / 60.0) << " мин, максимум "
              << epochs_this_run << " эпох в этом запуске..." << std::endl;
    sdetai::Trainer trainer(net);
    trainer.train_on_tokens(
        tokens, epochs_this_run, weights_path,
        /*checkpoint_every_n=*/20000, time_budget_seconds
    );

    std::cout << "\n=== Запуск обучения завершён. Всего пройдено эпох: "
              << net.completed_epochs() << " ===" << std::endl;
    return 0;
}
