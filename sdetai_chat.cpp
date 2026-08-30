#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"

// ВАЖНО: старая версия этого файла НИЧЕГО не генерировала — она
// создавала свежую случайную сеть (не грузила веса!), заталкивала весь
// промпт одним вызовом inject_input (который принимает только ОДИН
// float на слот, так что реально использовался только первый токен),
// и просто печатала диагностику. Настоящего цикла генерации не было
// вообще. Этот файл — первая версия, которая реально:
// 1. Загружает обученные веса (data/weights.bin)
// 2. Прогоняет промпт токен за токеном, чтобы "разогреть" состояние сети
// 3. Дальше генерирует НОВЫЕ токены один за другим (авторегрессивно):
//    каждое предсказание становится входом для следующего шага
// 4. Декодирует получившиеся токены обратно в текст

int main(int argc, char* argv[]) {
    std::string prompt = "привет";
    int num_tokens_to_generate = 30;
    if (argc > 1) prompt = argv[1];
    if (argc > 2) num_tokens_to_generate = std::max(1, std::atoi(argv[2]));

    // 1. Токенизатор
    sdetai::BPETokenizer tokenizer;
    std::string vocab_path = "data/vocab.bin";
    if (!tokenizer.load(vocab_path)) {
        vocab_path = "../data/vocab.bin";
        if (!tokenizer.load(vocab_path)) {
            std::cerr << "Словарь vocab.bin не найден!" << std::endl;
            return 1;
        }
    }
    const int32_t vocab_size = static_cast<int32_t>(tokenizer.size());
    std::cout << "Словарь загружен: " << vocab_size << " токенов." << std::endl;

    // 2. Сеть — ОБЯЗАТЕЛЬНО грузим обученные веса, иначе это просто
    // случайный шум, а не результат 50 эпох обучения.
    sparse_nn::SparseDynamicNetwork net;
    std::string weights_path = "data/weights.bin";
    bool loaded = net.load_weights(weights_path);
    if (!loaded) {
        weights_path = "../data/weights.bin";
        loaded = net.load_weights(weights_path);
    }
    if (!loaded) {
        std::cerr << "ПРЕДУПРЕЖДЕНИЕ: data/weights.bin не найден — сеть"
                  << " со случайными весами, результат будет бессмысленным."
                  << std::endl;
    } else {
        std::cout << "Веса загружены (" << weights_path << "), пройдено эпох: "
                  << net.completed_epochs() << std::endl;
    }

    // 3. Кодируем промпт
    auto prompt_tokens = tokenizer.encode(prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "Промпт закодировался в 0 токенов." << std::endl;
        return 1;
    }
    std::cout << "Промпт: \"" << prompt << "\" -> " << prompt_tokens.size()
              << " токенов" << std::endl;

    // 4. "Разогреваем" сеть промптом — прогоняем каждый токен промпта
    // через сеть по очереди (forward-only, БЕЗ обучения — просто
    // inject_input+run_cycle, без вызова train_step), чтобы persistent
    // состояние сети отражало контекст промпта перед тем как начнём
    // предсказывать продолжение.
    float last_predicted = 0.0f;
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        float input_token = static_cast<float>(prompt_tokens[i]);
        net.inject_input(&input_token, 1);
        net.run_cycle(1);
        net.read_output(&last_predicted, 1);
    }

    // 5. Авторегрессивная генерация: предсказание становится входом
    // для следующего шага. Округляем и клампим в валидный диапазон id
    // словаря, иначе decode() просто проигнорирует некорректные id
    // (тихо превратит их в пустую строку).
    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(num_tokens_to_generate);

    float current_input = last_predicted;
    for (int step = 0; step < num_tokens_to_generate; ++step) {
        net.inject_input(&current_input, 1);
        net.run_cycle(1);

        float predicted = 0.0f;
        net.read_output(&predicted, 1);

        int32_t token_id = static_cast<int32_t>(std::lround(predicted));
        token_id = std::clamp(token_id, 0, vocab_size - 1);

        generated_tokens.push_back(token_id);
        current_input = static_cast<float>(token_id);
    }

    std::string generated_text = tokenizer.decode(generated_tokens);

    std::cout << "\n=== Сгенерированные id токенов ===" << std::endl;
    for (int32_t t : generated_tokens) std::cout << t << " ";
    std::cout << std::endl;

    std::cout << "\n=== Декодированный текст ===" << std::endl;
    std::cout << generated_text << std::endl;

    return 0;
}
