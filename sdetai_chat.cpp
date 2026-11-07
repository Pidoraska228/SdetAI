#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
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
    sparse_nn::SparseDynamicNetwork net(0.1f, static_cast<size_t>(vocab_size));
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

    // 4. "Разогреваем" сеть промптом: каждый токен по очереди (forward-only).
    net.reset_state();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        net.forward_token(prompt_tokens[i]);
    }

    // 5. Авторегрессивная генерация: из распределения по словарю берём
    // токен (temperature + top-k), он становится входом следующего шага.
    const float temperature = 0.8f;
    const size_t top_k = 40;
    std::mt19937 rng(std::random_device{}());

    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(num_tokens_to_generate);

    for (int step = 0; step < num_tokens_to_generate; ++step) {
        const auto& logits = net.logits();
        std::vector<int32_t> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        const size_t k = std::min(top_k, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        std::vector<double> w(k);
        const double mx = logits[idx[0]];
        for (size_t j = 0; j < k; ++j) w[j] = std::exp((logits[idx[j]] - mx) / temperature);
        std::discrete_distribution<size_t> pick(w.begin(), w.end());
        const int32_t token_id = idx[pick(rng)];

        generated_tokens.push_back(token_id);
        net.forward_token(token_id);
    }

    std::string generated_text = tokenizer.decode(generated_tokens);

    std::cout << "\n=== Сгенерированные id токенов ===" << std::endl;
    for (int32_t t : generated_tokens) std::cout << t << " ";
    std::cout << std::endl;

    std::cout << "\n=== Декодированный текст ===" << std::endl;
    std::cout << generated_text << std::endl;

    return 0;
}
