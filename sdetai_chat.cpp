#include <iostream>
#include <vector>
#include <string>
#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"

int main(int argc, char* argv[]) {
    std::string prompt = "привет";
    if (argc > 1) {
        prompt = argv[1];
    }

    // 1. Загружаем токенизатор
    sdetai::BPETokenizer tokenizer;
    if (!tokenizer.load("data/vocab.bin")) {
        if (!tokenizer.load("../data/vocab.bin")) {
            std::cerr << "Словарь vocab.bin не найден!" << std::endl;
            return 1;
        }
    }

    // 2. Кодируем промпт в токены
    auto tokens = tokenizer.encode(prompt);
    std::vector<float> float_tokens(tokens.begin(), tokens.end());

    // 3. Запускаем разреженную сеть
    sparse_nn::SparseDynamicNetwork net;
    if (!float_tokens.empty()) {
        net.inject_input(float_tokens.data(), float_tokens.size());
    }
    net.run_cycle(3);

    // 4. Выводим результат работы сети
    std::cout << "SdetAI (Native C++ Engine) - Обработка успешна!" << std::endl;
    std::cout << "Входной промпт: " << prompt << std::endl;
    std::cout << "Токены (" << tokens.size() << "): ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\nГлобальный шаг нейросети: " << net.global_step() << std::endl;

    return 0;
}
