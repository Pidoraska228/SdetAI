#include <iostream>
#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"

int main(int argc, char* argv[]) {
    // 1. Загружаем твой обученный словарь
    sdetai::BPETokenizer tokenizer;
    tokenizer.load("data/vocab.bin");

    // 2. Берем вопрос из аргументов
    std::string prompt = (argc > 1) ? argv[1] : "Привет";

    // 3. Токенизируем
    auto tokens = tokenizer.encode(prompt);

    // 4. "Мышление" (Forward pass твоего движка)
    sparse_nn::SparseDynamicNetwork net;
    std::vector<float> float_tokens(tokens.begin(), tokens.end());
    net.inject_input(float_tokens.data(), float_tokens.size());
    net.run_cycle(1); // 1 цикл "мышления"

    // 5. Выводим ID токенов (как результат работы твоего мозга)
    std::cout << "SdetAI Think IDs: ";
    for(auto t : tokens) std::cout << t << " ";
    return 0;
}