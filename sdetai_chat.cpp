#include <iostream>
#include <vector>
#include "bpe_tokenizer.hpp"

int main(int argc, char* argv[]) {
    sdetai::BPETokenizer tokenizer;
    // Используем абсолютный путь, чтобы точно найти словарь
    if (!tokenizer.load("C:\\Users\\user\\Desktop\\sdet ai\\data\\vocab.bin")) {
        std::cerr << "Словарь не найден!" << std::endl;
        return 1;
    }

    std::string prompt = (argc > 1) ? argv[1] : "the";
    auto tokens = tokenizer.encode(prompt);

    std::cout << "Токены для '" << prompt << "': ";
    for(int32_t t : tokens) std::cout << t << " ";
    std::cout << std::endl;

    return 0;
}