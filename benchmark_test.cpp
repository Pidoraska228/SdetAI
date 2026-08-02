#include <iostream>
#include <vector>
#include <string>
#include "bpe_tokenizer.hpp"

int main() {
    sdetai::BPETokenizer tokenizer;
    // Загружаем обученный словарь
    if (!tokenizer.load("data/vocab.bin")) { 
        std::cerr << "Словарь не найден! Сначала обучись." << std::endl;
        return 1;
    }

    std::string test_code = "public class Agent { void think() { return; } }";
    auto tokens = tokenizer.encode(test_code, true, true);

    std::cout << "--- СТРЕСС-ТЕСТ АГЕНТА ---" << std::endl;
    std::cout << "Код: " << test_code << std::endl;
    std::cout << "Токенов: " << tokens.size() << std::endl;
    std::cout << "Сила агента (точность кодирования): ";
    
    // Если токенов больше 5, значит, он распознал структуры, а не просто буквы
    if (tokens.size() > 5) std::cout << "ВЫСОКАЯ (понимает синтаксис)" << std::endl;
    else std::cout << "НИЗКАЯ (нужно дообучение)" << std::endl;

    return 0;
}