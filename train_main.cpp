#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"
#include "training.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

int main() {
    std::cout << "=== Запуск полного цикла обучения SdetAI ===" << std::endl;

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

    // 2. Обучение токенизатора
    std::cout << "\nШаг 1: Обучение токенизатора..." << std::endl;
    sdetai::BPETokenizer tokenizer;
    tokenizer.train(text_data);

    std::filesystem::create_directories("data");
    tokenizer.save("data/vocab.bin");
    std::cout << "Словарь успешно сохранен в data/vocab.bin" << std::endl;

    // 3. Токенизация
    std::cout << "\nШаг 2: Токенизация текста..." << std::endl;
    auto tokens = tokenizer.encode(text_data);
    std::cout << "Получено токенов: " << tokens.size() << std::endl;

    if (tokens.size() > 25000) {
        tokens.resize(25000);
        std::cout << "Для стабильности обучения взяты первые 25000 токенов." << std::endl;
    }

    // 4. Обучение весов сети
    std::cout << "\nШаг 3: Обучение весов сети (Trainer)..." << std::endl;
    sparse_nn::SparseDynamicNetwork net;
    sdetai::Trainer trainer(net);
    trainer.train_on_tokens(tokens);
    trainer.save_weights("data/weights.bin");

    std::cout << "\n=== Полный цикл завершен успешно! ===" << std::endl;
    return 0;
}