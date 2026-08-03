#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

// Функция для сбора всех .jsonl файлов из папки data
std::string load_all_coding_datasets() {
    std::string combined_text = "";
    int file_count = 0;

    for (const auto& entry : std::filesystem::recursive_directory_iterator("data")) {
        if (entry.path().extension() == ".jsonl") {
            std::cout << "Читаю датасет кода: " << entry.path() << std::endl;
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                combined_text += buffer.str() + "\n";
                file_count++;

                // Ограничим пока для теста, чтобы не забить всю оперативку (например, первые 10 файлов)
                if (file_count >= 10) {
                    std::cout << "Загружено первых 10 файлов датасета для стабильности." << std::endl;
                    break;
                }
            }
        }
    }
    return combined_text;
}