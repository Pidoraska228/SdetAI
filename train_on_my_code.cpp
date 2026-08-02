#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "bpe_tokenizer.hpp"

namespace fs = std::filesystem;

int main() {
    sdetai::TokenizerConfig config;
    config.vocab_size = 5000; // Достаточно для начала
    sdetai::BPETokenizer tokenizer(config);

    std::vector<fs::path> code_files;
    // Собираем все .cpp и .hpp файлы в текущей папке
    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp") {
            code_files.push_back(entry.path());
        }
    }

    std::cout << "Training on " << code_files.size() << " source files..." << std::endl;
    tokenizer.train_from_files(code_files);
    tokenizer.save("data/my_code_vocab.bin");

    std::cout << "Success! Vocabulary saved to data/my_code_vocab.bin" << std::endl;
    return 0;
}