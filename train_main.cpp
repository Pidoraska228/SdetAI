#include <iostream>
#include <fstream>
#include <string>
#include "bpe_tokenizer.hpp"
#include "json.hpp"

using json = nlohmann::json;

int main() {
    std::string path = "C:/Users/user/Desktop/sdet ai/data/data.jsonl";
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open " << path << std::endl;
        return 1;
    }

    sdetai::BPETokenizer tokenizer;
    std::string line, all_text;
    int count = 0;

    std::cout << "Reading " << path << "..." << std::endl;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            // Пытаемся вытащить что угодно
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.value().is_string()) {
                    all_text += it.value().get<std::string>() + " ";
                }
            }
            count++;
        } catch (...) { continue; }
    }

    std::cout << "Processed " << count << " lines." << std::endl;
    std::cout << "Collected " << all_text.size() << " chars." << std::endl;

    if (all_text.size() > 0) {
        tokenizer.train(all_text);
        tokenizer.save("data/vocab.bin");
        std::cout << "Success! Saved vocab to data/vocab.bin" << std::endl;
    } else {
        std::cout << "No text found to train on." << std::endl;
    }
    return 0;
}