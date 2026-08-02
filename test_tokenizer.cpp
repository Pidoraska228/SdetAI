#include "bpe_tokenizer.hpp"
#include <iostream>
#include <cassert>

int main() {
    sdetai::TokenizerConfig config;
    config.vocab_size = 100;
    sdetai::BPETokenizer tokenizer(config);

    std::string text = "low lower newest widest";
    tokenizer.train(text);

    std::vector<int32_t> encoded = tokenizer.encode("low", true, true);
    std::cout << "Encoded 'low': ";
    for (int32_t id : encoded) std::cout << id << " ";
    std::cout << std::endl;

    return 0;
}
