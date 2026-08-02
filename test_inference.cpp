#include <iostream>
#include "bpe_tokenizer.hpp"

int main() {
    sdetai::BPETokenizer tokenizer;
    // (Если метод load еще не написан, просто выведи пару токенов напрямую)
    std::cout << "Токенизатор загружен. Проверяем токенизацию строки 'public class Test'..." << std::endl;

    auto tokens = tokenizer.encode("public class Test", true, true);
    for(auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    return 0;
}