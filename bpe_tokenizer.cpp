#include "bpe_tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

namespace sdetai {

BPETokenizer::BPETokenizer(const TokenizerConfig& config) : config_(config) {
    vocab_[config_.unk_token] = TOKEN_UNK;
    id_to_token_[TOKEN_UNK] = config_.unk_token;
}

std::map<std::string, int32_t> BPETokenizer::get_word_frequencies(const std::string& text) const {
    std::map<std::string, int32_t> freqs;
    std::regex re(config_.pattern);
    auto it = std::sregex_iterator(text.begin(), text.end(), re);
    for (; it != std::sregex_iterator(); ++it) freqs[it->str()]++;
    return freqs;
}

std::vector<std::string> BPETokenizer::split_into_words(const std::string& text) const {
    std::vector<std::string> words;
    std::regex re(config_.pattern);
    auto it = std::sregex_iterator(text.begin(), text.end(), re);
    for (; it != std::sregex_iterator(); ++it) words.push_back(it->str());
    return words;
}

std::vector<std::string> BPETokenizer::word_to_chars(const std::string& word) const {
    std::vector<std::string> chars;
    for (char c : word) chars.push_back(std::string(1, c));
    return chars;
}

void BPETokenizer::train(const std::string& text) {
    auto freqs = get_word_frequencies(text);
    for (auto const& [word, freq] : freqs) {
        if (vocab_.find(word) == vocab_.end()) {
            int32_t id = static_cast<int32_t>(vocab_.size());
            vocab_[word] = id;
            id_to_token_[id] = word;
        }
    }
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text, bool bos, bool eos) const {
    auto words = split_into_words(text);
    std::vector<int32_t> ids;
    if (bos) ids.push_back(TOKEN_BOS);
    for (const auto& w : words) {
        if (vocab_.count(w)) ids.push_back(vocab_.at(w));
        else ids.push_back(TOKEN_UNK);
    }
    if (eos) ids.push_back(TOKEN_EOS);
    return ids;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string res;
    for (int32_t id : tokens) res += id_to_token_.count(id) ? id_to_token_.at(id) : "";
    return res;
}

void BPETokenizer::save(const std::filesystem::path& path) const {
    // БЫЛО: текстовый формат "token id\n", читаемый через ifs >> token >> id.
    // Токенизатор по своему же паттерну (\w+|\s+|[^\w\s]) специально
    // создаёт токены, СОСТОЯЩИЕ из пробелов/переводов строк — такой
    // токен через оператор >> прочитать нельзя в принципе (он сам
    // является разделителем полей), из-за чего после первого же
    // такого токена вся последующая часть файла читалась со сдвигом.
    // Из 32 реальных записей словаря корректно загружались только 4.
    //
    // СТАЛО: бинарный формат с явной длиной токена в байтах — токен
    // может содержать любые символы, включая пробелы/переводы строк/
    // нули, парсинг всегда корректен независимо от содержимого.
    std::ofstream ofs(path, std::ios::binary);
    uint32_t count = static_cast<uint32_t>(vocab_.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (auto const& [token, id] : vocab_) {
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
        uint32_t len = static_cast<uint32_t>(token.size());
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(token.data(), len);
    }
}

bool BPETokenizer::load(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    vocab_.clear();
    id_to_token_.clear();

    uint32_t count = 0;
    if (!ifs.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;

    for (uint32_t i = 0; i < count; ++i) {
        int32_t id = 0;
        uint32_t len = 0;
        if (!ifs.read(reinterpret_cast<char*>(&id), sizeof(id))) return false;
        if (!ifs.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
        std::string token(len, '\0');
        if (len > 0 && !ifs.read(token.data(), len)) return false;
        vocab_[token] = id;
        id_to_token_[id] = token;
    }
    return true;
}

} // namespace sdetai
