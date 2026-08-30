#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <regex>

namespace sdetai {

constexpr int32_t TOKEN_UNK = 0;
constexpr int32_t TOKEN_PAD = 1;
constexpr int32_t TOKEN_BOS = 2;
constexpr int32_t TOKEN_EOS = 3;
constexpr int32_t TOKEN_SEP = 4;

struct TokenizerConfig {
    int32_t vocab_size = 5000;
    int32_t min_frequency = 2;
    std::string unk_token = "<unk>";
    std::string pad_token = "<pad>";
    std::string bos_token = "<bos>";
    std::string eos_token = "<eos>";
    std::string sep_token = "<sep>";
    std::string pattern = R"(\w+|\s+|[^\w\s])";
    bool lowercase = false;
};

class BPETokenizer {
public:
    BPETokenizer(const TokenizerConfig& config = TokenizerConfig());
    void train(const std::string& text);
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;
    std::string decode(const std::vector<int32_t>& tokens) const;
    size_t size() const { return id_to_token_.size(); }
    void save(const std::filesystem::path& path) const;
    bool load(const std::filesystem::path& path);

private:
    TokenizerConfig config_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;

    std::map<std::string, int32_t> get_word_frequencies(const std::string& text) const;
    std::vector<std::string> split_into_words(const std::string& text) const;
    std::vector<std::string> word_to_chars(const std::string& word) const;
    void finalize_vocab();
};
}