#pragma once
#include <vector>
#include <filesystem>
#include <cstdint>

// Предварительное объявление нейросети
namespace sparse_nn {
    class SparseDynamicNetwork;
}

namespace sdetai {

class Trainer {
public:
    explicit Trainer(sparse_nn::SparseDynamicNetwork& net);

    void train_on_tokens(const std::vector<int32_t>& tokens);
    bool save_weights(const std::filesystem::path& path) const;

private:
    sparse_nn::SparseDynamicNetwork& network_;
};

} // namespace sdetai