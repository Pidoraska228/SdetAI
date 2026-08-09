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

    // epochs_this_run    — сколько эпох пройти в ЭТОМ запуске (например, 2),
    //                       а не все 50 сразу — под лимит GitHub Actions.
    // checkpoint_path     — куда сохранять веса; используется и для
    //                       периодического checkpoint внутри эпохи.
    // checkpoint_every_n  — сохранять веса каждые N токенов (страховка
    //                       на случай, если запуск оборвётся посреди
    //                       эпохи — например, из-за 6-часового лимита
    //                       job'а). 0 — не сохранять внутри эпохи.
    void train_on_tokens(
        const std::vector<int32_t>& tokens,
        int epochs_this_run,
        const std::filesystem::path& checkpoint_path,
        size_t checkpoint_every_n = 20000
    );

    bool save_weights(const std::filesystem::path& path) const;

private:
    sparse_nn::SparseDynamicNetwork& network_;
};

} // namespace sdetai
