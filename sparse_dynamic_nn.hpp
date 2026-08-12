#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <filesystem>
#include <fstream>

namespace sparse_nn {

constexpr size_t TOTAL_NEURONS = 1'000'000;
constexpr size_t ACTIVE_NEURONS = 100'000;
constexpr size_t NUM_GROUPS = TOTAL_NEURONS / ACTIVE_NEURONS;
constexpr size_t STATE_DIM = 4;
constexpr size_t CACHE_LINE = 64;

struct alignas(CACHE_LINE) NeuronState {
    float h[STATE_DIM];
    uint32_t group_id;
    uint16_t active_step;
    uint16_t flags;
    
    NeuronState() noexcept { clear(); }
    
    void clear() noexcept {
        // Using default member initializers
    }

    float* data() noexcept { return h; }
    const float* data() const noexcept { return h; }
};

static_assert(sizeof(NeuronState) == 64, "NeuronState must be exactly one cache line");

struct alignas(CACHE_LINE) GroupState {
    NeuronState* neurons = nullptr;
    size_t count = 0;
    size_t group_id = 0;
    
    std::vector<float> state_buffer_a;
    std::vector<float> state_buffer_b;
    
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<float> weights;
    std::vector<float> projection_matrix;

    // Индексы нейронов, которые были реально активны (не пропущены
    // по activation_power < eps) на последнем вызове process_group.
    // Заполняется в process_group, используется в train_step, чтобы
    // обновлять веса только тех связей, которые реально сработали —
    // а не всех 1.6M весов группы на каждый токен.
    std::vector<uint32_t> active_this_step;

    // Скретч-буфер для top-k отбора самых значимых нейронов группы
    // (переиспользуется между вызовами, чтобы не аллоцировать на
    // каждый токен). first = activation_power, second = индекс нейрона.
    std::vector<std::pair<float, uint32_t>> activation_scratch;

    GroupState() = default;
    
    GroupState(size_t group_id_, size_t neuron_count, float sparsity = 0.1f)
        : count(neuron_count), group_id(group_id_) {
        
        size_t buf_size = count * STATE_DIM;
        state_buffer_a.resize(buf_size, 0.0f);
        state_buffer_b.resize(buf_size, 0.0f);
        
        active_this_step.reserve(count);
        activation_scratch.reserve(count);

        projection_matrix.resize(STATE_DIM * STATE_DIM, 0.0f);
        for(size_t i = 0; i < STATE_DIM; ++i) {
            projection_matrix[i * STATE_DIM + i] = 1.0f;
        }
        
        init_sparse_connectivity(sparsity);
    }
    
    void init_sparse_connectivity(float sparsity) {
        (void)sparsity;
        constexpr size_t FANOUT = 16;
        
        row_ptr.resize(count + 1);
        row_ptr[0] = 0;
        
        std::mt19937 rng(static_cast<unsigned>(group_id * 12345 + 67890));
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(ACTIVE_NEURONS - 1));
        
        size_t nnz = 0;
        for (size_t i = 0; i < count; ++i) {
            row_ptr[i + 1] = row_ptr[i] + FANOUT;
            nnz += FANOUT;
        }
        
        col_idx.resize(nnz);
        weights.resize(nnz);
        
        std::uniform_real_distribution<float> wdist(-0.1f, 0.1f);
        for (size_t i = 0; i < count; ++i) {
            for (size_t k = 0; k < FANOUT; ++k) {
                col_idx[row_ptr[i] + k] = dist(rng);
                weights[row_ptr[i] + k] = wdist(rng);
            }
        }
    }
    
    float* input_state(size_t i) noexcept { return &state_buffer_a[i * STATE_DIM]; }
    const float* input_state(size_t i) const noexcept { return &state_buffer_a[i * STATE_DIM]; }
    
    float* output_state(size_t i) noexcept { return &state_buffer_b[i * STATE_DIM]; }
    const float* output_state(size_t i) const noexcept { return &state_buffer_b[i * STATE_DIM]; }
    
    void swap_buffers() noexcept { std::swap(state_buffer_a, state_buffer_b); }
};

// Результат одного шага обучения — считается ЗА ОДИН forward pass,
// без повторного прогона сети ради статистики.
struct TrainStepResult {
    float predicted = 0.0f;
    float loss = 0.0f;
};

class SparseDynamicNetwork {
public:
    SparseDynamicNetwork(float sparsity = 0.1f);
    ~SparseDynamicNetwork() = default;
    
    void step(size_t group_idx);
    void run_cycle(size_t num_cycles = 1);
    
    size_t current_group() const noexcept { return current_group_; }
    uint64_t global_step() const noexcept { return global_step_; }
    
    const NeuronState* get_neurons() const noexcept { return neuron_pool_.data(); }
    const GroupState& get_group(size_t idx) const noexcept { return groups_[idx]; }
    
    static constexpr size_t total_neurons() { return TOTAL_NEURONS; }
    static constexpr size_t active_neurons() { return ACTIVE_NEURONS; }
    static constexpr size_t num_groups() { return NUM_GROUPS; }
    static constexpr size_t state_dim() { return STATE_DIM; }
    float sparsity() const { return 0.9f; }
    
    void inject_input(const float* input, size_t input_size);
    void read_output(float* output, size_t output_size) const;

    // Публичные методы обучения и сохранения.
    // Возвращает predicted+loss напрямую — отдельный вызов run_cycle
    // для статистики больше не нужен (был x2 лишней работы).
    TrainStepResult train_step(float input_token, float target_token, float learning_rate);

    bool save_weights(const std::filesystem::path& path) const;
    bool load_weights(const std::filesystem::path& path);

    // --- Состояние прогресса обучения (для checkpoint/resume) ---
    // Хранится вместе с весами, чтобы следующий GitHub Actions run
    // знал, с какой эпохи/токена продолжать.
    uint32_t completed_epochs() const noexcept { return completed_epochs_; }
    void set_completed_epochs(uint32_t e) noexcept { completed_epochs_ = e; }

private:
    std::vector<NeuronState> neuron_pool_;
    std::vector<GroupState> groups_;
    
    size_t current_group_ = 0;
    uint64_t global_step_ = 0;
    uint32_t completed_epochs_ = 0;
    
    void init_neuron_pool();
    void init_groups(float sparsity);
    void process_group(GroupState& current, GroupState& next);

    // Буферы для потоко-безопасного накопления scatter-записи в
    // next_input_buf при параллельной (OpenMP) обработке нейронов —
    // у каждого потока своя копия, без atomic/lock на каждую запись;
    // в конце суммируем в общий next_input_buf. Переиспользуются
    // между вызовами.
    std::vector<std::vector<float>> thread_scratch_buffers_;
    
    virtual void update_neuron(
        const float* input,
        float* output,
        NeuronState& state,
        const float* weights,
        const uint32_t* targets,
        size_t num_connections
    );
};

inline void SparseDynamicNetwork::step(size_t group_idx) {
    GroupState& current = groups_[group_idx];
    GroupState& next = groups_[(group_idx + 1) % NUM_GROUPS];

    // БЫЛО: process_group() разливает (scatter) свежий сигнал текущего
    // токена в next.state_buffer_a — а сразу следом next.swap_buffers()
    // уводил его в state_buffer_b и подставлял на его место СТАРЫЙ
    // output этой же группы с ПРОШЛОГО токена. В итоге каждая группа
    // в пределах одного run_cycle(1) обрабатывала протухший сигнал, а
    // не тот, что реально пришёл от предыдущей группы этим же токеном.
    // Группы обрабатываются последовательно (Gauss-Seidel), а не
    // параллельно — двойная буферизация со свопом тут не нужна вообще:
    // next.state_buffer_a должен остаться ровно тем, что в него
    // разлили, чтобы next прочитал именно это на своём шаге чуть ниже.
    process_group(current, next);

    current_group_ = (group_idx + 1) % NUM_GROUPS;
    ++global_step_;
}

inline void SparseDynamicNetwork::inject_input(const float* input, size_t input_size) {
    GroupState& first = groups_[0];
    size_t copy_size = std::min(input_size, first.count * STATE_DIM);
    std::copy(input, input + copy_size, first.state_buffer_a.begin());
}

inline void SparseDynamicNetwork::read_output(float* output, size_t output_size) const {
    const GroupState& last = groups_[NUM_GROUPS - 1];
    // БЫЛО: читали state_buffer_a — это ВХОД последней группы (то, что
    // в неё прилетело), а не её реальный посчитанный выход. Из-за
    // сочетания с багом в step() это давало ровно 0.0 всегда, что бы
    // сеть ни делала. Настоящий выход группы — state_buffer_b.
    size_t copy_size = std::min(output_size, last.count * STATE_DIM);
    std::copy(last.state_buffer_b.begin(), last.state_buffer_b.begin() + copy_size, output);
}

} // namespace sparse_nn