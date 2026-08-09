#include "sparse_dynamic_nn.hpp"
#include <immintrin.h>  // AVX2/SSE intrinsics
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>

namespace sparse_nn {

// =============================================================================
// SIMD-optimized neuron update
// =============================================================================
inline void simd_dense_mul_acc(
    const float* __restrict weights,
    const float* __restrict inputs,
    float* __restrict output,
    size_t rows, size_t cols, size_t nnz
) {
    // CSR SpMV: output[i] += sum_j weights[k] * inputs[col_idx[k]]
    // This is a simplified version - real impl would use proper AVX2 SpMV kernel
    for (size_t i = 0; i < rows; ++i) {
        // Accumulate into output[i * STATE_DIM : (i+1) * STATE_DIM]
        // For now, scalar fallback - replace with AVX2 SpMV kernel
    }
}

// Fast activation: SiLU/Swish approximation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
inline float silu(float x) {
    return x * sigmoid(x);
}

// Vectorized SiLU for 4 floats.
//
// ВАЖНО: _mm256_exp_ps НЕ является аппаратным AVX2-интринсиком — это
// функция из Intel SVML, которой нет ни в GCC/Clang, ни в стандартном
// MSVC без отдельной линковки. На GitHub Actions (ubuntu-latest,
// GCC/Clang) этот код вообще не собирался бы, если бы __AVX2__ был
// определён — а он и не был определён, т.к. CMakeLists не передавал
// -mavx2 для не-Windows сборки (см. правку в CMakeLists.txt).
// Ниже — честная скалярная версия с накидышем инструкции компилятору
// на автовекторизацию (restrict-указатели, простой цикл без ветвлений).
inline void silu4(float* __restrict x) {
    for (int i = 0; i < 4; ++i) {
        x[i] = silu(x[i]);
    }
}

// =============================================================================
// Constructor
// =============================================================================
SparseDynamicNetwork::SparseDynamicNetwork(float sparsity) {
    init_neuron_pool();
    init_groups(sparsity);
}

void SparseDynamicNetwork::init_neuron_pool() {
    neuron_pool_.resize(TOTAL_NEURONS);

    // Assign group IDs
    for (size_t g = 0; g < NUM_GROUPS; ++g) {
        size_t start = g * ACTIVE_NEURONS;
        for (size_t i = 0; i < ACTIVE_NEURONS; ++i) {
            neuron_pool_[start + i].group_id = static_cast<uint32_t>(g);
        }
    }
}

void SparseDynamicNetwork::init_groups(float sparsity) {
    groups_.reserve(NUM_GROUPS);

    for (size_t g = 0; g < NUM_GROUPS; ++g) {
        size_t start = g * ACTIVE_NEURONS;
        NeuronState* group_neurons = &neuron_pool_[start];

        groups_.emplace_back(g, ACTIVE_NEURONS, sparsity);
        groups_.back().neurons = group_neurons;
    }
}

// =============================================================================
// Hot path: process one group
// =============================================================================
void SparseDynamicNetwork::process_group(GroupState& current, GroupState& next) {
    // For each neuron in current group:
    // 1. Read its input state (from previous group's output)
    // 2. Apply neuron update (RNN/LSTM/GRU cell)
    // 3. Write to its persistent state
    // 4. Sparse project to next group's input buffer

    const float* __restrict input_buf = current.state_buffer_a.data();
    float* __restrict output_buf = current.state_buffer_b.data();
    float* __restrict next_input_buf = next.state_buffer_a.data();

    // Clear next group's input buffer
    std::fill(next_input_buf, next_input_buf + next.count * STATE_DIM, 0.0f);

    // Список активных на этом шаге нейронов переиспользуется между
    // вызовами (reserve один раз), чтобы не аллоцировать каждый токен.
    current.active_this_step.clear();

    // Process each neuron
    for (size_t i = 0; i < current.count; ++i) {
        NeuronState& ns = current.neurons[i];
        const float* neuron_input = &input_buf[i * STATE_DIM];
        float* neuron_output = &output_buf[i * STATE_DIM];

        // --- DYNAMIC SPARSITY: Skip inactive neurons ---
        float activation_power = 0.0f;
        for (int d = 0; d < STATE_DIM; ++d) activation_power += std::abs(neuron_input[d]);
        if (activation_power < 0.001f) {
            std::fill(neuron_output, neuron_output + STATE_DIM, 0.0f);
            continue;
        }

        current.active_this_step.push_back(static_cast<uint32_t>(i));

        // --- THINKING: Projection Matrix multiplication ---
        // Projected = W_proj * input
        std::array<float, STATE_DIM> projected = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < STATE_DIM; ++r) {
            for (int c = 0; c < STATE_DIM; ++c) {
                projected[r] += current.projection_matrix[r * STATE_DIM + c] * neuron_input[c];
            }
        }

        // Accumulate: state = state + projected_input (residual)
        for (int d = 0; d < STATE_DIM; ++d) {
            neuron_output[d] = ns.h[d] + projected[d];
        }

        // Non-linearity
        silu4(neuron_output);

        // Update persistent state
        for (int d = 0; d < STATE_DIM; ++d) {
            ns.h[d] = neuron_output[d];
        }

        ns.active_step = static_cast<uint16_t>(global_step_);
        ns.flags |= 0x1;  // active flag

        // ---- SPARSE PROJECTION TO NEXT GROUP ----
        for (uint32_t k = current.row_ptr[i]; k < current.row_ptr[i + 1]; ++k) {
            uint32_t target = current.col_idx[k];
            float weight = current.weights[k];
            float* target_input = &next_input_buf[target * STATE_DIM];

            for (int d = 0; d < STATE_DIM; ++d) {
                target_input[d] += neuron_output[d] * weight;
            }
        }
    }
}

void SparseDynamicNetwork::update_neuron(
    const float* input,
    float* output,
    NeuronState& state,
    const float* weights,
    const uint32_t* targets,
    size_t num_connections
) {
    // Virtual - override for custom neuron types (LSTM, GRU, etc.)
    // Default implementation in process_group above
}

void SparseDynamicNetwork::run_cycle(size_t num_cycles) {
    for (size_t c = 0; c < num_cycles; ++c) {
        for (size_t g = 0; g < NUM_GROUPS; ++g) {
            step(g);
        }
    }
}

// =============================================================================
// Training and serialization
// =============================================================================
TrainStepResult SparseDynamicNetwork::train_step(float input_token, float target_token, float learning_rate) {
    // 1. Inject input token
    inject_input(&input_token, 1);

    // 2. ОДИН forward pass через все группы (было: train_step делал
    //    свой run_cycle, а вызывающий код (training.cpp) делал ЕЩЁ
    //    ОДИН такой же проход ради predicted/loss — x2 работы на
    //    каждый токен. Теперь predicted/loss считаются прямо здесь
    //    и возвращаются наружу, второй проход не нужен вообще.)
    run_cycle(1);

    // 3. Читаем предсказание (выход последней группы)
    float predicted = 0.0f;
    read_output(&predicted, 1);

    // 4. Считаем ошибку
    float error = target_token - predicted;

    TrainStepResult result;
    result.predicted = predicted;
    result.loss = error * error;

    if (std::abs(error) < 1e-6f) return result;

    // 4b. БЫСТРЫЙ ПАТЧ СТАБИЛЬНОСТИ (без изменения архитектуры):
    //
    //    Раньше в шаге 5 использовался "сырой" error напрямую. Target —
    //    это id токена (может быть тысячи), поэтому error тоже мог быть
    //    порядка тысяч. При таком error каждый шаг толкает ВСЕ активные
    //    веса на большую величину в одну сторону — это и есть причина,
    //    почему loss не падал, а рос (5.68e7 → 6.19e7 за эпоху): сеть
    //    расходилась, а не сходилась.
    //
    //    Обрезаем error до разумного диапазона перед использованием в
    //    обновлении весов (сам loss в логах считается ДО обрезки, чтобы
    //    метрика честно показывала реальную ошибку предсказания).
    constexpr float ERROR_CLIP = 20.0f;
    float update_error = std::clamp(error, -ERROR_CLIP, ERROR_CLIP);

    // Небольшой weight decay — веса чуть-чуть "стягиваются" к нулю
    // каждый раз, когда обновляются. Без этого при синхронном толчке
    // всех активных весов в одну сторону (см. ниже) веса могут только
    // накапливать смещение и никогда не уменьшаться обратно.
    constexpr float WEIGHT_DECAY = 0.0001f;

    // 5. Обновляем обучаемые параметры: projection_matrix и sparse weights.
    //
    //    БЫЛО: цикл проходил ВСЕ 1,600,000 весов КАЖДОЙ группы (16M
    //    весов суммарно) на КАЖДЫЙ токен, независимо от того, был ли
    //    нейрон-источник вообще активен на этом шаге. Для 500,000
    //    токенов это ~8 триллионов операций записи за эпоху — и
    //    именно это по порядку величины совпадает с наблюдаемыми
    //    60-180 часами ETA. Обновление "мёртвых" связей (нейрон был
    //    пропущен, activation_power < eps, его выход == 0) к тому же
    //    бессмысленно: вклад такой связи в предсказание в этом шаге
    //    был ровно нулевым.
    //
    //    СТАЛО: обновляем только связи нейронов, которые process_group
    //    реально пометил активными на этом шаге (active_this_step) —
    //    то есть ровно те веса, которые физически влияли на output.
    //    Кол-во нейронов/токенов/качество модели не уменьшается —
    //    меняется только то, что переставшие участвовать в проходе
    //    веса больше не двигаются вслепую.
    for (auto& group : groups_) {
        // Projection matrix маленькая (STATE_DIM x STATE_DIM = 16
        // элементов) — её обновление и так дёшево, оставляем как есть,
        // но с обрезанным error и decay — по тем же причинам, что и ниже.
        for (size_t i = 0; i < group.projection_matrix.size(); ++i) {
            float& v = group.projection_matrix[i];
            v += learning_rate * update_error * 0.01f;
            v -= WEIGHT_DECAY * v;
        }

        // Обновляем sparse weights только активных на этом шаге нейронов
        for (uint32_t i : group.active_this_step) {
            for (uint32_t k = group.row_ptr[i]; k < group.row_ptr[i + 1]; ++k) {
                float& w = group.weights[k];
                w += learning_rate * update_error * 0.001f;
                w -= WEIGHT_DECAY * w;
            }
        }
    }

    return result;
}

bool SparseDynamicNetwork::save_weights(const std::filesystem::path& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    // File header: magic number
    uint32_t magic = 0x53444554; // "SDET"
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    // Number of groups
    uint32_t num_groups = static_cast<uint32_t>(groups_.size());
    ofs.write(reinterpret_cast<const char*>(&num_groups), sizeof(num_groups));

    // Сколько эпох уже пройдено — чтобы следующий запуск (новый
    // процесс, новая случайная сеть по умолчанию) знал, что нужно
    // load_weights() и с какой эпохи продолжать, а не начинать заново.
    ofs.write(reinterpret_cast<const char*>(&completed_epochs_), sizeof(completed_epochs_));

    for (const auto& group : groups_) {
        // Save projection_matrix
        uint32_t proj_size = static_cast<uint32_t>(group.projection_matrix.size());
        ofs.write(reinterpret_cast<const char*>(&proj_size), sizeof(proj_size));
        ofs.write(reinterpret_cast<const char*>(group.projection_matrix.data()), proj_size * sizeof(float));

        // Save sparse weights
        uint32_t weights_size = static_cast<uint32_t>(group.weights.size());
        ofs.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
        ofs.write(reinterpret_cast<const char*>(group.weights.data()), weights_size * sizeof(float));
    }

    return true;
}

bool SparseDynamicNetwork::load_weights(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t magic = 0;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x53444554) return false;

    uint32_t num_groups = 0;
    ifs.read(reinterpret_cast<char*>(&num_groups), sizeof(num_groups));
    if (num_groups != groups_.size()) return false;

    // Формат файла меняется этой правкой (добавилось поле
    // completed_epochs_). Старые data/weights.bin, сохранённые до
    // фикса, всё равно никогда реально не подгружались (load_weights
    // нигде не вызывался — см. train_main.cpp), поэтому сохранять
    // совместимость со старым форматом не нужно: первый запуск с
    // новым кодом просто стартует с нуля и дальше уже честно копит
    // прогресс.
    ifs.read(reinterpret_cast<char*>(&completed_epochs_), sizeof(completed_epochs_));
    if (!ifs) return false;

    for (auto& group : groups_) {
        // Load projection_matrix
        uint32_t proj_size = 0;
        ifs.read(reinterpret_cast<char*>(&proj_size), sizeof(proj_size));
        group.projection_matrix.resize(proj_size);
        ifs.read(reinterpret_cast<char*>(group.projection_matrix.data()), proj_size * sizeof(float));

        // Load sparse weights
        uint32_t weights_size = 0;
        ifs.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
        group.weights.resize(weights_size);
        ifs.read(reinterpret_cast<char*>(group.weights.data()), weights_size * sizeof(float));
    }

    return true;
}

} // namespace sparse_nn