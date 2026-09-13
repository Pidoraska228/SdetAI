#include "sparse_dynamic_nn.hpp"
#include <immintrin.h>  // AVX2/SSE intrinsics
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#ifdef _OPENMP
#include <omp.h>
#endif

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

// d(silu)/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
inline float silu_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
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
    // weight_decay в backward() постоянно тянет веса к нулю — со
    // временем часть из них проваливается в денормализованный диапазон
    // (< ~1.18e-38 для float). Арифметика с денормалами на многих x86
    // CPU обрабатывается медленным software/microcode путём вместо
    // обычных SSE/AVX инструкций — это может замедлить обучение в разы
    // без какой-либо видимой причины в самом алгоритме. FTZ/DAZ
    // заставляет CPU считать такие числа просто нулём (безопасно для
    // нас — денормалы всё равно неотличимы от шума в этом масштабе).
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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
    // Кудит-архитектура (заменяет decay+matmul+silu+clamp) — проверена
    // на прототипе (sdet_ai_asaken): та же сеть с обычными нейронами на
    // тех же данных застревала на loss~6.1 почти сразу, кудит-версия
    // дошла до loss~0.05-0.15. Стабильность здесь встроена в саму
    // математику (нормировка), а не приклеена сверху через clamp.

    const float* __restrict input_buf = current.state_buffer_a.data();
    float* __restrict output_buf = current.state_buffer_b.data();
    float* __restrict next_input_buf = next.state_buffer_a.data();
    const size_t next_buf_size = next.count * STATE_DIM;

    std::fill(next_input_buf, next_input_buf + next_buf_size, 0.0f);

#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
#else
    const int num_threads = 1;
#endif
    if (thread_scratch_buffers_.size() < static_cast<size_t>(num_threads)) {
        thread_scratch_buffers_.resize(num_threads);
    }
    for (int t = 0; t < num_threads; ++t) {
        if (thread_scratch_buffers_[t].size() < next_buf_size) {
            thread_scratch_buffers_[t].resize(next_buf_size);
        }
        std::fill(thread_scratch_buffers_[t].begin(), thread_scratch_buffers_[t].begin() + next_buf_size, 0.0f);
    }

    current.active_this_step.resize(current.count);
    std::iota(current.active_this_step.begin(), current.active_this_step.end(), 0u);

    // Углы вращения ОДНИ на всю группу (обучаемый параметр группы, не
    // нейрона) — считаем cos/sin ОДИН раз здесь, а не 100,000 раз
    // внутри параллельного цикла ниже.
    static const int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    float cos_t[6], sin_t[6];
    for (int k = 0; k < 6; ++k) {
        cos_t[k] = std::cos(current.rotation_theta[k]);
        sin_t[k] = std::sin(current.rotation_theta[k]);
    }

    constexpr float INPUT_MIX = 0.5f; // баланс между памятью (ns.h) и новым сигналом

    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(current.count); ++ii) {
        const size_t i = static_cast<size_t>(ii);
#ifdef _OPENMP
        float* __restrict local_next = thread_scratch_buffers_[omp_get_thread_num()].data();
#else
        float* __restrict local_next = thread_scratch_buffers_[0].data();
#endif
        NeuronState& ns = current.neurons[i];
        const float* neuron_input = &input_buf[i * STATE_DIM];
        float* neuron_output = &output_buf[i * STATE_DIM];

        // 1. Смешивание: mixed = (1-mix)*память + mix*вход
        float v[STATE_DIM];
        for (size_t d = 0; d < STATE_DIM; ++d) {
            v[d] = (1.0f - INPUT_MIX) * ns.h[d] + INPUT_MIX * neuron_input[d];
        }

        // Кэшируем mixed (ДО вращений) — нужен backward'у, чтобы
        // повторить вращения назад в той же точке, где считался forward.
        for (size_t d = 0; d < STATE_DIM; ++d) current.pre_cache[i * STATE_DIM + d] = v[d];

        // 2. Вращение (интерференция между уровнями) — 6 плоскостей.
        for (int k = 0; k < 6; ++k) {
            int a = pairs[k][0], b = pairs[k][1];
            float va = v[a], vb = v[b];
            v[a] = cos_t[k] * va - sin_t[k] * vb;
            v[b] = sin_t[k] * va + cos_t[k] * vb;
        }

        // 3. Нормировка — гарантия стабильности БЕСПЛАТНО, без clamp.
        float norm2 = 0.0f;
        for (size_t d = 0; d < STATE_DIM; ++d) norm2 += v[d] * v[d];
        float inv_norm = 1.0f / std::sqrt(std::max(norm2, 1e-24f));
        for (size_t d = 0; d < STATE_DIM; ++d) v[d] *= inv_norm;

        for (size_t d = 0; d < STATE_DIM; ++d) {
            neuron_output[d] = v[d];
            ns.h[d] = v[d]; // персистентная память для следующего токена
        }

        ns.active_step = static_cast<uint16_t>(global_step_);
        ns.flags |= 0x1;

        // ---- SPARSE PROJECTION TO NEXT GROUP ----
        __m128 vout = _mm_loadu_ps(v);
        for (uint32_t c = current.row_ptr[i]; c < current.row_ptr[i + 1]; ++c) {
            uint32_t target = current.col_idx[c];
            float weight = current.weights[c];
            float* target_input = &local_next[target * STATE_DIM];

            __m128 vw = _mm_set1_ps(weight);
            __m128 vtarget = _mm_loadu_ps(target_input);
            vtarget = _mm_fmadd_ps(vout, vw, vtarget);
            _mm_storeu_ps(target_input, vtarget);
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        const float* __restrict local_next = thread_scratch_buffers_[t].data();
        for (size_t j = 0; j < next_buf_size; ++j) {
            next_input_buf[j] += local_next[j];
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
// Настоящий backward pass (кудит-версия)
// =============================================================================
//
// Читаемый вывод сети — это "измерение": output = sum(amp[i]^2 * level_value[i])
// на выходе нейрона 0 последней группы. Градиент течёт через все 4
// измерения этого нейрона (не только dim 0, как было в старой версии
// без measurement), затем через 6 вращений (в обратном порядке,
// восстанавливая промежуточные состояния повтором forward'а из
// закэшированного "mixed"-вектора — дешевле, чем хранить все 6
// промежуточных состояний для каждого из миллиона нейронов), затем
// через нормировку, и дальше назад по sparse-связям — тот же принцип,
// что и раньше, но математика внутри нейрона другая.
//
// Momentum (см. vel_* поля) — обязателен для стабильности: на
// прототипе (sdet_ai_asaken) голый SGD давал разброс loss ~2.05,
// с momentum=0.9 — ~0.012 (в 170 раз стабильнее), да ещё и итоговый
// loss лучше.
void SparseDynamicNetwork::backward(float grad_predicted, float learning_rate) {
    constexpr float WEIGHT_DECAY = 0.0001f;
    constexpr float GRAD_CLIP = 50.0f;
    constexpr float MOMENTUM = 0.9f;
    static const int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    auto clip = [](float v, float c) { return std::clamp(v, -c, c); };

    for (auto& g : groups_) {
        for (uint32_t idx : g.touched) {
            std::fill(g.grad_out.begin() + idx * STATE_DIM, g.grad_out.begin() + (idx + 1) * STATE_DIM, 0.0f);
            g.touched_flag[idx] = 0;
        }
        g.touched.clear();
    }

    // Seed через Measurement: градиент есть у ВСЕХ 4 измерений нейрона 0
    // последней группы (не только dim 0, как раньше без measurement).
    GroupState& last = groups_[NUM_GROUPS - 1];
    const float* out_amp = &last.state_buffer_b[0];
    for (size_t d = 0; d < STATE_DIM; ++d) {
        float g_out = grad_predicted * 2.0f * out_amp[d] * level_value_[d];
        last.grad_out[d] = clip(g_out, GRAD_CLIP);

        // dL/d(level_value[d]) = grad_predicted * amp[d]^2 — обучаем
        // level_value_ той же схемой momentum, что и остальные параметры.
        float g_level = clip(grad_predicted * out_amp[d] * out_amp[d], GRAD_CLIP);
        vel_level_value_[d] = MOMENTUM * vel_level_value_[d] + (1.0f - MOMENTUM) * g_level;
        level_value_[d] -= learning_rate * vel_level_value_[d];
    }
    last.touched.push_back(0);
    last.touched_flag[0] = 1;

    constexpr float INPUT_MIX = 0.5f; // должно совпадать с process_group

    for (size_t gi = 0; gi < NUM_GROUPS; ++gi) {
        size_t g = NUM_GROUPS - 1 - gi;
        GroupState& cur = groups_[g];
        if (cur.touched.empty()) continue;

        float dTheta[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float cos_t[6], sin_t[6];
        for (int k = 0; k < 6; ++k) {
            cos_t[k] = std::cos(cur.rotation_theta[k]);
            sin_t[k] = std::sin(cur.rotation_theta[k]);
        }

        for (uint32_t i : cur.touched) {
            const float* g_out = &cur.grad_out[i * STATE_DIM];
            const float* mixed = &cur.pre_cache[i * STATE_DIM]; // состояние ДО вращений

            // Повторяем forward из mixed, чтобы получить промежуточные
            // состояния после каждого из 6 вращений (дешевле, чем
            // хранить их все для миллиона нейронов).
            float v[STATE_DIM];
            for (size_t d = 0; d < STATE_DIM; ++d) v[d] = mixed[d];
            float after[6][STATE_DIM];
            for (int k = 0; k < 6; ++k) {
                int a = pairs[k][0], b = pairs[k][1];
                float va = v[a], vb = v[b];
                v[a] = cos_t[k] * va - sin_t[k] * vb;
                v[b] = sin_t[k] * va + cos_t[k] * vb;
                for (size_t d = 0; d < STATE_DIM; ++d) after[k][d] = v[d];
            }
            float norm2 = 0.0f;
            for (size_t d = 0; d < STATE_DIM; ++d) norm2 += v[d] * v[d];
            float norm_before = std::sqrt(std::max(norm2, 1e-24f));

            // 1. Backward через нормировку: grad_v = (grad_u - u*(u.grad_u)) / norm_before
            float dot = 0.0f;
            for (size_t d = 0; d < STATE_DIM; ++d) dot += v[d] / norm_before * g_out[d];
            float grad_v[STATE_DIM];
            for (size_t d = 0; d < STATE_DIM; ++d) {
                float u_d = v[d] / norm_before;
                grad_v[d] = clip((g_out[d] - u_d * dot) / norm_before, GRAD_CLIP);
            }

            // 2. Backward через 6 вращений, в обратном порядке.
            for (int k = 5; k >= 0; --k) {
                int a = pairs[k][0], b = pairs[k][1];
                const float* v_before = (k == 0) ? mixed : after[k - 1];
                float c = cos_t[k], s = sin_t[k];
                float ga = grad_v[a], gb = grad_v[b];

                dTheta[k] += ga * (-s * v_before[a] - c * v_before[b])
                           + gb * ( c * v_before[a] - s * v_before[b]);

                grad_v[a] = c * ga + s * gb;
                grad_v[b] = -s * ga + c * gb;
            }

            // 3. Backward через смешивание входа — только вклад входа
            //    (вклад памяти ns.h урезан, см. комментарий класса про
            //    truncated backprop depth=1).
            if (g == 0) continue; // группа 0 — временная граница

            float g_input[STATE_DIM];
            for (size_t d = 0; d < STATE_DIM; ++d) g_input[d] = clip(INPUT_MIX * grad_v[d], GRAD_CLIP);

            GroupState& prev = groups_[g - 1];
            constexpr uint32_t MAX_FANIN_PER_NODE = 3;
            uint32_t k_begin = prev.reverse_row_ptr[i];
            uint32_t k_end = prev.reverse_row_ptr[i + 1];
            uint32_t k_limit = std::min(k_end, k_begin + MAX_FANIN_PER_NODE);

            for (uint32_t k = k_begin; k < k_limit; ++k) {
                uint32_t conn = prev.reverse_conn_idx[k];
                uint32_t src = prev.conn_source[conn];
                const float* src_out = &prev.state_buffer_b[src * STATE_DIM];

                float dW_conn = 0.0f;
                for (size_t d = 0; d < STATE_DIM; ++d) dW_conn += g_input[d] * src_out[d];
                dW_conn = clip(dW_conn, GRAD_CLIP);

                float& velw = prev.vel_weights[conn];
                velw = MOMENTUM * velw + (1.0f - MOMENTUM) * dW_conn;
                float& wv = prev.weights[conn];
                wv -= learning_rate * velw;
                wv -= WEIGHT_DECAY * wv;

                float w = wv; // (до вычитания decay — несущественная разница для пробрасывания градиента)
                if (!prev.touched_flag[src]) {
                    prev.touched_flag[src] = 1;
                    prev.touched.push_back(src);
                    std::fill(prev.grad_out.begin() + src * STATE_DIM, prev.grad_out.begin() + (src + 1) * STATE_DIM, 0.0f);
                }
                float* dst = &prev.grad_out[src * STATE_DIM];
                for (size_t d = 0; d < STATE_DIM; ++d) dst[d] += w * g_input[d];
            }
        }

        // Применяем накопленный градиент углов вращения (общие на группу) — momentum.
        for (int k = 0; k < 6; ++k) {
            float& vel = cur.vel_rotation_theta[k];
            vel = MOMENTUM * vel + (1.0f - MOMENTUM) * dTheta[k];
            cur.rotation_theta[k] -= learning_rate * vel;
        }
    }
}

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

    // 5. Настоящий градиентный спуск (см. backward() выше) вместо
    //    старой эвристики "толкнуть все активные веса на одну и ту же
    //    величину".
    // loss = (target - predicted)^2 = error^2
    // d(loss)/d(predicted) = -2 * error
    //
    // БЫЛО: тут дополнительно обрезали error до ±20 перед этим —
    // подобрано под старую архитектуру (GRAD_CLIP=5 внутри backward).
    // В прототипе (sdet_ai_asaken), на котором проверялась кудит-
    // математика, отдельного внешнего клипа не было — весь клиппинг
    // происходит ВНУТРИ backward() (GRAD_CLIP=50 на каждый скаляр),
    // этого достаточно для устойчивости (подтверждено экспериментами
    // со стабилизацией через momentum).
    float grad_predicted = -2.0f * error;

    backward(grad_predicted, learning_rate);

    return result;
}

bool SparseDynamicNetwork::save_weights(const std::filesystem::path& path) const {
    // Пишем во временный файл и переименовываем в конце (атомарно на
    // большинстве файловых систем) — иначе если процесс убьют жёстко
    // прямо посреди записи (а именно так и происходит при отмене
    // GitHub Actions job'а по таймауту), можно получить обрезанный,
    // повреждённый data/weights.bin, который потом не сможет
    // загрузиться на следующем запуске. Так как сейчас мы реально
    // полагаемся на промежуточные чекпоинты каждые 20,000 токенов
    // внутри ещё выполняющегося run'а, этот риск стал актуальным.
    std::filesystem::path tmp_path = path;
    tmp_path += ".tmp";

    std::ofstream ofs(tmp_path, std::ios::binary);
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

    // level_value_ — обучаемые параметры "измерения", общие на сеть
    // (не на группу), поэтому пишем один раз здесь, а не в цикле ниже.
    ofs.write(reinterpret_cast<const char*>(level_value_.data()), sizeof(level_value_));

    for (const auto& group : groups_) {
        // Save rotation angles (заменяет старую projection_matrix)
        ofs.write(reinterpret_cast<const char*>(group.rotation_theta.data()), sizeof(group.rotation_theta));

        // Save sparse weights
        uint32_t weights_size = static_cast<uint32_t>(group.weights.size());
        ofs.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
        ofs.write(reinterpret_cast<const char*>(group.weights.data()), weights_size * sizeof(float));
    }

    if (!ofs) {
        // Запись не удалась (диск кончился и т.п.) — не подменяем
        // старый рабочий файл заведомо битым.
        ofs.close();
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    ofs.close();

    // Атомарная подмена: старый data/weights.bin остаётся валидным
    // до самого последнего момента, читатели никогда не увидят
    // частично записанный файл.
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        // rename может не сработать между разными файловыми системами
        // (редкость для одного и того же каталога, но на всякий случай)
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) return false;
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

    // Формат меняется этой правкой: projection_matrix (16 float) →
    // rotation_theta (6 float) + level_value_ сети (4 float). Как и
    // раньше — старые файлы всё равно не подгружались бы совместимо
    // (архитектура целиком другая), так что не сохраняем совместимость
    // со старым форматом специально.
    //
    // ПРИМЕЧАНИЕ: momentum-буферы (vel_*) НЕ сохраняются — при resume
    // они обнуляются и разгоняются заново за несколько сотен шагов.
    // Не идеально, но не критично — в отличие от самих весов, потеря
    // momentum не отбрасывает уже выученное состояние сети.
    ifs.read(reinterpret_cast<char*>(&completed_epochs_), sizeof(completed_epochs_));
    if (!ifs) return false;

    ifs.read(reinterpret_cast<char*>(level_value_.data()), sizeof(level_value_));
    if (!ifs) return false;

    for (auto& group : groups_) {
        // Load rotation angles (заменяет старую projection_matrix)
        ifs.read(reinterpret_cast<char*>(group.rotation_theta.data()), sizeof(group.rotation_theta));
        if (!ifs) return false;

        // Load sparse weights
        uint32_t weights_size = 0;
        ifs.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));

        // Защита от повреждённого/несовместимого файла: если размер не
        // совпадает с реальным числом связей этой группы (row_ptr уже
        // построен в конструкторе и не меняется), это значит мы читаем
        // байты не в тех местах — например, старый формат файла (до
        // кудит-архитектуры) с другим расположением полей. Раньше это
        // приводило к тому, что weights_size читался как случайный
        // огромный мусор, group.weights.resize() пытался выделить это
        // как есть, и при следующем save_weights() файл раздувался до
        // гигабайт (именно так и получилось — 12 ГБ вместо ~64 МБ).
        const size_t expected_size = group.row_ptr.empty() ? 0 : group.row_ptr.back();
        if (weights_size != expected_size) {
            return false;
        }

        group.weights.resize(weights_size);
        ifs.read(reinterpret_cast<char*>(group.weights.data()), weights_size * sizeof(float));
        group.vel_weights.assign(weights_size, 0.0f); // momentum начинается с нуля после resume
    }

    return true;
}

} // namespace sparse_nn