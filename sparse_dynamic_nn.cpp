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
SparseDynamicNetwork::SparseDynamicNetwork(float sparsity, size_t vocab_size)
    : vocab_size_(std::max<size_t>(vocab_size, 2)) {
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
    init_token_io();
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
// Токенный ввод/вывод (v2): эмбеддинги -> сеть -> логиты по словарю
// =============================================================================
static_assert(IO_DIM % 8 == 0, "IO_DIM must be a multiple of 8 (AVX2 dot product)");
static_assert(IO_NEURONS <= ACTIVE_NEURONS, "IO_NEURONS must fit into a group");

void SparseDynamicNetwork::init_token_io() {
    emb_.assign(vocab_size_ * IO_DIM, 0.0f);
    head_w_.assign(vocab_size_ * IO_DIM, 0.0f);
    head_b_.assign(vocab_size_, 0.0f);

    std::mt19937 rng(20240601u);
    std::normal_distribution<float> emb_dist(0.0f, 0.5f);    // норма 4-вектора ~ 1
    std::normal_distribution<float> head_dist(0.0f, 0.02f);  // стартовые логиты ~ N(0, 0.16)
    for (auto& x : emb_) x = emb_dist(rng);
    for (auto& x : head_w_) x = head_dist(rng);

    feat_.assign(IO_DIM, 0.0f);
    logits_.assign(vocab_size_, 0.0f);
    probs_.assign(vocab_size_, 1.0f / static_cast<float>(vocab_size_));
}

void SparseDynamicNetwork::ensure_training_buffers() {
    if (training_ready_) return;
    auto init3 = [](std::vector<float>& g, std::vector<float>& m, std::vector<float>& v, size_t n) {
        g.assign(n, 0.0f);
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
    };
    init3(g_emb_, m_emb_, v_emb_, emb_.size());
    init3(g_head_w_, m_head_w_, v_head_w_, head_w_.size());
    init3(g_head_b_, m_head_b_, v_head_b_, head_b_.size());
    for (auto& g : groups_) init3(g.grad_w, g.m_w, g.v_w, g.weights.size());
    training_ready_ = true;
}

void SparseDynamicNetwork::reset_state() {
    for (auto& n : neuron_pool_) {
        for (size_t d = 0; d < STATE_DIM; ++d) n.h[d] = 0.0f;
        n.active_step = 0;
        n.flags = 0;
    }
    for (auto& g : groups_) {
        std::fill(g.state_buffer_a.begin(), g.state_buffer_a.end(), 0.0f);
        std::fill(g.state_buffer_b.begin(), g.state_buffer_b.end(), 0.0f);
    }
}

void SparseDynamicNetwork::forward_token(int32_t token) {
    const size_t t = (token < 0 || static_cast<size_t>(token) >= vocab_size_)
                         ? 0 : static_cast<size_t>(token);

    // Вход: эмбеддинг ДОБАВЛЯЕТСЯ к тому, что в группу 0 уже пришло по
    // кольцу от последней группы (рекуррентность), а не затирает его.
    float* in = groups_[0].state_buffer_a.data();
    const float* e = &emb_[t * IO_DIM];
    for (size_t k = 0; k < IO_DIM; ++k) in[k] += e[k];

    run_cycle(1);

    const GroupState& last = groups_[NUM_GROUPS - 1];
    std::copy(last.state_buffer_b.begin(), last.state_buffer_b.begin() + IO_DIM, feat_.begin());

    // Логиты: logits[v] = b[v] + <W[v], feat>
    const float* f = feat_.data();
    const long long V = static_cast<long long>(vocab_size_);
    #pragma omp parallel for schedule(static) if(V >= 2048)
    for (long long v = 0; v < V; ++v) {
        const float* w = &head_w_[static_cast<size_t>(v) * IO_DIM];
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < IO_DIM; k += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(f + k), acc);
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, acc);
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += lanes[i];
        logits_[static_cast<size_t>(v)] = head_b_[static_cast<size_t>(v)] + sum;
    }

    // Устойчивый softmax
    float mx = logits_[0];
    for (size_t v = 1; v < vocab_size_; ++v) mx = std::max(mx, logits_[v]);
    float denom = 0.0f;
    for (size_t v = 0; v < vocab_size_; ++v) {
        probs_[v] = std::exp(logits_[v] - mx);
        denom += probs_[v];
    }
    const float inv = 1.0f / denom;
    for (size_t v = 0; v < vocab_size_; ++v) probs_[v] *= inv;
}

TrainStepResult SparseDynamicNetwork::train_step(int32_t input_token, int32_t target_token) {
    ensure_training_buffers();
    forward_token(input_token);

    const size_t target = (target_token < 0 || static_cast<size_t>(target_token) >= vocab_size_)
                              ? 0 : static_cast<size_t>(target_token);

    TrainStepResult result;
    result.loss = -std::log(std::max(probs_[target], 1e-12f));
    result.predicted = static_cast<int32_t>(
        std::max_element(logits_.begin(), logits_.end()) - logits_.begin());

    // Backward через выходную проекцию: dL/dlogit[v] = p[v] - 1[v==target]
    float grad_feat[IO_DIM];
    std::fill(grad_feat, grad_feat + IO_DIM, 0.0f);
    const float* f = feat_.data();
    for (size_t v = 0; v < vocab_size_; ++v) {
        const float g = probs_[v] - (v == target ? 1.0f : 0.0f);
        g_head_b_[v] += g;
        float* gw = &g_head_w_[v * IO_DIM];
        const float* w = &head_w_[v * IO_DIM];
        for (size_t k = 0; k < IO_DIM; ++k) {
            gw[k] += g * f[k];
            grad_feat[k] += g * w[k];
        }
    }

    backward(grad_feat, input_token);
    ++pending_tokens_;
    return result;
}

// =============================================================================
// Backward (кудит-версия): градиент НАКАПЛИВАЕТСЯ, веса не меняются
// =============================================================================
//
// Градиент выходных признаков (IO_DIM) приходит от выходной проекции и
// сеет первые IO_NEURONS нейронов последней группы. Дальше он течёт через
// нормировку, 6 вращений (в обратном порядке, с повтором forward'а из
// закэшированного "mixed"), и назад по sparse-связям — как и раньше, но
// теперь результат копится в grad_w / grad_theta / g_emb_, а обновляет веса
// optimizer_step() (AdamW). Группа 0 — временная граница (truncated
// backprop depth=1): из неё градиент уходит только в эмбеддинг токена.
void SparseDynamicNetwork::backward(const float* grad_feat, int32_t input_token) {
    constexpr float GRAD_CLIP = 50.0f;
    static const int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    auto clip = [](float v, float c) { return std::clamp(v, -c, c); };

    const size_t tok = (input_token < 0 || static_cast<size_t>(input_token) >= vocab_size_)
                           ? 0 : static_cast<size_t>(input_token);

    for (auto& g : groups_) {
        for (uint32_t idx : g.touched) {
            std::fill(g.grad_out.begin() + idx * STATE_DIM, g.grad_out.begin() + (idx + 1) * STATE_DIM, 0.0f);
            g.touched_flag[idx] = 0;
        }
        g.touched.clear();
    }

    // Сид: градиент у выходных нейронов последней группы.
    GroupState& last = groups_[NUM_GROUPS - 1];
    for (size_t i = 0; i < IO_NEURONS; ++i) {
        for (size_t d = 0; d < STATE_DIM; ++d) {
            last.grad_out[i * STATE_DIM + d] = clip(grad_feat[i * STATE_DIM + d], GRAD_CLIP);
        }
        last.touched.push_back(static_cast<uint32_t>(i));
        last.touched_flag[i] = 1;
    }

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

            // Нейрон с (почти) нулевым вектором (например, на самом первом
            // токене после reset_state) — нормировка там не дифференцируема
            // осмысленно: деление на ~0 давало бы шум в ±GRAD_CLIP.
            if (norm_before < 1e-6f) continue;

            // 1. Backward через нормировку
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

            // 3. Через смешивание входа: dL/d(вход нейрона) = INPUT_MIX * grad_v
            if (g == 0) {
                // Первые IO_NEURONS нейронов группы 0 получают эмбеддинг токена.
                if (i < IO_NEURONS) {
                    float* ge = &g_emb_[tok * IO_DIM + static_cast<size_t>(i) * STATE_DIM];
                    for (size_t d = 0; d < STATE_DIM; ++d) ge[d] += clip(INPUT_MIX * grad_v[d], GRAD_CLIP);
                }
                continue; // дальше — временная граница
            }

            float g_input[STATE_DIM];
            for (size_t d = 0; d < STATE_DIM; ++d) g_input[d] = clip(INPUT_MIX * grad_v[d], GRAD_CLIP);

            GroupState& prev = groups_[g - 1];
            constexpr uint32_t MAX_FANIN_PER_NODE = SDETAI_BACKWARD_FANIN;
            uint32_t k_begin = prev.reverse_row_ptr[i];
            uint32_t k_end = prev.reverse_row_ptr[i + 1];
            uint32_t k_limit = std::min(k_end, k_begin + MAX_FANIN_PER_NODE);

            for (uint32_t k = k_begin; k < k_limit; ++k) {
                uint32_t conn = prev.reverse_conn_idx[k];
                uint32_t src = prev.conn_source[conn];
                const float* src_out = &prev.state_buffer_b[src * STATE_DIM];

                float dW_conn = 0.0f;
                for (size_t d = 0; d < STATE_DIM; ++d) dW_conn += g_input[d] * src_out[d];
                prev.grad_w[conn] += clip(dW_conn, GRAD_CLIP);

                const float w = prev.weights[conn];
                if (!prev.touched_flag[src]) {
                    prev.touched_flag[src] = 1;
                    prev.touched.push_back(src);
                    std::fill(prev.grad_out.begin() + src * STATE_DIM, prev.grad_out.begin() + (src + 1) * STATE_DIM, 0.0f);
                }
                float* dst = &prev.grad_out[src * STATE_DIM];
                for (size_t d = 0; d < STATE_DIM; ++d) dst[d] += w * g_input[d];
            }
        }

        for (int k = 0; k < 6; ++k) cur.grad_theta[k] += dTheta[k];
    }
}

// =============================================================================
// AdamW
// =============================================================================
static void adamw_update(float* p, const float* g, float* m, float* v, size_t n,
                         float lr, float wd, float inv_batch, float bc1, float bc2) {
    constexpr float B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f;
    const long long N = static_cast<long long>(n);
    #pragma omp parallel for schedule(static) if(N >= 65536)
    for (long long i = 0; i < N; ++i) {
        const float gi = g[i] * inv_batch;
        const float mi = B1 * m[i] + (1.0f - B1) * gi;
        const float vi = B2 * v[i] + (1.0f - B2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i];
        pi -= lr * wd * pi;                                          // decoupled weight decay
        pi -= lr * (mi / bc1) / (std::sqrt(vi / bc2) + EPS);
        p[i] = pi;
    }
}

void SparseDynamicNetwork::optimizer_step(float learning_rate, float weight_decay) {
    if (!training_ready_ || pending_tokens_ == 0) return;

    ++opt_step_;
    const float bc1 = 1.0f - static_cast<float>(std::pow(0.9, static_cast<double>(opt_step_)));
    const float bc2 = 1.0f - static_cast<float>(std::pow(0.999, static_cast<double>(opt_step_)));
    const float inv_b = 1.0f / static_cast<float>(pending_tokens_);

    adamw_update(emb_.data(), g_emb_.data(), m_emb_.data(), v_emb_.data(), emb_.size(),
                 learning_rate, weight_decay, inv_b, bc1, bc2);
    adamw_update(head_w_.data(), g_head_w_.data(), m_head_w_.data(), v_head_w_.data(), head_w_.size(),
                 learning_rate, weight_decay, inv_b, bc1, bc2);
    adamw_update(head_b_.data(), g_head_b_.data(), m_head_b_.data(), v_head_b_.data(), head_b_.size(),
                 learning_rate, 0.0f, inv_b, bc1, bc2);   // bias — без weight decay
    for (auto& g : groups_) {
        adamw_update(g.weights.data(), g.grad_w.data(), g.m_w.data(), g.v_w.data(), g.weights.size(),
                     learning_rate, weight_decay, inv_b, bc1, bc2);
        adamw_update(g.rotation_theta.data(), g.grad_theta.data(), g.m_theta.data(), g.v_theta.data(), 6,
                     learning_rate, 0.0f, inv_b, bc1, bc2); // углы — без weight decay
    }

    zero_grad();
}

void SparseDynamicNetwork::zero_grad() {
    if (!training_ready_) { pending_tokens_ = 0; return; }
    std::fill(g_emb_.begin(), g_emb_.end(), 0.0f);
    std::fill(g_head_w_.begin(), g_head_w_.end(), 0.0f);
    std::fill(g_head_b_.begin(), g_head_b_.end(), 0.0f);
    for (auto& g : groups_) {
        std::fill(g.grad_w.begin(), g.grad_w.end(), 0.0f);
        g.grad_theta.fill(0.0f);
    }
    pending_tokens_ = 0;
}

// =============================================================================
// Сохранение / загрузка (формат v2; старые файлы с float-регрессией по id
// токена несовместимы по архитектуре и не загружаются)
// =============================================================================
namespace {
constexpr uint32_t WEIGHTS_MAGIC = 0x53444554; // "SDET"
constexpr uint32_t WEIGHTS_VERSION = 2;

template <typename T>
void write_raw(std::ofstream& ofs, const T* data, size_t count) {
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}
template <typename T>
bool read_raw(std::ifstream& ifs, T* data, size_t count) {
    ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(ifs);
}
} // namespace

bool SparseDynamicNetwork::save_weights(const std::filesystem::path& path) const {
    // Пишем во временный файл и переименовываем — файл весов никогда не
    // обрывается посреди записи (важно при убийстве job'а по таймауту).
    std::filesystem::path tmp_path = path;
    tmp_path += ".tmp";

    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs) return false;

    const uint32_t magic = WEIGHTS_MAGIC;
    const uint32_t version = WEIGHTS_VERSION;
    const uint32_t num_groups = static_cast<uint32_t>(groups_.size());
    const uint32_t vocab = static_cast<uint32_t>(vocab_size_);
    const uint32_t io_neurons = static_cast<uint32_t>(IO_NEURONS);
    write_raw(ofs, &magic, 1);
    write_raw(ofs, &version, 1);
    write_raw(ofs, &num_groups, 1);
    write_raw(ofs, &vocab, 1);
    write_raw(ofs, &io_neurons, 1);
    write_raw(ofs, &completed_epochs_, 1);

    write_raw(ofs, emb_.data(), emb_.size());
    write_raw(ofs, head_w_.data(), head_w_.size());
    write_raw(ofs, head_b_.data(), head_b_.size());

    for (const auto& group : groups_) {
        write_raw(ofs, group.rotation_theta.data(), group.rotation_theta.size());
        const uint32_t weights_size = static_cast<uint32_t>(group.weights.size());
        write_raw(ofs, &weights_size, 1);
        write_raw(ofs, group.weights.data(), group.weights.size());
    }

    if (!ofs) {
        ofs.close();
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    ofs.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) return false;
    }
    return true;
}

bool SparseDynamicNetwork::load_weights(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t magic = 0, version = 0, num_groups = 0, vocab = 0, io_neurons = 0, epochs = 0;
    if (!read_raw(ifs, &magic, 1) || magic != WEIGHTS_MAGIC) return false;
    // Старый формат (v1) на этом месте хранил num_groups (=10), а не версию.
    if (!read_raw(ifs, &version, 1) || version != WEIGHTS_VERSION) return false;
    if (!read_raw(ifs, &num_groups, 1) || num_groups != groups_.size()) return false;
    if (!read_raw(ifs, &vocab, 1) || vocab != vocab_size_) return false;
    if (!read_raw(ifs, &io_neurons, 1) || io_neurons != IO_NEURONS) return false;
    if (!read_raw(ifs, &epochs, 1)) return false;

    // Читаем во временные буферы, чтобы битый файл не оставил сеть
    // наполовину перезаписанной.
    std::vector<float> emb(emb_.size()), head_w(head_w_.size()), head_b(head_b_.size());
    if (!read_raw(ifs, emb.data(), emb.size())) return false;
    if (!read_raw(ifs, head_w.data(), head_w.size())) return false;
    if (!read_raw(ifs, head_b.data(), head_b.size())) return false;

    std::vector<std::array<float, 6>> thetas(groups_.size());
    std::vector<std::vector<float>> weights(groups_.size());
    for (size_t gi = 0; gi < groups_.size(); ++gi) {
        if (!read_raw(ifs, thetas[gi].data(), 6)) return false;
        uint32_t weights_size = 0;
        if (!read_raw(ifs, &weights_size, 1)) return false;
        // Защита от несовместимого файла: размер обязан совпасть с числом
        // связей группы (структура связей фиксирована конструктором).
        const size_t expected = groups_[gi].row_ptr.empty() ? 0 : groups_[gi].row_ptr.back();
        if (weights_size != expected) return false;
        weights[gi].resize(weights_size);
        if (!read_raw(ifs, weights[gi].data(), weights[gi].size())) return false;
    }

    completed_epochs_ = epochs;
    emb_ = std::move(emb);
    head_w_ = std::move(head_w);
    head_b_ = std::move(head_b);
    for (size_t gi = 0; gi < groups_.size(); ++gi) {
        groups_[gi].rotation_theta = thetas[gi];
        groups_[gi].weights = std::move(weights[gi]);
    }
    // Моменты AdamW не сохраняются (файл был бы в 3 раза больше) — после
    // resume они разгоняются заново; Trainer делает warmup learning rate.
    training_ready_ = false;
    opt_step_ = 0;
    pending_tokens_ = 0;
    return true;
}

} // namespace sparse_nn
