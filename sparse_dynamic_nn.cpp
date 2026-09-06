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
    // For each neuron in current group:
    // 1. Read its input state (from previous group's output)
    // 2. Apply neuron update (RNN/LSTM/GRU cell)
    // 3. Write to its persistent state
    // 4. Sparse project to next group's input buffer

    const float* __restrict input_buf = current.state_buffer_a.data();
    float* __restrict output_buf = current.state_buffer_b.data();
    float* __restrict next_input_buf = next.state_buffer_a.data();
    const size_t next_buf_size = next.count * STATE_DIM;

    // Clear next group's input buffer
    std::fill(next_input_buf, next_input_buf + next_buf_size, 0.0f);

    // Нейроны внутри группы независимы друг от друга (каждый читает
    // только свой input_buf[i], пишет только в свой output_buf[i]) —
    // единственное разделяемое состояние — куда они "разливают"
    // (scatter) результат в next_input_buf, а туда возможны коллизии
    // индексов между разными нейронами. Вместо atomic на каждую из
    // 64 записей на нейрон (дорого) — у каждого потока своя копия
    // next_input_buf, складываем в конце (дёшево, один линейный проход).
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

    // Раз мы больше никого не пропускаем (см. примечание про top-k
    // ниже), active_this_step — это просто 0..count-1. Заполняем это
    // последовательно один раз (дёшево, O(count)), а не push_back из
    // параллельного цикла (была бы гонка).
    current.active_this_step.resize(current.count);
    std::iota(current.active_this_step.begin(), current.active_this_step.end(), 0u);

    // ПРИМЕЧАНИЕ: пробовал честный top-k отбор (nth_element по силе
    // активации, обрабатывать только топ 10%) — при такой дешёвой
    // работе на нейрон (~100 флопов) сам отбор (partition по 100,000
    // элементам, 10 групп, каждый токен) оказался ДОРОЖЕ, чем экономия
    // от пропуска 90% — итоговая скорость упала, а не выросла. Поэтому
    // здесь просто честно считаем всех — это одновременно и быстрее,
    // и корректно (никакого хрупкого порога, который может занулить
    // всё или не занулить ничего в зависимости от масштаба чисел).
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
        const float* __restrict proj = &current.projection_matrix[0];

        // --- THINKING: Projection Matrix multiplication (SSE) ---
        // STATE_DIM=4 укладывается ровно в один __m128 — считаем все
        // 4 выхода как 4 горизонтальные суммы вместо 16 скалярных
        // mult-add. Матрица хранится по строкам, так что просто
        // загружаем строку и умножаем на входной вектор.
        __m128 vin = _mm_loadu_ps(neuron_input);
        alignas(16) float projected[STATE_DIM];
        for (int r = 0; r < STATE_DIM; ++r) {
            __m128 vrow = _mm_loadu_ps(&proj[r * STATE_DIM]);
            __m128 vmul = _mm_mul_ps(vrow, vin);
            // horizontal sum of 4 floats
            __m128 shuf = _mm_movehdup_ps(vmul);
            __m128 sums = _mm_add_ps(vmul, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            projected[r] = _mm_cvtss_f32(sums);
        }

        // Accumulate: state = decay*state + projected_input (leaky residual)
        //
        // БЫЛО: neuron_output[d] = ns.h[d] + projected[d] — без затухания.
        // silu(x) ≈ x для больших x (сигмоида уходит в 1), поэтому это
        // фактически неограниченный интегратор: при активном нейроне на
        // каждом токене ns.h только растёт. Стресс-тест (500 шагов
        // подряд на одном и том же токене) подтвердил расхождение в inf.
        // HIDDEN_STATE_DECAY < 1 делает это "текущим" интегратором —
        // старое состояние забывается, а не накапливается бесконечно.
        constexpr float HIDDEN_STATE_DECAY = 0.9f;
        __m128 vh = _mm_loadu_ps(ns.h);
        __m128 vproj = _mm_load_ps(projected);
        __m128 vdecay = _mm_set1_ps(HIDDEN_STATE_DECAY);
        __m128 vout = _mm_fmadd_ps(vdecay, vh, vproj);
        _mm_storeu_ps(neuron_output, vout);

        // Кэшируем pre-активацию ДО silu/clamp — они необратимы
        // (clamp особенно), без этого backward не сможет посчитать
        // производную в точке, где реально был forward.
        _mm_storeu_ps(&current.pre_cache[i * STATE_DIM], vout);

        // Non-linearity — silu(x)=x*sigmoid(x) требует exp(), которого
        // нет как аппаратного AVX2-интринсика (см. правку в silu4) —
        // оставляем скалярным, это самая дешёвая часть по сравнению с
        // matvec/propagation выше и ниже.
        silu4(neuron_output);

        // Жёсткий backstop против расхождения (SSE clamp) — decay выше
        // СМЯГЧАЕТ рост, но не гарантирует границу — на стресс-тесте
        // (одинаковый токен много раз подряд) состояние всё равно
        // уходило в ~1e24. silu(x) для больших x близко к тождественной
        // функции, так что явный clamp — единственная настоящая
        // гарантия того, что numbers никогда не разойдутся.
        // БЫЛО: 50.0f. Id токенов в реальном словаре — это тысячи
        // (наблюдаемый plateau loss ~1.27e8 => sqrt ≈ 11,284 — то есть
        // сеть пытается предсказать ~11 тысяч, но физически не могла
        // выдать больше 50). Это создавало ЖЁСТКИЙ архитектурный
        // потолок ошибки, не зависящий от качества обучения вообще —
        // независимо от того, эвристика это была или настоящий
        // градиент, сеть НЕ МОГЛА подобраться к реальным значениям
        // target. Поднимаем потолок с большим запасом, оставляя его
        // всё ещё конечным (защита от true divergence в inf/NaN
        // остаётся), но не мешающим представить реальный диапазон id.
        constexpr float OUTPUT_CLAMP = 100000.0f;
        __m128 vclampmax = _mm_set1_ps(OUTPUT_CLAMP);
        __m128 vclampmin = _mm_set1_ps(-OUTPUT_CLAMP);
        vout = _mm_loadu_ps(neuron_output);
        vout = _mm_min_ps(_mm_max_ps(vout, vclampmin), vclampmax);
        _mm_storeu_ps(neuron_output, vout);

        // Update persistent state
        _mm_storeu_ps(ns.h, vout);

        ns.active_step = static_cast<uint16_t>(global_step_);
        ns.flags |= 0x1;  // active flag

        // ---- SPARSE PROJECTION TO NEXT GROUP (в локальный буфер потока) ----
        //
        // Раньше: FANOUT(16) x STATE_DIM(4) = 64 скалярных mult-add на
        // нейрон. Теперь: 16 векторных FMA — neuron_output загружаем
        // ОДИН раз (vout уже в регистре), на каждую связь — 1 load,
        // 1 FMA (умножить на broadcast веса и прибавить), 1 store.
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

    // Финальная редукция: суммируем буферы всех потоков в общий
    // next_input_buf. Дешёвый линейный проход, без гонок.
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
// Настоящий backward pass
// =============================================================================
//
// Читаемый вывод сети — это РОВНО ОДНО число: neuron[0], dim[0]
// последней группы (см. read_output). Значит прямой (внешний) градиент
// от loss есть только у ЭТОЙ единственной точки — всё остальное в сети
// получает градиент только если реально лежит на пути от неё назад
// через sparse-связи. Идём от последней группы к первой (обратный
// порядок ровно повторяет forward pass, только в другую сторону), на
// каждом шаге используя обратный индекс (reverse_row_ptr/conn_source),
// чтобы узнать, КТО из предыдущей группы реально стрелял в конкретный
// нейрон текущей группы.
//
// ВАЖНО (ограничение): это truncated backprop depth=1 — градиент НЕ
// течёт через persistent-состояние ns.h в предыдущие токены, и группа
// 0 считается "границей": её вход (инжектированный токен + утечка от
// group9 ПРЕДЫДУЩЕГО токена) не раскручивается назад дальше. Это
// стандартное упрощение для онлайн-обучения рекуррентных сетей одним
// токеном за раз — правильные ЛОКАЛЬНЫЕ градиенты, но без разматывания
// через время. Всё равно строго лучше, чем прежний "толчок всех весов
// на одну и ту же величину" — тут веса двигаются пропорционально
// РЕАЛЬНОМУ вкладу каждой связи в ошибку.
void SparseDynamicNetwork::backward(float grad_predicted, float learning_rate) {
    constexpr float WEIGHT_DECAY = 0.0001f;
    constexpr float GRAD_CLIP = 5.0f; // тот же принцип, что и OUTPUT_CLAMP в forward — защита от расхождения через цепочку

    // Сброс grad_out только там, где реально что-то трогали в прошлый раз.
    for (auto& g : groups_) {
        for (uint32_t idx : g.touched) {
            std::fill(g.grad_out.begin() + idx * STATE_DIM, g.grad_out.begin() + (idx + 1) * STATE_DIM, 0.0f);
            g.touched_flag[idx] = 0;
        }
        g.touched.clear();
    }

    // Seed: единственная прямая связь с loss — (последняя группа, neuron 0, dim 0).
    GroupState& last = groups_[NUM_GROUPS - 1];
    last.grad_out[0] = std::clamp(grad_predicted, -GRAD_CLIP, GRAD_CLIP);
    last.touched.push_back(0);
    last.touched_flag[0] = 1;

    for (size_t gi = 0; gi < NUM_GROUPS; ++gi) {
        size_t g = NUM_GROUPS - 1 - gi;
        GroupState& cur = groups_[g];
        if (cur.touched.empty()) continue;

        float dW[STATE_DIM * STATE_DIM] = {0.0f};

        for (uint32_t i : cur.touched) {
            const float* g_out = &cur.grad_out[i * STATE_DIM];
            const float* pre = &cur.pre_cache[i * STATE_DIM];
            const float* inp = &cur.state_buffer_a[i * STATE_DIM];

            // d(out)/d(pre) = clamp'(silu(pre)) * silu'(pre)
            float g_pre[STATE_DIM];
            for (size_t d = 0; d < STATE_DIM; ++d) {
                float s = silu(pre[d]);
                float clamp_deriv = (s > -50.0f && s < 50.0f) ? 1.0f : 0.0f;
                float v = g_out[d] * clamp_deriv * silu_derivative(pre[d]);
                g_pre[d] = std::clamp(v, -GRAD_CLIP, GRAD_CLIP);
            }

            // dL/dW[r][c] += g_pre[r] * input[c]  (накапливаем, W общая на группу)
            for (size_t r = 0; r < STATE_DIM; ++r) {
                for (size_t c = 0; c < STATE_DIM; ++c) {
                    dW[r * STATE_DIM + c] += g_pre[r] * inp[c];
                }
            }

            // Группа 0 — временная граница (см. комментарий к функции):
            // её вход не раскручиваем дальше назад, W0 всё равно обучаем
            // (через dW выше), а вот дальше по связям group9->group0 не
            // идём — это была бы утечка градиента в ДРУГОЙ токен.
            if (g == 0) continue;

            // dL/d(input)[c] = sum_r W[r][c] * g_pre[r]
            float g_input[STATE_DIM] = {0.0f};
            for (size_t r = 0; r < STATE_DIM; ++r) {
                for (size_t c = 0; c < STATE_DIM; ++c) {
                    g_input[c] += cur.projection_matrix[r * STATE_DIM + c] * g_pre[r];
                }
            }

            GroupState& prev = groups_[g - 1];

            // Без этого ограничения фронт распространения растёт
            // экспоненциально (ветвление по входящим связям): 1 → 14 →
            // 217 → 3447 → 42744 → ВСЕ 100,000 к группе 0 — backward
            // становится дороже forward. Берём не больше
            // MAX_FANIN_PER_NODE входящих связей на нейрон — этого
            // достаточно, чтобы градиент реально доходил до всех 10
            // групп, оставаясь на порядки дешевле полного разворота.
            constexpr uint32_t MAX_FANIN_PER_NODE = 3;
            uint32_t k_begin = prev.reverse_row_ptr[i];
            uint32_t k_end = prev.reverse_row_ptr[i + 1];
            uint32_t k_limit = std::min(k_end, k_begin + MAX_FANIN_PER_NODE);

            for (uint32_t k = k_begin; k < k_limit; ++k) {
                uint32_t conn = prev.reverse_conn_idx[k];
                uint32_t src = prev.conn_source[conn];
                float w = prev.weights[conn];
                const float* src_out = &prev.state_buffer_b[src * STATE_DIM];

                // dL/dweight_conn = dot(g_input, out_src) — вес умножает
                // весь 4-вектор выхода источника сразу (см. forward).
                float dW_conn = 0.0f;
                for (size_t d = 0; d < STATE_DIM; ++d) dW_conn += g_input[d] * src_out[d];
                dW_conn = std::clamp(dW_conn, -GRAD_CLIP, GRAD_CLIP);

                float& wv = prev.weights[conn];
                wv -= learning_rate * dW_conn;
                wv -= WEIGHT_DECAY * wv;

                // Пробрасываем градиент дальше назад в out_src.
                if (!prev.touched_flag[src]) {
                    prev.touched_flag[src] = 1;
                    prev.touched.push_back(src);
                    std::fill(prev.grad_out.begin() + src * STATE_DIM, prev.grad_out.begin() + (src + 1) * STATE_DIM, 0.0f);
                }
                float* dst = &prev.grad_out[src * STATE_DIM];
                for (size_t d = 0; d < STATE_DIM; ++d) {
                    dst[d] += w * g_input[d];
                }
            }
        }

        // Проекционная матрица общая для всех нейронов группы —
        // применяем накопленный градиент со всех задетых нейронов один раз.
        for (size_t idx = 0; idx < STATE_DIM * STATE_DIM; ++idx) {
            float& v = cur.projection_matrix[idx];
            v -= learning_rate * dW[idx];
            v -= WEIGHT_DECAY * v;
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
    //    величину". Та эвристика оказалась НЕ градиентным спуском в
    //    принципе — loss застревал (одно и то же число до 6 значащих
    //    цифр 5 эпох подряд на реальном тексте), потому что толчок не
    //    учитывал реальный вклад каждой связи в ошибку.
    //
    //    Ошибку по-прежнему обрезаем перед использованием в градиенте —
    //    target это id токена (тысячи), сырая ошибка такого масштаба
    //    рвала бы обучение вне зависимости от того, градиент это или
    //    эвристика.
    constexpr float ERROR_CLIP = 20.0f;
    float clipped_error = std::clamp(error, -ERROR_CLIP, ERROR_CLIP);

    // loss = (target - predicted)^2 = error^2
    // d(loss)/d(predicted) = -2 * error
    float grad_predicted = -2.0f * clipped_error;

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