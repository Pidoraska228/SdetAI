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
        constexpr float OUTPUT_CLAMP = 50.0f;
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