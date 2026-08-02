#include "sparse_dynamic_nn.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

using namespace sparse_nn;

void test_construction() {
    SparseDynamicNetwork net(0.9f);
    assert(net.total_neurons() == TOTAL_NEURONS);
    assert(net.active_neurons() == ACTIVE_NEURONS);
    assert(net.num_groups() == NUM_GROUPS);
    assert(std::abs(net.sparsity() - 0.9f) < 0.01f);
    std::cout << "✓ test_construction passed\n";
}

void test_inject_read() {
    SparseDynamicNetwork net(0.9f);
    const size_t input_size = net.state_dim() * net.active_neurons();
    
    std::vector<float> input(input_size, 0.5f);
    std::vector<float> output(input_size, 0.0f);
    
    net.inject_input(input.data(), input.size());
    net.run_cycle(1);
    net.read_output(output.data(), output.size());
    
    // Output should be non-zero after SiLU
    bool all_zero = true;
    for (float v : output) {
        if (std::abs(v) > 1e-6) { all_zero = false; break; }
    }
    assert(!all_zero);
    std::cout << "✓ test_inject_read passed\n";
}

void test_deterministic() {
    SparseDynamicNetwork net1(0.9f);
    SparseDynamicNetwork net2(0.9f);
    
    const size_t input_size = net1.state_dim() * net1.active_neurons();
    std::vector<float> input(input_size, 0.1f);
    std::vector<float> out1(input_size), out2(input_size);
    
    net1.inject_input(input.data(), input.size());
    net2.inject_input(input.data(), input.size());
    
    net1.run_cycle(5);
    net2.run_cycle(5);
    
    net1.read_output(out1.data(), out1.size());
    net2.read_output(out2.data(), out2.size());
    
    for (size_t i = 0; i < input_size; ++i) {
        assert(std::abs(out1[i] - out2[i]) < 1e-5f);
    }
    std::cout << "✓ test_deterministic passed\n";
}

void test_step_by_step() {
    SparseDynamicNetwork net(0.9f);
    const size_t input_size = net.state_dim() * net.active_neurons();
    std::vector<float> input(input_size, 0.2f);
    std::vector<float> output(input_size);
    
    net.inject_input(input.data(), input.size());
    
    // Step through each group manually
    for (int i = 0; i < 3; ++i) {
        net.step(0);
        assert(net.global_step() == i + 1);
    }
    
    net.read_output(output.data(), output.size());
    std::cout << "✓ test_step_by_step passed\n";
}

void test_silu() {
    // Test SiLU approximation
    float x[4] = {-2.0f, -1.0f, 0.0f, 1.0f};
    float expected[4];
    for (int i = 0; i < 4; ++i) {
        expected[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
    
    silu4(x);
    
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(x[i] - expected[i]) < 1e-5f);
    }
    std::cout << "✓ test_silu passed\n";
}

void test_sparsity_effect() {
    SparseDynamicNetwork dense(0.1f);   // 10% sparse = 90% connections
    SparseDynamicNetwork sparse(0.9f);  // 90% sparse = 10% connections
    
    // Dense should have more connections
    size_t dense_conn = 0, sparse_conn = 0;
    for (const auto& g : dense.groups_) {
        dense_conn += g.row_ptr[g.count];
    }
    for (const auto& g : sparse.groups_) {
        sparse_conn += g.row_ptr[g.count];
    }
    
    assert(dense_conn > sparse_conn * 5);  // Roughly 10x difference
    std::cout << "✓ test_sparsity_effect passed (dense: " << dense_conn 
              << ", sparse: " << sparse_conn << ")\n";
}

int main() {
    std::cout << "Running SparseDynamicNN tests...\n\n";
    
    test_silu();
    test_construction();
    test_inject_read();
    test_deterministic();
    test_step_by_step();
    test_sparsity_effect();
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}