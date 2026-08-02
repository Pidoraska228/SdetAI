#include "sparse_dynamic_nn.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace sparse_nn;

int main() {
    std::cout << "=== Sparse Dynamic Neural Network Demo ===\n\n";
    
    // Create network with 90% sparsity
    SparseDynamicNetwork net(0.9f);
    
    // Print architecture info
    std::cout << "Architecture:\n";
    std::cout << "  Total neurons: " << net.total_neurons() << "\n";
    std::cout << "  Active per cycle: " << net.active_neurons() << "\n";
    std::cout << "  Groups: " << net.num_groups() << "\n";
    std::cout << "  State dim: " << net.state_dim() << "\n";
    std::cout << "  Sparsity: " << net.sparsity() * 100 << "%\n";
    std::cout << "  Memory: ~" << (net.total_neurons() * net.state_dim() * 4) / 1024 << " KB\n\n";
    
    // Prepare random input
    const size_t input_size = net.state_dim() * net.active_neurons();
    std::vector<float> input(input_size);
    std::vector<float> output(input_size);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (auto& v : input) v = dist(rng);
    
    // Inject input
    net.inject_input(input.data(), input.size());
    
    // Run multiple cycles
    std::cout << "Running 10 cycles...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    net.run_cycle(10);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Read output
    net.read_output(output.data(), output.size());
    
    // Stats
    float sum = 0.0f, min_val = output[0], max_val = output[0];
    for (float v : output) {
        sum += v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    float mean = sum / output.size();
    
    std::cout << "\nResults after 10 cycles:\n";
    std::cout << "  Time: " << duration.count() << " µs (" << duration.count() / 10.0 << " µs/cycle)\n";
    std::cout << "  Output stats: mean=" << std::fixed << std::setprecision(6) << mean
              << ", min=" << min_val << ", max=" << max_val << "\n";
    std::cout << "  Global step: " << net.global_step() << "\n";
    
    // Demonstrate step-by-step control
    std::cout << "\n--- Step-by-step demo ---\n";
    net.inject_input(input.data(), input.size());
    
    for (int step = 0; step < 3; ++step) {
        net.step(0);  // Process group 0
        std::cout << "Step " << step + 1 << ": group 0 processed, global_step=" << net.global_step() << "\n";
    }
    
    net.read_output(output.data(), output.size());
    std::cout << "Sample output[0..7]: ";
    for (int i = 0; i < 8; ++i) std::cout << output[i] << " ";
    std::cout << "\n";
    
    return 0;
}