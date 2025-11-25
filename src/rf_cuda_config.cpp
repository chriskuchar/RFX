#include "rf_cuda_config.hpp"
#include <algorithm>
#include <iostream>
#include <cstdlib>

// Include CUDA headers only when actually needed
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rf {
namespace cuda {

bool CudaConfig::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Jupyter/IPython check removed - GPU safety features now handle notebook compatibility
    // Continue with CUDA initialization...
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        initialized_ = false;
        return false;
    }

    // Allocate device properties
    device_props_ = new cudaDeviceProp();
    
    // Get current device properties
    err = cudaGetDeviceProperties(device_props_, 0);
    if (err != cudaSuccess) {
        delete device_props_;
        device_props_ = nullptr;
        initialized_ = false;
        return false;
    }

    // Configure based on GPU capabilities
    configure_for_gpu();
    initialized_ = true;
    
    return true;
}

int CudaConfig::get_optimal_block_size(int problem_size) const {
    if (!initialized_ || !device_props_) return 256; // fallback

    // Adaptive block sizing based on GPU architecture
    int optimal_size = 256; // default
    
    if (device_props_->major >= 7) { // Volta/Turing/Ampere/Ada/Hopper
        optimal_size = std::min(1024, problem_size);
    } else if (device_props_->major >= 6) { // Pascal
        optimal_size = std::min(512, problem_size);
    } else if (device_props_->major >= 5) { // Maxwell
        optimal_size = std::min(256, problem_size);
    } else { // Older architectures
        optimal_size = std::min(128, problem_size);
    }

    // Ensure it's a multiple of warp size
    optimal_size = ((optimal_size + 31) / 32) * 32;
    return std::min(optimal_size, device_props_->maxThreadsPerBlock);
}

int CudaConfig::get_max_blocks() const {
    if (!initialized_ || !device_props_) return 65535; // fallback
    return device_props_->maxGridSize[0];
}

size_t CudaConfig::get_shared_memory_per_block() const {
    if (!initialized_ || !device_props_) return 48 * 1024; // 48KB fallback
    return device_props_->sharedMemPerBlock;
}

size_t CudaConfig::get_total_memory() const {
    if (!initialized_ || !device_props_) return 8ULL * 1024 * 1024 * 1024; // 8GB fallback
    return device_props_->totalGlobalMem;
}

bool CudaConfig::supports_fp16() const {
    if (!initialized_ || !device_props_) return false;
    return device_props_->major >= 6; // Pascal and newer
}

bool CudaConfig::supports_tensor_cores() const {
    if (!initialized_ || !device_props_) return false;
    return device_props_->major >= 7; // Volta and newer
}

const char* CudaConfig::get_gpu_name() const {
    if (!initialized_ || !device_props_) return "Unknown GPU";
    return device_props_->name;
}

int CudaConfig::get_compute_capability_major() const {
    if (!initialized_ || !device_props_) return 0;
    return device_props_->major;
}

int CudaConfig::get_compute_capability_minor() const {
    if (!initialized_ || !device_props_) return 0;
    return device_props_->minor;
}

size_t CudaConfig::get_available_memory() const {
    if (!initialized_ || !device_props_) return get_total_memory();
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    // Apply 20% safety margin to prevent WSL crashes
    // Use 80% of available memory instead of 100%
    return static_cast<size_t>(free * 0.8);
}

bool CudaConfig::can_handle_problem_size(int nsample, int mdim, int ntree) const {
    if (!initialized_) return true; // assume yes if not initialized
    
    // Estimate memory requirements
    size_t proximity_memory = nsample * nsample * sizeof(__half); // FP16
    size_t tree_memory = ntree * nsample * sizeof(integer_t);
    size_t feature_memory = nsample * mdim * sizeof(real_t);
    
    size_t total_estimated = proximity_memory + tree_memory + feature_memory;
    
    // get_available_memory() already applies 20% safety margin
    return total_estimated < get_available_memory();
}

// New function to check if execution should stop instead of falling back to CPU
bool CudaConfig::should_stop_on_insufficient_memory(int nsample, int mdim, int ntree) const {
    if (!initialized_) return false; // Don't stop if not initialized
    
    // Estimate memory requirements
    size_t proximity_memory = nsample * nsample * sizeof(__half); // FP16
    size_t tree_memory = ntree * nsample * sizeof(integer_t);
    size_t feature_memory = nsample * mdim * sizeof(real_t);
    
    size_t total_estimated = proximity_memory + tree_memory + feature_memory;
    size_t available_memory = get_available_memory();
    
    // If more than 50% of available memory is needed, it's risky
    return total_estimated > available_memory * 0.5;
}

// New function to check CPU memory requirements
bool CudaConfig::can_handle_cpu_problem_size(int nsample, int mdim, int ntree) const {
    // Estimate CPU memory requirements (more conservative)
    size_t proximity_memory = nsample * nsample * sizeof(dp_t); // FP32 for CPU
    size_t tree_memory = ntree * nsample * sizeof(integer_t);
    size_t feature_memory = nsample * mdim * sizeof(real_t);
    size_t bootstrap_memory = nsample * sizeof(integer_t);
    size_t oob_memory = nsample * sizeof(real_t);
    
    size_t total_estimated = proximity_memory + tree_memory + feature_memory + 
                           bootstrap_memory + oob_memory;
    
    // For CPU, check system memory (rough estimate)
    // Assume at least 1GB is available for the process
    size_t min_available_cpu_memory = 1024ULL * 1024ULL * 1024ULL; // 1GB
    
    // If more than 80% of minimum available CPU memory is needed, it's risky
    return total_estimated < min_available_cpu_memory * 0.8;
}

int CudaConfig::get_recommended_batch_size(int total_trees) const {
    if (!initialized_ || !device_props_) return std::min(100, total_trees);
    
    size_t available_memory = get_available_memory();
    int num_sms = device_props_->multiProcessorCount;
    
    // More conservative batch size calculation to avoid memory issues
    // Supports GPUs from 4GB to 250GB (consumer to data center)
    int optimal_batch_size;
    
    if (available_memory > 200ULL * 1024 * 1024 * 1024) { // > 200GB (H100 80GB+, A100 80GB+)
        optimal_batch_size = std::min(5000, total_trees);
    } else if (available_memory > 150ULL * 1024 * 1024 * 1024) { // > 150GB
        optimal_batch_size = std::min(4000, total_trees);
    } else if (available_memory > 100ULL * 1024 * 1024 * 1024) { // > 100GB
        optimal_batch_size = std::min(3000, total_trees);
    } else if (available_memory > 80ULL * 1024 * 1024 * 1024) { // > 80GB (A100 80GB, H100 80GB)
        optimal_batch_size = std::min(2500, total_trees);
    } else if (available_memory > 64ULL * 1024 * 1024 * 1024) { // > 64GB
        optimal_batch_size = std::min(2000, total_trees);
    } else if (available_memory > 48ULL * 1024 * 1024 * 1024) { // > 48GB (A6000, RTX 6000 Ada)
        optimal_batch_size = std::min(1500, total_trees);
    } else if (available_memory > 32ULL * 1024 * 1024 * 1024) { // > 32GB (V100 32GB)
        optimal_batch_size = std::min(1250, total_trees);
    } else if (available_memory > 24ULL * 1024 * 1024 * 1024) { // > 24GB (RTX 3090, 4090, A5000)
        optimal_batch_size = std::min(1000, total_trees);
    } else if (available_memory > 16ULL * 1024 * 1024 * 1024) { // > 16GB (V100 16GB)
        optimal_batch_size = std::min(800, total_trees);
    } else if (available_memory > 12ULL * 1024 * 1024 * 1024) { // > 12GB (RTX 3080 Ti, 3060 12GB)
        optimal_batch_size = std::min(750, total_trees);
    } else if (available_memory > 10ULL * 1024 * 1024 * 1024) { // > 10GB (RTX 3080 10GB)
        optimal_batch_size = std::min(600, total_trees);
    } else if (available_memory > 8ULL * 1024 * 1024 * 1024) { // > 8GB (RTX 3070, 2080)
        optimal_batch_size = std::min(500, total_trees);
    } else if (available_memory > 6ULL * 1024 * 1024 * 1024) { // > 6GB (RTX 2060)
        optimal_batch_size = std::min(300, total_trees);
    } else if (available_memory > 4ULL * 1024 * 1024 * 1024) { // > 4GB (GTX 1650)
        optimal_batch_size = std::min(200, total_trees);
    } else { // <= 4GB
        optimal_batch_size = std::min(100, total_trees);
    }
    
    // For very large numbers of trees, be more conservative
    if (total_trees > 100000) {
        optimal_batch_size = std::min(optimal_batch_size, 2000); // Cap at 2000 for very large forests
    }
    
    // SM-aware batch sizing: Ensure batch size utilizes SMs effectively
    // Target: 2 blocks per SM for good occupancy (each tree = 1 block)
    int target_blocks = num_sms * 2;
    
    if (total_trees >= 100 && optimal_batch_size < target_blocks) {
        // Increase batch size to better utilize SMs (memory permitting)
        optimal_batch_size = std::max(optimal_batch_size, 
                                      std::min(target_blocks, total_trees));
    }
    
    return optimal_batch_size;
}

const cudaDeviceProp* CudaConfig::get_device_props() const {
    return device_props_;
}

void CudaConfig::configure_for_gpu() {
    if (!device_props_) return;
    
    // Configure based on GPU architecture
    if (device_props_->major >= 8) { // Ampere/Ada/Hopper
        // Modern GPUs: prefer larger blocks, more parallelism
        default_block_size_ = 512;
        max_concurrent_kernels_ = 32;
    } else if (device_props_->major >= 7) { // Volta/Turing
        // Good GPUs: balanced configuration
        default_block_size_ = 256;
        max_concurrent_kernels_ = 16;
    } else if (device_props_->major >= 6) { // Pascal
        // Decent GPUs: moderate configuration
        default_block_size_ = 256;
        max_concurrent_kernels_ = 8;
    } else { // Older GPUs
        // Conservative configuration
        default_block_size_ = 128;
        max_concurrent_kernels_ = 4;
    }
}

} // namespace cuda
} // namespace rf
