#include <cuda_fp16.h>  // MUST be VERY FIRST - before ANY other includes
#include "rf_types.hpp"
#include "rf_config.hpp"
#include "rf_utils.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include "rf_testreebag.hpp"
#include "rf_varimp.hpp"
#include "rf_varimp.cuh"  // For gpu_compute_tnodewt_kernel declaration
#include "rf_proximity.hpp"
// Forward declarations to defer type resolution and avoid static initialization issues
// Include headers only when CUDA is available - this prevents static init crashes
#ifdef __CUDACC__
#include "rf_proximity_lowrank.hpp"
#include "rf_proximity_upper_triangle.hpp"
#endif
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>  // For memset
#include <algorithm>  // For std::fill
#include <iostream>
#include <cstdlib>
#include <memory>
#include <chrono>

namespace rf {

// Thread-local storage for low-rank proximity state
// MUST be at file scope (not inside function) to persist across all batch calls
// This fixes bug where 3000 trees (300 batches) would reset state, causing zero factors
thread_local struct LowRankState {
    void* lowrank_prox_ptr_raw;
    bool lowrank_initialized;
    integer_t total_trees_processed;
    integer_t saved_nsample;
    
    LowRankState() : lowrank_prox_ptr_raw(nullptr), lowrank_initialized(false), 
                   total_trees_processed(0), saved_nsample(0) {}
} g_lowrank_state;

// CUDA error checking macro - matches backup version (logs but doesn't throw)
// #ifndef CUDA_CHECK_VOID
// #define CUDA_CHECK_VOID(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
//                     cudaGetErrorString(err)); \
//         } \
//     } while(0)
// #endif

// Forward declaration
__device__ real_t gpu_calcs(const real_t* tmpclass, const real_t* tclasspop, real_t pdo,
                           integer_t nclass);

// GPU loss function helpers
__device__ real_t gpu_binary_logloss(real_t p, real_t w) {
    // Binary logloss: -w * (y*log(p) + (1-y)*log(1-p))
    // For split evaluation, we compute sum of logloss for left and right
    // p is probability of class 1, w is weight
    const real_t eps = 1e-15f;
    p = max(eps, min(1.0f - eps, p));  // Clamp to avoid log(0)
    return -w * (p > 0.5f ? logf(p) : logf(1.0f - p));
}

__device__ real_t gpu_multiclass_crossentropy(const real_t* class_probs, integer_t nclass, real_t total_weight) {
    // Multiclass crossentropy: -sum(w * log(p_i))
    // For split evaluation, compute sum for left and right
    const real_t eps = 1e-15f;
    real_t loss = 0.0f;
    for (integer_t j = 0; j < nclass; ++j) {
        real_t p = max(eps, min(1.0f - eps, class_probs[j] / total_weight));
        loss -= class_probs[j] * logf(p);
    }
    return loss;
}

__device__ real_t gpu_mae_loss(real_t mean_left, real_t mean_right, 
                               const integer_t* samples_left, integer_t n_left,
                               const integer_t* samples_right, integer_t n_right,
                               const integer_t* cl, const real_t* win) {
    // MAE: sum(|y - mean|) for left and right
    real_t mae_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mae_left += w * fabsf(y - mean_left);
        w_left += w;
    }
    
    real_t mae_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mae_right += w * fabsf(y - mean_right);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((mae_left + mae_right) / total_w) : 0.0f;
}

// Additional regression loss functions
__device__ real_t gpu_huber_loss(real_t mean_left, real_t mean_right,
                                  const integer_t* samples_left, integer_t n_left,
                                  const integer_t* samples_right, integer_t n_right,
                                  const integer_t* cl, const real_t* win, real_t delta = 1.0f) {
    // Huber loss: L_delta(y, mean) = 0.5*(y-mean)^2 if |y-mean| <= delta, else delta*|y-mean| - 0.5*delta^2
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_left);
        if (residual <= delta) {
            loss_left += w * 0.5f * residual * residual;
        } else {
            loss_left += w * (delta * residual - 0.5f * delta * delta);
        }
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_right);
        if (residual <= delta) {
            loss_right += w * 0.5f * residual * residual;
        } else {
            loss_right += w * (delta * residual - 0.5f * delta * delta);
        }
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_poisson_loss(real_t mean_left, real_t mean_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win) {
    // Poisson loss: -y*log(mean) + mean (for y >= 0)
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(0.0f, static_cast<real_t>(cl[idx]));  // Poisson requires non-negative
        real_t mean = max(eps, mean_left);
        loss_left += w * (-y * logf(mean) + mean);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(0.0f, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        loss_right += w * (-y * logf(mean) + mean);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_quantile_loss(real_t quantile_left, real_t quantile_right,
                                     const integer_t* samples_left, integer_t n_left,
                                     const integer_t* samples_right, integer_t n_right,
                                     const integer_t* cl, const real_t* win, real_t alpha = 0.5f) {
    // Quantile loss: alpha * max(y - q, 0) + (1-alpha) * max(q - y, 0)
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = y - quantile_left;
        loss_left += w * (alpha * max(0.0f, residual) + (1.0f - alpha) * max(0.0f, -residual));
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = y - quantile_right;
        loss_right += w * (alpha * max(0.0f, residual) + (1.0f - alpha) * max(0.0f, -residual));
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_mape_loss(real_t mean_left, real_t mean_right,
                                 const integer_t* samples_left, integer_t n_left,
                                 const integer_t* samples_right, integer_t n_right,
                                 const integer_t* cl, const real_t* win) {
    // MAPE: mean(|y - mean| / max(|y|, eps))
    const real_t eps = 1e-8f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t denom = max(fabsf(y), eps);
        loss_left += w * fabsf(y - mean_left) / denom;
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t denom = max(fabsf(y), eps);
        loss_right += w * fabsf(y - mean_right) / denom;
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_fair_loss(real_t mean_left, real_t mean_right,
                                const integer_t* samples_left, integer_t n_left,
                                const integer_t* samples_right, integer_t n_right,
                                const integer_t* cl, const real_t* win, real_t c = 1.0f) {
    // Fair loss: c^2 * (|y-mean|/c - log(1 + |y-mean|/c))
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_left) / c;
        loss_left += w * c * c * (residual - logf(1.0f + residual));
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_right) / c;
        loss_right += w * c * c * (residual - logf(1.0f + residual));
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_gamma_loss(real_t mean_left, real_t mean_right,
                                 const integer_t* samples_left, integer_t n_left,
                                 const integer_t* samples_right, integer_t n_right,
                                 const integer_t* cl, const real_t* win) {
    // Gamma loss: -log(y/mean) + y/mean - 1 (for y > 0, mean > 0)
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_left);
        loss_left += w * (-logf(y / mean) + y / mean - 1.0f);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        loss_right += w * (-logf(y / mean) + y / mean - 1.0f);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_tweedie_loss(real_t mean_left, real_t mean_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win, real_t p = 1.5f) {
    // Tweedie loss: -y*mean^(1-p)/(1-p) + mean^(2-p)/(2-p) for p != 1,2
    // For p=1: Poisson, p=2: Gamma
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_left);
        if (fabsf(p - 1.0f) < 1e-6f) {
            // Poisson: -y*log(mean) + mean
            loss_left += w * (-y * logf(mean) + mean);
        } else if (fabsf(p - 2.0f) < 1e-6f) {
            // Gamma: -log(y/mean) + y/mean - 1
            loss_left += w * (-logf(y / mean) + y / mean - 1.0f);
        } else {
            real_t mean_1p = powf(mean, 1.0f - p);
            real_t mean_2p = powf(mean, 2.0f - p);
            loss_left += w * (-y * mean_1p / (1.0f - p) + mean_2p / (2.0f - p));
        }
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        if (fabsf(p - 1.0f) < 1e-6f) {
            loss_right += w * (-y * logf(mean) + mean);
        } else if (fabsf(p - 2.0f) < 1e-6f) {
            loss_right += w * (-logf(y / mean) + y / mean - 1.0f);
        } else {
            real_t mean_1p = powf(mean, 1.0f - p);
            real_t mean_2p = powf(mean, 2.0f - p);
            loss_right += w * (-y * mean_1p / (1.0f - p) + mean_2p / (2.0f - p));
        }
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_ranking_loss(real_t score_left, real_t score_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win) {
    // Ranking loss (LambdaRank style): pairwise ranking loss
    // Simplified: use MSE on scores as proxy
    real_t loss = 0.0f;
    real_t total_w = 0.0f;
    
    // Compute mean score for left and right
    real_t mean_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mean_left += w * y;
        w_left += w;
    }
    if (w_left > 0.0f) mean_left /= w_left;
    
    real_t mean_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mean_right += w * y;
        w_right += w;
    }
    if (w_right > 0.0f) mean_right /= w_right;
    
    // Pairwise ranking loss: sum over pairs
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx_i = samples_left[i];
        real_t w_i = win[idx_i];
        real_t y_i = static_cast<real_t>(cl[idx_i]);
        for (integer_t j = 0; j < n_right; ++j) {
            integer_t idx_j = samples_right[j];
            real_t w_j = win[idx_j];
            real_t y_j = static_cast<real_t>(cl[idx_j]);
            if (y_i > y_j) {
                // i should rank higher than j
                real_t score_diff = score_left - score_right;
                loss += w_i * w_j * max(0.0f, 1.0f - score_diff);  // Hinge loss
            }
        }
    }
    
    total_w = w_left * w_right;
    return (total_w > 0.0f) ? (loss / total_w) : 0.0f;
}

__device__ real_t gpu_survival_loss(real_t hazard_left, real_t hazard_right,
                                     const integer_t* samples_left, integer_t n_left,
                                     const integer_t* samples_right, integer_t n_right,
                                     const integer_t* cl, const real_t* win) {
    // Survival loss (Cox proportional hazards): -log(hazard) + cumulative hazard
    // Simplified: use negative log-likelihood of survival model
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t hazard = max(eps, hazard_left);
        loss_left += w * (-logf(hazard) + hazard * y);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t hazard = max(eps, hazard_right);
        loss_right += w * (-logf(hazard) + hazard * y);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

// Initialize random states kernel
__global__ void init_random_states_kernel(curandState* random_states, integer_t num_trees, const integer_t* seeds) {
    integer_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_trees) {
        // Use tree-specific seed for reproducibility
        // Each tree gets a unique seed based on its index
        curand_init(seeds[tid], tid, 0, &random_states[tid]);
    }
}

// GPU bootstrap sampling kernel
__global__ void gpu_bootstrap_kernel(
    integer_t num_trees, integer_t nsample, const real_t* weight,
    integer_t* nin_all, real_t* win_all, integer_t* jinbag_all,
    curandState* random_states, const integer_t* seeds,
    bool use_casewise) {  // Pass use_casewise as parameter (g_config not accessible in device code)
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Debug output
    // if (tree_id == 0 && threadIdx.x == 0) {
    //     printf("GPU: Bootstrap kernel started for tree %d\n", tree_id);
    // }
    
    integer_t tid = threadIdx.x;
    integer_t stride = blockDim.x;
    
    // Calculate offsets for this tree
    integer_t nin_offset = tree_id * nsample;
    integer_t win_offset = tree_id * nsample;
    integer_t jinbag_offset = tree_id * nsample;
    
    integer_t* nin = nin_all + nin_offset;
    real_t* win = win_all + win_offset;
    integer_t* jinbag = jinbag_all + jinbag_offset;
    
    // Initialize arrays
    for (integer_t i = tid; i < nsample; i += stride) {
        nin[i] = 0;
        win[i] = 0.0f;
    }
    __syncthreads();
    
    // Bootstrap sampling - exact port from original Fortran boot.f
    // Original: i = int(randomu()*nsample) + 1  (1-based: generates 1..nsample)
    // GPU port: i = static_cast<integer_t>(rand_val * nsample)  (0-based: generates 0..nsample-1)
    // Note: Expected in-bag percentage = 1 - (1 - 1/n)^n
    //       For n→∞: approaches 1 - e^(-1) ≈ 0.6321 (63.2%)
    //       For finite n: slightly higher (e.g., n=150: ~63.21%, n=100: ~63.40%, n=10: ~65.13%)
    curandState local_state = random_states[tree_id];
    
    // Generate bootstrap samples (match Fortran: int(randomu() * nsample) + 1)
    // Each thread needs independent random numbers
    // Use thread ID to advance the random state differently for each thread
    for (integer_t s = tid; s < nsample; s += stride) {
        // Create thread-specific random state by advancing the base state
        curandState thread_state = local_state;
        // Advance state by a large offset based on thread ID and sample index
        // This ensures each thread gets a different random sequence
        integer_t advance_count = tid * 1000 + s;  // Large offset to ensure independence
        for (integer_t advance = 0; advance < advance_count; advance++) {
            curand_uniform(&thread_state);
        }
        
        real_t rand_val = curand_uniform(&thread_state);
        
        // Convert to sample index (0-based, matching CPU implementation)
        // Original Fortran: i = int(randomu()*nsample) + 1  (1-based: generates 1..nsample)
        // Fortran CUDA: i = int(rand_val * nsample) + 1, then if (i > nsample) i = nsample
        // C++ 0-based: i = static_cast<integer_t>(rand_val * nsample)
        // curand_uniform returns [0.0, 1.0), so rand_val * nsample is in [0.0, nsample)
        //           Casting to integer gives [0, nsample-1], which is correct for 0-based indexing
        integer_t i = static_cast<integer_t>(rand_val * static_cast<real_t>(nsample));
        // Clamp to valid range (should never be needed since curand_uniform < 1.0, but safety check)
        if (i >= nsample) i = nsample - 1;
        if (i < 0) i = 0;
        
        // Count occurrences using atomic operations
        // Use atomicAdd with proper memory ordering
        ::atomicAdd(&nin[i], 1);
    }
    
    // Ensure all atomic operations complete before reading
    __threadfence_block();  // Ensure all writes in this block are visible
    __syncthreads();
    
    // Additional sync to ensure all blocks complete their atomic operations
    __threadfence();  // Global memory fence - ensures all blocks see the updates
    __syncthreads();
    
    // Now populate jinbag with only in-bag samples (where nin[n] > 0)
    // This matches CPU implementation: jinbag[ninbag] = n for samples where nin[n] > 0
    // Use atomic counter to track position in jinbag array
    __shared__ integer_t s_ninbag;
    if (tid == 0) {
        s_ninbag = 0;
    }
    __syncthreads();
    
    // Each thread processes a subset of samples to find in-bag samples
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nin[n] > 0) {
            // In-bag sample: add to jinbag
            integer_t pos = ::atomicAdd(&s_ninbag, 1);
            if (pos < nsample) {
                jinbag[pos] = n;
            }
            
            // Compute win[n] = nin[n] * weight[n] (for case-wise) or 1.0 (for non-case-wise)
            // Note: weight is passed but may be nullptr, so use 1.0 as default
            if (use_casewise) {
                win[n] = static_cast<real_t>(nin[n]) * (weight != nullptr ? weight[n] : 1.0f);
            } else {
                win[n] = 1.0f;  // Non-case-wise: equal weighting
            }
        } else {
            // Out-of-bag sample
            win[n] = 0.0f;
        }
    }
    __syncthreads();
    
    // Debug: Check if nin was populated (only for tree 0, thread 0)
    // DEBUG output disabled for production
    // if (tree_id == 0 && tid == 0) {
    //     integer_t nin_sum = 0;
    //     integer_t non_zero_count = 0;
    //     integer_t max_val = 0;
    //     integer_t max_idx = -1;
    //     integer_t zero_count = 0;
    //     
    //     // Also check distribution of jinbag to see if sampling is uniform
    //     integer_t jinbag_counts[10] = {0};  // Count how many times indices 0-9 appear in jinbag
    //     
    //     for (integer_t j = 0; j < nsample; ++j) {
    //         nin_sum += nin[j];
    //         if (nin[j] > 0) {
    //             non_zero_count++;
    //             if (nin[j] > max_val) {
    //                 max_val = nin[j];
    //                 max_idx = j;
    //             }
    //         } else {
    //             zero_count++;
    //         }
    //         // Check jinbag distribution
    //         if (jinbag[j] >= 0 && jinbag[j] < 10) {
    //             jinbag_counts[jinbag[j]]++;
    //         }
    //     }
    //     
    //     // Calculate expected in-bag percentage
    //     real_t expected_inbag_pct = 1.0f - powf(1.0f - 1.0f / static_cast<real_t>(nsample), static_cast<real_t>(nsample));
    //     real_t actual_inbag_pct = static_cast<real_t>(non_zero_count) / static_cast<real_t>(nsample);
    //     real_t actual_oob_pct = static_cast<real_t>(zero_count) / static_cast<real_t>(nsample);
    //     
    //     printf("GPU Bootstrap tree 0: nin_sum=%d (should be %d)\n", nin_sum, nsample);
    //     printf("GPU Bootstrap tree 0: in-bag=%d (%.1f%%), out-of-bag=%d (%.1f%%)\n", 
    //            non_zero_count, actual_inbag_pct * 100.0f, zero_count, actual_oob_pct * 100.0f);
    //     printf("GPU Bootstrap tree 0: Expected in-bag=%.1f%%, Expected out-of-bag=%.1f%%\n",
    //            expected_inbag_pct * 100.0f, (1.0f - expected_inbag_pct) * 100.0f);
    //     printf("GPU Bootstrap tree 0: max_val=%d at idx=%d\n", max_val, max_idx);
    //     
    //     // Also print first 10 values to see if they're actually set
    //     printf("GPU Bootstrap tree 0: First 10 nin values: ");
    //     for (integer_t j = 0; j < (nsample < 10 ? nsample : 10); ++j) {
    //         printf("%d ", static_cast<int>(nin[j]));
    //     }
    //     printf("\n");
    //     
    //     // Print some non-zero values if they exist
    //     if (non_zero_count > 0) {
    //         printf("GPU Bootstrap tree 0: First 10 non-zero nin values (idx,val): ");
    //         integer_t printed = 0;
    //         for (integer_t j = 0; j < nsample && printed < 10; ++j) {
    //             if (nin[j] > 0) {
    //                 printf("(%d,%d) ", j, static_cast<int>(nin[j]));
    //                 printed++;
    //             }
    //         }
    //         printf("\n");
    //     }
    // }
    __syncthreads();
    
    // Compute weights based on counts
    // For case-wise: win = nin * weight, for non-case-wise: win = 1.0
    // Note: This code is already handled above in the jinbag population loop, so this is redundant
    // But keep it for compatibility with existing code structure
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nin[n] > 0) {
            if (use_casewise) {
                win[n] = static_cast<real_t>(nin[n]) * (weight != nullptr ? weight[n] : 1.0f);
            } else {
                win[n] = 1.0f;  // Non-case-wise: equal weighting
            }
        } else {
            win[n] = 0.0f;
        }
    }
    
//     // Debug output
//     if (tree_id == 0 && threadIdx.x == 0) {
//         printf("GPU: Bootstrap kernel completed for tree %d\n", tree_id);
//         printf("GPU: First 10 jinbag values: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
//                jinbag[0], jinbag[1], jinbag[2], jinbag[3], jinbag[4],
//                jinbag[5], jinbag[6], jinbag[7], jinbag[8], jinbag[9]);
//         printf("GPU: First 10 nin values: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
//                nin[0], nin[1], nin[2], nin[3], nin[4],
//                nin[5], nin[6], nin[7], nin[8], nin[9]);
//     }
}

// Simple GPU tree growing kernel - creates single-node trees with majority class
__global__ void gpu_simple_tree_kernel(
    integer_t num_trees, integer_t nsample, integer_t mdim, integer_t maxnode, integer_t nclass,
    integer_t minndsize, integer_t mtry, const real_t* x,
    integer_t* nodestatus_all, integer_t* nodeclass_all, integer_t* nnode_all,
    integer_t* bestvar_all, real_t* xbestsplit_all, integer_t* treemap_all,
    integer_t* jinbag_all, const integer_t* cl, const real_t* win) {
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t jinbag_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    integer_t* jinbag = jinbag_all + jinbag_offset;
    
    // Initialize tree arrays
    for (integer_t i = 0; i < maxnode; ++i) {
        nodestatus[i] = 0;
        nodeclass[i] = 0;
        bestvar[i] = 0;
        xbestsplit[i] = 0.0f;
    }
    for (integer_t i = 0; i < 2 * maxnode; ++i) {
        treemap[i] = 0;
    }
    
    // Count samples in jinbag for this tree
    integer_t ninbag = 0;
    for (integer_t i = 0; i < nsample; ++i) {
        if (jinbag[i] >= 0 && jinbag[i] < nsample) {
            ninbag++;
        } else {
            break;  // jinbag is packed, so first invalid index means end
        }
    }
    
    // Create simple single-node tree that predicts majority class
    if (ninbag > 0) {
        // Count class distribution for all bootstrap samples
        real_t class_counts[10] = {0.0f};
        
        for (integer_t i = 0; i < ninbag && i < nsample; ++i) {
            integer_t sample_idx = jinbag[i];
            if (sample_idx >= 0 && sample_idx < nsample) {
                integer_t class_label = cl[sample_idx];
                if (class_label >= 0 && class_label < nclass) {
                    class_counts[class_label] += 1.0f;
                }
            }
        }
        
        // Find majority class (in case of tie, pick the last one with max count)
        integer_t majority_class = 0;
        real_t max_count = class_counts[0];
        for (integer_t c = 1; c < nclass; ++c) {
            if (class_counts[c] >= max_count) {  // Use >= to pick last in case of tie
                max_count = class_counts[c];
                majority_class = c;
            }
        }
        
        // // Debug: Print class counts for verification
        // if (tree_id == 0 && threadIdx.x == 0) {
        //     printf("GPU Tree %d: Class counts: [%.1f, %.1f, %.1f], majority: %d\n", 
        //            tree_id, class_counts[0], class_counts[1], class_counts[2], majority_class);
        // }
        
        // Set root node as terminal with majority class
        nodestatus[0] = -1;  // Terminal node
        nodeclass[0] = majority_class;
        nnode_all[tree_id] = 1;
        
        // // Debug output
        // if (tree_id == 0 && threadIdx.x == 0) {
        //     printf("GPU Tree %d: Simple tree - majority class %d (counts: %.1f,%.1f,%.1f)\n", 
        //            tree_id, majority_class, class_counts[0], class_counts[1], class_counts[2]);
        //     printf("GPU Tree %d: First 10 jinbag samples: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        //            tree_id, jinbag[0], jinbag[1], jinbag[2], jinbag[3], jinbag[4],
        //            jinbag[5], jinbag[6], jinbag[7], jinbag[8], jinbag[9]);
        //     printf("GPU Tree %d: First 10 class labels: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        //            tree_id, cl[jinbag[0]], cl[jinbag[1]], cl[jinbag[2]], cl[jinbag[3]], cl[jinbag[4]],
        //            cl[jinbag[5]], cl[jinbag[6]], cl[jinbag[7]], cl[jinbag[8]], cl[jinbag[9]]);
        // }
    }
    
}

// Device function for categorical split optimization (small categories)
__device__ void gpu_catmax(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
                          integer_t nnz, integer_t numcat, integer_t nclass,
                          integer_t* igl, integer_t* itmp, real_t* tmpclass,
                          const real_t* tcat, const integer_t* icat, real_t& critmax,
                          integer_t* igoleft, integer_t& nhit) {
    // Simplified implementation - find best categorical split
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Try different category groupings (one category goes left)
    for (integer_t i = 0; i < nnz; ++i) {
        integer_t cat_idx = icat[i];
        
        // Initialize igoleft
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = 0;
        }
        igoleft[cat_idx] = 1;
        
        // Calculate criterion for this split
        real_t crit = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                }
            }
        }
        
        // Calculate Gini-based criterion (matching CPU implementation)
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
            sum_right += tclasspop[j] - tmpclass[j];
        }
        
        if (sum_left > 1e-5f && sum_right > 1e-5f) {
            // Calculate Gini: crit = (1-Gini_left) + (1-Gini_right)
            real_t pno_left = 0.0f;
            real_t pno_right = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno_left += tmpclass[j] * tmpclass[j];
                pno_right += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
            }
            crit = (pno_left / sum_left) + (pno_right / sum_right);
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Copy best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    igoleft[k] = (k == cat_idx) ? 1 : 0;
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    } else {
        nhit = 0;
    }
}

// Device function for categorical split optimization (large categories)
__device__ void gpu_catmaxr(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
                            integer_t numcat, integer_t nclass, integer_t* igl,
                            integer_t ncsplit, real_t* tmpclass,
                            real_t& critmax, integer_t* igoleft, integer_t& nhit,
                            curandState* local_state) {
    // Simplified implementation for large categorical variables
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Random sampling approach for large categories
    for (integer_t split = 0; split < ncsplit; ++split) {
        // Initialize igoleft randomly
        for (integer_t k = 0; k < numcat; ++k) {
            real_t rand_val = curand_uniform(local_state);
            igoleft[k] = (rand_val > 0.5f) ? 1 : 0;
        }
        
        // Calculate criterion
        real_t crit = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                }
            }
        }
        
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
            sum_right += tclasspop[j] - tmpclass[j];
        }
        
        if (sum_left > 1e-5f && sum_right > 1e-5f) {
            // Calculate Gini-based criterion
            real_t pno_left = 0.0f;
            real_t pno_right = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno_left += tmpclass[j] * tmpclass[j];
                pno_right += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
            }
            crit = (pno_left / sum_left) + (pno_right / sum_right);
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Copy best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    igoleft[k] = igoleft[k];  // Already set above
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    } else {
        nhit = 0;
    }
}

// Full GPU tree growing kernel based on original Fortran growtree.f algorithm
// Converted to row-major, 0-based indexing with batch parallel GPU execution
__global__ void gpu_full_tree_kernel(
    integer_t num_trees, integer_t nsample, integer_t mdim, integer_t maxnode, integer_t nclass,
    integer_t minndsize, integer_t mtry, const real_t* x,
    integer_t* nodestatus_all, integer_t* nodeclass_all, integer_t* nnode_all,
    integer_t* bestvar_all, real_t* xbestsplit_all, integer_t* treemap_all,
    integer_t* jinbag_all, const integer_t* cl, const real_t* win,
    const integer_t* cat,  // cat[m] == 1 means quantitative, cat[m] > 1 means categorical with cat[m] categories
    const integer_t* amat,  // For categorical: amat[m + n*mdim] stores category index (0-based)
    integer_t* catgoleft_all,  // catgoleft[k + node*maxcat] = 1 if category k goes left at node
    integer_t maxcat,  // Maximum number of categories
    integer_t ncsplit,  // Number of random splits for large categoricals
    integer_t ncmax,  // Threshold for large categorical
    integer_t loss_function,  // 0=Gini, 1=Binary logloss, 2=Multiclass crossentropy, 3=MSE, 4=MAE, etc.
    real_t min_gain_to_split,  // XGBoost/LightGBM style early stopping threshold
    integer_t growth_mode,  // 0=level-wise, 1=leaf-wise, 2=hybrid, 4=random selection
    const integer_t* allowed_modes,  // Allowed modes for random selection (when growth_mode=4)
    integer_t num_allowed_modes,  // Number of allowed modes
    real_t l1_reg,  // L1 regularization strength
    real_t l2_reg,  // L2 regularization strength
    bool gpu_parallel_mode0,
    curandState* random_states) {  // Pre-initialized random states (one per tree)
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Always use original RF growth mode (mode 0)
    integer_t actual_growth_mode = 0;

    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t jinbag_offset = tree_id * nsample;
    integer_t win_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    integer_t* nnode = nnode_all + tree_id;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    const integer_t* jinbag_tree = jinbag_all + jinbag_offset;
    const real_t* win_tree = win + win_offset;
    // Note: catgoleft is accessed per-node, not per-tree offset
    // catgoleft[k + node*maxcat] where node is the node index within the tree
    // We need to calculate the full offset: tree_offset * maxnode * maxcat + node * maxcat
    // For now, we'll access it directly using kgrow (node index) within the tree
    // The full array is: catgoleft_all[tree_id * maxnode * maxcat + node * maxcat + category]
    
    // Shared memory for thread synchronization
    __shared__ integer_t shared_left_child;
    __shared__ integer_t shared_right_child;
    __shared__ bool shared_should_split;
    __shared__ integer_t shared_jstat;
    __shared__ integer_t shared_best_var;
    __shared__ integer_t shared_best_split_idx;
    __shared__ real_t shared_best_crit;
    
    // OPTIMIZATION: Shared memory for incremental sample tracking
    __shared__ integer_t shared_node_samples[4096];  // Samples that reach current node
    __shared__ integer_t shared_node_sample_count;

    // Initialize tree structure (following original Fortran algorithm)
    if (threadIdx.x == 0) {
        // Initialize arrays (following original algorithm)
        for (integer_t i = 0; i < maxnode; ++i) {
            nodestatus[i] = 0;
            bestvar[i] = 0;
            xbestsplit[i] = 0.0f;
            treemap[i * 2] = 0;
            treemap[i * 2 + 1] = 0;
        }
        
        // Initialize root node (following original algorithm)
        nodestatus[0] = 2;  // Root node is active
        nnode[0] = 1;       // Start with 1 node
    }
    __syncthreads();
    
    // Helper function to calculate node depth (must be declared before use)
    auto calculate_node_depth = [&](integer_t node_idx) -> integer_t {
        integer_t depth = 0;
        integer_t current = node_idx;
        
        // Traverse up to root to count depth
        while (current > 0) {
            // Find parent by searching backwards
            bool parent_found = false;
            for (integer_t p = 0; p < current; ++p) {
                if (nodestatus[p] == 1) {  // Split node
                    if (treemap[p * 2] == current || treemap[p * 2 + 1] == current) {
                        current = p;
                        depth++;
                        parent_found = true;
                        break;
                    }
                }
            }
            if (!parent_found) break;
        }
        return depth;
    };
    
    // The loop condition checks nnode[0] which increases as nodes are split
    // SAFETY: Cache nnode[0] at start and add max iteration limit to prevent infinite loops
    integer_t initial_nnode = nnode[0];
    integer_t max_iterations = maxnode * 2;  // Safety limit: should never exceed maxnode nodes
    integer_t iteration_count = 0;
    
    for (integer_t kgrow = 0; kgrow < nnode[0] && iteration_count < max_iterations; ++kgrow, ++iteration_count) {
        // Safety check: if we've processed too many iterations, break
        if (iteration_count >= max_iterations) {
            if (threadIdx.x == 0) {
                // Reduced debug output - only print if safety break occurs
                // printf("DEBUG KERNEL: Tree %d SAFETY BREAK - iteration_count=%d >= max_iterations=%d, nnode[0]=%d\n",
                //        tree_id, iteration_count, max_iterations, nnode[0]);
            }
            break;
        }
        
        // Check if this node should be processed (status == 2 means active/not yet split)
        // Matching Fortran: if (nodestatus(kgrow).ne.2) goto 20 (skip to next iteration)
        if (nodestatus[kgrow] != 2) {
            continue;  // Skip nodes that aren't active (already split or terminal)
        }
        
        // Reduced debug output - disabled for production
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow % 20 == 0 && kgrow < 100) {
        //     printf("DEBUG KERNEL: Tree %d processing node %d/%d\n", tree_id, kgrow, nnode[0]);
        // }
        
        // Get samples that reach this node (following original node-wise filtering)
        // OPTIMIZATION: Increased array size and use shared memory for better performance
        integer_t node_samples[2048];  // Increased from 1024 to handle larger nodes
        integer_t node_sample_count = 0;
        
        // Reduced debug output
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50) {
        //     printf("DEBUG KERNEL: Starting sample collection for node %d\n", kgrow);
        // }
        
        // For root node, use all bootstrap samples (thread 0 only for simplicity)
        if (kgrow == 0) {
                if (threadIdx.x == 0) {
                    integer_t ninbag_count = 0;
                    for (integer_t i = 0; i < nsample; ++i) {
                        if (jinbag_tree[i] >= 0 && jinbag_tree[i] < nsample) {
                            if (node_sample_count < 2048) {
                                node_samples[node_sample_count] = jinbag_tree[i];
                                node_sample_count++;
                            }
                            ninbag_count++;
                        } else {
                            break;
                        }
                    }
                }
            } else {
            // For non-root nodes: Find parent and filter samples (thread 0 only for simplicity)
                if (threadIdx.x == 0) {
            // Find parent node by searching backwards through treemap
            integer_t parent_node = -1;
            
            // Search for parent: find node where treemap points to kgrow
            for (integer_t p = 0; p < kgrow; ++p) {
                if (nodestatus[p] == 1) {  // Only check split nodes
                        if (treemap[p * 2] == kgrow || treemap[p * 2 + 1] == kgrow) {
                        parent_node = p;
                        break;
                    }
                }
            }
            
            if (parent_node >= 0) {
                    // Traverse from root to this node for each sample
                    for (integer_t sample_idx = 0; sample_idx < nsample && node_sample_count < 2048; ++sample_idx) {
                    if (win_tree[sample_idx] <= 0.0f) {
                        continue;
                    }
                    
                        // Traverse from root to this node
                        integer_t current_node = 0;
                        bool reaches_node = true;
                        
                        while (current_node != kgrow && reaches_node) {
                            if (nodestatus[current_node] == 1) {
                                integer_t split_var = bestvar[current_node];
                                integer_t split_numcat = (cat != nullptr && split_var < mdim) ? cat[split_var] : 1;
                                bool split_is_categorical = (split_numcat > 1);
                                
                                if (split_is_categorical) {
                        integer_t category_idx = 0;
                        if (amat != nullptr) {
                                        category_idx = amat[split_var + sample_idx * mdim];
                        } else {
                                        category_idx = static_cast<integer_t>(x[sample_idx * mdim + split_var] + 0.5f);
                                    }
                                    integer_t catgoleft_offset = tree_id * maxnode * maxcat + current_node * maxcat;
                                    if (category_idx >= 0 && category_idx < maxcat && 
                                        catgoleft_all[catgoleft_offset + category_idx] == 1) {
                                        current_node = treemap[current_node * 2];
                    } else {
                                        current_node = treemap[current_node * 2 + 1];
                                    }
                                } else {
                                    real_t split_point = xbestsplit[current_node];
                                    real_t sample_value = x[sample_idx * mdim + split_var];
                                    
                                    if (sample_value <= split_point) {
                                        current_node = treemap[current_node * 2];
                                    } else {
                                        current_node = treemap[current_node * 2 + 1];
                                    }
                                }
                                
                                if (current_node > kgrow || current_node < 0 || current_node >= maxnode) {
                                    reaches_node = false;
                                    break;
                                }
                            } else if (nodestatus[current_node] == -1) {
                                reaches_node = false;
                                break;
                            } else {
                                reaches_node = false;
                                break;
                            }
                        }
                        
                        if (reaches_node && current_node == kgrow) {
                            node_samples[node_sample_count] = sample_idx;
                            node_sample_count++;
                        }
                    }
                }
                // If parent_node < 0, node_sample_count stays at 0
            }
        }
        
        // Reduced debug output - disabled for production
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50) {
        //     printf("DEBUG KERNEL: Node %d has %d samples\n", kgrow, node_sample_count);
        // }
        
        // Check if node should be split (following original algorithm)
        // For regression: need at least 2*minndsize to split (minndsize in each child)
        integer_t min_size_to_split = (nclass == 1) ? (2 * minndsize) : minndsize;
        if (node_sample_count < min_size_to_split) {
            // Node is too small, make it terminal
            // ALL threads must execute this to avoid divergence
            if (threadIdx.x == 0) {
                // Reduced debug output
                // if (tree_id == 0 && kgrow < 10) {
                //     printf("DEBUG KERNEL: Node %d too small (%d < %d), making terminal\n", kgrow, node_sample_count, min_size_to_split);
                // }
                nodestatus[kgrow] = -1;  // Terminal node
            }
            // ALL threads continue together
            continue;
        }
        
        // Handle regression vs classification differently
        integer_t non_zero_classes = 0;
        integer_t majority_class = 0;
        real_t max_count = 0.0f;
        real_t variance = 0.0f;
        real_t crit0 = 0.0f;
        
        // Declare class_counts outside so it's available for crit0 calculation
        real_t class_counts[10] = {0.0f};  // Max 10 classes
        
        if (nclass == 1) {
            // Regression: calculate weighted variance
            real_t sum_y = 0.0f;
            real_t sum_y2 = 0.0f;
            real_t sum_w = 0.0f;
            for (integer_t i = 0; i < node_sample_count; ++i) {
                integer_t sample_idx = node_samples[i];
                real_t w = win_tree[sample_idx];  // Use case weight
                real_t y_val = static_cast<real_t>(cl[sample_idx]);  // Convert from integer
                sum_y += w * y_val;
                sum_y2 += w * y_val * y_val;
                sum_w += w;
            }
            real_t mean_y = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
            variance = (sum_w > 0.0f) ? ((sum_y2 / sum_w) - mean_y * mean_y) : 0.0f;
            crit0 = variance;  // Initial MSE for regression
            
            // If variance is very small, make terminal
            if (variance < 1.0e-6f) {
                if (threadIdx.x == 0) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = 0;
                }
                continue;
            }
        } else {
            // Classification: calculate weighted class distribution
        for (integer_t j = 0; j < nclass; ++j) {
            class_counts[j] = 0.0f;
        }
        
        for (integer_t i = 0; i < node_sample_count; ++i) {
            integer_t sample_idx = node_samples[i];
                real_t w = win_tree[sample_idx];  // Use case weight
            integer_t class_label = cl[sample_idx];
            if (class_label >= 0 && class_label < nclass) {
                    class_counts[class_label] += w;  // Weighted count
            }
        }
        
        for (integer_t j = 0; j < nclass; ++j) {
            if (class_counts[j] > 0.0f) {
                non_zero_classes++;
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
        }
        
            // Calculate initial criterion based on loss function
            if (nclass == 1) {
                // Regression: use MSE (loss_function == 3)
                crit0 = variance;
            } else {
                // Classification: use Gini impurity 
                real_t pno = 0.0f;
                real_t pdo = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    pno += class_counts[j] * class_counts[j];
                    pdo += class_counts[j];
                }
                crit0 = pdo > 0.0f ? (pno / pdo) : 0.0f;
            }
            
            // Check if node is pure (all samples same class)
        if (non_zero_classes <= 1) {
            // Node is pure, make it terminal
            if (threadIdx.x == 0) {
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
            }
            continue;
            }
        }
        
        // Find best split using MSE (regression) or Gini (classification)
        // For classification: crit = (rln/rld)+(rrn/rrd) = (1-Gini_left) + (1-Gini_right)
        // We MAXIMIZE crit (which minimizes Gini), matching Fortran algorithm
        real_t best_crit = (nclass == 1) ? 1.0e20f : -2.0f;  // Regression: minimize MSE, Classification: maximize crit
        integer_t best_var = 0;
        integer_t best_split_idx = 0;
        integer_t jstat = 0;
        
        // Try mtry random variables (following original algorithm)
    for (integer_t mv = 0; mv < mtry; ++mv) {
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == 0) {
            //     printf("DEBUG KERNEL: Node %d: Starting split search, mtry=%d\n", kgrow, mtry);
            // }
            
            // Generate random variable (following original: int(mdim*randomu())+1)
            // Use pre-initialized random state with hash-based skip
            // Advancing RNG state in a loop is still too slow for parallel mode
            curandState local_state = random_states[tree_id];
            // Use skipahead to efficiently jump to the right position
            // Each node*mtry combination gets a unique position in the sequence
            skipahead(kgrow * mtry + mv, &local_state);
            real_t rand_val = curand_uniform(&local_state);
            integer_t mvar = static_cast<integer_t>(rand_val * mdim);
            if (mvar >= mdim) mvar = mdim - 1;
            
            // Check if this is a categorical variable
            integer_t numcat = (cat != nullptr && mvar < mdim) ? cat[mvar] : 1;
            bool is_categorical = (numcat > 1);
            
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == 0) {
            //     printf("DEBUG KERNEL: Node %d: Selected variable %d, is_categorical=%d, numcat=%d\n", kgrow, mvar, is_categorical, numcat);
            // }
            
            // CATEGORICAL VARIABLE HANDLING
            if (is_categorical && nclass > 1) {  // Only classification supports categoricals for now
                // Build tclasscat matrix: tclasscat[class + category*nclass]
                real_t tclasscat[10 * 10] = {0.0f};  // Max 10 classes, 10 categories
                real_t tclasspop[10] = {0.0f};
                real_t tcat[10] = {0.0f};
                integer_t icat[10] = {0};
                
                // Count class distribution per category
                for (integer_t i = 0; i < node_sample_count; ++i) {
                    integer_t sample_idx = node_samples[i];
                    real_t w = win_tree[sample_idx];
                    integer_t class_label = cl[sample_idx];
                    
                    // Get category index from amat (or from x if amat not available)
                    integer_t category_idx = 0;
                    if (amat != nullptr) {
                        category_idx = amat[mvar + sample_idx * mdim];
                    } else {
                        // Fallback: use x value rounded to integer
                        category_idx = static_cast<integer_t>(x[sample_idx * mdim + mvar] + 0.5f);
                    }
                    
                    if (category_idx >= 0 && category_idx < numcat && class_label >= 0 && class_label < nclass) {
                        tclasscat[class_label + category_idx * nclass] += w;
                        tclasspop[class_label] += w;
                    }
                }
                
                // Find non-zero categories
                integer_t nnz = 0;
                for (integer_t k = 0; k < numcat; ++k) {
                    tcat[k] = 0.0f;
                    for (integer_t j = 0; j < nclass; ++j) {
                        tcat[k] += tclasscat[j + k * nclass];
                    }
                    if (tcat[k] > 0.0f) {
                        icat[nnz] = k;
                        nnz++;
                    }
                }
                
                // Only try categorical split if more than one category present
                if (nnz > 1) {
                    integer_t igoleft[10] = {0};
                    integer_t igl[10] = {0};
                    integer_t itmp[10] = {0};
                    real_t tmpclass[10] = {0.0f};
                    integer_t nhit = 0;
                    real_t cat_critmax = best_crit;
                    
                    // Calculate pdo (total weight)
                    real_t pdo = 0.0f;
                    for (integer_t j = 0; j < nclass; ++j) {
                        pdo += tclasspop[j];
                    }
                    
                    // Use appropriate categorical split algorithm
                    if (numcat < ncmax) {
                        // Small categorical: try all category groupings
                        gpu_catmax(tclasscat, tclasspop, pdo, nnz, numcat, nclass,
                                  igl, itmp, tmpclass, tcat, icat, cat_critmax, igoleft, nhit);
                    } else {
                        // Large categorical: random sampling
                        gpu_catmaxr(tclasscat, tclasspop, pdo, numcat, nclass, igl,
                                   ncsplit, tmpclass, cat_critmax, igoleft, nhit, &local_state);
                    }
                    
                    // Update best split if categorical split is better
                    if (nhit == 1 && cat_critmax > best_crit) {
                        best_crit = cat_critmax;
                        best_var = mvar;
                        jstat = 1;
                        
                        // Store catgoleft for this node
                        integer_t catgoleft_offset = tree_id * maxnode * maxcat + kgrow * maxcat;  // Full offset
                        for (integer_t k = 0; k < numcat; ++k) {
                            if (k < maxcat) {
                                catgoleft_all[catgoleft_offset + k] = igoleft[k];
                            }
                        }
                        
                        // Set xbestsplit to 0.0 for categorical (not used)
                        if (threadIdx.x == 0) {
                            xbestsplit[kgrow] = 0.0f;
                        }
                    }
                }
                
                // Continue to next variable
                continue;
            }
            
            // QUANTITATIVE (CONTINUOUS) VARIABLE HANDLING
            // Initialize variables for splitting
            real_t rrn = 0.0f;
            real_t rrd = 0.0f;
            real_t rln = 0.0f;
            real_t rld = 0.0f;

            // Initialize left and right class weights (only for classification)
            real_t wl[10] = {0.0f};
            real_t wr[10] = {0.0f};
            if (nclass > 1) {
                // Calculate initial Gini values for classification
            for (integer_t j = 0; j < nclass; ++j) {
                    rrn += class_counts[j] * class_counts[j];
                    rrd += class_counts[j];
                wr[j] = class_counts[j];
                }
            }
            
            // OPTIMIZATION: Use insertion sort for small arrays, parallel approach for larger
            // Insertion sort is faster for small arrays (< 32 elements)
            if (node_sample_count <= 32) {
                // Insertion sort (faster for small arrays)
                for (integer_t i = 1; i < node_sample_count; ++i) {
                    integer_t key = node_samples[i];
                    real_t key_val = x[key * mdim + mvar];
                    integer_t j = i - 1;
                    
                    while (j >= 0 && x[node_samples[j] * mdim + mvar] > key_val) {
                        node_samples[j + 1] = node_samples[j];
                        j--;
                    }
                    node_samples[j + 1] = key;
                }
            } else {
                // Parallel bubble sort using threads (better than sequential for larger arrays)
                integer_t n = node_sample_count;
                for (integer_t i = 0; i < n - 1; ++i) {
                    // Each thread handles pairs starting at different offsets
                    integer_t start = threadIdx.x * 2;
                    integer_t stride = blockDim.x * 2;
                    
                    for (integer_t j = start; j < n - 1 - i; j += stride) {
                        if (j + 1 < n - i) {
                            real_t val1 = x[node_samples[j] * mdim + mvar];
                            real_t val2 = x[node_samples[j + 1] * mdim + mvar];
                            if (val1 > val2) {
                                integer_t temp = node_samples[j];
                                node_samples[j] = node_samples[j + 1];
                                node_samples[j + 1] = temp;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == 0) {
            //     printf("DEBUG KERNEL: Node %d: Sort completed, trying splits\n", kgrow);
            // }
            
            // Try splits at sample boundaries
            for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                integer_t nc = node_samples[ii];
                integer_t ncnext = node_samples[ii + 1];
                real_t val1 = x[nc * mdim + mvar];
                real_t val2 = x[ncnext * mdim + mvar];
                
                // Skip ties
                if (val1 >= val2) {
                    if (nclass > 1) {
                        // Still update Gini for classification even if tie
                integer_t k = cl[nc];
                        real_t u = win_tree[nc];  // Use case weight
                rln += u * (u + 2.0f * wl[k]);
                rrn += u * (u - 2.0f * wr[k]);
                rld += u;
                rrd -= u;
                wl[k] += u;
                wr[k] -= u;
                    }
                    continue;
                }
                
                if (nclass == 1) {
                    // Regression: Calculate gradients and Hessians
                    // For MSE: gradient g_i = y_pred - y_i, Hessian h_i = 1
                    
                    // Calculate parent node gradients and Hessians
                    real_t G_parent = 0.0f;  // Sum of gradients
                    real_t H_parent = 0.0f;  // Sum of Hessians
                    real_t sum_w_parent = 0.0f;
                    real_t mean_parent = 0.0f;
                    
                    // First pass: calculate parent mean for gradient calculation
                    for (integer_t i = 0; i < node_sample_count; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        mean_parent += w * y_val;
                        sum_w_parent += w;
                    }
                    mean_parent = (sum_w_parent > 0.0f) ? (mean_parent / sum_w_parent) : 0.0f;
                    
                    // Calculate parent gradients and Hessians
                    for (integer_t i = 0; i < node_sample_count; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        real_t g_i = mean_parent - y_val;  // Gradient: residual
                        real_t h_i = 1.0f;  // Hessian: constant for MSE
                        G_parent += w * g_i;
                        H_parent += w * h_i;
                    }
                    
                    // Calculate left child gradients and Hessians
                    real_t G_left = 0.0f;
                    real_t H_left = 0.0f;
                    real_t sum_w_left = 0.0f;
                    real_t mean_left = 0.0f;
                    
                    for (integer_t i = 0; i <= ii; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        mean_left += w * y_val;
                        sum_w_left += w;
                    }
                    mean_left = (sum_w_left > 0.0f) ? (mean_left / sum_w_left) : 0.0f;
                    
                    for (integer_t i = 0; i <= ii; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        real_t g_i = mean_left - y_val;
                        real_t h_i = 1.0f;
                        G_left += w * g_i;
                        H_left += w * h_i;
                    }
                    
                    // Calculate right child gradients and Hessians
                    real_t G_right = 0.0f;
                    real_t H_right = 0.0f;
                    real_t sum_w_right = 0.0f;
                    real_t mean_right = 0.0f;
                    
                    for (integer_t i = ii + 1; i < node_sample_count; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        mean_right += w * y_val;
                        sum_w_right += w;
                    }
                    mean_right = (sum_w_right > 0.0f) ? (mean_right / sum_w_right) : 0.0f;
                    
                    for (integer_t i = ii + 1; i < node_sample_count; ++i) {
                        integer_t sample_idx = node_samples[i];
                        real_t w = win_tree[sample_idx];
                        real_t y_val = static_cast<real_t>(cl[sample_idx]);
                        real_t g_i = mean_right - y_val;
                        real_t h_i = 1.0f;
                        G_right += w * g_i;
                        H_right += w * h_i;
                    }
                    
                    real_t crit = 0.0f;
                    
                    // Mode 0 (Normal RF): Use pure MSE (Breiman 2001) - no gradient boosting gain
                    if (actual_growth_mode == 0) {
                        // Pure Breiman (2001): Use MSE (variance) only, no gradient boosting gain or regularization
                        // For regression, crit is the weighted variance (MSE)
                        // mean_left and mean_right are already calculated above (lines 1797-1831)
                        // Calculate variance for left and right children
                        real_t var_left = 0.0f;
                        real_t var_right = 0.0f;
                        for (integer_t i = 0; i <= ii; ++i) {
                            integer_t sample_idx = node_samples[i];
                            real_t w = win_tree[sample_idx];
                            real_t y_val = static_cast<real_t>(cl[sample_idx]);
                            real_t diff = y_val - mean_left;
                            var_left += w * diff * diff;
                        }
                        for (integer_t i = ii + 1; i < node_sample_count; ++i) {
                            integer_t sample_idx = node_samples[i];
                            real_t w = win_tree[sample_idx];
                            real_t y_val = static_cast<real_t>(cl[sample_idx]);
                            real_t diff = y_val - mean_right;
                            var_right += w * diff * diff;
                        }
                        
                        // Normalize by weights
                        var_left = (sum_w_left > 0.0f) ? (var_left / sum_w_left) : 0.0f;
                        var_right = (sum_w_right > 0.0f) ? (var_right / sum_w_right) : 0.0f;
                        
                        // Total variance (minimize this) - Original RF (mode 0)
                        crit = var_left + var_right;
                    }
                    
                    // Update best split (for regression, lower crit = higher gain is better)
                    if (crit < best_crit) {
                        best_split_idx = ii;
                        best_crit = crit;
                        best_var = mvar;
                    }
                } else {
                    // Classification: Use Gini impurity (loss_function == 0)
                    integer_t k = cl[nc];
                    real_t u = win_tree[nc];  // Use case weight
                    
                    // Gini impurity calculation (Original RF - mode 0)
                    rln += u * (u + 2.0f * wl[k]);
                    rrn += u * (u - 2.0f * wr[k]);
                    rld += u;
                    rrd -= u;
                    wl[k] += u;
                    wr[k] -= u;
                    
                    if (min(rrd, rld) > 1.0e-5f) {
                        // Original RF (mode 0): Use Gini impurity only
                        // Gini crit = (rln / rld) + (rrn / rrd)
                        real_t crit = (rln / rld) + (rrn / rrd);
                        
                        if (crit > best_crit) {
                            best_split_idx = ii;
                            best_crit = crit;
                            best_var = mvar;
                        }
                    }
                }
                }
            
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == mtry - 1) {
            //     printf("DEBUG KERNEL: Node %d: Finished trying all %d variables, best_crit=%.6f\n", kgrow, mtry, best_crit);
            // }
        }
        
        // Reduced debug output
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50) {
        //     printf("DEBUG KERNEL: Node %d: Split search completed, jstat=%d, best_crit=%.6f, crit0=%.6f\n", kgrow, jstat, best_crit, crit0);
        // }
        
        // Calculate gain for this node
        real_t node_gain = 0.0f;
        if (nclass == 1) {
            // Regression: gain = loss reduction (MSE/MAE reduction)
            node_gain = crit0 - best_crit;
            if (node_gain < 1.0e-6f) {
                jstat = 1;  // No significant improvement
            }
            // Original RF (mode 0): no gain threshold check
        } else {
            // Classification: calculate gain using Gini impurity (loss_function == 0)
            // Gini: gain = improvement in crit
            node_gain = best_crit - crit0;
            
        if (best_crit < -1.0f) {
                jstat = 1;  // No good split found (matching Fortran and CPU algorithm)
            }
            // Original RF (mode 0): no gain threshold check
        }
        
        // Original RF (mode 0): no gain accumulation needed
        
        // Check if we found a good split and create children if so
        // Only thread 0 modifies nnode[0] and tree structure
        if (threadIdx.x == 0) {
            // Reduced debug output
            // if (tree_id == 0 && kgrow < 10) {
            //     printf("DEBUG KERNEL: Node %d split check: jstat=%d, best_crit=%.6f, crit0=%.6f\n", 
            //            kgrow, jstat, best_crit, crit0);
            // }
            
            if (jstat == 0 && ((nclass == 1 && best_crit < crit0) || (nclass > 1 && best_crit > -1.0f))) {
            // Create children
            if (nnode[0] + 2 <= maxnode) {
                integer_t left_child = nnode[0];
                integer_t right_child = nnode[0] + 1;
                    
                    // Reduced debug output
                    // if (tree_id == 0 && kgrow < 10) {
                    //     printf("DEBUG KERNEL: Splitting node %d, creating children %d,%d, nnode[0] will be %d\n", 
                    //            kgrow, left_child, right_child, nnode[0] + 2);
                    // }
                
                // Set up split
                nodestatus[kgrow] = 1;  // Split node
                bestvar[kgrow] = best_var;
                
                    // Calculate split point (matching Fortran: average of two consecutive values)
                    if (best_split_idx < node_sample_count - 1) {
                        integer_t sample_idx1 = node_samples[best_split_idx];
                        integer_t sample_idx2 = node_samples[best_split_idx + 1];
                        real_t val1 = x[sample_idx1 * mdim + best_var];
                        real_t val2 = x[sample_idx2 * mdim + best_var];
                        xbestsplit[kgrow] = (val1 + val2) / 2.0f;  // Average of two consecutive values
                    } else if (best_split_idx < node_sample_count) {
                    integer_t sample_idx = node_samples[best_split_idx];
                    xbestsplit[kgrow] = x[sample_idx * mdim + best_var];
                } else {
                    xbestsplit[kgrow] = 0.0f;
                }
                
                // Set up children
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;   // Active
                nodestatus[right_child] = 2;  // Active
                nnode[0] += 2;
                
                    // Original RF (mode 0): no gain accumulation
                    
                    // Reduced debug output
                    // if (tree_id == 0 && kgrow < 100) {
                    //     printf("DEBUG KERNEL: After split, nnode[0]=%d, node_gain=%.6f, depth=%d\n", 
                    //            nnode[0], node_gain, calculate_node_depth(kgrow));
                // }
            } else {
                // No more nodes available, make terminal
                    // Reduced debug output
                    // if (tree_id == 0 && kgrow < 10) {
                    //     printf("DEBUG KERNEL: Node %d max nodes reached, making terminal\n", kgrow);
                    // }
                    nodestatus[kgrow] = -1;  // Terminal node
                    nodeclass[kgrow] = majority_class;
            }
        } else {
            // No good split found, make terminal
            // Reduced debug output
            // if (tree_id == 0 && kgrow < 10) {
            //     printf("DEBUG KERNEL: Node %d no good split, making terminal\n", kgrow);
            // }
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
        }
        }
    }  // End of kgrow loop
    
    // Reduced debug output - disabled for production
    if (threadIdx.x == 0) {
        // Only print for first tree to reduce output
        // if (tree_id < 3) {
        //     const char* mode_str = (actual_growth_mode == 0) ? "level-wise" : 
        //                           (actual_growth_mode == 1) ? "leaf-wise" : 
        //                           (actual_growth_mode == 2) ? "hybrid" : "random";
        //     printf("DEBUG KERNEL: Tree %d loop completed, final nnode[0]=%d, growth_mode=%s\n", 
        //            tree_id, nnode[0], mode_str);
        // }
    }
    
    // Finalize terminal nodes (set node classes for terminal nodes)
    // OPTIMIZATION: Skip this expensive O(nodes × samples × depth) traversal
    // nodeclass is already correctly set when nodes are made terminal above
    // This was causing hangs on large datasets (10k+ samples)
    // The redundant finalization is removed - nodeclass is set at lines 2121 and 2129
    if (threadIdx.x == 0) {
        // Just verify terminal nodes exist (for debugging)
        integer_t terminal_count = 0;
        for (integer_t node = 0; node < nnode[0]; ++node) {
            if (nodestatus[node] == -1) {
                terminal_count++;
            }
        }
        // Reduced debug output - disabled for production
        // Only print for first few trees to reduce output
        // if (tree_id < 3) {
        //     printf("DEBUG KERNEL: Tree %d completed with %d terminal nodes (classes already set), total nodes=%d\n", 
        //            tree_id, terminal_count, nnode[0]);
        // }
    }
}

// Device function to calculate Gini impurity (based on Fortran calcs subroutine)
__device__ real_t gpu_calcs(const real_t* tmpclass, const real_t* tclasspop, real_t pdo,
                           integer_t nclass) {
    real_t pln = 0.0f;
    real_t pld = 0.0f;
    real_t prn = 0.0f;
    
    // Calculate left side statistics
    for (integer_t j = 0; j < nclass; ++j) {
        pln += tmpclass[j] * tmpclass[j];
        pld += tmpclass[j];
        prn += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
    }
    
    real_t prd = pdo - pld;
    
    // Calculate left and right Gini ratios
    real_t rl = (pld > 0.0f) ? (pln / pld) : 0.0f;
    real_t rr = (prd > 0.0f) ? (prn / prd) : 0.0f;
    
    // Return total Gini impurity (lower is better)
    return rl + rr;
}

// GPU OOB vote accumulation kernel
// GPU kernel to compute nodextr for in-bag samples in parallel
// This replaces the expensive sequential CPU loop for 100k+ samples
__global__ void gpu_compute_nodextr_inbag_kernel(
    const real_t* x,  // Feature matrix (nsample × mdim)
    const integer_t* treemap_all,  // Tree mapping (num_trees × 2 × maxnode)
    const integer_t* bestvar_all,  // Best variable per node (num_trees × maxnode)
    const real_t* xbestsplit_all,  // Best split value per node (num_trees × maxnode)
    const integer_t* nodestatus_all,  // Node status (num_trees × maxnode)
    const integer_t* nin,  // In-bag indicator (num_trees × nsample)
    const integer_t* cat,  // Categorical indicator (mdim)
    const integer_t* catgoleft_all,  // Category go-left array (num_trees × maxnode × maxcat)
    integer_t* nodextr_all,  // Output: terminal node for each sample (num_trees × nsample)
    integer_t nsample,  // Number of samples
    integer_t mdim,  // Number of features
    integer_t maxnode,  // Maximum nodes per tree
    integer_t maxcat,  // Maximum categories
    integer_t tree_id,  // Current tree ID
    integer_t tree_offset,  // Offset for this tree's nodes (tree_id * maxnode)
    integer_t nin_offset,  // Offset for this tree's nin array (tree_id * nsample)
    integer_t nnode) {  // Actual number of nodes in this tree
    
    // Each thread processes one sample
    integer_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nsample) return;
    
    // Process ALL samples (both in-bag and OOB) to compute nodextr
    // This replaces both cpu_testreebag (OOB) and the sequential CPU loop (in-bag)
    if (nodextr_all[nin_offset + n] == 0) {
        // Traverse tree from root to terminal node
        integer_t kt = 0;  // Start at root node
        integer_t traversal_count = 0;
        const integer_t max_traversal = 1000;  // Safety limit
        
        while (nodestatus_all[tree_offset + kt] == 1 && traversal_count < max_traversal) {
            traversal_count++;
            integer_t mvar = bestvar_all[tree_offset + kt];
            
            if (mvar < 0 || mvar >= mdim) {
                break;
            }
            
            // Check if this is a quantitative or categorical variable
            bool is_categorical = (cat != nullptr && cat[mvar] > 1);
            
            if (!is_categorical) {
                // Quantitative variable
                real_t sample_value = x[n * mdim + mvar];
                real_t split_point = xbestsplit_all[tree_offset + kt];
                
                if (sample_value <= split_point) {
                    kt = treemap_all[tree_offset * 2 + kt * 2];  // Left child
                } else {
                    kt = treemap_all[tree_offset * 2 + kt * 2 + 1];  // Right child
                }
            } else {
                // Categorical variable
                integer_t cat_value = static_cast<integer_t>(x[n * mdim + mvar] + 0.5f);
                integer_t numcat = cat[mvar];
                
                if (cat_value >= 0 && cat_value < numcat) {
                    integer_t catgoleft_offset = tree_id * maxnode * maxcat + kt * maxcat;
                    if (catgoleft_all[catgoleft_offset + cat_value] == 1) {
                        kt = treemap_all[tree_offset * 2 + kt * 2];  // Left child
                    } else {
                        kt = treemap_all[tree_offset * 2 + kt * 2 + 1];  // Right child
                    }
                } else {
                    // Invalid category value, go right
                    kt = treemap_all[tree_offset * 2 + kt * 2 + 1];  // Right child
                }
            }
            
            // Safety checks
            if (kt < 0 || kt >= nnode) {
                kt = 0;
                break;
            }
            
            // Check if we've reached a terminal node
            if (nodestatus_all[tree_offset + kt] != 1) {
                break;
            }
        }
        
        // Store terminal node index
        if (nodestatus_all[tree_offset + kt] == -1 || nodestatus_all[tree_offset + kt] == 2) {
            nodextr_all[nin_offset + n] = kt;
        }
    }
}

__global__ void gpu_oob_vote_kernel(
    integer_t num_trees, integer_t nsample, integer_t nclass, integer_t maxnode, integer_t mdim,
    const integer_t* nodestatus_all, const integer_t* nodeclass_all, 
    const integer_t* bestvar_all, const real_t* xbestsplit_all,
    const integer_t* treemap_all, const integer_t* nin_all, const integer_t* cl,
    const real_t* x, real_t* q_all) {
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    integer_t tid = threadIdx.x;
    integer_t stride = blockDim.x;
    
    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t nin_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    const integer_t* nodestatus = nodestatus_all + node_offset;
    const integer_t* nodeclass = nodeclass_all + node_offset;
    const integer_t* bestvar = bestvar_all + node_offset;
    const real_t* xbestsplit = xbestsplit_all + node_offset;
    const integer_t* treemap = treemap_all + treemap_offset;
    const integer_t* nin = nin_all + nin_offset;
    
    // For each sample, if it's OOB (nin[sample] == 0), add vote to q array
    for (integer_t sample = tid; sample < nsample; sample += stride) {
        // Check if this sample is OOB (not in bootstrap sample)
        if (nin[sample] == 0) {
            // Sample is OOB, traverse tree to get prediction
            integer_t prediction = 0;  // Default prediction
            integer_t current_node = 0;  // Start at root
            
            // Traverse tree until we reach a terminal node
            // Add safety check to prevent infinite loops
            integer_t max_traversal_steps = maxnode;
            integer_t traversal_steps = 0;
            while (nodestatus[current_node] != -1 && traversal_steps < max_traversal_steps) {
                traversal_steps++;
                if (nodestatus[current_node] == 1) {
                    // Node has been split, traverse to child
                    integer_t split_var = bestvar[current_node];
                    real_t split_point = xbestsplit[current_node];
                    
                    // Get sample's value for split variable
                    // Use actual feature values from x matrix (row-major layout)
                    real_t sample_value = x[sample * mdim + split_var];
                    
                    if (sample_value <= split_point) {
                        // Go left
                        current_node = treemap[current_node * 2];
                } else {
                        // Go right
                        current_node = treemap[current_node * 2 + 1];
                    }
                    
                    // Bounds check
                    if (current_node < 0 || current_node >= maxnode) {
                        // Invalid node index, use default prediction
                        prediction = 0;
                        break;
                    }
                } else if (nodestatus[current_node] == 2) {
                    // Node exists but hasn't been split yet (shouldn't happen in final tree)
                    // Use majority class
                    prediction = nodeclass[current_node];
                    break;
                } else {
                    // Unknown node status, use default prediction
                    prediction = 0;
                    break;
                }
            }
            
            // If we reached a terminal node, use its class
            if (nodestatus[current_node] == -1) {
                prediction = nodeclass[current_node];
            } else if (traversal_steps >= max_traversal_steps) {
                // Hit traversal limit, use default prediction
                prediction = 0;
            }
            
            // Add vote to q array (shared across all trees)
            if (prediction >= 0 && prediction < nclass) {
                integer_t q_idx = sample * nclass + prediction;
                ::atomicAdd(&q_all[q_idx], 1.0f);
                
                // DEBUG: Print first few predictions for first tree
                if (tree_id == 0 && sample < 5 && tid == 0) {
                    // printf("DEBUG: Tree %d OOB sample %d predicted class %d (nodeclass[%d]=%d)\n", 
                    //        tree_id, sample, prediction, current_node, nodeclass[current_node]);
                }
            }
        }
    }
    
    // Debug output
    // if (tree_id == 0 && threadIdx.x == 0) {
    //     printf("GPU: OOB vote kernel completed for tree %d\n", tree_id);
    // }
}

// GPU batch tree growing function
void gpu_growtree_batch(integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
                        integer_t task_type,  // 0=CLASSIFICATION, 1=REGRESSION, 2=UNSUPERVISED
                        const integer_t* cat, const integer_t* ties, integer_t* nin,
                        const real_t* y_regression,  // Original continuous y values for regression (nullptr for classification)
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                        const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
                        integer_t* amat_all, integer_t* jinbag_all, real_t* tnodewt_all, real_t* xbestsplit_all,
                        integer_t* nodestatus_all, integer_t* bestvar_all, integer_t* treemap_all,
                        integer_t* catgoleft_all, integer_t* nodeclass_all, integer_t* nnode_all,
                        integer_t* jtr_all, integer_t* nodextr_all, integer_t* idmove_all,
                        integer_t* igoleft_all, integer_t* igl_all, integer_t* nodestart_all,
                        integer_t* nodestop_all, integer_t* itemp_all, integer_t* itmp_all,
                        integer_t* icat_all, real_t* tcat_all, real_t* classpop_all, real_t* wr_all,
                        real_t* wl_all, real_t* tclasscat_all, real_t* tclasspop_all, real_t* tmpclass_all, 
                        real_t* q_all, integer_t* nout_all, real_t* avimp_all,
                        real_t* qimp_all, real_t* qimpm_all, dp_t* proximity_all,
                        real_t* oob_predictions_all,  // For regression: accumulate OOB predictions (sum across trees)
                        void** lowrank_proximity_ptr_out) {  // Output: LowRankProximityMatrix pointer
    
    // std::cout << "[GPU_GROWTREE_BATCH] ENTRY - num_trees=" << num_trees << std::endl;
    // std::cout.flush();
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] ENTRY - nsample=" << nsample 
    //           << ", num_trees=" << num_trees 
    //           << ", mdim=" << mdim << std::endl;
    // std::cout.flush();
    
    // Validate basic parameters before any operations (prevents crashes)
    if (nsample <= 0 || num_trees <= 0 || mdim <= 0) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: Invalid parameters - nsample=" << nsample 
        //           << ", num_trees=" << num_trees << ", mdim=" << mdim << std::endl;
        return;
    }
    
    // Validate pointer parameters are not null (if they're required)
    if (x == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: x is nullptr" << std::endl;
        return;
    }
    if (win == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: win is nullptr" << std::endl;
        return;
    }
    if (seeds == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: seeds is nullptr" << std::endl;
        return;
    }
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] Parameters validated, setting g_config..." << std::endl;
    // std::cout.flush();
    
    // Set global config to GPU mode for proximity computation
    g_config.use_gpu = true;
    g_config.force_cpu = false;
    
    // Ensure use_casewise is set BEFORE any GPU operations (importance, proximity, etc.)
    // This is already set in fit_batch_gpu, but verify it's set correctly here
    // Note: g_config.use_casewise should be set by caller (fit_batch_gpu), but we ensure it's valid here
    // The value is read by gpu_varimp (for overall and local importance), gpu_proximity, and gpu_proximity_upper_triangle
    // All GPU functions check g_config.use_casewise to determine weighting:
    //   - use_casewise=true: case-wise (weight=tnodewt[node_idx] - bootstrap frequency weighted)
    //   - use_casewise=false: non-case-wise (weight=1.0 for all samples - UC Berkeley standard)
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] g_config.use_casewise=" << (g_config.use_casewise ? "true" : "false") 
    //           << " (for importance and proximity)" << std::endl;
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] g_config set, about to check CUDA context..." << std::endl;
// std::cout.flush();
    
    // Matches backup version - flush output before CUDA operations
// std::cout.flush();
    
    // IMPORTANT: Set use_qlora and quant_mode from g_config BEFORE proximity computation
    // These should already be set by RandomForest constructor via g_config, but ensure they're valid
    // Default values if not set (shouldn't happen, but safety check)
    // Note: g_config.use_qlora and g_config.quant_mode are set in rf_random_forest.cpp before calling gpu_growtree_batch
    
    // Check CUDA context first before synchronizing (matches backup version exactly)
    // This prevents crashes in Jupyter when context is corrupted
    // Check CUDA context first before synchronizing
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] About to call cudaGetDevice..." << std::endl;
// std::cout.flush();
    
    int device;
    cudaError_t ctx_err = cudaGetDevice(&device);
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaGetDevice returned: " << ctx_err << " (device=" << device << ")" << std::endl;
// std::cout.flush();
    if (ctx_err != cudaSuccess) {
        // Context might be invalid - try to set device 0 (matches backup)
// Debug prints removed
        // std::cout << "[GPU_GROWTREE_BATCH] cudaGetDevice failed, trying cudaSetDevice(0)..." << std::endl;
// std::cout.flush();
        ctx_err = cudaSetDevice(0);
        if (ctx_err != cudaSuccess) {
            // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: Failed to set CUDA device: " << ctx_err << std::endl;
            return;
        }
        device = 0;
    }
    
    // Ensure we're on the correct device (matches backup)
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] About to call cudaSetDevice(" << device << ")..." << std::endl;
// std::cout.flush();
    cudaSetDevice(device);
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaSetDevice completed, about to call cudaGetLastError()..." << std::endl;
// std::cout.flush();
    
    // Clear any previous errors (matches backup)
    cudaGetLastError();
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaGetLastError completed" << std::endl;
// std::cout.flush();
    
    // Try a lightweight operation to verify context is working (matches backup version)
    cudaError_t test_err = cudaFree(0);  // Freeing nullptr is safe and tests context
    if (test_err != cudaSuccess && test_err != cudaErrorInvalidValue) {
        // Context test failed - return early to prevent crash (matches backup)
        return;
    }
    
    // Clear any previous errors (matches backup)
    cudaGetLastError();
    
    // Test that context is ready with a simple non-blocking operation (matches backup version)
    void* test_alloc_ptr = nullptr;
    cudaError_t test_alloc_err = cudaMalloc(&test_alloc_ptr, 1);  // Try to allocate 1 byte
    if (test_alloc_err == cudaSuccess && test_alloc_ptr != nullptr) {
        cudaFree(test_alloc_ptr);  // Free immediately
    } else {
        // Context test failed - clear error and continue anyway (matches backup)
        cudaGetLastError();  // Clear error
    }
    
    // Report GPU memory status - required for .ipynb to work properly (matches backup version)
    size_t free_memory = 0, total_memory = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
    
    if (mem_err != cudaSuccess) {
        // Memory info failed - clear error and continue with default values (matches backup)
        cudaGetLastError();
    }
    // std::cout << "GPU: Memory Status - Total: " << (total_memory / (1024*1024)) << "MB, "
              // << "Free: " << (free_memory / (1024*1024)) << "MB, "
              // << "Used: " << ((total_memory - free_memory) / (1024*1024)) << "MB" << std::endl;
    
    // Allocate GPU memory for input data (matches backup version with explicit error checking)
    real_t* x_gpu;
    integer_t* cl_gpu;
    real_t* win_gpu;
    integer_t* seeds_gpu;
    
    // Ensure device is set and clear any errors before allocation (matches backup version)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    cudaError_t data_alloc_error;
    data_alloc_error = cudaMalloc(&x_gpu, nsample * mdim * sizeof(real_t));
    
    // std::cout << "[GPU_GROWTREE_BATCH] cudaMalloc complete" << std::endl;
    // std::cout.flush();
    if (data_alloc_error != cudaSuccess) {
        // Allocation failed - return early to prevent crash
        return;
    }
    
    // Ensure device is still set and clear any errors (matches backup version)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    data_alloc_error = cudaMalloc(&cl_gpu, nsample * sizeof(integer_t));
    if (data_alloc_error != cudaSuccess) {
        // Clean up x_gpu before returning
        cudaFree(x_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    data_alloc_error = cudaMalloc(&win_gpu, nsample * sizeof(real_t));
    if (data_alloc_error != cudaSuccess) {
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    data_alloc_error = cudaMalloc(&seeds_gpu, num_trees * sizeof(integer_t));
    if (data_alloc_error != cudaSuccess) {
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        return;
    }
    
    // Debug: Check host data before copying to GPU
    // std::cout << "GPU: Host class labels first 10: ";
    // for (integer_t i = 0; i < 10; i++) {
    //     std::cout << cl[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Copy data to GPU (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    cudaError_t copy_err = cudaMemcpy(x_gpu, x, nsample * mdim * sizeof(real_t), cudaMemcpyHostToDevice);
    
    // std::cout << "[GPU_GROWTREE_BATCH] cudaMemcpy complete" << std::endl;
    // std::cout.flush();  // DISABLED - causes Jupyter crash
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    // Check if cl is valid before copying (important for unsupervised learning)
    if (cl != nullptr) {
        cudaSetDevice(device);
        cudaGetLastError();  // Clear any errors
        
        copy_err = cudaMemcpy(cl_gpu, cl, nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            // Copy failed - clean up and return
            cudaFree(x_gpu);
            cudaFree(cl_gpu);
            cudaFree(win_gpu);
            cudaFree(seeds_gpu);
            return;
        }
    } else {
        // For unsupervised learning, initialize with zeros
        cudaSetDevice(device);
        cudaGetLastError();  // Clear any errors
        cudaMemset(cl_gpu, 0, nsample * sizeof(integer_t));
    }
    
    // Copy win and seeds to GPU (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    copy_err = cudaMemcpy(win_gpu, win, nsample * sizeof(real_t), cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    copy_err = cudaMemcpy(seeds_gpu, seeds, num_trees * sizeof(integer_t), cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    // Matches backup version - no debug output
    
    // Debug: Verify GPU data after copying
    // integer_t* cl_test = new integer_t[10];
    // cudaMemcpy(cl_test, cl_gpu, 10 * sizeof(integer_t), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: GPU class labels first 10: ";
    // for (integer_t i = 0; i < 10; i++) {
    //     std::cout << cl_test[i] << " ";
    // }
    // std::cout << std::endl;
    // delete[] cl_test;
    
    // Allocate GPU memory for tree arrays
    
    integer_t* nodestatus_all_gpu;
    integer_t* nodeclass_all_gpu;
    integer_t* nnode_all_gpu;
    integer_t* bestvar_all_gpu;
    real_t* xbestsplit_all_gpu;
    integer_t* treemap_all_gpu;
    integer_t* jinbag_all_gpu;
    
    // Matches backup version - direct allocation, no error checking, no debug output
    // Allocate GPU memory for tree arrays (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();
    
    cudaError_t tree_alloc_error;
    tree_alloc_error = cudaMalloc(&nodestatus_all_gpu, num_trees * maxnode * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    tree_alloc_error = cudaMalloc(&nodeclass_all_gpu, num_trees * maxnode * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        cudaFree(nodestatus_all_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    tree_alloc_error = cudaMalloc(&nnode_all_gpu, num_trees * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        cudaFree(nodestatus_all_gpu);
        cudaFree(nodeclass_all_gpu);
        return;
    }
    
    // Matches backup version - direct allocation, no error checking, no debug output
    CUDA_CHECK_VOID(cudaMalloc(&bestvar_all_gpu, num_trees * maxnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&xbestsplit_all_gpu, num_trees * maxnode * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&treemap_all_gpu, num_trees * 2 * maxnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&jinbag_all_gpu, num_trees * nsample * sizeof(integer_t)));
    
    // Allocate GPU memory for categorical arrays
    integer_t* cat_gpu = nullptr;
    integer_t* amat_gpu = nullptr;
    integer_t* catgoleft_all_gpu = nullptr;
    
    // Allocate cat array if provided
    if (cat != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&cat_gpu, mdim * sizeof(integer_t)));
            cudaMemcpy(cat_gpu, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice);
    }
    
    // Allocate amat array if provided
    if (amat_all != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&amat_gpu, mdim * nsample * sizeof(integer_t)));
            cudaMemcpy(amat_gpu, amat_all, mdim * nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
    }
    
    // Allocate catgoleft array (always allocate, initialize to zeros)
    CUDA_CHECK_VOID(cudaMalloc(&catgoleft_all_gpu, num_trees * maxnode * maxcat * sizeof(integer_t)));
        cudaMemset(catgoleft_all_gpu, 0, num_trees * maxnode * maxcat * sizeof(integer_t));
    
    // Matches backup version - direct allocation, no test allocation, no error checking
    real_t* q_all_gpu;
    size_t q_size = nsample * nclass * sizeof(real_t);
    CUDA_CHECK_VOID(cudaMalloc(&q_all_gpu, q_size));
    
    // std::cout << "DEBUG: q_all_gpu allocation successful, proceeding to memset\n";
    
    // Initialize q array to zeros
    // std::cout << "DEBUG: About to call cudaMemset for q_all_gpu\n";
    
    // Clear any previous CUDA errors before memset to prevent false positives
    // Error code 1 (cudaErrorInvalidValue) often comes from stale error state
    cudaGetLastError(); // Clear any previous errors
    
    cudaError_t memset_err = cudaMemset(q_all_gpu, 0, nsample * nclass * sizeof(real_t));
    
    // std::cout << "DEBUG: After cudaMemset q_all_gpu, error=" << memset_err << "\n";
    
    if (memset_err != cudaSuccess) {
        // Silently handle memset errors - don't print in Jupyter (can cause crashes)
        // Clear error state and continue
        cudaGetLastError(); // Clear error state
        // Safe free to prevent segfaults
        if (q_all_gpu) { cudaError_t err = cudaFree(q_all_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // std::cout << "DEBUG: q_all_gpu memset successful\n";
    
    // Clear error state after memset to prevent false positives in Jupyter
    // Error code 1 (cudaErrorInvalidValue) is often a false positive from previous operations
    // We clear it silently to prevent Jupyter kernel crashes
    cudaGetLastError(); // Clear any error state (including false positives)
    
    // std::cout << "DEBUG: After error check, before variable declarations\n";
    
    // Allocate GPU memory for importance arrays (if needed)
    real_t* qimp_all_gpu = nullptr;
    real_t* qimpm_all_gpu = nullptr;
    
    // std::cout << "DEBUG: Variable declarations complete, before check\n";
    
    if (qimp_all != nullptr && qimpm_all != nullptr) {
        // std::cout << "DEBUG: qimp_all and qimpm_all are not null, allocating importance arrays\n";
        
        // std::cout << "DEBUG: About to allocate qimp_all_gpu\n";
        
        CUDA_CHECK_VOID(cudaMalloc(&qimp_all_gpu, nsample * sizeof(real_t)));
        
        // std::cout << "DEBUG: About to allocate qimpm_all_gpu\n";
        
        CUDA_CHECK_VOID(cudaMalloc(&qimpm_all_gpu, nsample * mdim * sizeof(real_t)));
        
        // Initialize importance arrays to zeros
        // std::cout << "DEBUG: About to memset importance arrays\n";
        
        cudaMemset(qimp_all_gpu, 0, nsample * sizeof(real_t));
        cudaMemset(qimpm_all_gpu, 0, nsample * mdim * sizeof(real_t));
        
        // std::cout << "DEBUG: Importance arrays memset completed\n";
    } else {
        // std::cout << "DEBUG: qimp_all or qimpm_all is null, skipping importance arrays\n";
    }
    
    // std::cout << "DEBUG: About to allocate bootstrap arrays\n";
    
    // Allocate additional arrays for bootstrap
    integer_t* nin_all_gpu;
    real_t* win_all_gpu;
    curandState* random_states;
    
    // Matches backup version - direct allocation, no error checking
    CUDA_CHECK_VOID(cudaMalloc(&nin_all_gpu, num_trees * nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&win_all_gpu, num_trees * nsample * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&random_states, num_trees * sizeof(curandState)));
    
    // Initialize random states
    dim3 init_block_size(256);
    dim3 init_grid_size((num_trees + init_block_size.x - 1) / init_block_size.x);
    
    init_random_states_kernel<<<init_grid_size, init_block_size>>>(
        random_states, num_trees, seeds_gpu
    );
    
    // Matches backup version - no kernel launch error checking
    
    // Ensure random state initialization finishes before bootstrapping
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // Launch bootstrap kernel
    dim3 bootstrap_block_size(256);
    dim3 bootstrap_grid_size(num_trees);
    
    // std::cout << "GPU: Launching bootstrap kernel with " << num_trees << " trees..." << std::endl;
    // std::cout << "GPU: Bootstrap grid size: " << bootstrap_grid_size.x << ", Block size: " << bootstrap_block_size.x << std::endl;
    
    gpu_bootstrap_kernel<<<bootstrap_grid_size, bootstrap_block_size>>>(
        num_trees, nsample, win_gpu, nin_all_gpu, win_all_gpu, jinbag_all_gpu,
        random_states, seeds_gpu,
        g_config.use_casewise  // Pass use_casewise as parameter
    );
    
    // Matches backup version - no kernel launch error checking
    
    // std::cout << "DEBUG: Bootstrap kernel launched, about to sync\n";
    
    // OPTIMIZATION: Remove unnecessary sync and debug copy before tree kernel
    // The bootstrap kernel will complete before tree kernel starts (implicit ordering)
    // Only sync if we need to check results (debug mode disabled)
    // CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Removed - let kernels pipeline
    
    // Launch tree growing kernel
    // std::cout << "GPU: Launching tree growing kernel with " << num_trees << " trees..." << std::endl;
    
    // OPTIMIZATION: Increased block size for better parallelism
    // Use 256 threads per block for better GPU utilization and parallel sample filtering
    // Modern GPUs (Ampere+) can handle 256 threads efficiently
    dim3 tree_block_size(256);
    dim3 tree_grid_size(num_trees);
    
    // std::cout << "GPU: Grid size: " << tree_grid_size.x << ", Block size: " << tree_block_size.x << std::endl;
    
    // Allocate GPU memory for allowed modes if using random selection
    integer_t* allowed_modes_gpu = nullptr;
    // Always use original RF growth mode (mode 0) - no need for allowed_modes
    integer_t num_allowed_modes = 0;
    
    // std::cout << "[GPU_GROWTREE_BATCH] About to launch gpu_full_tree_kernel, parallel_mode=" << g_config.gpu_parallel_mode0 << std::endl;
    // std::cout.flush();  // DISABLED - causes Jupyter crash
    
    gpu_full_tree_kernel<<<tree_grid_size, tree_block_size>>>(
        num_trees, nsample, mdim, maxnode, nclass,
        minndsize, mtry, x_gpu,
        nodestatus_all_gpu, nodeclass_all_gpu, nnode_all_gpu,
        bestvar_all_gpu, xbestsplit_all_gpu, treemap_all_gpu,
        jinbag_all_gpu, cl_gpu, win_all_gpu,
        cat_gpu,  // Categorical feature array (nullptr if none)
        amat_gpu,  // Categorical matrix (nullptr if none)
        catgoleft_all_gpu,  // Category go-left array
        maxcat,  // Maximum categories
        ncsplit,  // Number of random splits for large categoricals
        ncmax,  // Threshold for large categorical
        g_config.gpu_loss_function,  // Loss function: 0=Gini (classification), 3=MSE (regression)
        0.0f,  // min_gain_to_split not used in original RF
        0,  // Always use original RF growth mode (mode 0)
        nullptr,  // No allowed modes needed
        num_allowed_modes,  // Number of allowed modes (always 0)
        0.0f,  // L1 regularization not used in original RF
        0.0f,  // L2 regularization not used in original RF
        g_config.gpu_parallel_mode0,
        random_states  // Pass pre-initialized random states to avoid slow curand_init in kernel
    );
    
    // OPTIMIZATION: Only sync once after tree kernel (removed duplicate sync)
    // Check for kernel launch errors (non-blocking check before sync)
    cudaError_t tree_kernel_err = cudaGetLastError();
    if (tree_kernel_err != cudaSuccess) {
        // std::cerr << "ERROR: Tree building kernel error: " << cudaGetErrorString(tree_kernel_err) << std::endl;
    }
    
    // Single synchronization point after tree kernel
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // Compute tnodewt for all trees IMMEDIATELY after tree growing completes
    // This must be done while win_all_gpu and nin_all_gpu are still in scope
    // OPTIMIZATION: Skip tnodewt if not needed (non-casewise + no importance)
    // tnodewt is only needed for: (1) casewise OOB weighting, (2) casewise importance
    // DISABLED: GPU tnodewt kernel has bugs - use CPU fallback instead (matches arxiv)
    bool need_tnodewt = false;  // DISABLED - CPU fallback below works correctly
    if (tnodewt_all != nullptr && need_tnodewt) {
        // Allocate GPU memory for tnodewt (one allocation for all trees)
        real_t* tnodewt_all_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&tnodewt_all_gpu, num_trees * maxnode * sizeof(real_t)));
        CUDA_CHECK_VOID(cudaMemset(tnodewt_all_gpu, 0, num_trees * maxnode * sizeof(real_t)));
        
        // OPTIMIZATION: nin is already on CPU - use it directly instead of copying from GPU
        // Copy nin from GPU once for all trees
        std::vector<integer_t> nin_all_host(num_trees * nsample);
        CUDA_CHECK_VOID(cudaMemcpy(nin_all_host.data(), nin_all_gpu, num_trees * nsample * sizeof(integer_t), cudaMemcpyDeviceToHost));
        
        // Launch GPU kernel for each tree sequentially to avoid hangs
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            integer_t tree_offset = tree_id * maxnode;
            integer_t jinbag_offset = tree_id * nsample;
            
            // Get nnode for this tree
            integer_t nnode_tree = 0;
            CUDA_CHECK_VOID(cudaMemcpy(&nnode_tree, nnode_all_gpu + tree_id, sizeof(integer_t), cudaMemcpyDeviceToHost));
            
            if (nnode_tree > 0) {
                // Get tree-specific GPU pointers
                integer_t* treemap_tree_gpu = treemap_all_gpu + tree_offset * 2;
                integer_t* nodestatus_tree_gpu = nodestatus_all_gpu + tree_offset;
                real_t* xbestsplit_tree_gpu = xbestsplit_all_gpu + tree_offset;
                integer_t* bestvar_tree_gpu = bestvar_all_gpu + tree_offset;
                integer_t* jinbag_tree_gpu = jinbag_all_gpu + jinbag_offset;
                integer_t* nin_tree_gpu = nin_all_gpu + jinbag_offset;
                real_t* win_tree_gpu = win_all_gpu + jinbag_offset;
                real_t* tnodewt_tree_gpu = tnodewt_all_gpu + tree_offset;
                
                // Count in-bag samples for this tree using CPU-side nin data
                const integer_t* nin_tree_host = nin_all_host.data() + jinbag_offset;
                integer_t ninbag = 0;
                for (integer_t i = 0; i < nsample; ++i) {
                    if (nin_tree_host[i] > 0) ninbag++;
                }
                
                // Launch GPU kernel to compute tnodewt for all nodes in this tree
                dim3 block_size(256);
                dim3 grid_size((nnode_tree + block_size.x - 1) / block_size.x);
                
                gpu_compute_tnodewt_kernel<<<grid_size, block_size>>>(
                    x_gpu, nsample, mdim, nnode_tree,
                    treemap_tree_gpu, nodestatus_tree_gpu, xbestsplit_tree_gpu, bestvar_tree_gpu,
                    cl_gpu, nullptr,  // y_regression=nullptr for classification
                    win_tree_gpu,
                    nin_tree_gpu,
                    jinbag_tree_gpu, ninbag,
                    nclass, task_type,
                    tnodewt_tree_gpu);
                
            }
        }
        
        // Synchronize once after launching kernels for all trees to ensure completion
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));
        
        // Copy results back to host
        CUDA_CHECK_VOID(cudaMemcpy(tnodewt_all, tnodewt_all_gpu, num_trees * maxnode * sizeof(real_t), cudaMemcpyDeviceToHost));
        
        // Free GPU memory
        CUDA_CHECK_VOID(cudaFree(tnodewt_all_gpu));
    }
    
    // Copy nnode_all back to check tree sizes
    cudaMemcpy(nnode_all, nnode_all_gpu, num_trees * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // DEBUG: Check tree structure after growing
    std::vector<integer_t> nnode_host(num_trees);
    cudaMemcpy(nnode_host.data(), nnode_all_gpu, num_trees * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // std::cout << "DEBUG: Tree node counts (first 10 trees): ";
    // for (integer_t i = 0; i < num_trees && i < 10; ++i) {
    //     std::cout << "tree" << i << "=" << nnode_host[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Check if trees have splits (not just single-node trees)
    integer_t single_node_count = 0;
    integer_t multi_node_count = 0;
    for (integer_t i = 0; i < num_trees; ++i) {
        if (nnode_host[i] == 1) {
            single_node_count++;
        } else {
            multi_node_count++;
        }
    }
    // std::cout << "DEBUG: Single-node trees: " << single_node_count 
    //           << ", Multi-node trees: " << multi_node_count << " out of " << num_trees << std::endl;
    
    // DEBUG: Check first tree's structure
    if (num_trees > 0) {
        std::vector<integer_t> nodestatus_tree0(maxnode);
        std::vector<integer_t> nodeclass_tree0(maxnode);
        cudaMemcpy(nodestatus_tree0.data(), nodestatus_all_gpu, maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(nodeclass_tree0.data(), nodeclass_all_gpu, maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
        
        // std::cout << "DEBUG: Tree 0 - nodestatus[0]=" << nodestatus_tree0[0] 
        //           << ", nodeclass[0]=" << nodeclass_tree0[0] << std::endl;
        if (nnode_host[0] > 1) {
            // std::cout << "DEBUG: Tree 0 has " << nnode_host[0] << " nodes" << std::endl;
            // for (integer_t i = 0; i < nnode_host[0] && i < 10; ++i) {
            //     std::cout << "  node " << i << ": status=" << nodestatus_tree0[i] 
            //               << ", class=" << nodeclass_tree0[i] << std::endl;
            // }
        }
    }
    
    // Copy tree data back to host BEFORE any post-processing (importance or proximity)
    // This copy is needed for BOTH importance computation AND proximity computation
    // std::cout << "DEBUG: About to copy tree data back to host\n";
    
    // Copy tree data back to host (needed for both importance and proximity computation)
    // Matches backup version - direct calls, no error checking
    cudaMemcpy(nodestatus_all, nodestatus_all_gpu, num_trees * maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeclass_all, nodeclass_all_gpu, num_trees * maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nnode_all, nnode_all_gpu, num_trees * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // DEBUG: Verify nodestatus was copied correctly (after nnode_all is available)
    integer_t terminal_count = 0;
    integer_t split_count = 0;
    integer_t active_count = 0;
    integer_t zero_count = 0;
    for (integer_t k = 0; k < std::min(maxnode, nnode_all[0]); ++k) {
        if (nodestatus_all[k] == -1) terminal_count++;
        else if (nodestatus_all[k] == 1) split_count++;
        else if (nodestatus_all[k] == 2) active_count++;
        else if (nodestatus_all[k] == 0) zero_count++;
    }
    // std::cout << "DEBUG: After copy - Tree 0 nodestatus: terminal=" << terminal_count 
    //           << ", split=" << split_count << ", active=" << active_count 
    //           << ", zero=" << zero_count << ", total_nodes=" << nnode_all[0] << std::endl;
    
    // DEBUG: Check first few nodestatus values
    // std::cout << "DEBUG: Tree 0 nodestatus (first 10): ";
    // for (integer_t k = 0; k < std::min(10, nnode_all[0]); ++k) {
    //     std::cout << static_cast<int>(nodestatus_all[k]) << " ";
    // }
    // std::cout << std::endl;
    
    // Matches backup version - direct calls, no error checking
        cudaMemcpy(bestvar_all, bestvar_all_gpu, num_trees * maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(xbestsplit_all, xbestsplit_all_gpu, num_trees * maxnode * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(treemap_all, treemap_all_gpu, num_trees * 2 * maxnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(jinbag_all, jinbag_all_gpu, num_trees * nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(nin, nin_all_gpu, num_trees * nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
        
        // Copy catgoleft from GPU to host if categorical features are used
    integer_t* catgoleft_all_host = nullptr;
        if (catgoleft_all != nullptr && catgoleft_all_gpu != nullptr && maxcat > 1) {
            cudaMemcpy(catgoleft_all, catgoleft_all_gpu, 
                                  num_trees * maxnode * maxcat * sizeof(integer_t), 
                                  cudaMemcpyDeviceToHost);
            catgoleft_all_host = catgoleft_all;
        }
        
    // Debug: Verify copy worked by checking GPU memory directly
    std::vector<integer_t> nin_gpu_check(nsample);
    cudaError_t check_err = cudaMemcpy(nin_gpu_check.data(), nin_all_gpu, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
    if (check_err == cudaSuccess) {
        // std::cout << "DEBUG: Direct GPU copy (first tree, first 20): ";
        // for (integer_t i = 0; i < std::min(20, nsample); ++i) {
        //     std::cout << static_cast<int>(nin_gpu_check[i]) << " ";
        // }
        // std::cout << std::endl;
        
        integer_t gpu_inbag = 0;
        for (integer_t i = 0; i < nsample; ++i) {
            if (nin_gpu_check[i] > 0) gpu_inbag++;
        }
        // std::cout << "DEBUG: Direct GPU copy in-bag count: " << gpu_inbag << " / " << nsample << std::endl;
    }
    
    // Debug: Check first tree's nin values
    // std::cout << "DEBUG: First tree nin values (first 20): ";
    // for (integer_t i = 0; i < std::min(20, nsample); ++i) {
    //     std::cout << static_cast<int>(nin[i]) << " ";
    // }
    // std::cout << std::endl;
    
    // Count in-bag samples for first tree
    integer_t inbag_count = 0;
    for (integer_t i = 0; i < nsample; ++i) {
        if (nin[i] > 0) inbag_count++;
    }
    // std::cout << "DEBUG: First tree in-bag count: " << inbag_count << " / " << nsample << std::endl;
    
    // std::cout << "DEBUG: Tree data copied back to host successfully\n";
    
    // Initialize host importance arrays to zero before accumulation (for both classification and regression)
    // NOTE: qimp_all and qimpm_all are reset per batch (they're accumulated per batch in GPU)
    // BUT: avimp_all should NOT be reset here - it needs to accumulate across ALL batches
    // The reset happens in fit_batch_gpu() before the first batch, and we accumulate across batches
    if (qimp_all != nullptr && qimpm_all != nullptr) {
        std::fill(qimp_all, qimp_all + nsample, 0.0f);
        std::fill(qimpm_all, qimpm_all + nsample * mdim, 0.0f);
    }
    // REMOVED: avimp_all reset - it must accumulate across all batches
    // if (avimp_all != nullptr) {
    //     std::fill(avimp_all, avimp_all + mdim, 0.0f);
    // }
    
    // Compute nodextr_all ONCE for BOTH proximity AND importance computation
    // nodextr_all stores the terminal node index for each sample for each tree
    // It's needed by BOTH proximity (to know which terminal node each sample ends up in)
    // AND importance (to compute node weights for importance calculation)
    // AND OOB vote accumulation (to compute jtr_tree from nodextr_tree)
    // Compute it ONCE before all use it, not separately in each block
    bool need_nodextr = ((proximity_all != nullptr || lowrank_proximity_ptr_out != nullptr) || 
                         avimp_all != nullptr || 
                         (task_type == 0 && q_all != nullptr));  // Classification needs nodextr for OOB votes
    if (need_nodextr && nodextr_all != nullptr) {
        auto time_nodextr_start = std::chrono::high_resolution_clock::now();
        
        // OPTIMIZATION: Allocate GPU memory for nodextr_all once (outside loop) for all trees
        // This avoids repeated allocations and enables GPU-accelerated computation
        integer_t* nodextr_all_gpu = nullptr;
        cudaError_t nodextr_alloc_err = cudaMalloc(&nodextr_all_gpu, num_trees * nsample * sizeof(integer_t));
        bool use_gpu_nodextr = (nodextr_alloc_err == cudaSuccess);
        
        if (use_gpu_nodextr) {
            // Initialize to zero
            cudaMemset(nodextr_all_gpu, 0, num_trees * nsample * sizeof(integer_t));
        } else {
            // std::cerr << "WARNING: Failed to allocate GPU memory for nodextr_all_gpu, using CPU fallback" << std::endl;
        }
        
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            auto time_tree_nodextr_start = std::chrono::high_resolution_clock::now();
            // Get tree-specific data
            integer_t tree_offset = tree_id * maxnode;
            integer_t jinbag_offset = tree_id * nsample;
            
            // Create temporary arrays for this tree's nodextr computation
            std::vector<integer_t> jtr_tree(nsample, 0);
            std::vector<integer_t> nodextr_tree(nsample, 0);
            
            // Get catgoleft offset for this tree: tree_id * maxnode * maxcat
            const integer_t* catgoleft_tree = nullptr;
            if (catgoleft_all_host != nullptr) {
                integer_t catgoleft_offset = tree_id * maxnode * maxcat;
                catgoleft_tree = catgoleft_all_host + catgoleft_offset;
            }
            
            // OPTIMIZATION: Use GPU kernel to compute nodextr for ALL samples (both in-bag and OOB) in parallel
            // This replaces both cpu_testreebag AND the expensive sequential CPU loop
            // For importance computation, we only need nodextr (not jtr), so we can skip cpu_testreebag entirely
            if (use_gpu_nodextr) {
                // Initialize nodextr_tree to zero (will be filled by GPU kernel)
                std::fill(nodextr_tree.begin(), nodextr_tree.end(), 0);
                
                // Launch GPU kernel to compute nodextr for ALL samples in parallel (both in-bag and OOB)
                dim3 block_size(256);  // 256 threads per block
                dim3 grid_size((nsample + block_size.x - 1) / block_size.x);  // Enough blocks to cover all samples
                
                gpu_compute_nodextr_inbag_kernel<<<grid_size, block_size>>>(
                    x_gpu,  // Feature matrix on GPU
                    treemap_all_gpu,  // Tree mapping on GPU
                    bestvar_all_gpu,  // Best variable per node on GPU
                    xbestsplit_all_gpu,  // Best split value per node on GPU
                    nodestatus_all_gpu,  // Node status on GPU
                    nin_all_gpu,  // In-bag indicator on GPU
                    cat_gpu,  // Categorical indicator on GPU (nullptr if none)
                    catgoleft_all_gpu,  // Category go-left array on GPU (nullptr if none)
                    nodextr_all_gpu,  // Output: terminal node for each sample on GPU
                    nsample, mdim, maxnode, maxcat,
                    tree_id, tree_offset, jinbag_offset, nnode_all[tree_id]
                );
                
                // Check for kernel launch errors (non-blocking check)
                // cudaError_t kernel_err = cudaGetLastError();
                // if (kernel_err != cudaSuccess) {
                //     std::cerr << "WARNING: GPU kernel launch failed for tree " << tree_id 
                //               << ": " << cudaGetErrorString(kernel_err) << ", falling back to CPU" << std::endl;
                //     use_gpu_nodextr = false;  // Disable GPU for remaining trees
                // }
                // NOTE: Don't synchronize here - batch all launches, sync once at end
            } else {
                // CPU fallback: use cpu_testreebag for OOB samples, then sequential loop for in-bag
                auto time_cpu_testreebag_start = std::chrono::high_resolution_clock::now();
                cpu_testreebag(x, xbestsplit_all + tree_offset, nin + jinbag_offset, 
                              treemap_all + tree_offset * 2, bestvar_all + tree_offset,
                              nodeclass_all + tree_offset, cat, nodestatus_all + tree_offset, 
                              catgoleft_tree,  // Pass tree-specific catgoleft array
                              nsample, mdim, nnode_all[tree_id], maxcat,
                              jtr_tree.data(), nodextr_tree.data());
                auto time_cpu_testreebag_end = std::chrono::high_resolution_clock::now();
                auto cpu_testreebag_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_cpu_testreebag_end - time_cpu_testreebag_start);
            }
            
            // CPU fallback: sequential loop for in-bag samples (if GPU failed or not available)
            if (!use_gpu_nodextr) {
                // cpu_testreebag already processed OOB samples, now process in-bag samples
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] > 0 && nodextr_tree[n] == 0) {
                        integer_t kt = 0;
                        integer_t traversal_count = 0;
                        const integer_t max_traversal = 1000;
                        while (nodestatus_all[tree_offset + kt] == 1 && traversal_count < max_traversal) {
                            traversal_count++;
                            integer_t mvar = bestvar_all[tree_offset + kt];
                            if (mvar < 0 || mvar >= mdim) break;
                            if (cat == nullptr || cat[mvar] == 1) {
                                if (x[n * mdim + mvar] <= xbestsplit_all[tree_offset + kt]) {
                                    kt = treemap_all[tree_offset * 2 + kt * 2];
                                } else {
                                    kt = treemap_all[tree_offset * 2 + kt * 2 + 1];
                                }
                            } else {
                                integer_t cat_value = static_cast<integer_t>(x[n * mdim + mvar]);
                                integer_t numcat = cat[mvar];
                                if (cat_value >= 0 && cat_value < numcat) {
                                    integer_t catgoleft_offset = tree_id * maxnode * maxcat + kt * maxcat;
                                    if (catgoleft_all_host[catgoleft_offset + cat_value] == 1) {
                                        kt = treemap_all[tree_offset * 2 + kt * 2];
                                    } else {
                                        kt = treemap_all[tree_offset * 2 + kt * 2 + 1];
                                    }
                                } else {
                                    kt = treemap_all[tree_offset * 2 + kt * 2 + 1];
                                }
                            }
                            if (kt < 0 || kt >= nnode_all[tree_id]) { kt = 0; break; }
                            if (nodestatus_all[tree_offset + kt] != 1) break;
                        }
                        if (nodestatus_all[tree_offset + kt] == -1 || nodestatus_all[tree_offset + kt] == 2) {
                            nodextr_tree[n] = kt;
                        }
                    }
                }
                
                // Copy nodextr_tree to nodextr_all (CPU path)
                std::copy(nodextr_tree.begin(), nodextr_tree.end(), nodextr_all + jinbag_offset);
            }
        }
        
        // OPTIMIZATION: Batch synchronization and copy-back for all trees (GPU path)
        if (use_gpu_nodextr) {
            // Synchronize once after all kernels have been launched (batch all trees)
            CUDA_CHECK_VOID(cudaDeviceSynchronize());
            
            // Copy all results back in batch
            for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                integer_t jinbag_offset = tree_id * nsample;
                cudaMemcpy(nodextr_all + jinbag_offset, nodextr_all_gpu + jinbag_offset,
                          nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            }
            
            // DEBUG: Check if nodextr_all was computed correctly (first tree only)
            if (task_type == 0 && num_trees > 0) {
                integer_t non_zero_nodextr = 0;
                integer_t zero_nodextr = 0;
                for (integer_t n = 0; n < nsample && n < 20; ++n) {
                    if (nodextr_all[n] > 0) {
                        non_zero_nodextr++;
                    } else if (nodextr_all[n] == 0) {
                        zero_nodextr++;
                    }
                }
// Debug prints removed
                // std::cout << "[DEBUG NODEXTR] Tree 0: non_zero=" << non_zero_nodextr 
                //           << ", zero=" << zero_nodextr 
                //           << " (first 20 samples)" << std::endl;
                // Print first few nodextr values
// Debug prints removed
                // std::cout << "[DEBUG NODEXTR] Tree 0 first 10 nodextr values: ";
                // for (integer_t n = 0; n < 10; ++n) {
                //     std::cout << nodextr_all[n] << " ";
                // }
                // std::cout << std::endl;
            }
            auto time_copy_end = std::chrono::high_resolution_clock::now();
        }
        
        // Free GPU memory for nodextr_all_gpu (allocated once outside loop)
        if (nodextr_all_gpu != nullptr) {
            // Safe free to prevent segfaults
            if (nodextr_all_gpu) { cudaError_t err = cudaFree(nodextr_all_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        }
        
        auto time_nodextr_end = std::chrono::high_resolution_clock::now();
    }
    
    // OLD tnodewt computation moved to line 2764 (right after tree growing)
    // This section is now empty - tnodewt is computed on GPU immediately after trees finish
    
    // Compute variable importance for each tree individually (CPU fallback)
    // This must be done before OOB vote accumulation to ensure correct OOB predictions
    if (avimp_all != nullptr) {
        auto time_varimp_start = std::chrono::high_resolution_clock::now();
        
        // OPTIMIZATION: Allocate GPU memory once for all trees (avoids 30K+ malloc/free calls)
        rf::GPUVarImpContext* gpu_ctx = nullptr;
        if (task_type == 0) {  // Classification uses GPU
            // Use nsample as max_ninbag (worst-case: all samples in-bag)
            // We can't scan nin array here because it's not initialized until trees are grown
            // In practice, bootstrap samples ~63% unique, but we need to handle worst case
            integer_t max_ninbag = nsample;
            
            // Allocate context ONCE for all trees
            gpu_ctx = rf::gpu_varimp_alloc_context(nsample, mdim, maxnode, maxcat, max_ninbag, nclass, mdim, 256);
        }
        
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
        auto time_tree_start = std::chrono::high_resolution_clock::now();
        // std::cout << "DEBUG: Processing tree " << tree_id << " for variable importance..." << std::endl;
        
        // Get tree-specific data
        integer_t tree_offset = tree_id * maxnode;
        integer_t jinbag_offset = tree_id * nsample;
        
        // std::cout << "DEBUG: About to call cpu_testreebag for tree " << tree_id << std::endl;
        
        // OPTIMIZATION: Reuse nodextr_all that was already computed above (GPU-accelerated)
        // Instead of calling cpu_testreebag again, just extract nodextr_tree from nodextr_all
        auto time_extract_start = std::chrono::high_resolution_clock::now();
        std::vector<real_t> q_tree(nsample * nclass, 0.0f);
        std::vector<integer_t> jtr_tree(nsample, 0);
        std::vector<integer_t> nodextr_tree(nsample, 0);
        
        // Copy nodextr from nodextr_all (already computed above)
        std::copy(nodextr_all + jinbag_offset, nodextr_all + jinbag_offset + nsample, 
                  nodextr_tree.begin());
        
        // OPTIMIZATION: Only compute jtr_tree for OOB samples (needed for importance)
        // We can do this quickly by traversing tree only for OOB samples
        
        // Compute jtr_tree from nodextr_tree for ALL samples (needed for OOB vote accumulation)
        // jtr_tree[n] = nodeclass of terminal node that sample n reaches
        // nodextr_tree must be valid (copied from nodextr_all above)
        // Simplified logic: just look up nodeclass from nodextr (terminal node index)
        for (integer_t n = 0; n < nsample; ++n) {
            integer_t kt = nodextr_tree[n];  // Use pre-computed terminal node
            if (kt >= 0 && kt < nnode_all[tree_id] && 
                nodestatus_all[tree_offset + kt] == -1) {  // Valid terminal node
                jtr_tree[n] = nodeclass_all[tree_offset + kt];
            } else {
                // Invalid node or not terminal, use default (shouldn't happen if nodextr is correct)
                jtr_tree[n] = 0;
            }
        }
        auto time_extract_end = std::chrono::high_resolution_clock::now();
        auto extract_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_extract_end - time_extract_start);
        
        // DEBUG STEP 1: Verify OOB per tree - check nodextr_tree and jtr_tree for OOB samples
        if (task_type == 0 && tree_id < 3) {  // Classification, first 3 trees only
            integer_t oob_samples_checked = 0;
            integer_t oob_samples_with_valid_jtr = 0;
            integer_t oob_samples_with_zero_nodextr = 0;
            for (integer_t n = 0; n < nsample && oob_samples_checked < 10; ++n) {
                if (nin[jinbag_offset + n] == 0) {  // OOB sample
                    oob_samples_checked++;
                    if (nodextr_tree[n] == 0) {
                        oob_samples_with_zero_nodextr++;
                    }
                    if (jtr_tree[n] > 0 || (jtr_tree[n] == 0 && nodextr_tree[n] >= 0)) {
                        oob_samples_with_valid_jtr++;
                    }
                    if (oob_samples_checked <= 5) {
// Debug prints removed
                        // std::cout << "[DEBUG TREE " << tree_id << "] OOB sample " << n 
                        //           << ": nodextr=" << nodextr_tree[n] 
                        //           << ", jtr=" << jtr_tree[n]
                        //           << ", nodeclass=" << (nodextr_tree[n] >= 0 && nodextr_tree[n] < nnode_all[tree_id] ? 
                        //               nodeclass_all[tree_offset + nodextr_tree[n]] : -1)
                        //           << ", nodestatus=" << (nodextr_tree[n] >= 0 && nodextr_tree[n] < nnode_all[tree_id] ? 
                        //               nodestatus_all[tree_offset + nodextr_tree[n]] : -999)
                        //           << std::endl;
                    }
                }
            }
// Debug prints removed
            // std::cout << "[DEBUG TREE " << tree_id << "] OOB samples checked: " << oob_samples_checked
            //           << ", valid jtr: " << oob_samples_with_valid_jtr
            //           << ", zero nodextr: " << oob_samples_with_zero_nodextr << std::endl;
        }
        
        // Count OOB samples for this tree
        integer_t noob_count = 0;
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[jinbag_offset + n] == 0) {
                noob_count++;
            }
        }
        
        // For regression: compute tnodewt_all before OOB accumulation
        // tnodewt stores unscaled regression predictions (mean of y in terminal node)
        // This matches CPU behavior where tnodewt is computed during tree growing
        if (task_type == 1 && oob_predictions_all != nullptr && tnodewt_all != nullptr) {  // REGRESSION
            integer_t tree_offset = tree_id * maxnode;
            real_t* tnodewt_tree = tnodewt_all + tree_offset;
            
            // Compute tnodewt for this tree using GPU kernel
            // This computes the mean of y_regression values in each terminal node
            integer_t nnode_tree = nnode_all[tree_id];
            if (nnode_tree > 0) {
                // Get tree-specific pointers
                integer_t* treemap_tree = treemap_all + tree_offset * 2;
                integer_t* nodestatus_tree = nodestatus_all + tree_offset;
                real_t* xbestsplit_tree = xbestsplit_all + tree_offset;
                integer_t* bestvar_tree = bestvar_all + tree_offset;
                
                // For regression, jinbag is not used (kernel iterates through all samples)
                // Pass nullptr for jinbag and 0 for ninbag
                const integer_t* jinbag_tree = nullptr;
                integer_t ninbag_regression = 0;
                
                // Launch GPU kernel to compute tnodewt for all nodes in this tree
                dim3 block_size(256);
                dim3 grid_size((nnode_tree + block_size.x - 1) / block_size.x);
                
                gpu_compute_tnodewt_kernel<<<grid_size, block_size>>>(
                    x, nsample, mdim, nnode_tree,
                    treemap_tree, nodestatus_tree, xbestsplit_tree, bestvar_tree,
                    cl, y_regression, win,
                    nin + jinbag_offset,  // Pass nin for correct tnodewt computation
                    jinbag_tree, ninbag_regression,  // Not used for regression
                    nclass, task_type,
                    tnodewt_tree);
                
                // OPTIMIZATION: Don't sync per-tree - batch sync after all trees
                // CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Removed - sync once after loop
            }
        }
        
        // OPTIMIZATION: Batch sync after all tnodewt kernels complete (for regression GPU path)
        // This ensures all tnodewt computations finish before proceeding to importance
        if (task_type == 1 && tnodewt_all != nullptr && tree_id == num_trees - 1) {
            // Sync once after the last tree's kernel is launched
            CUDA_CHECK_VOID(cudaDeviceSynchronize());
        }
        
        // Accumulate OOB votes/predictions for this tree only
        // For all task types, increment nout_all for OOB samples to match CPU behavior
        // This ensures correct normalization in finalize_training() and compute_regression_mse()
        if (task_type == 0 || task_type == 2) {  // CLASSIFICATION or UNSUPERVISED
            // For classification/unsupervised: accumulate class votes in q_tree
            // Also increment nout_all for OOB samples (needed for normalization)
            
            // Compute tnodewt for classification casewise mode (CPU fallback)
            // For classification with casewise=True, tnodewt[node] = mean bootstrap weight of in-bag samples in that node
            std::vector<real_t> tnodewt_tree(maxnode, 0.0f);
            
            if (g_config.use_casewise && tnodewt_all != nullptr) {
                // Count samples and sum weights for each terminal node
                std::vector<real_t> node_weight_sum(maxnode, 0.0f);
                std::vector<integer_t> node_sample_count(maxnode, 0);
                
                // For each IN-BAG sample, traverse tree to find terminal node
                const integer_t* nodestatus_tree = nodestatus_all + tree_offset;
                const integer_t* bestvar_tree = bestvar_all + tree_offset;
                const real_t* xbestsplit_tree = xbestsplit_all + tree_offset;
                const integer_t* treemap_tree = treemap_all + tree_offset * 2;
                const integer_t* catgoleft_tree = (catgoleft_all_host != nullptr) ? 
                    catgoleft_all_host + tree_id * maxnode * maxcat : nullptr;
                
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] > 0) {  // In-bag sample
                        // Traverse tree from root to find terminal node
                        integer_t current_node = 0;
                        while (nodestatus_tree[current_node] == 1) {  // While not terminal
                            integer_t split_var = bestvar_tree[current_node];
                            bool goes_left = false;
                            
                            // Check if categorical
                            integer_t split_numcat = (cat != nullptr && split_var < mdim) ? cat[split_var] : 1;
                            if (split_numcat > 1) {  // Categorical
                                integer_t category_idx = static_cast<integer_t>(x[n * mdim + split_var] + 0.5f);
                                integer_t catgoleft_offset = current_node * maxcat;
                                goes_left = (category_idx >= 0 && category_idx < maxcat && 
                                           catgoleft_tree[catgoleft_offset + category_idx] == 1);
                            } else {  // Quantitative
                                real_t split_point = xbestsplit_tree[current_node];
                                real_t sample_value = x[n * mdim + split_var];
                                goes_left = (sample_value <= split_point);
                            }
                            
                            current_node = treemap_tree[current_node * 2 + (goes_left ? 0 : 1)];
                            if (current_node < 0 || current_node >= nnode_all[tree_id]) break;
                        }
                        
                        // Add to terminal node statistics
                        if (current_node >= 0 && current_node < nnode_all[tree_id] && 
                            nodestatus_tree[current_node] == -1) {
                            node_weight_sum[current_node] += static_cast<real_t>(nin[jinbag_offset + n]);
                            node_sample_count[current_node]++;
                        }
                    }
                }
                
                // Compute mean weight for each terminal node
                integer_t nonzero_tnodewt = 0;
                for (integer_t node = 0; node < nnode_all[tree_id]; ++node) {
                    if (node_sample_count[node] > 0) {
                        tnodewt_tree[node] = node_weight_sum[node] / static_cast<real_t>(node_sample_count[node]);
                        if (tnodewt_tree[node] > 0.0f) nonzero_tnodewt++;
                    }
                }
                
                
                // Copy to global tnodewt_all array
                for (integer_t node = 0; node < maxnode; ++node) {
                    tnodewt_all[tree_offset + node] = tnodewt_tree[node];
                }
            }
            
            // DEBUG STEP 1: Check jtr_tree for OOB samples (before vote accumulation)
            if (task_type == 0 && tree_id < 3) {
                integer_t oob_with_valid_jtr = 0;
                integer_t oob_with_zero_jtr = 0;
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] == 0) {  // OOB sample
                        if (jtr_tree[n] >= 0 && jtr_tree[n] < nclass) {
                            oob_with_valid_jtr++;
                            if (oob_with_valid_jtr <= 5) {
// Debug prints removed
                                // std::cout << "[DEBUG TREE " << tree_id << "] STEP 1 - OOB sample " << n 
                                //           << ": nodextr=" << nodextr_tree[n]
                                //           << ", jtr=" << jtr_tree[n] << std::endl;
                            }
                        } else {
                            oob_with_zero_jtr++;
                        }
                    }
                }
// Debug prints removed
                // std::cout << "[DEBUG TREE " << tree_id << "] STEP 1 - OOB samples: valid_jtr=" 
                //           << oob_with_valid_jtr << ", zero_jtr=" << oob_with_zero_jtr << std::endl;
            }
            
            for (integer_t n = 0; n < nsample; ++n) {
                if (nin[jinbag_offset + n] == 0) {  // Sample is out-of-bag
                    // Increment OOB count for this sample (matches CPU: nout_[n]++)
                    if (nout_all != nullptr) {
                        nout_all[n]++;
                    }
                    
                    if (jtr_tree[n] >= 0 && jtr_tree[n] < nclass) {
                        // Casewise vs non-casewise weighting for OOB votes (GPU path)
                        // Non-casewise: weight = 1.0 (UC Berkeley standard)
                        // Casewise: weight = tnodewt[terminal_node] (bootstrap frequency weighted)
                        real_t vote_weight = 1.0f;
                        if (g_config.use_casewise && tnodewt_all != nullptr) {
                            integer_t terminal_node = nodextr_tree[n];
                            if (terminal_node >= 0 && terminal_node < nnode_all[tree_id]) {
                                vote_weight = tnodewt_tree[terminal_node];
                            }
                        }
                        q_tree[n * nclass + jtr_tree[n]] += vote_weight;
                    }
                }
            }
            
            // Accumulate q_tree into global q_all (needed for OOB predictions)
            // Note: q_all is host memory, q_all_gpu is GPU memory
            // We accumulate into host q_all here, and gpu_oob_vote_kernel will also write to q_all_gpu
            // For classification, we use the per-tree accumulation (more accurate)
            if (q_all != nullptr) {
                // DEBUG STEP 2: Verify vote accumulation per tree
                if (task_type == 0 && tree_id < 3) {  // Classification, first 3 trees
                    integer_t votes_before = 0;
                    integer_t votes_after = 0;
                    integer_t oob_samples_with_votes = 0;
                    for (integer_t n = 0; n < nsample && n < 10; ++n) {
                        if (nin[jinbag_offset + n] == 0) {  // OOB sample
                            integer_t votes_in_tree = 0;
                            for (integer_t j = 0; j < nclass; ++j) {
                                votes_before += static_cast<integer_t>(q_all[n * nclass + j]);
                                votes_in_tree += static_cast<integer_t>(q_tree[n * nclass + j]);
                            }
                            if (votes_in_tree > 0) {
                                oob_samples_with_votes++;
                                if (oob_samples_with_votes <= 3) {
// Debug prints removed
                                    // std::cout << "[DEBUG TREE " << tree_id << "] STEP 2 - Sample " << n 
                                    //           << " votes in q_tree: ";
                                    // for (integer_t j = 0; j < nclass; ++j) {
                                    //     std::cout << "class" << j << "=" << q_tree[n * nclass + j] << " ";
                                    // }
                                    // std::cout << std::endl;
                                }
                            }
                        }
                    }
                    // Accumulate
                    for (integer_t n = 0; n < nsample; ++n) {
                        for (integer_t j = 0; j < nclass; ++j) {
                            q_all[n * nclass + j] += q_tree[n * nclass + j];
                        }
                    }
                    // Check after accumulation
                    for (integer_t n = 0; n < nsample && n < 10; ++n) {
                        if (nin[jinbag_offset + n] == 0) {
                            for (integer_t j = 0; j < nclass; ++j) {
                                votes_after += static_cast<integer_t>(q_all[n * nclass + j]);
                            }
                        }
                    }
// Debug prints removed
                    // std::cout << "[DEBUG TREE " << tree_id << "] STEP 2 - Votes before: " << votes_before
                    //           << ", after: " << votes_after
                    //           << ", OOB samples with votes: " << oob_samples_with_votes << std::endl;
                } else {
                    // Normal accumulation without debug
                    for (integer_t n = 0; n < nsample; ++n) {
                        for (integer_t j = 0; j < nclass; ++j) {
                            q_all[n * nclass + j] += q_tree[n * nclass + j];
                        }
                    }
                }
            }
        } else if (task_type == 1) {  // REGRESSION
            // For regression: accumulate regression predictions from tnodewt[nodextr_tree[n]]
            // oob_predictions_all stores the SUM of predictions across all trees
            // Final prediction = oob_predictions_all[n] / nout_all[n] (average across trees)
            // Increment nout_all for OOB samples to match CPU behavior
            // This ensures correct normalization in compute_regression_mse()
            if (oob_predictions_all != nullptr && tnodewt_all != nullptr && nout_all != nullptr) {
                integer_t tree_offset = tree_id * maxnode;
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] == 0) {  // Sample is out-of-bag
                        // Increment OOB count for this sample (matches CPU: nout_[n]++)
                        nout_all[n]++;
                        
                        integer_t terminal_node = nodextr_tree[n];
                        if (terminal_node >= 0 && terminal_node < nnode_all[tree_id]) {
                            // tnodewt stores unscaled regression predictions (mean of y in terminal node)
                            // This matches CPU: tnodewt[terminal_node] contains unscaled mean prediction
                            real_t regression_prediction = tnodewt_all[tree_offset + terminal_node];
                            oob_predictions_all[n] += regression_prediction;
                        }
                    }
                }
            }
        }
        
        // Compute variable importance for this tree using its individual OOB predictions
        // Create temporary arrays per tree (following CPU pattern)
        std::vector<real_t> qimp_temp(nsample, 0.0f);  // Temporary per-tree array
        std::vector<real_t> qimpm_temp(nsample * mdim, 0.0f);  // Temporary per-tree array
        std::vector<real_t> avimp_temp(mdim, 0.0f);
        std::vector<real_t> sqsd_temp(mdim, 0.0f);
        std::vector<integer_t> jvr_temp(nsample, 0);
        std::vector<integer_t> nodexvr_temp(nsample, 0);
        
        // Call appropriate variable importance function based on task type
        // Use global arrays for local importance accumulation
        auto time_varimp_func_start = std::chrono::high_resolution_clock::now();
        // Only require qimp_all for importance - qimpm_all is only needed for LOCAL importance
        if (qimp_all != nullptr) {
            if (task_type == 1) {  // 1=REGRESSION
                // Use regression-specific importance (MSE-based)
                // Use the original continuous y values passed as parameter
                if (y_regression != nullptr) {
                    // Set impn based on compute_local_importance flag
                    // impn = 1 means compute local importance, impn = 0 means skip local importance
                    integer_t impn = g_config.impn;  // Use global config flag (set from config_.compute_local_importance)
                    
                    // For regression, tnodewt_ptr should point to tnodewt_all if available
                    // (tnodewt_all is computed for OOB predictions and contains mean y values in terminal nodes)
                    // Note: tnodewt_ptr would be used when regression varimp is implemented
                    (void)tnodewt_all;  // Suppress unused variable warning
                    
                    // REGRESSION VARIMP NOT IMPLEMENTED IN THIS RELEASE
                    throw std::runtime_error("Regression variable importance not supported in this release");
                    /*
                    // Use temporary arrays per tree (following CPU pattern)
                    // tnodewt_ptr should already be computed above (after tree growing)
                    cpu_varimp_regression(x, nsample, mdim,
                                         y_regression, nin + jinbag_offset, jtr_tree.data(),
                                     impn,
                                     qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                                     treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                                     xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                                     nodeclass_all + tree_offset, nnode_all[tree_id],
                                     cat, jvr_temp.data(), nodexvr_temp.data(),
                                     maxcat, nullptr, // catgoleft not used
                                     tnodewt_ptr, nodextr_tree.data());
                    */
                    
                    // Accumulate into global arrays (following CPU pattern)
                    if (qimp_all != nullptr) {
                        for (integer_t i = 0; i < nsample; i++) {
                            qimp_all[i] += qimp_temp[i];
                        }
                    }
                    // Only accumulate qimpm_all if local importance was requested
                    if (qimpm_all != nullptr && g_config.impn == 1) {
                        for (integer_t i = 0; i < nsample * mdim; i++) {
                            qimpm_all[i] += qimpm_temp[i];
                        }
                    }
                } else {
                    // y_regression is nullptr - fallback to classification importance (should not happen for regression)
                    // This is a safety fallback
                }
            } else {
                // CLASSIFICATION PATH: Use GPU variable importance with temporary host arrays
                // gpu_varimp expects host pointers and handles GPU allocation internally
                // Use temp arrays per tree (following CPU pattern) then accumulate
                // Set impn based on compute_local_importance flag
                integer_t impn = g_config.impn;  // Use global config flag (set from config_.compute_local_importance)
            // gpu_varimp will compute tnodewt internally if needed (when use_casewise=true for classification)
            // Pass nullptr for tnodewt to indicate it should be computed internally
            // Compute ninbag (number of in-bag samples) from nin - matches CPU implementation
            integer_t ninbag = 0;
            const integer_t* nin_tree = nin + jinbag_offset;
            for (integer_t i = 0; i < nsample; ++i) {
                if (nin_tree[i] > 0) {
                    ninbag++;
                }
            }
            const integer_t* jinbag_tree = jinbag_all + jinbag_offset;
            
            // DEBUG: Check jinbag_tree values for first tree
            if (task_type == 0 && tree_id < 2 && ninbag > 0) {
// Debug prints removed
                // std::cout << "[DEBUG JINBAG TREE " << tree_id << "] ninbag=" << ninbag 
                //           << ", first 10 jinbag values: ";
                // for (integer_t i = 0; i < std::min(10, ninbag); ++i) {
                //     std::cout << jinbag_tree[i] << " ";
                // }
                // std::cout << std::endl;
// Debug prints removed
                // std::cout << "[DEBUG JINBAG TREE " << tree_id << "] first 10 nin values: ";
                // for (integer_t i = 0; i < std::min(10, nsample); ++i) {
                //     std::cout << nin_tree[i] << " ";
                // }
                // std::cout << std::endl;
            }
            
            // tnodewt should already be computed above (after tree growing)
            real_t* tnodewt_ptr = nullptr;
            if (tnodewt_all != nullptr) {
                tnodewt_ptr = tnodewt_all + tree_offset;
            }
            
            // OPTIMIZED: Use pre-allocated GPU context (avoids 20+ malloc/free per tree)
            if (gpu_ctx != nullptr) {
                gpu_varimp_with_context(gpu_ctx, x, nsample, mdim, cl, nin + jinbag_offset, jtr_tree.data(),
                           impn, qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                           treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                           xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                           nodeclass_all + tree_offset, nnode_all[tree_id],
                           cat, jvr_temp.data(), nodexvr_temp.data(),
                           maxcat, nullptr, tnodewt_ptr, nodextr_tree.data(),
                           y_regression, win, jinbag_tree, ninbag, nclass, task_type);
            } else {
                // Context allocation failed - compute OVERALL importance only (skip local)
                // Local importance would be too slow without optimized context (30× slower)
                if (tree_id == 0) {
                    std::cerr << "Warning: GPU memory insufficient for local importance context." << std::endl;
                    std::cerr << "         Computing overall importance only (local importance disabled)." << std::endl;
                }
                
                // Just compute overall importance using the old path
                gpu_varimp(x, nsample, mdim, cl, nin + jinbag_offset, jtr_tree.data(),
                           0,  // impn=0: disable local importance (too slow without context)
                           qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                           treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                           xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                           nodeclass_all + tree_offset, nnode_all[tree_id],
                           cat, jvr_temp.data(), nodexvr_temp.data(),
                           maxcat, nullptr, tnodewt_ptr, nodextr_tree.data(),
                           y_regression, win, jinbag_tree, ninbag, nclass, task_type);
            }
            
            // Debug prints removed to avoid potential stream corruption issues
            // if (task_type == 0 && tree_id < 3) {
            //     std::cout << "[DEBUG IMPORTANCE TREE " << tree_id << "] After gpu_varimp: "
            //               << "avimp_temp: ";
            //     for (integer_t i = 0; i < mdim; ++i) {
            //         std::cout << avimp_temp[i] << " ";
            //     }
            //     std::cout << std::endl;
            //     integer_t non_zero_qimp = 0;
            //     for (integer_t i = 0; i < nsample; ++i) {
            //         if (qimp_temp[i] != 0.0f) non_zero_qimp++;
            //     }
            //     std::cout << "[DEBUG IMPORTANCE TREE " << tree_id << "] qimp_temp non-zero: " 
            //               << non_zero_qimp << " / " << nsample << std::endl;
            // }
                
                // Check for CUDA errors after gpu_varimp (it uses GPU internally)
                cudaError_t varimp_err = cudaGetLastError();
                if (varimp_err != cudaSuccess) {
                    // std::cerr << "Warning: CUDA error after gpu_varimp for tree " << tree_id << ": " << cudaGetErrorString(varimp_err) << std::endl;
                    // std::cerr << "Warning: Clearing error and recovering context..." << std::endl;
                    // Aggressive recovery: sync and clear errors
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Wait for all operations
                    // Ensure device is still valid
                    int device = 0;
                    cudaGetDevice(&device);
                }
                
                // Accumulate into global host arrays (following CPU pattern)
                if (qimp_all != nullptr) {
                    for (integer_t i = 0; i < nsample; i++) {
                        qimp_all[i] += qimp_temp[i];
                    }
                }
                // Only accumulate qimpm_all if local importance was requested
                if (qimpm_all != nullptr && g_config.impn == 1) {
                    for (integer_t i = 0; i < nsample * mdim; i++) {
                        qimpm_all[i] += qimpm_temp[i];
                    }
                }
            }
        } else {
            // std::cout << "GPU: Skipping variable importance for tree " << tree_id << " (arrays not allocated)" << std::endl;
        }
        // std::cout << "GPU: gpu_varimp completed for tree " << tree_id << std::endl;
        
        // Accumulate importance values into the global arrays
        // Ensure avimp_all is not nullptr before accumulating
        if (avimp_all != nullptr) {
        for (integer_t i = 0; i < mdim; ++i) {
            avimp_all[i] += avimp_temp[i];
            }
        }
        
        auto time_varimp_func_end = std::chrono::high_resolution_clock::now();
        auto time_tree_end = std::chrono::high_resolution_clock::now();
        }
        
        auto time_varimp_end = std::chrono::high_resolution_clock::now();
        
        // CLEANUP: Free GPU context after all trees are processed
        if (gpu_ctx != nullptr) {
            rf::gpu_varimp_free_context(gpu_ctx);
        }
        
        // Synchronize after importance computation loop completes
        // Use stream sync for better Jupyter compatibility
        if (avimp_all != nullptr) {
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));
        }
    }
    
    // Accumulate OOB votes for classification/unsupervised
    // This MUST run even if avimp_all is nullptr (when importance is not computed)
    // The vote accumulation was previously inside the avimp_all block, causing it to be skipped
    if ((task_type == 0 || task_type == 2) && q_all != nullptr && nodextr_all != nullptr) {
        // Classification/unsupervised: accumulate votes per tree
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            integer_t tree_offset = tree_id * maxnode;
            integer_t jinbag_offset = tree_id * nsample;
            
            // Extract nodextr_tree and compute jtr_tree for this tree
            std::vector<integer_t> nodextr_tree(nsample, 0);
            std::vector<integer_t> jtr_tree(nsample, 0);
            std::vector<real_t> q_tree(nsample * nclass, 0.0f);
            
            // Copy nodextr from nodextr_all
            std::copy(nodextr_all + jinbag_offset, nodextr_all + jinbag_offset + nsample, 
                      nodextr_tree.begin());
            
            // Compute jtr_tree from nodextr_tree
            for (integer_t n = 0; n < nsample; ++n) {
                integer_t kt = nodextr_tree[n];
                if (kt >= 0 && kt < nnode_all[tree_id] && 
                    nodestatus_all[tree_offset + kt] == -1) {
                    jtr_tree[n] = nodeclass_all[tree_offset + kt];
                } else {
                    jtr_tree[n] = 0;
                }
            }
            
            // Compute tnodewt for casewise mode (CPU fallback)
            // This must be done here for OOB vote accumulation to work correctly with casewise
            std::vector<real_t> tnodewt_tree(maxnode, 0.0f);
            if (g_config.use_casewise && tnodewt_all != nullptr) {
                // Count samples and sum weights for each terminal node
                std::vector<real_t> node_weight_sum(maxnode, 0.0f);
                std::vector<integer_t> node_sample_count(maxnode, 0);
                
                const integer_t* nodestatus_tree = nodestatus_all + tree_offset;
                const integer_t* bestvar_tree = bestvar_all + tree_offset;
                const real_t* xbestsplit_tree = xbestsplit_all + tree_offset;
                const integer_t* treemap_tree = treemap_all + tree_offset * 2;
                const integer_t* catgoleft_tree = (catgoleft_all != nullptr) ? 
                    catgoleft_all + tree_id * maxnode * maxcat : nullptr;
                
                // For each IN-BAG sample, traverse tree to find terminal node
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] > 0) {  // In-bag sample
                        integer_t current_node = 0;
                        while (nodestatus_tree[current_node] == 1) {  // While not terminal
                            integer_t split_var = bestvar_tree[current_node];
                            bool goes_left = false;
                            
                            // Check if categorical
                            integer_t split_numcat = (cat != nullptr && split_var < mdim) ? cat[split_var] : 1;
                            if (split_numcat > 1) {  // Categorical
                                integer_t category_idx = static_cast<integer_t>(x[n * mdim + split_var] + 0.5f);
                                integer_t catgoleft_offset = current_node * maxcat;
                                goes_left = (category_idx >= 0 && category_idx < maxcat && 
                                           catgoleft_tree[catgoleft_offset + category_idx] == 1);
                            } else {  // Quantitative
                                real_t split_point = xbestsplit_tree[current_node];
                                real_t sample_value = x[n * mdim + split_var];
                                goes_left = (sample_value <= split_point);
                            }
                            
                            current_node = treemap_tree[current_node * 2 + (goes_left ? 0 : 1)];
                            if (current_node < 0 || current_node >= nnode_all[tree_id]) break;
                        }
                        
                        // Add to terminal node statistics
                        if (current_node >= 0 && current_node < nnode_all[tree_id] && 
                            nodestatus_tree[current_node] == -1) {
                            node_weight_sum[current_node] += static_cast<real_t>(nin[jinbag_offset + n]);
                            node_sample_count[current_node]++;
                        }
                    }
                }
                
                // Compute mean weight for each terminal node
                for (integer_t node = 0; node < nnode_all[tree_id]; ++node) {
                    if (node_sample_count[node] > 0) {
                        tnodewt_tree[node] = node_weight_sum[node] / static_cast<real_t>(node_sample_count[node]);
                    }
                }
                
                // Copy to global tnodewt_all array
                for (integer_t node = 0; node < maxnode; ++node) {
                    tnodewt_all[tree_offset + node] = tnodewt_tree[node];
                }
            }
            
            // Accumulate votes for OOB samples
            for (integer_t n = 0; n < nsample; ++n) {
                if (nin[jinbag_offset + n] == 0) {  // OOB sample
                    if (nout_all != nullptr) {
                        nout_all[n]++;
                    }
                    if (jtr_tree[n] >= 0 && jtr_tree[n] < nclass) {
                        // Casewise vs non-casewise weighting for OOB votes
                        // Non-casewise: weight = 1.0 (UC Berkeley standard)
                        // Casewise: weight = tnodewt[terminal_node] (bootstrap frequency weighted)
                        real_t vote_weight = 1.0f;
                        if (g_config.use_casewise && tnodewt_all != nullptr) {
                            integer_t kt = nodextr_tree[n];
                            if (kt >= 0 && kt < nnode_all[tree_id] && nodestatus_all[tree_offset + kt] == -1) {
                                vote_weight = tnodewt_all[tree_offset + kt];
                            }
                        }
                        q_tree[n * nclass + jtr_tree[n]] += vote_weight;
                    }
                }
            }
            
            // Accumulate q_tree into global q_all
            for (integer_t n = 0; n < nsample; ++n) {
                for (integer_t j = 0; j < nclass; ++j) {
                    q_all[n * nclass + j] += q_tree[n * nclass + j];
                }
            }
        }
    }
    
    // Launch OOB vote accumulation kernel
    // For classification (task_type == 0), we skip the GPU kernel because
    // we already accumulate votes per-tree in the loop above (q_tree -> q_all).
    // The GPU kernel would overwrite our correct per-tree accumulation.
    // For regression/unsupervised, we can use the GPU kernel if needed.
    if (task_type != 0) {  // Not classification - use GPU kernel for regression/unsupervised
        // std::cout << "GPU: Launching OOB vote kernel..." << std::endl;
        
        dim3 oob_block_size(256);
        dim3 oob_grid_size(num_trees);
        
        gpu_oob_vote_kernel<<<oob_grid_size, oob_block_size>>>(
            num_trees, nsample, nclass, maxnode, mdim,
            nodestatus_all_gpu, nodeclass_all_gpu,
            bestvar_all_gpu, xbestsplit_all_gpu, treemap_all_gpu,
            nin_all_gpu, cl_gpu, x_gpu, q_all_gpu
        );
        
        // Check for kernel launch errors immediately
        // cudaError_t kernel_err = cudaGetLastError();
        // if (kernel_err != cudaSuccess) {
        //     std::cerr << "Warning: OOB vote kernel launch error: error code " << kernel_err << std::endl;
        //     std::cerr << "Warning: Clearing error and continuing..." << std::endl;
        // }
        
        // std::cout << "DEBUG: About to sync stream after OOB vote kernel\n";
        
        // Check for kernel launch errors before syncing
        // cudaError_t launch_err = cudaGetLastError();
        // if (launch_err != cudaSuccess) {
        //     std::cerr << "ERROR: OOB vote kernel launch failed: " << cudaGetErrorString(launch_err) << std::endl;
        //     return;  // Exit early to prevent hang
        // }
        
        // Use device sync like backup version
        CUDA_CHECK_VOID(cudaDeviceSynchronize());
        
        // Check for kernel execution errors after sync
        // cudaError_t exec_err = cudaGetLastError();
        // if (exec_err != cudaSuccess) {
        //     std::cerr << "ERROR: OOB vote kernel execution error: " << cudaGetErrorString(exec_err) << std::endl;
        //     std::cerr << "ERROR: Exiting to prevent further issues." << std::endl;
        //     return;  // Exit early
        // }
        
        // std::cout << "GPU: OOB vote accumulation completed!" << std::endl;
        
        // Copy q array back to host - matches backup version (direct call, no error checking)
        cudaMemcpy(q_all, q_all_gpu, nsample * nclass * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else {
        // Classification: q_all already contains correct votes from per-tree accumulation
        // No need to copy from GPU - our host-side accumulation is correct
        // std::cout << "GPU: Skipping OOB vote kernel for classification (using per-tree accumulation)" << std::endl;
        
        // DEBUG STEP 3 & 4: Verify multiple trees accumulation and overall prediction
        if (task_type == 0 && q_all != nullptr) {  // Classification
// Debug prints removed
            // std::cout << "[DEBUG STEP 3&4] Final q_all after all trees:" << std::endl;
            integer_t samples_with_votes = 0;
            integer_t samples_with_zero_votes = 0;
            for (integer_t n = 0; n < nsample && n < 20; ++n) {
                integer_t total_votes = 0;
                real_t max_votes = -1.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    total_votes += static_cast<integer_t>(q_all[n * nclass + j]);
                    if (q_all[n * nclass + j] > max_votes) {
                        max_votes = q_all[n * nclass + j];
                    }
                }
                // Debug prints removed
                // if (total_votes > 0) {
                //     samples_with_votes++;
                //     if (samples_with_votes <= 10) {
                //         std::cout << "  Sample " << n << ": total_votes=" << total_votes 
                //                   << ", predicted_class=" << max_class
                //                   << ", votes: ";
                //         for (integer_t j = 0; j < nclass; ++j) {
                //             std::cout << "class" << j << "=" << q_all[n * nclass + j] << " ";
                //         }
                //         std::cout << std::endl;
                //     }
                // } else {
                //     samples_with_zero_votes++;
                // }
            }
// Debug prints removed
            // std::cout << "[DEBUG STEP 3&4] Samples with votes: " << samples_with_votes
            //           << ", samples with zero votes: " << samples_with_zero_votes << std::endl;
        }
    }
    {
        // DEBUG: Check if q_all has non-zero values
        // DEBUG output disabled for production
        // std::cout << "DEBUG: Checking q_all after copy from GPU..." << std::endl;
        // int non_zero_count = 0;
        // int max_class_votes[7] = {0};
        // for (integer_t n = 0; n < nsample && n < 10; ++n) {  // Check first 10 samples
        //     for (integer_t j = 0; j < nclass; ++j) {
        //         integer_t idx = n * nclass + j;
        //         if (q_all[idx] > 0.0f) {
        //             non_zero_count++;
        //             if (q_all[idx] > max_class_votes[j]) {
        //                 max_class_votes[j] = static_cast<int>(q_all[idx]);
        //             }
        //         }
        //     }
        // }
        // std::cout << "DEBUG: q_all non-zero entries (first 10 samples): " << non_zero_count << std::endl;
        // std::cout << "DEBUG: Max votes per class (first 10 samples): ";
        // for (integer_t j = 0; j < nclass && j < 7; ++j) {
        //     std::cout << "class" << j << "=" << max_class_votes[j] << " ";
        // }
        // std::cout << std::endl;
    }
    
    // NOTE: Local importance (qimp_all and qimpm_all) is accumulated on HOST side
    // during the tree loop using temporary arrays (qimp_temp, qimpm_temp).
    // The GPU arrays (qimp_all_gpu, qimpm_all_gpu) are allocated but NOT used for accumulation.
    // Therefore, we do NOT copy from GPU arrays back to host - the host arrays already have
    // the accumulated values from the per-tree accumulation loop.
    // 
    // If we copied from GPU arrays (which are all zeros), we would overwrite the accumulated values!
    
    // Matches backup version - no error checking before proximity
    
    // Synchronize to ensure all previous operations complete
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // NOTE: nout_all is now incremented per-tree in the OOB accumulation loop above
    // (lines 3558-3591) to match CPU behavior and ensure correct normalization.
    // This duplicate loop has been removed to avoid double-counting.
    // The per-tree incrementation ensures nout_all[n] correctly counts how many trees
    // had sample n as OOB, which is needed for normalization in compute_regression_mse()
    // and finalize_training().
    
    // Compute proximity for each tree if requested
    // std::cout << "DEBUG: About to check proximity computation" << std::endl;
    
    // std::cout << "DEBUG: Entering proximity computation section" << std::endl;
    
    // Skip proximity entirely if use_qlora is requested but not enabled
    // Low-rank mode (use_qlora=True) is required for GPU proximity computation
    if ((proximity_all != nullptr || g_config.iprox) && !g_config.use_qlora && g_config.use_gpu) {
        // Warning removed for Jupyter compatibility (don't print to stderr)
        // std::cerr << "WARNING: Proximity computation requested on GPU but use_qlora=False..." << std::endl;
        // Continue anyway - will use traditional approach if proximity_all is allocated
    }
    
    // Matches backup version - no try-catch, simple CUDA check
    // Note: device_count already checked at function entry, so just check if CUDA is available
    int device_count_check = 0;
    cudaGetDeviceCount(&device_count_check);
    if (device_count_check == 0) {
        return;  // Skip proximity computation if CUDA is not available
    }
    
    // NOW we can safely check proximity flags
    // std::cout << "DEBUG: Checking g_config.iprox=" << g_config.iprox 
    //           << ", proximity_all=" << (proximity_all != nullptr ? "non-null" : "null") << std::endl;
    
    // Early return if proximity is not requested at all (skip all proximity code)
    // For low-rank (QLORA), proximity_all will be nullptr (we only keep factors)
    // For full matrix, proximity_all must be allocated
    if (!g_config.iprox) {
        // std::cout << "DEBUG: Proximity not requested (iprox=0), returning early" << std::endl;
        return;  // No proximity computation needed - skip all proximity code
    }
    
    // If proximity is requested but we're not using QLORA and proximity_all is null, warn and return
    if (!g_config.use_qlora && proximity_all == nullptr) {
        std::cerr << "Warning: Proximity requested (iprox=True) but proximity_all is nullptr and use_qlora=false. "
                  << "Enable use_qlora=True for low-rank proximity, or allocate proximity_all for full matrix." << std::endl;
        return;
    }
    
    // Use fully qualified names instead of namespace to avoid initialization issues
    // For low-rank mode, we can process even if proximity_all is nullptr (reconstruction deferred)
    bool use_lowrank = false;
    bool use_upper_triangle = true;
    
    // Debug: Print config values
    // std::cout << "GPU Proximity Debug: g_config.use_qlora=" << g_config.use_qlora 
    //           << ", g_config.quant_mode=" << g_config.quant_mode 
    //           << ", g_config.iprox=" << g_config.iprox 
    //           << ", proximity_all=" << (proximity_all != nullptr ? "non-null" : "null")
    //           << ", nsample=" << nsample << std::endl;
    
    // If proximity is requested but low-rank is disabled on GPU,
    // warn but allow it (user may have allocated proximity_all themselves)
    // if ((proximity_all != nullptr || g_config.iprox) && !g_config.use_qlora && g_config.use_gpu) {
    //     std::cerr << "WARNING: Proximity requested on GPU but use_qlora=false. "
    //               << "Low-rank mode (use_qlora=True) is recommended for GPU proximity to avoid huge temp buffer allocation. "
    //               << "Continuing with traditional approach if proximity_all is allocated." << std::endl;
    //     std::cerr.flush();
    // }
    
    // Only check low-rank if proximity is requested and CUDA is available
    if (proximity_all != nullptr || g_config.iprox) {
        // MODIFIED: Always use low-rank mode on GPU when use_qlora is enabled, regardless of dataset size
        // This enables low-rank factors for all datasets when using GPU with QLoRA
        // NOTE: Low-rank mode is GPU-only (requires CUDA kernels, cuBLAS, cuSolver)
        // CPU mode will never use low-rank - it always computes full proximity matrix
        // 
        // g_config.use_gpu is set to true at the start of this function (line 2214)
        // and g_config.use_qlora should be set by the caller (rf_random_forest.cpp) before calling this function
        if (g_config.use_gpu && g_config.use_qlora) {
            use_lowrank = true;  // Force low-rank for all GPU datasets when QLoRA is enabled
            use_upper_triangle = true;
        } else if (!g_config.use_gpu) {
            // CPU mode: use full matrix (low-rank not available on CPU)
            use_lowrank = false;  // CPU cannot use low-rank (CUDA-only feature)
            use_upper_triangle = true;
        } else {
            // GPU mode without qlora: use traditional full matrix approach if proximity_all is allocated
            use_lowrank = false;
            use_upper_triangle = true;
            // if (proximity_all == nullptr && g_config.iprox) {
            //     std::cerr << "Warning: Proximity requested (iprox=True) but proximity_all is nullptr and use_qlora=false. "
            //               << "Enable use_qlora=True for low-rank proximity, or allocate proximity_all for full matrix." << std::endl;
            // }
        }
    }
    
    // ========================================================================
    // SECTION 3: PROXIMITY COMPUTATION
    // ========================================================================
    // GPU proximity ALWAYS uses low-rank (QLoRA) - no full matrix option on GPU
    // Two types of GPU proximity:
    //   1. Standard proximity: compute_proximity=True, use_rfgap=False
    //   2. RF-GAP proximity: compute_proximity=True, use_rfgap=True
    // These are mutually exclusive
    
    bool should_compute_gpu_proximity = g_config.iprox && g_config.use_gpu && g_config.use_qlora;
    bool should_compute_rfgap = g_config.use_rfgap && should_compute_gpu_proximity;
    bool should_compute_standard_proximity = should_compute_gpu_proximity && !g_config.use_rfgap;
    
    if (should_compute_standard_proximity) {
        // Standard low-rank proximity (GPU only)
        
        // Use file-scope thread-local storage (g_lowrank_state declared at top of file)
        // This ensures state persists across all batch calls, fixing the bug where
        // 3000 trees (300 batches) would reset state and produce zero factors
        
        void*& lowrank_prox_ptr_raw = g_lowrank_state.lowrank_prox_ptr_raw;
        bool& lowrank_initialized = g_lowrank_state.lowrank_initialized;
        integer_t& total_trees_processed = g_lowrank_state.total_trees_processed;
        integer_t& saved_nsample = g_lowrank_state.saved_nsample;
        
        // Allocate proximity workspace arrays
        std::vector<integer_t> nod_workspace(maxnode);
        std::vector<integer_t> ncount_workspace(nsample);
        std::vector<integer_t> ncn_workspace(nsample);
        std::vector<integer_t> nodexb_workspace(nsample);
        std::vector<integer_t> ndbegin_workspace(nsample + 1);
        std::vector<integer_t> npcase_workspace(nsample);
        
        // std::cout << "GPU Proximity: use_qlora=" << g_config.use_qlora 
        //           << ", nsample=" << nsample 
        //           << ", use_lowrank=" << use_lowrank 
        //           << ", use_upper_triangle=" << use_upper_triangle << std::endl;
        
        // Ensure use_lowrank is consistent with use_qlora
        // If use_qlora is true and we're on GPU, use_lowrank MUST be true
        if (g_config.use_gpu && g_config.use_qlora && !use_lowrank) {
            // std::cerr << "WARNING: use_qlora=true but use_lowrank=false - forcing use_lowrank=true" << std::endl;
            use_lowrank = true;
            use_upper_triangle = true;
        }
        
        // For nsample > 10000, ONLY use low-rank AB' factors - NO temp buffer!
        if (use_lowrank && use_upper_triangle) {
            #ifdef __CUDACC__
            // Get quantization level from config - use fully qualified name
            // Delay type resolution until we actually need it
            int quant_mode_int = g_config.quant_mode;
            // REMOVED: quant_level determination - we always use FP16 during training
            // Quantization happens only at the end via finalize_accumulation()
            
            // Detect if this is a NEW model instance
            // If lowrank_proximity_ptr_out points to nullptr, it means RandomForest just allocated
            // a fresh lowrank_proximity_ptr_ member, so we're starting a new model
            // In this case, reset thread_local state to prevent reusing stale pointers from previous models
            bool is_new_model = (lowrank_proximity_ptr_out != nullptr && 
                                *lowrank_proximity_ptr_out == nullptr &&
                                lowrank_prox_ptr_raw != nullptr);
            
            // Check if nsample changed OR new model detected - need to reset
            if (lowrank_prox_ptr_raw != nullptr && (saved_nsample != nsample || is_new_model)) {
                    // Don't delete - the previous model's destructor already did that
                    // Just reset our thread_local pointers
                lowrank_prox_ptr_raw = nullptr;
                lowrank_initialized = false;
                total_trees_processed = 0;
                saved_nsample = 0;
            }
            
            // Initialize LowRankProximityMatrix on first batch (persists across batches)
            // DEFER initialization until AFTER first tree is processed to avoid crashes during static initialization
            if (!lowrank_initialized || lowrank_prox_ptr_raw == nullptr) {
                // std::cout << "GPU Proximity: Creating persistent LowRankProximityMatrix object..." << std::endl;
                // std::cout << "GPU Proximity: Parameters - nsample=" << nsample 
                //           << ", rank=100, quant_level=" << static_cast<int>(quant_level) << std::endl;
                
                // Validate inputs before creating
                if (nsample <= 0 || nsample > 1000000) {
                    // std::cerr << "Error: Invalid nsample for LowRankProximityMatrix: " << nsample << std::endl;
                    use_lowrank = false;
                } else {
                    // Ensure CUDA context is valid before creating handles
                    cudaError_t ctx_check = cudaSetDevice(0);
                    if (ctx_check != cudaSuccess) {
                        // std::cerr << "Error: Failed to set CUDA device: " << cudaGetErrorString(ctx_check) << std::endl;
                        use_lowrank = false;
                    } else {
                        // Matches backup version - direct call, no try-catch
                            integer_t max_rank = g_config.lowrank_rank > 0 ? g_config.lowrank_rank : 1000;
                            integer_t initial_rank = 0;  // Start with rank 0, will grow as trees are added
                            rf::cuda::QuantizationLevel quant_level_explicit = 
                                static_cast<rf::cuda::QuantizationLevel>(quant_mode_int);
                            lowrank_prox_ptr_raw = static_cast<void*>(
                                new rf::cuda::LowRankProximityMatrix(nsample, initial_rank, quant_level_explicit, max_rank)
                            );
                            saved_nsample = nsample;
                    }
                }
                
                if (lowrank_prox_ptr_raw != nullptr) {
                    // Initialize GPU memory for low-rank factors
                    // std::cout << "GPU Proximity: Initializing LowRankProximityMatrix..." << std::endl;
                    // Matches backup version - direct call, no try-catch
                        rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                            static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                        if (!lowrank_prox_ptr->initialize()) {
                            // If initialization fails, fall back to traditional approach
                            // std::cerr << "Warning: Low-rank proximity initialization failed, falling back to full matrix" << std::endl;
                            delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                            lowrank_prox_ptr_raw = nullptr;
                            use_lowrank = false;
                            lowrank_initialized = false;
                        } else {
                            lowrank_initialized = true;
                            total_trees_processed = 0;  // Reset tree counter
                    }
                }
            }
            
            if (lowrank_prox_ptr_raw != nullptr && lowrank_initialized) {
                // Cast void* to actual type when needed
                rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                    static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    
                // std::cout << "GPU Proximity: Processing " << num_trees << " trees (total trees processed so far: " 
                //           << total_trees_processed << ") with low-rank mode" << std::endl;
                
                // Process each tree for proximity computation
                for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                    // Get tree-specific data
                    integer_t tree_offset = tree_id * maxnode;
                    integer_t nin_offset = tree_id * nsample;
                    
                    // Safety check: ensure tree actually grew
                    if (nnode_all[tree_id] == 0) {
                        // std::cerr << "Warning: Tree " << tree_id << " has 0 nodes, skipping proximity" << std::endl;
                        continue;
                    }
                    
                    // For nsample > 10000: NO CPU temp buffers! Compute directly on GPU
                    // Allocate GPU memory directly (no CPU allocation)
                    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
                    
                    // ALWAYS compute per-tree proximity in FP16 during training
                    // Quantization to INT8/NF4 happens ONLY at the very end via finalize_accumulation()
                    // This prevents compounding quantization errors during training
                    __half* tree_prox_upper_gpu = nullptr;
                    CUDA_CHECK_VOID(cudaMalloc(&tree_prox_upper_gpu, upper_triangle_size * sizeof(__half)));
                    cudaMemset(tree_prox_upper_gpu, 0, upper_triangle_size * sizeof(__half));
                    
                    // Matches backup version - assume arrays are on host (they're copied from GPU earlier)
                    // Copy data to host arrays for gpu_proximity_upper_triangle (it expects host pointers)
                    std::vector<integer_t> nodestatus_host(nnode_all[tree_id]);
                    std::vector<integer_t> nodextr_host(nsample);
                    std::vector<integer_t> nin_host(nsample);
                    
                    // nodextr_all uses jinbag_offset = tree_id * nsample (same as nin_offset)
                    integer_t nodextr_offset = tree_id * nsample;  // nodextr_all uses same offset as jinbag
                    
                    
                    // Direct memcpy - arrays are already on host
                    std::memcpy(nodestatus_host.data(), nodestatus_all + tree_offset, 
                                nnode_all[tree_id] * sizeof(integer_t));
                    std::memcpy(nodextr_host.data(), nodextr_all + nodextr_offset, 
                                nsample * sizeof(integer_t));
                    std::memcpy(nin_host.data(), nin + nin_offset, 
                                nsample * sizeof(integer_t));
                    
                    // Always compute proximity in FP16 during training
                    // Quantization to INT8/NF4 happens only at the end
                    rf::cuda::gpu_proximity_upper_triangle_fp16(
                        nodestatus_host.data(),
                        nodextr_host.data(),
                        nin_host.data(),
                        nsample, nnode_all[tree_id],  // Use actual number of nodes, not maxnode!
                        tree_prox_upper_gpu,  // FP16 output buffer
                        nod_workspace.data(), ncount_workspace.data(), ncn_workspace.data(),
                        nodexb_workspace.data(), ndbegin_workspace.data(), npcase_workspace.data()
                    );
                    
                    // Increment tree counter for this tree (needed for finalize_accumulation)
                    total_trees_processed++;
                    
                    // Add to low-rank matrix using FP16 directly (no quantization during training!)
                    lowrank_prox_ptr->add_tree_contribution_incremental_upper_triangle_fp16(
                        tree_prox_upper_gpu, nsample
                    );
                    
                    // Free GPU memory
                    CUDA_CHECK_VOID(cudaFree(tree_prox_upper_gpu));
                }
                
                // RF-GAP proximity computation with QLoRA/low-rank (if enabled)
                if (g_config.use_rfgap && g_config.use_qlora && lowrank_prox_ptr_raw != nullptr && lowrank_initialized) {
                    rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                        static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    
                    // Process each tree for RF-GAP proximity computation
                    for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                        integer_t nin_offset = tree_id * nsample;
                        integer_t nodextr_offset = tree_id * nsample;
                        
                        // Safety check: ensure tree actually grew
                        if (nnode_all[tree_id] == 0) {
                            continue;  // Skip trees with 0 nodes
                        }
                        
                        // Allocate GPU memory for RF-GAP upper triangle
                        size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
                        __half* tree_rfgap_upper_gpu = nullptr;
                        
                        // Matches backup version - direct allocation, no error checking
                        CUDA_CHECK_VOID(cudaMalloc(&tree_rfgap_upper_gpu, upper_triangle_size * sizeof(__half)));
                        cudaMemset(tree_rfgap_upper_gpu, 0, upper_triangle_size * sizeof(__half));
                        
                        // Ensure nodextr_all is computed before using it
                        if (nodextr_all == nullptr) {
                            // std::cerr << "ERROR: nodextr_all is nullptr but needed for RF-GAP proximity computation!" << std::endl;
                            // Safe free to prevent segfaults
                            if (tree_rfgap_upper_gpu) { cudaError_t err = cudaFree(tree_rfgap_upper_gpu); if (err != cudaSuccess) cudaGetLastError(); }
                            continue;  // Skip this tree
                        }
                        
                        // Compute RF-GAP contributions for this tree in upper triangle format
                        // Note: RF-GAP uses nin (bootstrap multiplicities) and nodextr (terminal nodes)
                        // Both are already available from GPU tree growing
                        rf::cuda::gpu_proximity_rfgap_upper_triangle_fp16(
                            nin + nin_offset,           // Bootstrap multiplicities for this tree
                            nodextr_all + nodextr_offset,  // Terminal node assignments for this tree
                            nsample,
                            tree_rfgap_upper_gpu        // Output: Packed upper triangle in FP16
                        );
                        
                        // Add to low-rank matrix using GPU memory directly (no CPU copy!)
                        lowrank_prox_ptr->add_tree_contribution_incremental_upper_triangle_fp16(
                            tree_rfgap_upper_gpu, nsample  // Pass GPU pointer directly
                        );
                        
                        // Free GPU memory - matches backup version
                        CUDA_CHECK_VOID(cudaFree(tree_rfgap_upper_gpu));
                    }
                }
                
                // Synchronize device after all tree contributions are added
                // This ensures all cuBLAS/cuSolver operations from all trees complete
                // before we finalize or move to next batch. Required for correct accumulation across batches.
                CUDA_CHECK_VOID(cudaDeviceSynchronize());
                
                // total_trees_processed is now incremented per-tree in the loop above
                // Don't increment here to avoid double-counting
                
                // For low-rank mode, DO NOT reconstruct automatically - keep in low-rank form!
                // Reconstruction would require 80GB (100k × 100k × 8 bytes) which would crash memory.
                // Instead, keep LowRankProximityMatrix alive for on-demand reconstruction.
                // Only reconstruct if explicitly requested (proximity_all is allocated AND this is the final batch).
                // But for low-rank mode, we should NOT allocate proximity_all at all to avoid 80GB allocation.
                
                if (proximity_all != nullptr) {
                    // This should only happen if user explicitly requests full matrix (not recommended for large datasets)
                    // std::cout << "GPU Proximity: WARNING - Reconstructing full 80GB matrix (not recommended for large datasets)" << std::endl;
                    // std::cout << "GPU Proximity: All batches complete (" << total_trees_processed 
                    //           << " total trees). Reconstructing full proximity matrix..." << std::endl;
                    // Finalize accumulation and reconstruct full proximity matrix
                    // final_call=true: This path is for full reconstruction, so it's the end
                    lowrank_prox_ptr->finalize_accumulation(total_trees_processed, /*final_call=*/true);
                    // Sync after finalize_accumulation to ensure all operations complete
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());
                    lowrank_prox_ptr->get_accumulated_proximity(proximity_all, nsample);
                    // std::cout << "GPU Proximity: Low-rank accumulation completed and reconstructed" << std::endl;
                    
                    // Clean up persistent low-rank matrix after reconstruction
                    delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    lowrank_prox_ptr_raw = nullptr;
                    lowrank_initialized = false;
                    total_trees_processed = 0;
                } else {
                    // Keep LowRankProximityMatrix alive - don't delete it!
                    // Store pointer so RandomForest can access it later
                    if (lowrank_proximity_ptr_out != nullptr) {
                        *lowrank_proximity_ptr_out = lowrank_prox_ptr_raw;
                        // std::cout << "GPU Proximity: Stored LowRankProximityMatrix pointer for later access" << std::endl;
                    }
                    // Finalize accumulation so factors are available for retrieval
                    // This sets trees_processed_ so get_factors() can work correctly
                    // final_call=true: Quantize factors now for storage (only once, saves 8× memory)
                    lowrank_prox_ptr->finalize_accumulation(total_trees_processed, /*final_call=*/true);
                    // Sync after finalize_accumulation to ensure all operations complete before returning
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());
                    // std::cout << "GPU Proximity: Low-rank accumulation completed for this batch (" 
                    //           << num_trees << " trees). Total trees processed: " << total_trees_processed 
                    //           << " (keeping in low-rank form - ~40MB instead of 80GB)" << std::endl;
                    // std::cout << "GPU Proximity: Full matrix reconstruction not available for low-rank mode (would require 80GB)" << std::endl;
                    // std::cout << "GPU Proximity: Use get_lowrank_factors() or compute_distances_from_factors() instead" << std::endl;
                }
            }
            #endif  // __CUDACC__ - End of low-rank proximity code block
            // Matches backup version - no try-catch
            
        } else if (proximity_all != nullptr && nsample <= 10000) {
            // Traditional full matrix approach - ONLY for small datasets (nsample <= 10000)
            // For larger datasets, this would allocate huge temp buffers (800MB+ per tree)
            // std::cout << "GPU Proximity: Using traditional full matrix approach (nsample=" << nsample << " <= 10000)" << std::endl;
            // Use traditional full matrix approach (for smaller datasets or when qlora disabled)
        // Process each tree for proximity computation
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            // Get tree-specific data
            integer_t tree_offset = tree_id * maxnode;
            integer_t nin_offset = tree_id * nsample;
            
            // Create temporary proximity matrix for this tree
            std::vector<dp_t> tree_proximity(nsample * nsample, 0.0);
            
                // Call GPU proximity computation for this tree
                gpu_proximity(
                    nodestatus_all + tree_offset,  // nodestatus for this tree
                    nodextr_all + nin_offset,  // nodextr for this tree
                    nin + nin_offset,  // nin for this tree
                    nsample, maxnode,
                    tree_proximity.data(),  // Use temporary matrix for this tree
                    nod_workspace.data(), ncount_workspace.data(), ncn_workspace.data(),
                    nodexb_workspace.data(), ndbegin_workspace.data(), npcase_workspace.data()
                );
            
            // Accumulate this tree's proximity into the global matrix
                // This accumulates across all batches (proximity_all persists across batch calls)
                // Ensure GPU operations complete before CPU accumulation
                // std::cout << "DEBUG: About to sync before CPU proximity accumulation\n";
                
                // Synchronize before CPU accumulation
                CUDA_CHECK_VOID(cudaDeviceSynchronize());
                
            for (integer_t i = 0; i < nsample * nsample; ++i) {
                proximity_all[i] += tree_proximity[i];
                }
            }
        }
        
        // std::cout << "GPU: Standard proximity computation completed!" << std::endl;
    }
    
    // ========================================================================
    // SECTION 4: RF-GAP PROXIMITY COMPUTATION
    // ========================================================================
    // RF-GAP is a separate proximity algorithm (mutually exclusive with standard)
    // Only runs when: use_rfgap=True AND use_qlora=True
    if (should_compute_rfgap && lowrank_proximity_ptr_out != nullptr) {
        // RF-GAP low-rank proximity computation
        // Reserved for future implementation (v2.0)
    }
    
    // Matches backup version - no try-catch
    
    // Debug: Print first few q values
    // std::cout << "GPU: First 10 q values: ";
    // for (integer_t i = 0; i < 10; ++i) {
    //     std::cout << q_all[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Debug: Print first few nout values
    // std::cout << "GPU: First 10 nout values: ";
    // for (integer_t i = 0; i < 10; ++i) {
    //     std::cout << nout_all[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Ensure all GPU operations complete before cleanup
    // Use device sync like backup version
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // Clean up GPU memory - matches backup version approach
    if (x_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(x_gpu));
    if (cl_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(cl_gpu));
    if (win_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(win_gpu));
    if (seeds_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(seeds_gpu));
    if (nodestatus_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nodestatus_all_gpu));
    if (nodeclass_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nodeclass_all_gpu));
    if (nnode_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nnode_all_gpu));
    if (bestvar_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(bestvar_all_gpu));
    if (xbestsplit_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(xbestsplit_all_gpu));
    if (treemap_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(treemap_all_gpu));
    if (jinbag_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(jinbag_all_gpu));
    if (q_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(q_all_gpu));
    if (qimp_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(qimp_all_gpu));
    if (qimpm_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(qimpm_all_gpu));
    if (nin_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nin_all_gpu));
    if (win_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(win_all_gpu));
    if (random_states != nullptr) CUDA_CHECK_VOID(cudaFree(random_states));
}

} // namespace rf