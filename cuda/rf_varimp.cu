#include "rf_varimp.cuh"
#include "rf_config.hpp"
#include "rf_utils.hpp"
#include "rf_memory.cuh"
#include "rf_quantization_kernels.hpp"
#include "rf_cuda_config.hpp"
#include "rf_varimp.hpp"  // Include CPU function declarations
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>

namespace rf {

// ============================================================================
// CUDA Device Functions (internal)
// ============================================================================

namespace {  // Anonymous namespace for internal functions

// Device function for random permutation (permobmr equivalent)
__device__ void cuda_permobmr_device(const integer_t* joob, integer_t* pjoob,
                                     integer_t noob, curandState* random_states,
                                     integer_t thread_id) {
    // Copy joob to pjoob
    for (integer_t j = 0; j < noob; j++) {
        pjoob[j] = joob[j];
    }

    // Fisher-Yates shuffle (exact copy of original permobmr)
    integer_t j = noob - 1;  // 0-based
    while (j > 0) {
        // Use thread-specific random state to avoid race conditions
        real_t rnd = curand_uniform(&random_states[thread_id]);
        integer_t k = static_cast<integer_t>((j + 1) * rnd);
        if (k > j) k = j;

        // Swap j and k
        integer_t jt = pjoob[j];
        pjoob[j] = pjoob[k];
        pjoob[k] = jt;
        j--;
    }
}

// Device function for tree testing with importance (testreeimp equivalent)
__device__ void cuda_testreeimp_device(const real_t* x, integer_t nsample, integer_t mdim,
                                       const integer_t* joob, integer_t* pjoob, integer_t noob,
                                       integer_t mr, const integer_t* treemap,
                                       const integer_t* nodestatus, const real_t* xbestsplit,
                                       const integer_t* bestvar, const integer_t* nodeclass,
                                       integer_t nnode, const integer_t* cat, integer_t maxcat,
                                       const integer_t* catgoleft, integer_t* jvr,
                                       integer_t* nodexvr, curandState* random_states,
                                       integer_t thread_id) {
    // First permute the out-of-bag indices
    cuda_permobmr_device(joob, pjoob, noob, random_states, thread_id);

    // Exact copy of original testreeimp algorithm
    for (integer_t n = 0; n < noob; n++) {
        integer_t kt = 0;  // 0-based (root node)

        for (integer_t k = 0; k < nnode; k++) {
            if (nodestatus[kt] == -1) {
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                break;
            }

            integer_t m = bestvar[kt];
            real_t xmn;

            // Use permuted value for variable mr, original for others
            if (m == mr) {
                xmn = x[m + pjoob[n] * mdim];  // Column-major: x(m, pjoob(n))
            } else {
                xmn = x[m + joob[n] * mdim];   // Column-major: x(m, joob(n))
            }

            // See whether case n goes right or left
            if (cat[m] == 1) {
                // Quantitative variable
                if (xmn <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];  // Column-major: treemap(1, kt)
                } else {
                    kt = treemap[1 + kt * 2];  // Column-major: treemap(2, kt)
                }
            } else {
                // Categorical variable
                integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
                if (catgoleft[jcat + kt * maxcat] == 1) {  // Column-major
                    kt = treemap[0 + kt * 2];
                } else {
                    kt = treemap[1 + kt * 2];
                }
            }
        }
    }
}

} // anonymous namespace

} // namespace rf

// Device function to check if a sample reaches a specific node
// Must be outside namespace so kernel can access it
__device__ bool sample_reaches_node(rf::integer_t sample_idx, rf::integer_t target_node,
                                    const rf::real_t* x, rf::integer_t mdim,
                                    const rf::integer_t* treemap, const rf::integer_t* nodestatus,
                                    const rf::real_t* xbestsplit, const rf::integer_t* bestvar,
                                    rf::integer_t nnode) {
    rf::integer_t current_node = 0;  // Start at root
    
    while (current_node != target_node) {
        if (nodestatus[current_node] == -1) {
            // Reached a terminal node that's not the target
            return false;
        }
        
        if (nodestatus[current_node] == 1) {
            // Node is split, traverse to child
            rf::integer_t split_var = bestvar[current_node];
            rf::real_t split_value = xbestsplit[current_node];
            rf::real_t sample_value = x[split_var + sample_idx * mdim];  // Column-major (matches CPU)
            
            if (sample_value <= split_value) {
                current_node = treemap[current_node * 2];  // Left child
            } else {
                current_node = treemap[current_node * 2 + 1];  // Right child
            }
            
            if (current_node < 0 || current_node >= nnode) {
                return false;
            }
        } else {
            // Node not split, can't reach target
            return false;
        }
    }
    
    return true;  // Reached target node
}

// GPU kernel to compute tnodewt (node weights) in parallel
// Each thread block handles one node, threads within block parallelize over samples
// Must be outside namespace for CUDA kernel linkage
__global__ void gpu_compute_tnodewt_kernel(
    const rf::real_t* x, rf::integer_t nsample, rf::integer_t mdim, rf::integer_t nnode,
    const rf::integer_t* treemap, const rf::integer_t* nodestatus,
    const rf::real_t* xbestsplit, const rf::integer_t* bestvar,
    const rf::integer_t* cl, const rf::real_t* y_regression, const rf::real_t* win,
    const rf::integer_t* nin,  // Bootstrap frequency - needed for correct tnodewt computation
    const rf::integer_t* jinbag,  // In-bag sample indices (like CPU)
    rf::integer_t ninbag,  // Number of in-bag samples (like CPU)
    rf::integer_t nclass, rf::integer_t task_type,
    rf::real_t* tnodewt) {
    
    rf::integer_t node_id = blockIdx.x;
    if (node_id >= nnode) return;
    
    // Only compute for terminal nodes
    if (nodestatus[node_id] != -1) {
        if (threadIdx.x == 0) {
            tnodewt[node_id] = 0.0f;
        }
        return;
    }
    
    rf::integer_t tid = threadIdx.x;
    rf::integer_t stride = blockDim.x;
    
    // Shared memory for parallel reduction
    __shared__ rf::real_t s_sum_win[256];  // sum(win) for classification, sum(y) for regression
    __shared__ rf::real_t s_sum_nin[256];  // sum(nin) for in-bag samples (needed for classification)
    __shared__ rf::real_t s_count[256];    // count of samples (for regression mean)
    
    rf::real_t local_sum_win = 0.0f;
    rf::real_t local_sum_nin = 0.0f;
    rf::real_t local_count = 0.0f;
    
    if (task_type == 1) {
        // Regression: iterate through all samples (like CPU)
        for (rf::integer_t i = tid; i < nsample; i += stride) {
            // Check if sample reaches this node
            bool reaches = sample_reaches_node(i, node_id, x, mdim,
                                              treemap, nodestatus, xbestsplit, bestvar, nnode);
            
            if (reaches) {
                // Regression: sum target values and count (for mean)
                if (y_regression != nullptr) {
                    local_sum_win += y_regression[i];
                } else {
                    local_sum_win += static_cast<rf::real_t>(cl[i]);
                }
                local_count += 1.0f;
            }
        }
    } else {
        // Classification: iterate through jinbag (in-bag samples only) - matches CPU exactly
        for (rf::integer_t i = tid; i < ninbag; i += stride) {
            if (i < 0 || i >= ninbag) continue;
            if (jinbag == nullptr) continue;
            
            rf::integer_t sample_idx = jinbag[i];
            if (sample_idx < 0 || sample_idx >= nsample) continue;
            
            // Check if sample reaches this terminal node
            rf::integer_t current_node = 0;
            bool reaches_node = true;
            
            while (current_node != node_id && reaches_node) {
                if (nodestatus[current_node] == 1) {
                    rf::integer_t split_var = bestvar[current_node];
                    rf::real_t split_point = xbestsplit[current_node];
                    rf::real_t sample_value = x[split_var + sample_idx * mdim];  // Column-major (matches CPU and line 124)
                    
                    if (sample_value <= split_point) {
                        current_node = treemap[0 + current_node * 2];  // Column-major: treemap(1, current_node)
                    } else {
                        current_node = treemap[1 + current_node * 2];  // Column-major: treemap(2, current_node)
                    }
                    
                    if (current_node > node_id || current_node < 0 || current_node >= nnode) {
                        reaches_node = false;
                        break;
                    }
                } else {
                    reaches_node = false;
                    break;
                }
            }
            
            if (reaches_node && current_node == node_id) {
                // Classification: sum nin and count samples for in-bag samples in this node
                // This matches CPU exactly: node_weight_sum[node] += nin[n], node_sample_count[node]++
                // Then tnodewt[node] = node_weight_sum[node] / node_sample_count[node]
                local_sum_win += static_cast<rf::real_t>(nin[sample_idx]);  // Sum of bootstrap frequencies
                local_count += 1.0f;  // Count of samples
            }
        }
    }
    
    // Store local sums in shared memory
    if (tid < 256) {
        s_sum_win[tid] = local_sum_win;
        s_sum_nin[tid] = local_sum_nin;
        s_count[tid] = local_count;
    }
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (rf::integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < 256) {
            s_sum_win[tid] += s_sum_win[tid + s];
            s_sum_nin[tid] += s_sum_nin[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes final result
    if (tid == 0) {
        if (task_type == 1) {
            // Regression: mean of y values
            if (s_count[0] > 0.0f) {
                tnodewt[node_id] = s_sum_win[0] / s_count[0];
            } else {
                tnodewt[node_id] = 0.0f;
            }
        } else {
            // Classification: tnodewt = sum(nin) / count for in-bag samples in terminal node
            // This matches CPU exactly: tnodewt[node] = node_weight_sum[node] / node_sample_count[node]
            // where node_weight_sum = sum(nin), node_sample_count = count of samples
            if (s_count[0] > 0.0f) {
                tnodewt[node_id] = s_sum_win[0] / s_count[0];
            } else {
                tnodewt[node_id] = 0.0f;
            }
        }
    }
}

namespace rf {

// Kernel to initialize random states
__global__ void init_random_states_kernel(curandState* random_states, integer_t seed) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Initialize random state with unique seed per thread
    curand_init(seed + tid, tid, 0, &random_states[tid]);
}

// Kernel 1: Initialize OOB samples, original accuracy, and iv array
// This kernel runs with a single block to ensure proper initialization
__global__ void init_varimp_kernel(const integer_t* nin, const integer_t* jtr,
                                   const integer_t* cl, const integer_t* nodextr,
                                   const real_t* tnodewt, integer_t nsample,
                                   integer_t* joob, integer_t* noob_ptr, real_t* right_ptr,
                                   integer_t impn, real_t* qimp, integer_t nnode,
                                   const integer_t* nodestatus, const integer_t* bestvar,
                                   integer_t mdim, integer_t* iv, integer_t use_casewise) {
    integer_t tid = threadIdx.x;
    
    // Only thread 0 initializes shared values
    if (tid == 0) {
        integer_t noob = 0;
        real_t right = 0.0f;
        
        // Step 1: Find OOB samples and calculate original accuracy
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[n] == 0) {  // OOB sample
                // Update count of correct OOB classifications
                if (jtr[n] == cl[n]) {
                    // Case-wise: use bootstrap frequency weighted (tnodewt)
                    // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                    real_t weight = 1.0f;
                    if (use_casewise && tnodewt != nullptr && nodextr[n] >= 0 && nodextr[n] < nnode) {
                        weight = tnodewt[nodextr[n]];
                    }
                    right += weight;
                }
                joob[noob] = n;  // Store OOB sample index
                noob++;
            }
        }
        
        // Store results in global memory
        *noob_ptr = noob;
        *right_ptr = right;
    }
    __syncthreads();
    
    // Step 2: Update qimp for local importance (if impn == 1) - parallelize across threads
    if (impn == 1) {
        integer_t noob = *noob_ptr;
        for (integer_t n = tid; n < noob; n += blockDim.x) {
            // Bounds check
            if (n < 0 || n >= nsample) continue;
            
            integer_t nn = joob[n];
            if (nn < 0 || nn >= nsample) continue;
            
            if (jtr[nn] == cl[nn]) {
                integer_t node_idx = nodextr[nn];
                if (node_idx >= 0 && node_idx < nnode) {
                    // Case-wise: use bootstrap frequency weighted (tnodewt)
                    // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                    real_t weight = 1.0f;
                    if (use_casewise && tnodewt != nullptr) {
                        weight = tnodewt[node_idx];
                    }
                    if (nn >= 0 && nn < nsample) {
                        ::atomicAdd(&qimp[nn], weight / static_cast<real_t>(noob));
                    }
                }
            }
        }
    }
    
    // Step 3: Mark which variables were used in splits - parallelize across threads
    // Use atomic operations to ensure thread-safe writes
    // Only mark variables used in actual split nodes (nodestatus == 1), not unsplit nodes (status 2)
    for (integer_t jj = tid; jj < nnode; jj += blockDim.x) {
        if (nodestatus[jj] == 1) {  // Split node (not terminal, not unsplit)
            integer_t var_idx = bestvar[jj];
            if (var_idx >= 0 && var_idx < mdim) {
                // Use atomic operation to ensure thread-safe write
                // Multiple threads may write to same iv[var_idx]
                atomicExch(&iv[var_idx], 1);
            }
        }
    }
}

// Kernel 2: Process variable importance for each variable in parallel
// Each block processes one variable
__global__ void cuda_varimp_kernel(const real_t* x, integer_t nsample, integer_t mdim,
                                   const integer_t* cl, const integer_t* nin,
                                   const integer_t* jtr, integer_t impn,
                                   real_t* qimp, real_t* qimpm, real_t* avimp, real_t* sqsd,
                                   const integer_t* treemap, const integer_t* nodestatus,
                                   const real_t* xbestsplit, const integer_t* bestvar,
                                   const integer_t* nodeclass, integer_t nnode,
                                   const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                                   integer_t maxcat, const integer_t* catgoleft,
                                   const real_t* tnodewt, const integer_t* nodextr,
                                   const integer_t* joob, integer_t* pjoob, const integer_t* iv,
                                   const integer_t* noob_ptr, const real_t* right_ptr,
                                   curandState* random_states, integer_t use_casewise) {
    integer_t tid = threadIdx.x;
    integer_t bid = blockIdx.x;  // Block ID = variable index
    integer_t k = bid;  // This block processes variable k
    
    // Calculate random state index for this thread (block-specific)
    integer_t random_state_idx = bid * blockDim.x + tid;
    
    // All blocks: Process assigned variable k
    if (k >= mdim) return;  // Out of bounds
    
    // Read values initialized by init_varimp_kernel
    integer_t noob = *noob_ptr;
    real_t right = *right_ptr;
    
    if (iv[k] == 1) {  // Variable was used in splits
        // Test tree with permuted variable k - parallelize OOB sample processing
        // Each block needs its own permutation - use shared memory
        __shared__ integer_t s_pjoob[256];  // Shared memory for this block's permutation
        
        // Generate permutation for this block's OOB samples
        // Thread 0 generates the full permutation for this block using Fisher-Yates
        if (tid == 0) {
            curandState local_state = random_states[random_state_idx];
            integer_t n_perm = (noob < 256) ? noob : 256;
            // Initialize with first n_perm OOB samples
            for (integer_t i = 0; i < n_perm; ++i) {
                s_pjoob[i] = joob[i];
            }
            // Fisher-Yates shuffle
            for (integer_t i = n_perm - 1; i > 0; --i) {
                real_t rand_val = curand_uniform(&local_state);
                integer_t j = static_cast<integer_t>(rand_val * (i + 1));
                if (j > i) j = i;
                integer_t temp = s_pjoob[i];
                s_pjoob[i] = s_pjoob[j];
                s_pjoob[j] = temp;
            }
        }
        __syncthreads();
        
        // Now test tree with permuted values - parallelize across OOB samples
        for (integer_t n = tid; n < noob; n += blockDim.x) {
            // Bounds check for array indices
            if (n < 0 || n >= nsample) continue;
            
            integer_t kt = 0;  // Start at root
            integer_t nn = joob[n];  // Original sample index
            
            // Bounds check for sample index
            if (nn < 0 || nn >= nsample) continue;
            
            // Traverse tree with permuted value for variable k
            for (integer_t depth = 0; depth < nnode && kt >= 0 && kt < nnode; ++depth) {
                if (nodestatus[kt] == -1) {
                    if (n < nsample && kt >= 0 && kt < nnode) {
                        jvr[n] = nodeclass[kt];
                        nodexvr[n] = kt;
                    }
                    break;
                }
                
                integer_t m = bestvar[kt];
                if (m < 0 || m >= mdim) break;
                real_t xmn;
                
                // Use permuted value for variable k, original for others
                // Column-major indexing: column m, row sample_idx (matches CPU)
                if (m == k) {
                    // Use permuted sample index from shared memory
                    // If n >= 256, wrap around (use modulo)
                    integer_t perm_idx = (n < 256) ? n : (n % 256);
                    if (perm_idx >= 256) perm_idx = 255;  // Safety check
                    integer_t perm_sample_idx = s_pjoob[perm_idx];
                    // Bounds check
                    if (perm_sample_idx >= 0 && perm_sample_idx < nsample) {
                        xmn = x[m + perm_sample_idx * mdim];  // Permuted value (column-major)
                    } else {
                        xmn = x[m + nn * mdim];  // Fallback to original (column-major)
                    }
                } else {
                    xmn = x[m + nn * mdim];  // Original value (column-major)
                }
                
                // Traverse to child
                if (cat[m] == 1) {
                    // Quantitative
                    if (xmn <= xbestsplit[kt]) {
                        kt = treemap[0 + kt * 2];
                    } else {
                        kt = treemap[1 + kt * 2];
                    }
                } else {
                    // Categorical
                    integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
                    if (jcat >= 0 && jcat < maxcat && catgoleft != nullptr) {
                        if (catgoleft[jcat + kt * maxcat] == 1) {
                            kt = treemap[0 + kt * 2];
                        } else {
                            kt = treemap[1 + kt * 2];
                        }
                    } else {
                        // Invalid category, go right
                        kt = treemap[1 + kt * 2];
                    }
                }
                
                if (kt < 0 || kt >= nnode) break;
            }
        }
        __syncthreads();
        
        // Calculate permuted accuracy - parallelize reduction
        __shared__ real_t s_rightimp[256];
        real_t local_rightimp = 0.0f;
        
        for (integer_t n = tid; n < noob; n += blockDim.x) {
            // Bounds check
            if (n < 0 || n >= nsample) continue;
            
            integer_t nn = joob[n];
            if (nn < 0 || nn >= nsample) continue;
            
            if (jvr[n] == cl[nn]) {
                integer_t node_idx = nodexvr[n];
                if (node_idx >= 0 && node_idx < nnode) {
                    // Case-wise: use bootstrap frequency weighted (tnodewt)
                    // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                    real_t weight = 1.0f;
                    if (use_casewise && tnodewt != nullptr) {
                        weight = tnodewt[node_idx];
                    }
                    local_rightimp += weight;
                    
                    if (impn == 1) {
                        integer_t qimpm_idx = nn * mdim + k;
                        if (qimpm_idx >= 0 && qimpm_idx < nsample * mdim) {
                            ::atomicAdd(&qimpm[qimpm_idx], weight / static_cast<real_t>(noob));
                        }
                    }
                }
            }
        }
        
        // Store local sum
        if (tid < 256) {
            s_rightimp[tid] = local_rightimp;
        }
        __syncthreads();
        
        // Reduce in shared memory
        for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < 256) {
                s_rightimp[tid] += s_rightimp[tid + s];
            }
            __syncthreads();
        }
        
        // Thread 0 calculates final importance
        if (tid == 0 && noob > 0) {
            real_t rightimp = s_rightimp[0];
            real_t rr = (right - rightimp) / static_cast<real_t>(noob);
            avimp[k] += rr;
            sqsd[k] += rr * rr;
        }
    } else {
        // Variable not used in splits - use original predictions
        if (impn == 1) {
            for (integer_t n = tid; n < noob; n += blockDim.x) {
                // Bounds check
                if (n < 0 || n >= nsample) continue;
                
                integer_t nn = joob[n];
                if (nn < 0 || nn >= nsample) continue;
                
                if (jtr[nn] == cl[nn]) {
                    integer_t node_idx = nodextr[nn];
                    if (node_idx >= 0 && node_idx < nnode) {
                        // Case-wise: use bootstrap frequency weighted (tnodewt)
                        // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                        real_t weight = 1.0f;
                        if (use_casewise && tnodewt != nullptr) {
                            weight = tnodewt[node_idx];
                        }
                        integer_t qimpm_idx = nn * mdim + k;
                        if (qimpm_idx >= 0 && qimpm_idx < nsample * mdim) {
                            ::atomicAdd(&qimpm[qimpm_idx], weight / static_cast<real_t>(noob));
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GPU VARIABLE IMPORTANCE IMPLEMENTATION
// ============================================================================

void gpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
            const integer_t* cl, const integer_t* nin, const integer_t* jtr,
            integer_t impn, real_t* qimp, real_t* qimpm,
            real_t* avimp, real_t* sqsd,
            const integer_t* treemap, const integer_t* nodestatus,
            const real_t* xbestsplit, const integer_t* bestvar,
            const integer_t* nodeclass, integer_t nnode,
            const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
            integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr,
               const real_t* y_regression, const real_t* win,
               const integer_t* jinbag, integer_t ninbag,  // In-bag samples (like CPU)
               integer_t nclass, integer_t task_type) {
    // Get use_casewise flag from global config
    integer_t use_casewise = g_config.use_casewise ? 1 : 0;
    
    auto time_total_start = std::chrono::high_resolution_clock::now();
    
    // Use GPU for variable importance computation following varimp_cuda.cuf algorithm
    // std::cout << "GPU: Computing variable importance with parallel permutation testing..." << std::endl;
    
    // Use the existing CUDA kernel for full variable importance computation
    // OPTIMIZATION: Parallelize across variables - each block handles one variable
    dim3 block_size(256);
    dim3 grid_size(mdim);  // One block per variable for parallel processing
    
    auto time_alloc_start = std::chrono::high_resolution_clock::now();
    // Allocate device memory for all arrays
    real_t* x_d;
    integer_t* cl_d;
    integer_t* nin_d;
    integer_t* jtr_d;
    real_t* qimp_d;
    real_t* qimpm_d;
    real_t* avimp_d;
    real_t* sqsd_d;
    integer_t* treemap_d;
    integer_t* nodestatus_d;
    real_t* xbestsplit_d;
    integer_t* bestvar_d;
    integer_t* nodeclass_d;
    integer_t* cat_d;
    integer_t* jvr_d;
    integer_t* nodexvr_d;
    integer_t* catgoleft_d;
    real_t* tnodewt_d;
    real_t* y_regression_d;
    real_t* win_d;
    integer_t* jinbag_d;  // In-bag sample indices (needed for tnodewt computation)
    integer_t* nodextr_d;
    integer_t* joob_d;
    integer_t* pjoob_d;
    integer_t* iv_d;
    integer_t* noob_ptr_d;
    real_t* right_ptr_d;
    curandState* random_states_d;
    
    // Determine if we need to compute tnodewt
    // For classification: needed when use_casewise=true
    // For regression: tnodewt is typically pre-computed, but we can compute it if needed
    bool need_compute_tnodewt = (tnodewt == nullptr) && 
                                ((task_type == 0 && use_casewise) || (task_type == 1));
    
    // Allocate memory
    CUDA_CHECK_VOID(cudaMalloc(&x_d, nsample * mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&cl_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nin_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&jtr_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&qimp_d, nsample * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&qimpm_d, nsample * mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&avimp_d, mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&sqsd_d, mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&treemap_d, 2 * nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodestatus_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&xbestsplit_d, nnode * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&bestvar_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodeclass_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&cat_d, mdim * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&jvr_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodexvr_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&catgoleft_d, maxcat * nnode * sizeof(integer_t)));
    
    // Allocate tnodewt_d - either use provided tnodewt or compute it
    if (tnodewt != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&tnodewt_d, nnode * sizeof(real_t)));
    } else if (need_compute_tnodewt) {
        CUDA_CHECK_VOID(cudaMalloc(&tnodewt_d, nnode * sizeof(real_t)));
        CUDA_CHECK_VOID(cudaMemset(tnodewt_d, 0, nnode * sizeof(real_t)));
    } else {
        tnodewt_d = nullptr;
    }
    
    // Allocate y_regression_d, win_d, and jinbag_d if needed for tnodewt computation
    if (need_compute_tnodewt) {
        if (y_regression != nullptr) {
            CUDA_CHECK_VOID(cudaMalloc(&y_regression_d, nsample * sizeof(real_t)));
        } else {
            y_regression_d = nullptr;
        }
        if (win != nullptr) {
            CUDA_CHECK_VOID(cudaMalloc(&win_d, nsample * sizeof(real_t)));
        } else {
            win_d = nullptr;
        }
        // Allocate jinbag_d for classification (needed to match CPU implementation)
        if (task_type == 0 && jinbag != nullptr && ninbag > 0) {
            CUDA_CHECK_VOID(cudaMalloc(&jinbag_d, ninbag * sizeof(integer_t)));
        } else {
            jinbag_d = nullptr;
        }
    } else {
        y_regression_d = nullptr;
        win_d = nullptr;
        jinbag_d = nullptr;
    }
    
    CUDA_CHECK_VOID(cudaMalloc(&nodextr_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&joob_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&pjoob_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&iv_d, mdim * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&noob_ptr_d, sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&right_ptr_d, sizeof(real_t)));
    // Allocate random states for all blocks (mdim blocks × block_size threads)
    CUDA_CHECK_VOID(cudaMalloc(&random_states_d, grid_size.x * block_size.x * sizeof(curandState)));
    auto time_alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_alloc_end - time_alloc_start);
    
    auto time_copy_start = std::chrono::high_resolution_clock::now();
    // Copy data to device
    CUDA_CHECK_VOID(cudaMemcpy(x_d, x, nsample * mdim * sizeof(real_t), cudaMemcpyHostToDevice));
    
    // Check if cl is valid before copying (important for unsupervised learning)
    if (cl != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpy(cl_d, cl, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    } else {
        // For unsupervised learning, initialize with zeros
        CUDA_CHECK_VOID(cudaMemset(cl_d, 0, nsample * sizeof(integer_t)));
    }
    
    CUDA_CHECK_VOID(cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(jtr_d, jtr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(treemap_d, treemap, 2 * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodestatus_d, nodestatus, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(xbestsplit_d, xbestsplit, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(bestvar_d, bestvar, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodeclass_d, nodeclass, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(cat_d, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(jvr_d, jvr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodexvr_d, nodexvr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    if (catgoleft != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpy(catgoleft_d, catgoleft, maxcat * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    }
    if (tnodewt != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpy(tnodewt_d, tnodewt, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    }
    
    // Copy y_regression, win, and jinbag if needed for tnodewt computation
    if (need_compute_tnodewt) {
        if (y_regression_d != nullptr && y_regression != nullptr) {
            CUDA_CHECK_VOID(cudaMemcpy(y_regression_d, y_regression, nsample * sizeof(real_t), cudaMemcpyHostToDevice));
        }
        if (win_d != nullptr && win != nullptr) {
            CUDA_CHECK_VOID(cudaMemcpy(win_d, win, nsample * sizeof(real_t), cudaMemcpyHostToDevice));
        }
        // Copy jinbag for classification (needed to match CPU implementation)
        if (jinbag_d != nullptr && jinbag != nullptr && ninbag > 0) {
            CUDA_CHECK_VOID(cudaMemcpy(jinbag_d, jinbag, ninbag * sizeof(integer_t), cudaMemcpyHostToDevice));
        }
    }
    
    CUDA_CHECK_VOID(cudaMemcpy(nodextr_d, nodextr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Initialize output arrays (these are global arrays that accumulate across trees)
    // Copy current values from host to device for accumulation
    CUDA_CHECK_VOID(cudaMemcpy(qimp_d, qimp, nsample * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(qimpm_d, qimpm, nsample * mdim * sizeof(real_t), cudaMemcpyHostToDevice));
    // Initialize avimp_d and sqsd_d to zero on device before accumulation
    // Do NOT copy from host - we want to accumulate fresh for each tree
    CUDA_CHECK_VOID(cudaMemset(avimp_d, 0, mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMemset(sqsd_d, 0, mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMemset(iv_d, 0, mdim * sizeof(integer_t)));
    auto time_copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_copy_end - time_copy_start);
    
    auto time_kernel_start = std::chrono::high_resolution_clock::now();
    
    // STEP 0: Compute tnodewt if needed (before initialization kernel)
    if (need_compute_tnodewt && tnodewt_d != nullptr) {
        dim3 tnodewt_block_size(256);
        dim3 tnodewt_grid_size((nnode + tnodewt_block_size.x - 1) / tnodewt_block_size.x);
        
        // DEBUG: Check parameters for tnodewt computation
        // if (task_type == 0 && use_casewise) {
        //     std::cout << "[DEBUG TNODEWT] Computing tnodewt: task_type=" << task_type 
        //               << ", use_casewise=" << use_casewise
        //               << ", ninbag=" << ninbag
        //               << ", jinbag_d=" << (jinbag_d != nullptr ? "valid" : "nullptr")
        //               << ", win_d=" << (win_d != nullptr ? "valid" : "nullptr")
        //               << ", nnode=" << nnode << std::endl;
        // }
        
        gpu_compute_tnodewt_kernel<<<tnodewt_grid_size, tnodewt_block_size>>>(
            x_d, nsample, mdim, nnode,
            treemap_d, nodestatus_d, xbestsplit_d, bestvar_d,
            cl_d, y_regression_d, win_d,
            nin_d,  // Pass nin_d for correct tnodewt computation (sum(win)/sum(nin) for classification)
            jinbag_d, ninbag,  // Pass jinbag and ninbag to match CPU implementation
            nclass, task_type,
            tnodewt_d);
        
        CUDA_CHECK_VOID(cudaDeviceSynchronize());
        
        // DEBUG: Check computed tnodewt values
        // if (task_type == 0 && use_casewise && nnode > 0) {
        //     std::vector<real_t> tnodewt_host(nnode);
        //     CUDA_CHECK_VOID(cudaMemcpy(tnodewt_host.data(), tnodewt_d, nnode * sizeof(real_t), cudaMemcpyDeviceToHost));
        //     integer_t non_zero_tnodewt = 0;
        //     for (integer_t i = 0; i < nnode; ++i) {
        //         if (tnodewt_host[i] != 0.0f) non_zero_tnodewt++;
        //     }
        //     std::cout << "[DEBUG TNODEWT] After computation: non-zero=" << non_zero_tnodewt 
        //               << "/" << nnode << ", first 5 values: ";
        //     for (integer_t i = 0; i < 5 && i < nnode; ++i) {
        //         std::cout << tnodewt_host[i] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
    
    // STEP 1: Initialize random states for all blocks
    // Each block needs its own set of random states
    dim3 init_grid_size((grid_size.x * block_size.x + 255) / 256);  // Enough blocks to cover all threads
    init_random_states_kernel<<<init_grid_size, block_size>>>(random_states_d, 42);
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // STEP 2: Initialize OOB samples, original accuracy, and iv array
    // Run initialization kernel with single block to ensure proper setup
    // This kernel initializes: joob, noob_ptr, right_ptr, qimp (if impn==1), and iv
    dim3 init_block_size(256);
    dim3 init_kernel_grid_size(1);  // Single block for initialization
    init_varimp_kernel<<<init_kernel_grid_size, init_block_size>>>(
        nin_d, jtr_d, cl_d, nodextr_d, tnodewt_d, nsample,
        joob_d, noob_ptr_d, right_ptr_d,
        impn, qimp_d, nnode,
        nodestatus_d, bestvar_d, mdim, iv_d, use_casewise
    );
    
    // Synchronize after initialization to ensure all writes are visible
    // This ensures iv array and other initialized values are visible to all blocks
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // STEP 3: Launch variable importance processing kernel
    // Each block processes one variable in parallel
    cuda_varimp_kernel<<<grid_size, block_size>>>(
        x_d, nsample, mdim, cl_d, nin_d, jtr_d, impn, qimp_d, qimpm_d,
        avimp_d, sqsd_d, treemap_d, nodestatus_d, xbestsplit_d, bestvar_d,
        nodeclass_d, nnode, cat_d, jvr_d, nodexvr_d, maxcat, catgoleft_d,
        tnodewt_d, nodextr_d, joob_d, pjoob_d, iv_d, noob_ptr_d, right_ptr_d, random_states_d, use_casewise
    );
    
    // Synchronize to ensure all blocks complete before copying results
    CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Full device sync for cross-block visibility
    auto time_kernel_end = std::chrono::high_resolution_clock::now();
    auto kernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_kernel_end - time_kernel_start);
    
    auto time_copyback_start = std::chrono::high_resolution_clock::now();
    // Copy results back to host
    CUDA_CHECK_VOID(cudaMemcpy(qimp, qimp_d, nsample * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(qimpm, qimpm_d, nsample * mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(avimp, avimp_d, mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(sqsd, sqsd_d, mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
    auto time_copyback_end = std::chrono::high_resolution_clock::now();
    auto copyback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_copyback_end - time_copyback_start);
    
    auto time_free_start = std::chrono::high_resolution_clock::now();
    // Cleanup device memory (safe free to prevent segfaults)
    if (x_d) { cudaError_t err = cudaFree(x_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (cl_d) { cudaError_t err = cudaFree(cl_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (jtr_d) { cudaError_t err = cudaFree(jtr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (qimp_d) { cudaError_t err = cudaFree(qimp_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (qimpm_d) { cudaError_t err = cudaFree(qimpm_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (avimp_d) { cudaError_t err = cudaFree(avimp_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (sqsd_d) { cudaError_t err = cudaFree(sqsd_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (treemap_d) { cudaError_t err = cudaFree(treemap_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodestatus_d) { cudaError_t err = cudaFree(nodestatus_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (xbestsplit_d) { cudaError_t err = cudaFree(xbestsplit_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (bestvar_d) { cudaError_t err = cudaFree(bestvar_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodeclass_d) { cudaError_t err = cudaFree(nodeclass_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (cat_d) { cudaError_t err = cudaFree(cat_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (jvr_d) { cudaError_t err = cudaFree(jvr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodexvr_d) { cudaError_t err = cudaFree(nodexvr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (catgoleft_d) { cudaError_t err = cudaFree(catgoleft_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (tnodewt_d != nullptr) {
        cudaError_t err = cudaFree(tnodewt_d); if (err != cudaSuccess) cudaGetLastError();
    }
    if (y_regression_d != nullptr) { cudaError_t err = cudaFree(y_regression_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (win_d != nullptr) { cudaError_t err = cudaFree(win_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (jinbag_d != nullptr) { cudaError_t err = cudaFree(jinbag_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodextr_d) { cudaError_t err = cudaFree(nodextr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (joob_d) { cudaError_t err = cudaFree(joob_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (pjoob_d) { cudaError_t err = cudaFree(pjoob_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (iv_d) { cudaError_t err = cudaFree(iv_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (noob_ptr_d) { cudaError_t err = cudaFree(noob_ptr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (right_ptr_d) { cudaError_t err = cudaFree(right_ptr_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (random_states_d) { cudaError_t err = cudaFree(random_states_d); if (err != cudaSuccess) cudaGetLastError(); }
    auto time_free_end = std::chrono::high_resolution_clock::now();
    auto free_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_free_end - time_free_start);
    
    auto time_total_end = std::chrono::high_resolution_clock::now();
    
    // std::cout << "GPU: Variable importance computation completed." << std::endl;
}

// ============================================================================
// GPU VARIMP CONTEXT - Reusable GPU memory for batched importance computation  
// ============================================================================

struct GPUVarImpContext {
    real_t* x_d; integer_t* cl_d; integer_t* nin_d; integer_t* jtr_d;
    real_t* qimp_d; real_t* qimpm_d; real_t* avimp_d; real_t* sqsd_d;
    integer_t* treemap_d; integer_t* nodestatus_d; real_t* xbestsplit_d; integer_t* bestvar_d;
    integer_t* nodeclass_d; integer_t* cat_d; integer_t* jvr_d; integer_t* nodexvr_d;
    integer_t* catgoleft_d; real_t* tnodewt_d; integer_t* nodextr_d;
    integer_t* joob_d; integer_t* pjoob_d; integer_t* iv_d;
    integer_t* noob_ptr_d; real_t* right_ptr_d; curandState* random_states_d;
    real_t* y_regression_d; real_t* win_d; integer_t* jinbag_d;
    integer_t nsample, mdim, maxnode, maxcat, max_ninbag;
    bool allocated, static_data_copied, random_states_initialized, importance_arrays_initialized;
    GPUVarImpContext() : allocated(false), static_data_copied(false), random_states_initialized(false), importance_arrays_initialized(false) {}
};

GPUVarImpContext* gpu_varimp_alloc_context(integer_t nsample, integer_t mdim, integer_t maxnode, integer_t maxcat,
                                           integer_t max_ninbag, integer_t nclass, integer_t grid_size_x, integer_t block_size_x) {
    
    // Check available GPU memory BEFORE allocating to avoid blocking
    size_t free_mem, total_mem;
    cudaError_t mem_check = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_check != cudaSuccess) {
        std::cerr << "Warning: Failed to check GPU memory, skipping GPU local importance" << std::endl;
        return nullptr;
    }
    
    // Estimate required memory (rough calculation)
    size_t required_mem = (
        nsample * mdim * sizeof(real_t) +           // x_d
        nsample * mdim * sizeof(real_t) +           // qimpm_d
        2 * maxnode * sizeof(integer_t) +           // treemap_d
        maxcat * maxnode * sizeof(integer_t) +      // catgoleft_d
        grid_size_x * block_size_x * sizeof(curandState) + // random_states_d
        nsample * 10 * sizeof(integer_t) +          // various sample arrays
        maxnode * 5 * sizeof(real_t)                // various node arrays
    );
    
    // Need at least 100MB safety margin
    if (free_mem < required_mem + 100 * 1024 * 1024) {
        std::cerr << "Warning: Insufficient GPU memory for local importance context" << std::endl;
        std::cerr << "  Required: " << (required_mem / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "  Available: " << (free_mem / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "  Falling back to old per-tree allocation (slower)" << std::endl;
        return nullptr;  // Gracefully fail - will use old gpu_varimp() path
    }
    
    GPUVarImpContext* ctx = new GPUVarImpContext();
    ctx->nsample = nsample; ctx->mdim = mdim; ctx->maxnode = maxnode; ctx->maxcat = maxcat; ctx->max_ninbag = max_ninbag;
    
    // Initialize all pointers to nullptr FIRST (safe for cudaFree)
    ctx->x_d = nullptr; ctx->cl_d = nullptr; ctx->nin_d = nullptr; ctx->jtr_d = nullptr;
    ctx->qimp_d = nullptr; ctx->qimpm_d = nullptr; ctx->avimp_d = nullptr; ctx->sqsd_d = nullptr;
    ctx->treemap_d = nullptr; ctx->nodestatus_d = nullptr; ctx->xbestsplit_d = nullptr; ctx->bestvar_d = nullptr;
    ctx->nodeclass_d = nullptr; ctx->cat_d = nullptr; ctx->jvr_d = nullptr; ctx->nodexvr_d = nullptr;
    ctx->catgoleft_d = nullptr; ctx->tnodewt_d = nullptr; ctx->nodextr_d = nullptr;
    ctx->joob_d = nullptr; ctx->pjoob_d = nullptr; ctx->iv_d = nullptr;
    ctx->noob_ptr_d = nullptr; ctx->right_ptr_d = nullptr; ctx->random_states_d = nullptr;
    ctx->y_regression_d = nullptr; ctx->win_d = nullptr; ctx->jinbag_d = nullptr;
    
    // Try to allocate - if ANY allocation fails, clean up and return nullptr
    cudaError_t err;
    
    #define TRY_ALLOC(ptr, size, name) \
        err = cudaMalloc((void**)&ptr, size); \
        if (err != cudaSuccess) { \
            std::cerr << "Warning: Failed to allocate " << name << " (" << (size/1024/1024) << " MB)" << std::endl; \
            gpu_varimp_free_context(ctx); \
            return nullptr; \
        }
    
    TRY_ALLOC(ctx->x_d, nsample * mdim * sizeof(real_t), "x_d");
    TRY_ALLOC(ctx->cl_d, nsample * sizeof(integer_t), "cl_d");
    TRY_ALLOC(ctx->nin_d, nsample * sizeof(integer_t), "nin_d");
    TRY_ALLOC(ctx->jtr_d, nsample * sizeof(integer_t), "jtr_d");
    TRY_ALLOC(ctx->qimp_d, nsample * sizeof(real_t), "qimp_d");
    TRY_ALLOC(ctx->qimpm_d, nsample * mdim * sizeof(real_t), "qimpm_d");
    TRY_ALLOC(ctx->avimp_d, mdim * sizeof(real_t), "avimp_d");
    TRY_ALLOC(ctx->sqsd_d, mdim * sizeof(real_t), "sqsd_d");
    TRY_ALLOC(ctx->treemap_d, 2 * maxnode * sizeof(integer_t), "treemap_d");
    TRY_ALLOC(ctx->nodestatus_d, maxnode * sizeof(integer_t), "nodestatus_d");
    TRY_ALLOC(ctx->xbestsplit_d, maxnode * sizeof(real_t), "xbestsplit_d");
    TRY_ALLOC(ctx->bestvar_d, maxnode * sizeof(integer_t), "bestvar_d");
    TRY_ALLOC(ctx->nodeclass_d, maxnode * sizeof(integer_t), "nodeclass_d");
    TRY_ALLOC(ctx->cat_d, mdim * sizeof(integer_t), "cat_d");
    TRY_ALLOC(ctx->jvr_d, nsample * sizeof(integer_t), "jvr_d");
    TRY_ALLOC(ctx->nodexvr_d, nsample * sizeof(integer_t), "nodexvr_d");
    TRY_ALLOC(ctx->catgoleft_d, maxcat * maxnode * sizeof(integer_t), "catgoleft_d");
    TRY_ALLOC(ctx->tnodewt_d, maxnode * sizeof(integer_t), "tnodewt_d");
    TRY_ALLOC(ctx->nodextr_d, nsample * sizeof(integer_t), "nodextr_d");
    TRY_ALLOC(ctx->joob_d, nsample * sizeof(integer_t), "joob_d");
    TRY_ALLOC(ctx->pjoob_d, nsample * sizeof(integer_t), "pjoob_d");
    TRY_ALLOC(ctx->iv_d, mdim * sizeof(integer_t), "iv_d");
    TRY_ALLOC(ctx->noob_ptr_d, sizeof(integer_t), "noob_ptr_d");
    TRY_ALLOC(ctx->right_ptr_d, sizeof(real_t), "right_ptr_d");
    TRY_ALLOC(ctx->random_states_d, grid_size_x * block_size_x * sizeof(curandState), "random_states_d");
    TRY_ALLOC(ctx->y_regression_d, nsample * sizeof(real_t), "y_regression_d");
    TRY_ALLOC(ctx->win_d, nsample * sizeof(real_t), "win_d");
    TRY_ALLOC(ctx->jinbag_d, max_ninbag * sizeof(integer_t), "jinbag_d");
    
    #undef TRY_ALLOC
    
    ctx->allocated = true;
    return ctx;
}

void gpu_varimp_free_context(GPUVarImpContext* ctx) {
    if (!ctx || !ctx->allocated) return;
    cudaFree(ctx->x_d); cudaFree(ctx->cl_d); cudaFree(ctx->nin_d); cudaFree(ctx->jtr_d);
    cudaFree(ctx->qimp_d); cudaFree(ctx->qimpm_d); cudaFree(ctx->avimp_d); cudaFree(ctx->sqsd_d);
    cudaFree(ctx->treemap_d); cudaFree(ctx->nodestatus_d); cudaFree(ctx->xbestsplit_d); cudaFree(ctx->bestvar_d);
    cudaFree(ctx->nodeclass_d); cudaFree(ctx->cat_d); cudaFree(ctx->jvr_d); cudaFree(ctx->nodexvr_d);
    cudaFree(ctx->catgoleft_d); cudaFree(ctx->tnodewt_d); cudaFree(ctx->nodextr_d);
    cudaFree(ctx->joob_d); cudaFree(ctx->pjoob_d); cudaFree(ctx->iv_d);
    cudaFree(ctx->noob_ptr_d); cudaFree(ctx->right_ptr_d); cudaFree(ctx->random_states_d);
    cudaFree(ctx->y_regression_d); cudaFree(ctx->win_d); cudaFree(ctx->jinbag_d);
    delete ctx;
}

void gpu_varimp_with_context(GPUVarImpContext* ctx, const real_t* x, integer_t nsample, integer_t mdim,
                             const integer_t* cl, const integer_t* nin, const integer_t* jtr, integer_t impn,
                             real_t* qimp, real_t* qimpm, real_t* avimp, real_t* sqsd,
                             const integer_t* treemap, const integer_t* nodestatus, const real_t* xbestsplit,
                             const integer_t* bestvar, const integer_t* nodeclass, integer_t nnode,
                             const integer_t* cat, integer_t* jvr, integer_t* nodexvr, integer_t maxcat,
                             const integer_t* catgoleft, const real_t* tnodewt, const integer_t* nodextr,
                             const real_t* y_regression, const real_t* win, const integer_t* jinbag,
                             integer_t ninbag, integer_t nclass, integer_t task_type) {
    if (!ctx || !ctx->allocated) return;
    
    if (!ctx->static_data_copied) {
        CUDA_CHECK_VOID(cudaMemcpy(ctx->x_d, x, nsample * mdim * sizeof(real_t), cudaMemcpyHostToDevice));
        if (cl) CUDA_CHECK_VOID(cudaMemcpy(ctx->cl_d, cl, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(ctx->cat_d, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice));
        ctx->static_data_copied = true;
    }
    
    CUDA_CHECK_VOID(cudaMemcpy(ctx->nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->jtr_d, jtr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->treemap_d, treemap, 2 * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->nodestatus_d, nodestatus, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->xbestsplit_d, xbestsplit, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->bestvar_d, bestvar, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->nodeclass_d, nodeclass, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ctx->nodextr_d, nodextr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    if (tnodewt) CUDA_CHECK_VOID(cudaMemcpy(ctx->tnodewt_d, tnodewt, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    if (jinbag && ninbag > 0) CUDA_CHECK_VOID(cudaMemcpy(ctx->jinbag_d, jinbag, ninbag * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Only zero importance arrays ONCE at start, then accumulate across all trees
    if (!ctx->importance_arrays_initialized) {
        CUDA_CHECK_VOID(cudaMemset(ctx->qimp_d, 0, nsample * sizeof(real_t)));
        CUDA_CHECK_VOID(cudaMemset(ctx->qimpm_d, 0, nsample * mdim * sizeof(real_t)));
        CUDA_CHECK_VOID(cudaMemset(ctx->avimp_d, 0, mdim * sizeof(real_t)));
        CUDA_CHECK_VOID(cudaMemset(ctx->sqsd_d, 0, mdim * sizeof(real_t)));
        ctx->importance_arrays_initialized = true;
    }
    
    dim3 block_size(256);
    dim3 grid_size(mdim);
    
    if (!ctx->random_states_initialized) {
        init_random_states_kernel<<<grid_size, block_size>>>(ctx->random_states_d, 12345);
        CUDA_CHECK_VOID(cudaDeviceSynchronize());
        ctx->random_states_initialized = true;
    }
    
    integer_t use_casewise = g_config.use_casewise ? 1 : 0;
    
    // Must call init kernel first to set up noob_ptr, right_ptr, and iv arrays
    dim3 init_block_size(256);
    dim3 init_kernel_grid_size(1);  // Single block for initialization
    init_varimp_kernel<<<init_kernel_grid_size, init_block_size>>>(
        ctx->nin_d, ctx->jtr_d, ctx->cl_d, ctx->nodextr_d, ctx->tnodewt_d, nsample,
        ctx->joob_d, ctx->noob_ptr_d, ctx->right_ptr_d,
        impn, ctx->qimp_d, nnode,
        ctx->nodestatus_d, ctx->bestvar_d,
        mdim, ctx->iv_d, use_casewise);
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    cuda_varimp_kernel<<<grid_size, block_size>>>(ctx->x_d, nsample, mdim, ctx->cl_d, ctx->nin_d, ctx->jtr_d,
        impn, ctx->qimp_d, ctx->qimpm_d, ctx->avimp_d, ctx->sqsd_d, ctx->treemap_d, ctx->nodestatus_d,
        ctx->xbestsplit_d, ctx->bestvar_d, ctx->nodeclass_d, nnode, ctx->cat_d, ctx->jvr_d, ctx->nodexvr_d,
        maxcat, nullptr, ctx->tnodewt_d, ctx->nodextr_d, ctx->joob_d, ctx->pjoob_d, ctx->iv_d,
        ctx->noob_ptr_d, ctx->right_ptr_d, ctx->random_states_d, use_casewise);
    
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    CUDA_CHECK_VOID(cudaMemcpy(qimp, ctx->qimp_d, nsample * sizeof(real_t), cudaMemcpyDeviceToHost));
    if (impn == 1) {
        CUDA_CHECK_VOID(cudaMemcpy(qimpm, ctx->qimpm_d, nsample * mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_VOID(cudaMemcpy(avimp, ctx->avimp_d, mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(sqsd, ctx->sqsd_d, mdim * sizeof(real_t), cudaMemcpyDeviceToHost));
}

} // namespace rf





