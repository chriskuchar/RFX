#include "rf_random_forest.hpp"
#include "rf_config.hpp"
#include "rf_arrays.hpp"
#include "rf_utils.hpp"
#include "rf_utilities.hpp"
#include "rf_bootstrap.hpp"
#include "rf_getamat.hpp"
#include "rf_testreebag.hpp"
#include "rf_varimp.hpp"
#include "rf_predict.hpp"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>
#include <sstream>
#include <iomanip>
#include <memory>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

// CRITICAL: Avoid including CUDA headers directly in .cpp file to prevent static initialization issues
// Use forward declarations for GPU functions - they're in separate .cu files that are compiled separately
// Include only CPU headers
#include "rf_growtree.hpp"  // CPU tree growing functions (cpu_growtree_wrapper)
#include "rf_proximity.hpp"  // CPU proximity implementation
#include "rf_finishprox.hpp"

// Forward declare GPU functions - they're in separate .cu files
namespace rf {
    extern void gpu_growtree_batch(integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
                                    integer_t task_type, const integer_t* cat, const integer_t* ties, integer_t* nin,
                                    const real_t* y_regression, integer_t mdim, integer_t nsample, integer_t nclass,
                                    integer_t maxnode, integer_t minndsize, integer_t mtry, integer_t ninbag,
                                    integer_t maxcat, const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
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
                                    real_t* oob_predictions_all,  // For regression: accumulate OOB predictions
                                    void** lowrank_proximity_ptr_out);
    
    #ifdef CUDA_FOUND
    // CLIQUE removed for v1.0 (v2.0 feature)
    // Forward declaration for GPU CLIQUE (defined in cuda/rf_clique.cu) - REMOVED
    /* extern void compute_clique_gpu(...); */
    #endif
}

// Include CUDA headers only when CUDA is available (for RF-GAP to low-rank conversion)
#ifdef CUDA_FOUND
#include "rf_proximity_lowrank.hpp"
#include "rf_lowrank_helpers.hpp"
#endif

namespace rf {
namespace cuda {
    // Forward declaration for CUDA config functions (defined in cuda/rf_config_cuda.cu)
    bool cuda_init_runtime(bool force_cpu);
    bool cuda_is_available();
    integer_t get_recommended_batch_size(integer_t ntree);  // Non-inline implementation in rf_config_cuda.cu
    // Forward declarations for CUDA context management helpers (defined in cuda/rf_config_cuda.cu)
    void cuda_clear_errors();           // Clear any stale CUDA errors before operations
    void cuda_ensure_context_ready();   // Ensure CUDA context is ready before operations
    void cuda_finalize_operations();    // Finalize CUDA operations after fit/predict/etc.
    // Forward declaration for low-rank helper function (defined in cuda/rf_lowrank_helpers.cu)
    bool get_lowrank_factors_host(void* lowrank_proximity_ptr, 
                                   dp_t** A_host, dp_t** B_host, 
                                   integer_t* r, integer_t nsamples);
    std::vector<double> compute_mds_3d_from_factors_host(void* lowrank_proximity_ptr);
    // Forward declaration for LowRankProximityMatrix (defined in cuda/rf_proximity_lowrank.cu)
    class LowRankProximityMatrix;
} // namespace cuda

// Forward declaration for GPU RF-GAP (defined in cuda/rf_proximity.cu)
void gpu_proximity_rfgap(integer_t ntree, integer_t nsample,
                        const integer_t* tree_nin_flat,
                        const integer_t* tree_nodextr_flat,
                        dp_t* prox);
} // namespace rf

namespace rf {

// Standalone train_test_split function (doesn't require RandomForest instance)
void train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                     real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                     integer_t& n_train, integer_t& n_test, 
                     real_t test_size, integer_t random_state, bool stratify) {
    
    // Validate inputs
    if (!X || !y || !X_train || !y_train || !X_test || !y_test) {
        // std::cout << "ERROR: Null pointer in train_test_split" << std::endl;
        return;
    }
    
    if (nsamples <= 0 || mdim <= 0 || test_size <= 0.0 || test_size >= 1.0) {
        // std::cout << "ERROR: Invalid parameters in train_test_split" << std::endl;
        return;
    }
    
    // Calculate split sizes
    n_test = static_cast<integer_t>(nsamples * test_size);
    n_train = nsamples - n_test;
    
    // Validate split sizes
    if (n_train <= 0 || n_test <= 0 || n_train >= nsamples || n_test >= nsamples) {
        // std::cout << "ERROR: Invalid split sizes: n_train=" << n_train << ", n_test=" << n_test << ", nsamples=" << nsamples << std::endl;
        return;
    }
    
    // Simple random split without stratification for now
    // Use heap allocation to avoid stack overflow in Jupyter kernels
    std::unique_ptr<integer_t[]> indices(new integer_t[nsamples]);
    
    for (integer_t i = 0; i < nsamples; i++) {
        indices[i] = i;
    }
    
    // Simple shuffle
    MT19937 rng;
    rng.sgrnd(random_state);
    
    for (integer_t i = nsamples - 1; i > 0; i--) {
        integer_t j = static_cast<integer_t>(rng.grnd() * (i + 1));
        std::swap(indices[i], indices[j]);
    }
    
    // Copy data to train/test arrays
    for (integer_t i = 0; i < n_train; i++) {
        integer_t idx = indices[i];
        if (idx < 0 || idx >= nsamples) {
            // std::cout << "ERROR: Invalid index " << idx << " in train split" << std::endl;
            return;
        }
        // Copy features
        for (integer_t j = 0; j < mdim; j++) {
            X_train[i * mdim + j] = X[idx * mdim + j];
        }
        // Copy labels - cast through real_t first to handle float input
        static_cast<integer_t*>(y_train)[i] = static_cast<integer_t>(static_cast<const real_t*>(y)[idx]);
    }
    
    for (integer_t i = 0; i < n_test; i++) {
        integer_t idx = indices[n_train + i];
        if (idx < 0 || idx >= nsamples) {
            // std::cout << "ERROR: Invalid index " << idx << " in test split" << std::endl;
            return;
        }
        // Copy features
        for (integer_t j = 0; j < mdim; j++) {
            X_test[i * mdim + j] = X[idx * mdim + j];
        }
        // Copy labels - cast through real_t first to handle float input
        static_cast<integer_t*>(y_test)[i] = static_cast<integer_t>(static_cast<const real_t*>(y)[idx]);
    }
    
    // std::cout << "Train-test split: " << n_train << " train, " << n_test << " test samples\n";  // Commented to avoid stream conflicts with Python progress bars
}

// Helper function to get memory information
void print_memory_info(integer_t nsample, integer_t mdim, integer_t ntree, integer_t nclass, integer_t maxnode = 1000, bool use_gpu = false) {
    // Get system memory info from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long total_mem_kb = 0, available_mem_kb = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> total_mem_kb;
        } else if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> available_mem_kb;
        }
    }
    
    // Convert to MB
    double total_mem_mb = total_mem_kb / 1024.0;
    double available_mem_mb = available_mem_kb / 1024.0;
    double used_mem_mb = total_mem_mb - available_mem_mb;
    
    // Estimate training memory usage
    double estimated_mb = 0;
    estimated_mb += nsample * mdim * 4 / (1024.0 * 1024.0);  // x array
    estimated_mb += nsample * 4 / (1024.0 * 1024.0);  // y array
    estimated_mb += 2 * maxnode * ntree * 4 / (1024.0 * 1024.0);  // treemap
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // nodestatus
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // xbestsplit
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // bestvar
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // nodeclass
    estimated_mb += nclass * maxnode * ntree * 4 / (1024.0 * 1024.0);  // classpop
    estimated_mb += nsample * 4 / (1024.0 * 1024.0);  // qimp
    estimated_mb += nsample * mdim * 4 / (1024.0 * 1024.0);  // qimpm
    estimated_mb += mdim * 4 / (1024.0 * 1024.0);  // feature_importances
    estimated_mb += nsample * nsample * 4 / (1024.0 * 1024.0);  // proximity
    
    // Print memory information
    // NOTE: GPU memory info is now printed via Python bindings (print_gpu_memory_status())
    // to avoid std::cout stream conflicts with Python progress bars. This section is disabled.
    if (use_gpu) {
        // GPU memory info printing moved to Python bindings for safety
        // The Python bindings call print_gpu_memory_status() which uses Python's print()
        // This avoids std::codecvt crashes when C++ streams conflict with Python stdout/stderr
        (void)estimated_mb;  // Suppress unused variable warning
        /*
        std::cout << "\n🖥️  GPU MEMORY INFORMATION\n";
        std::cout << "==================================================\n";
        
        // Get GPU memory info safely (may fail in Jupyter, so wrap in try-catch)
        size_t free_mem = 0, total_mem = 0;
        bool memory_info_available = false;
        
        try {
            // Skip GPU memory check in CPU compilation path to avoid CUDA header dependency
            // Memory info will be available when GPU code is actually called
            // GPU memory queries require CUDA headers, which cause static init issues
            memory_info_available = false;  // Disable GPU memory check in CPU path
        } catch (...) {
            // Ignore - CUDA might not be ready or context might be invalid
        }
        
        if (memory_info_available) {
            std::cout << "📊 GPU Memory:\n";
            std::cout << "   Total: " << std::fixed << std::setprecision(1) << (total_mem / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "   Available: " << (free_mem / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "   Used: " << ((total_mem - free_mem) / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "\n";
            std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
            
            if (free_mem > 0) {
                std::cout << "📈 GPU Memory Usage: " << (estimated_mb / (free_mem / (1024.0 * 1024.0)) * 100) << "% of available\n";
                
                if (estimated_mb < (free_mem / (1024.0 * 1024.0)) * 0.8) {
                    std::cout << "📈 GPU Memory Safety: ✅ SAFE\n";
                } else {
                    std::cout << "📈 GPU Memory Safety: ⚠️  HIGH USAGE\n";
                }
            } else {
                std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
                std::cout << "📈 GPU Memory Safety: ⚠️  Unable to query memory\n";
            }
        } else {
            // Fallback when memory info not available
            std::cout << "📊 GPU Memory: Info not available (CUDA context may not be fully initialized)\n";
            std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
            std::cout << "📈 GPU Memory Safety: ⚠️  Unable to verify\n";
        }
        */
    } else {
        // Commented out to avoid stream conflicts with Python progress bars
        // std::cout << "\n💻 CPU MEMORY INFORMATION\n";
        // std::cout << "==================================================\n";
        // std::cout << "📊 System Memory:\n";
        // std::cout << "   Total: " << std::fixed << std::setprecision(1) << total_mem_mb << " MB\n";
        // std::cout << "   Available: " << available_mem_mb << " MB\n";
        // std::cout << "   Used: " << used_mem_mb << " MB\n";
        // std::cout << "\n";
        // std::cout << "📈 Estimated CPU Training Memory: " << estimated_mb << " MB\n";
        // std::cout << "📈 CPU Memory Usage: " << (estimated_mb / available_mem_mb * 100) << "% of available\n";
        // 
        // if (estimated_mb < available_mem_mb * 0.8) {
        //     std::cout << "📈 CPU Memory Safety: ✅ SAFE\n";
        // } else {
        //     std::cout << "📈 CPU Memory Safety: ⚠️  HIGH USAGE\n";
        // }
    }
    // std::cout << "\n";  // Commented to avoid stream conflicts with Python progress bars
}

// Train-test split method (like XGBoost) - now just calls the standalone function
void RandomForest::train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                                   real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                                   integer_t& n_train, integer_t& n_test, 
                                   real_t test_size, integer_t random_state, bool stratify) {
    // Delegate to standalone function
    rf::train_test_split(X, y, nsamples, mdim, X_train, y_train, X_test, y_test, 
                        n_train, n_test, test_size, random_state, stratify);
}

RandomForest::RandomForest(const RandomForestConfig& config) :
    config_(config),
    oob_error_(0.0),
    oob_mse_(0.0),
    proximity_normalized_(false),
    progress_callback_(nullptr),
    lowrank_proximity_ptr_(nullptr) {

    // Use only use_gpu flag - execution_mode is deprecated
    // config_.use_gpu is already set from config.use_gpu
    // Don't override use_gpu here - CUDA will be initialized lazily during fit()
    // The user's use_gpu setting from Python should be preserved
    initialize_arrays();
}

RandomForest::~RandomForest() {
    // Minimal destructor - let std::vector destructors handle cleanup automatically
    // Explicit clearing can cause issues during Python GC, so we rely on RAII
    // The vectors will be destroyed in reverse order of declaration automatically
    
    // Just clear the progress callback to break any circular references
    try {
        progress_callback_ = nullptr;
    } catch (...) {
        // Ignore any errors during destruction
    }
    
    // Clean up low-rank proximity matrix if it exists
    // Use exception-safe cleanup for Jupyter notebook compatibility
    if (lowrank_proximity_ptr_ != nullptr) {
        #ifdef CUDA_FOUND
        try {
            // Cast to LowRankProximityMatrix and delete
            // LowRankProximityMatrix destructor will handle GPU cleanup safely
            // The destructor is exception-safe and won't throw
            rf::cuda::LowRankProximityMatrix* lowrank_prox = 
                static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
            delete lowrank_prox;
        } catch (const std::exception& e) {
            // Log but don't rethrow - we're in a destructor
            // In Jupyter, CUDA context might be corrupted, which is OK
        } catch (...) {
            // Ignore all other errors during cleanup - process might be shutting down
            // CUDA runtime might already be destroyed
        }
        #endif
        lowrank_proximity_ptr_ = nullptr;
    }
    
    // CRITICAL: Do NOT explicitly clear vectors in destructor
    // This can cause double-free errors if Python still holds references
    // Let the vector destructors handle cleanup automatically
    // Only clear if we're sure no references exist (which we can't guarantee)
    // The vectors will be automatically destroyed when the object is destroyed
    
    // Don't explicitly clear other vectors - let their destructors handle it
    // This is safer during Python garbage collection
    // Don't call cuda_cleanup() in destructor - it's global and should only be called explicitly
}

void RandomForest::initialize_arrays() {
    // Allocate tree storage (ntree trees × maxnode nodes)
    tnodewt_.resize(config_.ntree * config_.maxnode, 0.0f);
    xbestsplit_.resize(config_.ntree * config_.maxnode, 0.0f);
    nodestatus_.resize(config_.ntree * config_.maxnode, 0);
    bestvar_.resize(config_.ntree * config_.maxnode, 0);
    treemap_.resize(config_.ntree * 2 * config_.maxnode, 0);
    catgoleft_.resize(config_.ntree * config_.maxcat * config_.maxnode, 0);
    nodeclass_.resize(config_.ntree * config_.maxnode, 0);
    nnode_.resize(config_.ntree, 0);

    // Allocate OOB tracking
    q_.resize(config_.nsample * config_.nclass, 0.0f);
    nout_.resize(config_.nsample, 0);

    // Allocate variable type array (default: all quantitative)
    cat_.resize(config_.mdim, 1);

    // Allocate ties and sorted indices
    ties_.resize(config_.mdim * config_.nsample, 0);
    asave_.resize(config_.mdim * config_.nsample, 0);

    // Initialize workspace arrays (to avoid repeated allocation/deallocation)
    initialize_workspace_arrays();

    // Allocate proximity if requested
    // NOTE: For low-rank/upper-triangle mode, defer allocation until after training
    // to avoid allocating full 80GB matrix for 100k samples
    if (config_.compute_proximity) {
        // Only allocate full matrix if NOT using low-rank
        // For low-rank mode, allocation will happen in GPU code
        // Check global config for flags (they're set via g_config, not config_)
        bool use_lowrank = config_.use_qlora;  // QLoRA is flag-based, not sample-size based
        bool use_upper_triangle = true;  // Default to true for memory efficiency
        
        if (!(use_lowrank && use_upper_triangle)) {
            // Traditional full matrix allocation for small datasets or when qlora disabled
            proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
        }
        // For low-rank mode, proximity_matrix_ will be allocated later in GPU code
    }

    // Initialize CUDA configuration if using GPU
    // Note: CudaConfig is forward declared, so we can't call initialize() here
    // CUDA initialization will happen lazily when GPU code is actually called
    if (config_.use_gpu) {
        // CUDA initialization deferred to avoid static init issues
    }

    // Allocate feature importances if requested
    if (config_.compute_importance) {
        // std::cout << "DEBUG: compute_importance is TRUE, allocating arrays" << std::endl;
        // std::cout.flush();  // Commented out to avoid stream conflicts with Python progress bars
        integer_t mimp = config_.mdim;
        feature_importances_.resize(mimp, 0.0f);
        
        // Allocate global importance arrays for accumulation across trees
        qimp_.resize(config_.nsample, 0.0f);
        // Only allocate qimpm_ if local importance is requested
        // qimpm_ is only needed for local importance, not for overall importance
        if (config_.compute_local_importance) {
        qimpm_.resize(config_.nsample * config_.mdim, 0.0f);
        }
        avimp_.resize(config_.mdim, 0.0f);
        sqsd_.resize(config_.mdim, 0.0f);
    } else {
        // std::cout << "DEBUG: compute_importance is FALSE" << std::endl;
    }
}

void RandomForest::detect_categorical_features(const real_t* X, integer_t nsample, integer_t mdim, integer_t maxcat) {
    // Auto-detect categorical features:
    // A feature is categorical if:
    // 1. All values are integers (or very close to integers)
    // 2. Number of unique values <= maxcat
    // 3. Values are in range [0, maxcat-1] or can be mapped to that range
    
    cat_.resize(mdim, 1);  // Initialize all as quantitative (1)
    
    for (integer_t m = 0; m < mdim; ++m) {
        std::set<real_t> unique_values;
        bool all_integers = true;
        real_t min_val = 1e10f;
        real_t max_val = -1e10f;
        
        // Check all samples for this feature
        for (integer_t n = 0; n < nsample; ++n) {
            real_t val = X[n * mdim + m];
            unique_values.insert(val);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            
            // Check if value is close to an integer
            real_t rounded = std::round(val);
            if (std::abs(val - rounded) > 1e-5f) {
                all_integers = false;
            }
        }
        
        // Check if feature is categorical
        integer_t num_unique = static_cast<integer_t>(unique_values.size());
        if (all_integers && num_unique > 1 && num_unique <= maxcat && 
            min_val >= 0.0f && max_val < static_cast<real_t>(maxcat)) {
            // Feature is categorical - set cat[m] to number of categories
            cat_[m] = num_unique;
        }
    }
}

void RandomForest::set_categorical_features(const integer_t* cat_array, integer_t mdim) {
    // Manually set categorical features from provided array
    cat_.resize(mdim);
    for (integer_t m = 0; m < mdim; ++m) {
        cat_[m] = cat_array[m];
    }
}

// Unified fit method that dispatches based on task type
void RandomForest::fit(const real_t* X, const void* y, const real_t* sample_weight) {
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[FIT] RandomForest::fit() called - task_type=" << static_cast<int>(config_.task_type) 
    //           << ", use_casewise=" << config_.use_casewise << std::endl;
    // std::cout.flush();
    
    // Set global config values that CPU/GPU code needs
    rf::g_config.use_casewise = config_.use_casewise;
    // std::cout << "[FIT] Set g_config.use_casewise=" << rf::g_config.use_casewise << std::endl;
    // std::cout.flush();
    //     // std::cout << "DEBUG: RandomForest::fit ENTRY - nsample=" << config_.nsample 
    //           << ", compute_proximity=" << config_.compute_proximity 
    //           << ", use_qlora=" << config_.use_qlora << std::endl;
    // std::cout.flush();
    
    switch (config_.task_type) {
        case TaskType::CLASSIFICATION:
            fit_classification(X, static_cast<const integer_t*>(y), sample_weight);
            break;
        case TaskType::REGRESSION:
            fit_regression(X, static_cast<const real_t*>(y), sample_weight);
            break;
        case TaskType::UNSUPERVISED:
            fit_unsupervised(X, sample_weight);
            break;
    }
}

// Classification training
void RandomForest::fit_classification(const real_t* X, const integer_t* y, const real_t* sample_weight) {
    // NOTE: Removed debug prints to avoid potential stack issues during return
    
    // Debug: Check y data
    // std::cout << "C++ fit_classification: y data first " << std::min(10, (int)config_.nsample) << ": ";
    // for (int i = 0; i < std::min(10, (int)config_.nsample); i++) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;
    // if (config_.nsample > 50) {
    //     std::cout << "C++ fit_classification: y data samples 50-59: ";
    //     for (int i = 50; i < std::min(60, (int)config_.nsample); i++) {
    //         std::cout << y[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "DEBUG: fit_classification called, compute_importance=" << config_.compute_importance << std::endl;
    
    // Training message handled by Python wrapper
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
        try {
            // std::cout << "DEBUG: Attempting CUDA initialization..." << std::endl;
            // std::cout.flush();
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
            // Commented out to avoid stream conflicts with Python progress bars
            // if (config_.use_gpu) {
            //     std::cout << "CUDA: Using GPU acceleration\n";
            // } else {
            //     std::cout << "CUDA: Initialization failed, falling back to CPU\n";
            // }
        } catch (const std::exception& e) {
            // std::cerr << "CUDA: Initialization failed (" << e.what() << "), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        } catch (...) {
            // std::cerr << "CUDA: Initialization failed (unknown error), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        }
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // Print memory information (skip in CPU mode to reduce output)
    // Only print for GPU mode or if explicitly requested
    // if (config_.use_gpu) {
    //     print_memory_info(config_.nsample, config_.mdim, config_.ntree, config_.nclass, config_.maxnode, config_.use_gpu);
    // }

    // Store training data
    // std::cout << "DEBUG: About to assign X_train_..." << std::endl;
    // std::cout.flush();  // Commented out to avoid stream conflicts with Python progress bars
    X_train_.assign(X, X + config_.mdim * config_.nsample);
    
    // std::cout << "DEBUG: About to assign y_train_classification_..." << std::endl;
    // std::cout.flush();
    y_train_classification_.assign(y, y + config_.nsample);

    // std::cout << "DEBUG: About to resize sample_weight_..." << std::endl;
    // std::cout.flush();
    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // std::cout << "DEBUG: About to call setup_classification_task..." << std::endl;
    // std::cout.flush();
    // Setup classification-specific parameters
    setup_classification_task(y);
    
    // std::cout << "DEBUG: About to call prepare_data..." << std::endl;
    // std::cout.flush();
    // Prepare data (sort, create ties)
    prepare_data();
    
    // std::cout << "DEBUG: Completed prepare_data" << std::endl;
    // std::cout.flush();
    
    // CRITICAL: Add barrier to ensure prepare_data() completed fully
    // std::cout << "DEBUG: prepare_data() returned successfully" << std::endl;
    // std::cout.flush();
    
    // CRITICAL: Check config values BEFORE accessing them in condition
    // This ensures we're not accessing corrupted memory
    // std::cout << "DEBUG: About to read config_.use_gpu..." << std::endl;
    // std::cout.flush();
    bool use_gpu_val = config_.use_gpu;
    // std::cout << "DEBUG: config_.use_gpu read successfully: " << use_gpu_val << std::endl;
    // std::cout.flush();
    
    // std::cout << "DEBUG: About to read config_.ntree..." << std::endl;
    // std::cout.flush();
    integer_t ntree_val = config_.ntree;
    // std::cout << "DEBUG: config_.ntree read successfully: " << ntree_val << std::endl;
    // std::cout.flush();

    // std::cout << "DEBUG: Checking GPU batch mode - use_gpu=" << use_gpu_val 
    //           << ", ntree=" << ntree_val << std::endl;
    // std::cout.flush();
    
    // std::cout << "DEBUG: About to check if condition..." << std::endl;
    // std::cout.flush();
    
    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!use_gpu_val) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // CRITICAL: Low-rank proximity is GPU-only! Disable it in CPU mode for large datasets
        if (config_.compute_proximity && config_.use_qlora) {
            // std::cerr << "WARNING: Low-rank proximity (use_qlora=True) requires GPU, but GPU is not available." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Disabling proximity computation for CPU mode with use_qlora=True." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Use GPU mode (use_gpu=True) for low-rank proximity." << std::endl;  // Commented to avoid stream conflicts
            config_.compute_proximity = false;  // Disable proximity in CPU mode
            proximity_matrix_.clear();  // Clear any allocated proximity matrix
        }
        
        // std::cout << "DEBUG: Using CPU sequential mode (classification) - ntree=" << ntree_val << std::endl;
        // std::cout.flush();
        // std::cout << "Growing trees:\n";
        for (integer_t itree = 0; itree < ntree_val; ++itree) {
            // std::cout << "DEBUG: Loop iteration " << itree << " of " << ntree_val << std::endl;
            // std::cout.flush();
            // std::cout << "DEBUG: About to call grow_tree_single for tree " << itree << std::endl;
            // std::cout.flush();
            // CRITICAL: Cache config_.iseed to avoid accessing config_ member during crash
            integer_t seed_val = config_.iseed + itree;
            // std::cout << "DEBUG: seed_val=" << seed_val << std::endl;
            // std::cout.flush();
            grow_tree_single(itree, seed_val);
            // std::cout << "DEBUG: Completed grow_tree_single for tree " << itree << std::endl;
            // std::cout.flush();

            // Progress update for CPU training (callback)
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        // std::cout << "\n";
    } else {
        // GPU mode: Choose between sequential (batch_size=1) or batch (batch_size>1)
        integer_t batch_size;
        bool is_auto_scaled = false;
        
        // std::cout << "[DEBUG CLASSIFICATION] config_.batch_size = " << config_.batch_size << std::endl;
        std::cout.flush();
        
        // Determine batch size first
        if (config_.batch_size > 0) {
            // Use user-specified batch size
            batch_size = std::min(config_.batch_size, ntree_val);
        } else {
            // Auto-scale batch size using get_recommended_batch_size
            try {
                batch_size = rf::cuda::get_recommended_batch_size(ntree_val);
            } catch (...) {
                // Fallback: use default batch size if CUDA function fails
                batch_size = std::min(static_cast<integer_t>(10), ntree_val);
            }
            batch_size = std::min(batch_size, ntree_val);
            is_auto_scaled = true;
        }
        
        // CRITICAL: Set gpu_parallel_mode0 based on batch_size
        // Parallel mode is enabled when batch_size > 1
        // Fixed: Removed expensive curand_init from kernel, now uses pre-initialized states
        // This must be set BEFORE calling fit_batch_gpu
        rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
        
        if (is_auto_scaled){
            // std::cout << "GPU: Auto-scaling selected batch size = " << batch_size << " trees (out of " << ntree_val << " total)" << std::endl;  // Commented for Jupyter safety
        } else {
            // std::cout << "GPU: Using manual batch size = " << batch_size << " trees (out of " << ntree_val << " total, config_.batch_size=" << config_.batch_size << ")" << std::endl;  // Commented for Jupyter safety
        }
        // std::cout.flush();  // Commented for Jupyter safety
        fit_batch_gpu(X, y, sample_weight_.data(), batch_size);
        
        // CRITICAL: Validate object state after GPU operations to catch corruption
        // Check that essential member variables are still valid
        if (config_.nsample <= 0 || config_.nclass <= 0) {
            throw std::runtime_error("Object state corrupted after fit_batch_gpu: invalid config");
        }
        if (nout_.size() < static_cast<size_t>(config_.nsample) || 
            q_.size() < static_cast<size_t>(config_.nsample * config_.nclass)) {
            throw std::runtime_error("Object state corrupted after fit_batch_gpu: invalid member arrays");
        }
    }

    // Compute OOB error and predictions BEFORE normalization (using raw vote counts in q_)
    // Match repo version order: compute_classification_error, compute_oob_predictions, then finalize_training
    compute_classification_error();
    compute_oob_predictions();
    
    // Finalize training (normalizes OOB votes for probabilities, but we've already computed error/predictions)
    // Match repo version: call finalize_training AFTER compute_oob_predictions
    // NOTE: fit_batch_gpu() should NOT call finalize_training() - we call it here instead
    finalize_training();

    // CRITICAL: After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells and ensures clean return
    // This matches the pattern in fit_regression() and fit_unsupervised()
    // CRITICAL: Store use_gpu in local variable to avoid accessing config_ after GPU operations
    bool use_gpu_local = config_.use_gpu;
    if (use_gpu_local) {
        rf::cuda::cuda_finalize_operations();
        // CRITICAL: Clear any CUDA errors after finalization to ensure clean state
        #ifdef CUDA_FOUND
        cudaGetLastError();  // Clear any pending errors
        #endif
    }
    
    // CRITICAL: Add memory barrier before return to ensure all operations complete
    // This helps prevent stack corruption during return in exec() context
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // CRITICAL: Flush all output streams before return to ensure clean state
    // std::cout.flush();  // Commented out to avoid stream conflicts with Python progress bars
    // std::cerr.flush();  // Commented out to avoid stream conflicts with Python progress bars

    // OOB error handled by Python wrapper
    // Match repo version exactly - simple return, no debug prints
}

// Regression training
void RandomForest::fit_regression(const real_t* X, const real_t* y, const real_t* sample_weight) {
    // Training message handled by Python wrapper
    // std::cout << "Training Random Forest Regressor with " << config_.ntree << " trees...\n";
    
    // CRITICAL: Clear any stale CUDA errors before fit (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_clear_errors();
    }
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
        try {
            // std::cout << "DEBUG: Attempting CUDA initialization..." << std::endl;
            // std::cout.flush();
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
            // Commented out to avoid stream conflicts with Python progress bars
            // if (config_.use_gpu) {
            //     std::cout << "CUDA: Using GPU acceleration\n";
            // } else {
            //     std::cout << "CUDA: Initialization failed, falling back to CPU\n";
            // }
        } catch (const std::exception& e) {
            // std::cerr << "CUDA: Initialization failed (" << e.what() << "), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        } catch (...) {
            // std::cerr << "CUDA: Initialization failed (unknown error), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        }
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // CRITICAL: Before fit, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Print memory information (skip in CPU mode to reduce output)
    // Only print for GPU mode or if explicitly requested
    // if (config_.use_gpu) {
    //     print_memory_info(config_.nsample, config_.mdim, config_.ntree, config_.nclass, config_.maxnode, config_.use_gpu);
    // }
    integer_t ntree_val = config_.ntree;
    // Store training data
    X_train_.assign(X, X + config_.mdim * config_.nsample);
    y_train_regression_.assign(y, y + config_.nsample);

    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // Setup regression-specific parameters
    setup_regression_task(y);

    // Prepare data (sort, create ties)
    prepare_data();

    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!config_.use_gpu) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // CRITICAL: Low-rank proximity is GPU-only! Disable it in CPU mode for large datasets
        if (config_.compute_proximity && config_.use_qlora) {
            // std::cerr << "WARNING: Low-rank proximity (use_qlora=True) requires GPU, but GPU is not available." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Disabling proximity computation for CPU mode with use_qlora=True." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Use GPU mode (use_gpu=True) for low-rank proximity." << std::endl;  // Commented to avoid stream conflicts
            config_.compute_proximity = false;  // Disable proximity in CPU mode
            proximity_matrix_.clear();  // Clear any allocated proximity matrix
        }
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            grow_tree_single(itree, config_.iseed + itree);
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
    } else {
        // GPU mode: Choose between sequential (batch_size=1) or batch (batch_size>1)
        integer_t batch_size;
        bool is_auto_scaled = false;
        
        // Determine batch size first
        if (config_.batch_size > 0) {
            // Use user-specified batch size
            batch_size = std::min(config_.batch_size, config_.ntree);
        } else {
            // Auto-scale batch size using get_recommended_batch_size
            try {
                batch_size = rf::cuda::get_recommended_batch_size(config_.ntree);
            } catch (...) {
                // Fallback: use default batch size if CUDA function fails
                batch_size = std::min(static_cast<integer_t>(10), config_.ntree);
            }
            batch_size = std::min(batch_size, config_.ntree);
            is_auto_scaled = true;
        }
        
        // CRITICAL: Set gpu_parallel_mode0 based on batch_size
        // Parallel mode is enabled when batch_size > 1
        // This must be set BEFORE calling fit_batch_gpu
        rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
        
        // CRITICAL: Set g_config.use_casewise BEFORE calling fit_batch_gpu
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (is_auto_scaled){
            // std::cout << "GPU: Auto-scaling selected batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        } else {
            // std::cout << "GPU: Using manual batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        }
        // std::cout.flush();
        // std::cout << "[FIT_REGRESSION] About to call fit_batch_gpu with batch_size=" << batch_size 
        //           << ", use_casewise=" << config_.use_casewise << std::endl;
        // std::cout.flush();
        
        fit_batch_gpu(X, y, sample_weight_.data(), batch_size);
        // std::cout << "[FIT_REGRESSION] fit_batch_gpu completed successfully" << std::endl;
        // std::cout.flush();
    }

    // Finalize training
    finalize_training();

    // Compute regression MSE
    compute_regression_mse();

    // CRITICAL: After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells
    if (config_.use_gpu) {
        rf::cuda::cuda_finalize_operations();
    }

    // OOB MSE handled by Python wrapper
    // std::cout << "Training complete. OOB MSE: " << oob_mse_ << "\n";
}

// Unsupervised training
void RandomForest::fit_unsupervised(const real_t* X, const real_t* sample_weight) {
    // Set global config values that CPU/GPU code needs
    rf::g_config.use_casewise = config_.use_casewise;
    
    // Training message handled by Python wrapper
    // std::cout << "Training Random Forest Unsupervised with " << config_.ntree << " trees...\n";
    
    // CRITICAL: Clear any stale CUDA errors before fit (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_clear_errors();
    }
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
        try {
            // std::cout << "DEBUG: Attempting CUDA initialization..." << std::endl;
            // std::cout.flush();
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
            // Commented out to avoid stream conflicts with Python progress bars
            // if (config_.use_gpu) {
            //     std::cout << "CUDA: Using GPU acceleration\n";
            // } else {
            //     std::cout << "CUDA: Initialization failed, falling back to CPU\n";
            // }
        } catch (const std::exception& e) {
            // std::cerr << "CUDA: Initialization failed (" << e.what() << "), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        } catch (...) {
            // std::cerr << "CUDA: Initialization failed (unknown error), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        }
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // CRITICAL: Before fit, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    integer_t ntree_val = config_.ntree;

    // Print memory information (skip in CPU mode to reduce output)
    // Only print for GPU mode or if explicitly requested
    // if (config_.use_gpu) {
    //     print_memory_info(config_.nsample, config_.mdim, config_.ntree, config_.nclass, config_.maxnode, config_.use_gpu);
    // }

    // Store training data
    X_train_.assign(X, X + config_.mdim * config_.nsample);

    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // Initialize synthetic labels for unsupervised learning
    y_train_unsupervised_.resize(config_.nsample, 0);

    // Setup unsupervised-specific parameters (MUST be called before initialize_arrays)
    // This sets nclass=1 which affects workspace array sizing
    setup_unsupervised_task();
    
    // Re-size arrays that depend on nclass (which was just changed to 1)
    // This ensures arrays are properly sized for unsupervised (nclass=1)
    q_.resize(config_.nsample * config_.nclass, 0.0f);
    
    // Re-initialize workspace arrays with correct nclass=1 for unsupervised
    // This ensures workspace arrays are properly sized
    initialize_workspace_arrays();

    // Prepare data (sort, create ties)
    prepare_data();

    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!config_.use_gpu) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // CRITICAL: Low-rank proximity is GPU-only! Disable it in CPU mode for large datasets
        if (config_.compute_proximity && config_.use_qlora) {
            // std::cerr << "WARNING: Low-rank proximity (use_qlora=True) requires GPU, but GPU is not available." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Disabling proximity computation for CPU mode with use_qlora=True." << std::endl;  // Commented to avoid stream conflicts
            // std::cerr << "WARNING: Use GPU mode (use_gpu=True) for low-rank proximity." << std::endl;  // Commented to avoid stream conflicts
            config_.compute_proximity = false;  // Disable proximity in CPU mode
            proximity_matrix_.clear();  // Clear any allocated proximity matrix
        }
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            grow_tree_single(itree, config_.iseed + itree);
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Finalize training for CPU mode (this also computes outlier scores for unsupervised)
        finalize_training();

        // Compute cluster labels after proximity matrix is finalized (CPU only)
        if (config_.compute_proximity && !proximity_matrix_.empty() && proximity_normalized_) {
            compute_cluster_labels();
        }
    } else {
        // GPU mode: Choose between sequential (grow_gpu_parallel=False) or batch (grow_gpu_parallel=True)
        integer_t batch_size;
        bool is_auto_scaled = false;
        
        // Determine batch size first
        if (config_.batch_size > 0) {
            // Use user-specified batch size
            batch_size = std::min(config_.batch_size, config_.ntree);
        } else {
            // Auto-scale batch size using get_recommended_batch_size
            try {
                batch_size = rf::cuda::get_recommended_batch_size(config_.ntree);
            } catch (...) {
                // Fallback: use default batch size if CUDA function fails
                batch_size = std::min(static_cast<integer_t>(10), config_.ntree);
            }
            batch_size = std::min(batch_size, config_.ntree);
            is_auto_scaled = true;
        }
        
        // CRITICAL: Set gpu_parallel_mode0 based on batch_size
        // Parallel mode is enabled when batch_size > 1
        // This must be set BEFORE calling fit_batch_gpu
        rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
        
        // CRITICAL: Set g_config.use_casewise BEFORE calling fit_batch_gpu
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (is_auto_scaled){
            // std::cout << "GPU: Auto-scaling selected batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        } else {
            // std::cout << "GPU: Using manual batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        }
        // std::cout.flush();  // Commented to avoid stream conflicts with Python progress bars
        // std::cout << "[FIT_UNSUPERVISED] About to call fit_batch_gpu with batch_size=" << batch_size 
        //           << ", use_casewise=" << config_.use_casewise << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        // std::cout.flush();
        
        // For unsupervised, pass y_train_unsupervised_ instead of nullptr
        fit_batch_gpu(X, y_train_unsupervised_.data(), sample_weight_.data(), batch_size);
        // std::cout << "[FIT_UNSUPERVISED] fit_batch_gpu completed successfully" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        // std::cout.flush();
    }

    // CRITICAL: After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells
    if (config_.use_gpu) {
        rf::cuda::cuda_finalize_operations();
    }

    // Completion message handled by Python wrapper
    // std::cout << "Training complete. Proximity matrix computed.\n";
}

void RandomForest::prepare_data() {
    // Prepare data: sort and create ties array
    // std::cout << "DEBUG: About to allocate isort and v arrays" << std::endl;
    // std::cout.flush();
    
    std::vector<integer_t> isort(config_.nsample);
    std::vector<real_t> v(config_.nsample);
    
    // std::cout << "DEBUG: About to call prepdata" << std::endl;
    // std::cout.flush();

    prepdata(X_train_.data(), config_.mdim, config_.nsample,
             cat_.data(), isort.data(), v.data(), asave_.data(), ties_.data());
             
    // std::cout << "DEBUG: Completed prepdata" << std::endl;
    // std::cout.flush();
}

void RandomForest::grow_tree_single(integer_t tree_id, integer_t seed) {
    // std::cout << "DEBUG: grow_tree_single ENTRY - tree_id=" << tree_id 
    //           << ", seed=" << seed << ", nsample=" << config_.nsample << std::endl;
    // std::cout.flush();
    
    // std::cerr << "DEBUG: grow_tree_single called for tree " << tree_id << " with seed " << seed << std::endl;
    
    // Use pre-allocated workspace arrays to avoid repeated allocation/deallocation
    integer_t ninbag, noobag;

    // std::cout << "DEBUG: About to call cpu_bootstrap..." << std::endl;
    // std::cout.flush();
    
    // Use printf to avoid potential iostream corruption issues
    // printf("DEBUG: sample_weight_.size()=%zu\n", sample_weight_.size());
    // fflush(stdout);
    
    if (sample_weight_.empty()) {
        // printf("ERROR: sample_weight_ is empty!\n");  // Commented to avoid stream conflicts
        // fflush(stdout);
        return;
    }
    
    // printf("DEBUG: config_.nsample=%d\n", static_cast<int>(config_.nsample));
    // printf("DEBUG: tree_id=%d\n", static_cast<int>(tree_id));
    // fflush(stdout);
    
    // Bootstrap sampling
    // printf("DEBUG: Actually calling cpu_bootstrap now...\n");
    // fflush(stdout);
    cpu_bootstrap(sample_weight_.data(), config_.nsample, win_workspace_.data(), nin_workspace_.data(),
              nout_.data(), jinbag_workspace_.data(), joobag_workspace_.data(), ninbag, noobag, tree_id);
    // printf("DEBUG: cpu_bootstrap returned successfully\n");
    // fflush(stdout);
    
    // printf("DEBUG: About to call cpu_getamat...\n");
    // fflush(stdout);
    
    // Create sorted index matrix for this bootstrap sample
    cpu_getamat(asave_.data(), nin_workspace_.data(), cat_.data(), config_.nsample,
            config_.mdim, amat_workspace_.data());
    
    // printf("DEBUG: cpu_getamat returned successfully\n");
    // fflush(stdout);

    // Clear workspace arrays for this tree
    // printf("DEBUG: About to clear workspace arrays\n");
    // fflush(stdout);
    std::fill(jtr_workspace_.begin(), jtr_workspace_.end(), 0);
    std::fill(nodextr_workspace_.begin(), nodextr_workspace_.end(), 0);
    // printf("DEBUG: Workspace arrays cleared\n");
    // fflush(stdout);

    // Get pointers to this tree's storage
    // printf("DEBUG: About to get pointers to tree storage\n");
    // fflush(stdout);
    real_t* tnodewt = tnodewt_.data() + tree_id * config_.maxnode;
    real_t* xbestsplit = xbestsplit_.data() + tree_id * config_.maxnode;
    integer_t* nodestatus = nodestatus_.data() + tree_id * config_.maxnode;
    integer_t* bestvar = bestvar_.data() + tree_id * config_.maxnode;
    integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
    integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
    integer_t* nodeclass = nodeclass_.data() + tree_id * config_.maxnode;
    
    // Initialize tree arrays to zero (matching Fortran initialization)
    std::fill(nodestatus, nodestatus + config_.maxnode, 0);
    std::fill(bestvar, bestvar + config_.maxnode, 0);
    std::fill(xbestsplit, xbestsplit + config_.maxnode, 0.0f);
    std::fill(treemap, treemap + 2 * config_.maxnode, 0);
    std::fill(nodeclass, nodeclass + config_.maxnode, 0);
    std::fill(catgoleft, catgoleft + config_.maxcat * config_.maxnode, 0);
    std::fill(tnodewt, tnodewt + config_.maxnode, 0.0f);  // Initialize tnodewt for this tree

    // printf("DEBUG: Tree arrays initialized\n");
    // fflush(stdout);

    // Grow tree
    // printf("DEBUG: About to check task_type\n");
    // fflush(stdout);
    const integer_t* y_data;
    if (config_.task_type == TaskType::CLASSIFICATION) {
        //         // printf("DEBUG: Task type is CLASSIFICATION\n");
        // fflush(stdout);
        // Use 0-based class labels directly (matching reference setup_classification_task)
        // Note: cpu_growtree expects 0-based labels (checks class_idx >= 0 && class_idx < nclass)
        //         // printf("DEBUG: Using y_train_classification_1based_ (0-based labels)\n");
        // fflush(stdout);
        y_data = y_train_classification_1based_.data();
        //         // printf("DEBUG: y_data set to y_train_classification_1based_.data()\n");
        // fflush(stdout);
    } else if (config_.task_type == TaskType::REGRESSION) {
        // For regression, we need to convert real_t to integer_t
        // Scale y values to preserve precision (small values like 0.1 become 100)
        // This prevents all values from becoming 0 when cast to integers
        const real_t y_scale = 1000.0f;
        if (y_train_regression_int_.empty()) {
            y_train_regression_int_.resize(y_train_regression_.size());
            for (size_t i = 0; i < y_train_regression_.size(); ++i) {
                y_train_regression_int_[i] = static_cast<integer_t>(y_train_regression_[i] * y_scale);
            }
        }
        y_data = y_train_regression_int_.data();
    } else {
        // For unsupervised, use synthetic labels
        if (y_train_unsupervised_.empty()) {
            y_train_unsupervised_.resize(config_.nsample, 0);
        }
        y_data = y_train_unsupervised_.data();
    }
    
    // printf("DEBUG: About to call cpu_growtree_wrapper\n");
    // fflush(stdout);
    
    // Use the standard tree growing function for all task types
    // The implementation in rf_growtree.cpp now handles regression, classification, and unsupervised
    cpu_growtree_wrapper(X_train_.data(), win_workspace_.data(), y_data, cat_.data(),
         ties_.data(), nin_workspace_.data(), config_.mdim, config_.nsample,
         config_.nclass, config_.maxnode, config_.minndsize, config_.mtry,
         ninbag, config_.maxcat, seed, config_.ncsplit, config_.ncmax,
         amat_workspace_.data(), jinbag_workspace_.data(), tnodewt, xbestsplit, nodestatus,
         bestvar, treemap, catgoleft, nodeclass, jtr_workspace_.data(), nodextr_workspace_.data(),
         nnode_[tree_id], idmove_workspace_.data(), igoleft_workspace_.data(), igl_workspace_.data(),
         nodestart_workspace_.data(), nodestop_workspace_.data(), itemp_workspace_.data(), itmp_workspace_.data(),
         icat_workspace_.data(), tcat_workspace_.data(), classpop_workspace_.data(), wr_workspace_.data(), wl_workspace_.data(),
         tclasscat_workspace_.data(), tclasspop_workspace_.data(), tmpclass_workspace_.data());

    // Test tree on OOB samples - temporarily disabled due to segmentation fault
    // testreebag(X_train_.data(), xbestsplit, nin.data(), treemap, bestvar,
    //            nodeclass, cat_.data(), nodestatus, catgoleft, config_.nsample,
    //            config_.mdim, nnode_[tree_id], config_.maxcat, jtr.data(), nodextr.data());
    
    // Debug: Check if tree was properly grown
    
    // Call testreebag to get OOB predictions and terminal nodes for this tree
    // NOTE: cpu_testreebag traverses the tree for OOB samples and:
    //   - For classification: populates jtr with class predictions, nodextr with terminal nodes
    //   - For regression: we still need nodextr (terminal nodes) to get predictions from tnodewt[nodextr[n]]
    //                     jtr is not used for regression (we use tnodewt[nodextr[n]] directly)
    //   - For unsupervised: same as classification (uses class predictions)
    cpu_testreebag(X_train_.data(), xbestsplit, nin_workspace_.data(), treemap, bestvar,
               nodeclass, cat_.data(), nodestatus, catgoleft,
               config_.nsample, config_.mdim, nnode_[tree_id], config_.maxcat,
               jtr_workspace_.data(), nodextr_workspace_.data());
    
    // CRITICAL: Compute tnodewt for classification casewise mode
    // For classification with casewise=True, tnodewt[node] = mean bootstrap weight of in-bag samples in that node
    // For classification with casewise=False, we don't use tnodewt (always weight=1.0)
    // For regression, tnodewt is already computed during tree growing (mean y value in node)
    if ((config_.task_type == TaskType::CLASSIFICATION || config_.task_type == TaskType::UNSUPERVISED) && config_.use_casewise) {
        // Initialize tnodewt to 0
        std::fill(tnodewt, tnodewt + config_.maxnode, 0.0f);
        
        // Count samples and sum weights for each terminal node
        std::vector<real_t> node_weight_sum(config_.maxnode, 0.0f);
        std::vector<integer_t> node_sample_count(config_.maxnode, 0);
        
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {  // In-bag sample
                // Traverse tree from root to find terminal node
                integer_t current_node = 0;
                while (nodestatus[current_node] == 1) {  // While not terminal
                    integer_t split_var = bestvar[current_node];
                    bool goes_left = false;
                    
                    // Check if categorical
                    integer_t split_numcat = (cat_.empty() || split_var >= config_.mdim) ? 1 : cat_[split_var];
                    if (split_numcat > 1) {  // Categorical
                        integer_t category_idx = static_cast<integer_t>(X_train_[n * config_.mdim + split_var] + 0.5f);
                        integer_t catgoleft_offset = current_node * config_.maxcat;
                        goes_left = (category_idx >= 0 && category_idx < config_.maxcat && 
                                   catgoleft[catgoleft_offset + category_idx] == 1);
                    } else {  // Quantitative
                        real_t split_point = xbestsplit[current_node];
                        real_t sample_value = X_train_[n * config_.mdim + split_var];
                        goes_left = (sample_value <= split_point);
                    }
                    
                    current_node = treemap[current_node * 2 + (goes_left ? 0 : 1)];
                    if (current_node < 0 || current_node >= nnode_[tree_id]) break;
                }
                
                // Add to terminal node statistics
                if (current_node >= 0 && current_node < nnode_[tree_id] && nodestatus[current_node] == -1) {
                    node_weight_sum[current_node] += static_cast<real_t>(nin_workspace_[n]);
                    node_sample_count[current_node]++;
                }
            }
        }
        
        // Compute mean weight for each terminal node
        for (integer_t node = 0; node < nnode_[tree_id]; ++node) {
            if (node_sample_count[node] > 0) {
                tnodewt[node] = node_weight_sum[node] / static_cast<real_t>(node_sample_count[node]);
            }
        }
    }
    
    // For regression, jtr_workspace_ contains class predictions (not meaningful for regression)
    // We use tnodewt[nodextr_workspace_[n]] for regression predictions instead
    // But we still need nodextr_workspace_ to know which terminal node each OOB sample reached


    // Accumulate OOB votes/predictions
    integer_t oob_count = 0;
    for (integer_t n = 0; n < config_.nsample; ++n) {
        if (nin_workspace_[n] == 0) {  // Sample is out-of-bag
            nout_[n]++;  // Increment OOB count for this sample
            oob_count++;
            
            if (config_.task_type == TaskType::CLASSIFICATION || config_.task_type == TaskType::UNSUPERVISED) {
                // For classification/unsupervised: accumulate class votes in q_ array
            if (jtr_workspace_[n] >= 0 && jtr_workspace_[n] < config_.nclass) {
                // Use 0-based indexing directly
                // q_ array is indexed as [sample * nclass + class]
                // CRITICAL: Casewise vs non-casewise weighting for OOB votes
                // Non-casewise: weight = 1.0 (UC Berkeley standard)
                // Casewise: weight = tnodewt[terminal_node] (bootstrap frequency weighted)
                real_t vote_weight = 1.0f;
                if (config_.use_casewise) {
                    integer_t terminal_node = nodextr_workspace_[n];
                    if (terminal_node >= 0 && terminal_node < nnode_[tree_id]) {
                        vote_weight = tnodewt[terminal_node];
                    }
                }
                integer_t class_idx = n * config_.nclass + jtr_workspace_[n];
                if (class_idx >= 0 && class_idx < q_.size()) {
                    q_[class_idx] += vote_weight;
                    }
                }
            } else if (config_.task_type == TaskType::REGRESSION) {
                // For regression: accumulate regression predictions from tnodewt[nodextr[n]]
                // oob_predictions_ stores the sum of predictions across all trees
                // Final prediction = oob_predictions_[n] / nout_[n] (average across trees)
                integer_t terminal_node = nodextr_workspace_[n];
                if (terminal_node >= 0 && terminal_node < nnode_[tree_id]) {
                    // tnodewt stores unscaled regression predictions (mean of y in terminal node)
                    real_t regression_prediction = tnodewt[terminal_node];
                    if (n < oob_predictions_.size()) {
                        oob_predictions_[n] += regression_prediction;
                    }
                }
            }
            
            // Debug: Print OOB vote accumulation for first few samples
            // if (tree_id < 3 && n < 10) {
            //     std::cout << "CPU Tree " << tree_id << " OOB sample " << n 
            //              << " predicted class " << jtr_workspace_[n] << std::endl;
            // }
        }
    }

    // Variable importance computation if requested
    if (config_.compute_importance) {
        // Set global config for importance computation (must be set before cpu_varimp)
        rf::g_config.use_casewise = config_.use_casewise;
        
        // printf("DEBUG: Starting variable importance computation for CPU\n");
        // printf("DEBUG: compute_importance flag is TRUE in grow_tree_single\n");
        // fflush(stdout);
        std::vector<real_t> qimp_temp(config_.nsample);
        std::vector<real_t> qimpm_temp(config_.nsample * config_.mdim);
        std::vector<real_t> avimp_temp(config_.mdim);
        std::vector<real_t> sqsd_temp(config_.mdim);
        
        // Initialize temporary arrays
        std::fill(qimp_temp.begin(), qimp_temp.end(), 0.0f);
        std::fill(qimpm_temp.begin(), qimpm_temp.end(), 0.0f);
        std::fill(avimp_temp.begin(), avimp_temp.end(), 0.0f);
        std::fill(sqsd_temp.begin(), sqsd_temp.end(), 0.0f);
        
        // Compute variable importance for this tree
        // Create temporary arrays for non-const parameters
        std::vector<integer_t> temp_jvr(config_.nsample, 0);  // jvr(nsample) in Fortran - FIXED!
        std::vector<integer_t> temp_nodexvr(config_.nsample, 0);  // nodexvr(nsample) in Fortran - FIXED!
        std::vector<integer_t> temp_joob(config_.nsample, 0);  // joob(nsample) in Fortran
        std::vector<integer_t> temp_pjoob(config_.nsample, 0);  // pjoob(nsample) in Fortran
        std::vector<integer_t> temp_iv(config_.mdim, 0);  // iv(mdim) in Fortran
        
        // Set impn based on compute_local_importance flag
        // impn = 1 means compute local importance, impn = 0 means skip local importance
        integer_t impn = config_.compute_local_importance ? 1 : 0;
        
        // Dispatch to appropriate importance calculation based on task type
        if (config_.task_type == TaskType::REGRESSION) {
            // REGRESSION NOT SUPPORTED IN THIS RELEASE
            throw std::runtime_error("Regression variable importance not implemented in this release. Classification only.");
            /*
            // Use regression-specific importance (MSE-based)
            // Use actual number of nodes for this tree (nnode_[tree_id]) instead of maxnode
            integer_t actual_nnode = nnode_[tree_id];
            if (actual_nnode == 0) {
                // Tree not grown yet or empty tree - skip importance computation
                actual_nnode = 1;  // Use minimum value to avoid division by zero
            }
            // cpu_varimp_regression(...) NOT IMPLEMENTED
            */
        } else {
            // Use classification importance for classification and unsupervised
            // Use actual number of nodes for this tree (nnode_[tree_id]) instead of maxnode
            integer_t actual_nnode = nnode_[tree_id];
            if (actual_nnode == 0) {
                // Tree not grown yet or empty tree - skip importance computation
                actual_nnode = 1;  // Use minimum value to avoid division by zero
            }
            cpu_varimp(X_train_.data(), config_.nsample, config_.mdim,  
                   y_data, nin_workspace_.data(), jtr_workspace_.data(), impn,
                   qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                   treemap, nodestatus, xbestsplit, bestvar, nodeclass, actual_nnode,
                   reinterpret_cast<const integer_t*>(cat_.data()),
                   temp_jvr.data(), temp_nodexvr.data(),
                   config_.maxcat, catgoleft, tnodewt, nodextr_workspace_.data());
        }
        
        // Accumulate into global importance arrays
        for (integer_t i = 0; i < config_.nsample; i++) {
            qimp_[i] += qimp_temp[i];
        }
        // Only accumulate qimpm_ if local importance was requested
        if (config_.compute_local_importance) {
        for (integer_t i = 0; i < config_.nsample * config_.mdim; i++) {
            qimpm_[i] += qimpm_temp[i];
            }
        }
        for (integer_t i = 0; i < config_.mdim; i++) {
            avimp_[i] += avimp_temp[i];
            sqsd_[i] += sqsd_temp[i];
        }
    }

    // Store RF-GAP data if requested
    if (config_.compute_proximity && config_.use_rfgap) {
        // Ensure storage is allocated - resize to at least tree_id+1 to avoid out-of-bounds
        if (tree_nin_rfgap_.size() <= static_cast<size_t>(tree_id)) {
            // Resize to ntree to avoid repeated reallocations
            size_t new_size = static_cast<size_t>(config_.ntree);
            tree_nin_rfgap_.resize(new_size);
            tree_nodextr_rfgap_.resize(new_size);
        }
        
        // Store bootstrap multiplicities for this tree
        tree_nin_rfgap_[tree_id].resize(config_.nsample);
        std::copy(nin_workspace_.begin(), nin_workspace_.begin() + config_.nsample,
                  tree_nin_rfgap_[tree_id].begin());
        
        // Store terminal nodes for OOB samples (already computed by cpu_testreebag)
        tree_nodextr_rfgap_[tree_id].resize(config_.nsample, -1);
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] == 0) {
                // OOB sample - terminal node already computed
                tree_nodextr_rfgap_[tree_id][n] = nodextr_workspace_[n];
            }
        }
        
        // Compute terminal nodes for in-bag samples
        // We need to run samples through the tree to get their terminal nodes
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {
                // In-bag sample - compute terminal node
                integer_t terminal_node = find_terminal_node(
                    X_train_.data() + n * config_.mdim,
                    nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                tree_nodextr_rfgap_[tree_id][n] = terminal_node;
            }
        }
    }
    
    // Proximity computation if requested (standard proximity, not RF-GAP)
    // NOTE: proximity_matrix_ should be initialized ONCE before all trees, not per tree
    // This function is called per tree, so we should NOT resize/reset here
    // Only allocate if empty (first tree), otherwise accumulate across trees
    if (config_.compute_proximity && !config_.use_rfgap) {
        // CRITICAL: Only initialize on first tree (when empty), then accumulate across trees
        if (proximity_matrix_.empty()) {
            proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            proximity_matrix_.reserve(static_cast<size_t>(config_.nsample) * config_.nsample);
        }
        // If not empty, proximity_matrix_ already exists and we'll accumulate into it
        
        std::vector<integer_t> nod(config_.maxnode);  // nod(nnode) in Fortran
        std::vector<integer_t> ncount(config_.nsample);  // ncount(nsample) in Fortran
        std::vector<integer_t> ncn(config_.nsample);  // ncn(nsample) in Fortran
        std::vector<integer_t> nodexb(config_.nsample);  // nodexb(nsample) in Fortran
        // CRITICAL: Allocate ndbegin with size maxnode+1 to prevent out-of-bounds access
        // The proximity code accesses ndbegin[k+1], so we need at least maxnode+1 elements
        // But we'll use nterm+1 where nterm <= maxnode, so maxnode+1 is safe
        std::vector<integer_t> ndbegin(config_.maxnode + 1);  // ndbegin(nterm+1) in Fortran - FIXED!
        std::vector<integer_t> npcase(config_.nsample);  // npcase(nsample) in Fortran - FIXED!

        // CRITICAL: cpu_testreebag only sets nodextr_workspace_ for OOB samples
        // For standard proximity, we need terminal nodes for ALL samples (both OOB and in-bag)
        // Compute terminal nodes for in-bag samples
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {
                // In-bag sample - compute terminal node by running it through the tree
                integer_t terminal_node = find_terminal_node(
                    X_train_.data() + n * config_.mdim,
                    nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                nodextr_workspace_[n] = terminal_node;
            }
        }

        cpu_proximity(nodestatus, nodextr_workspace_.data(), nin_workspace_.data(), config_.nsample,
                 nnode_[tree_id], proximity_matrix_.data(), nod.data(),
                 ncount.data(), ncn.data(), nodexb.data(), ndbegin.data(),
                 npcase.data());
    }
}

void RandomForest::fit_batch_gpu(const real_t* X, const void* y,
                                   const real_t* sample_weight, integer_t batch_size) {
    // CRITICAL: Before fit_batch_gpu, ensure CUDA context is ready (handles context between Jupyter cells)
    // This is especially important for GPU sequential mode (batch_size=1)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // std::cout << "DEBUG: fit_batch_gpu ENTRY - nsample=" << config_.nsample 
    //           << ", ntree=" << config_.ntree 
    //           << ", compute_proximity=" << config_.compute_proximity
    //           << ", use_qlora=" << config_.use_qlora << std::endl;
    // std::cout.flush();
    
    // Debug: Check y data
    // if (y != nullptr) {
    //     const integer_t* y_data = static_cast<const integer_t*>(y);
    //     std::cout << "C++ fit_batch_gpu: y data first " << std::min(10, (int)config_.nsample) << ": ";
    //     for (int i = 0; i < std::min(10, (int)config_.nsample); i++) {
    //         std::cout << y_data[i] << " ";
    //     }
        // std::cout << std::endl;
    //     if (config_.nsample > 50) {
    //         std::cout << "C++ fit_batch_gpu: y data samples 50-59: ";
    //         for (int i = 50; i < std::min(60, (int)config_.nsample); i++) {
    //             std::cout << y_data[i] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // } else {
    //     std::cout << "C++ fit_batch_gpu: y data is nullptr" << std::endl;
    // }
    
    // std::cout << "Using GPU batch mode with batch size: " << batch_size << "\n";
    
    // CRITICAL: Initialize variable importance array ONCE at the start of fit_batch_gpu
    // Do NOT reset it between batches - it needs to accumulate across all batches
    // This initialization happens once per call to fit_batch_gpu (which is called once per fit)
    if (config_.compute_importance) {
        // Only initialize if not already initialized (avimp_ should be empty or zero at first call)
        if (avimp_.empty() || std::all_of(avimp_.begin(), avimp_.end(), [](real_t v) { return v == 0.0f; })) {
            std::fill(avimp_.begin(), avimp_.end(), 0.0f);
        }
        // Otherwise, avimp_ already has values from previous batches - keep accumulating
    }

    // Process trees in batches
    // std::cout << "DEBUG: fit_batch_gpu START - ntree=" << config_.ntree 
    //           << ", batch_size=" << batch_size << std::endl;
    // std::cout.flush();
    
    for (integer_t batch_start = 0; batch_start < config_.ntree; batch_start += batch_size) {
        integer_t batch_end = std::min(batch_start + batch_size, config_.ntree);
        integer_t num_trees_batch = batch_end - batch_start;

        // std::cout << "[FIT_BATCH] LOOP ITERATION START - batch_start=" << batch_start 
        //           << ", batch_end=" << batch_end << ", num_trees_batch=" << num_trees_batch << std::endl;
        // std::cout.flush();
        
        // std::cout << "DEBUG: fit_batch_gpu LOOP START - batch_start=" << batch_start 
        //           << ", batch_end=" << batch_end << ", num_trees_batch=" << num_trees_batch << std::endl;
        // std::cout.flush();
        
        // std::cout << "  Processing trees " << batch_start << "-" << (batch_end - 1) << "...\n";
        
        // Resize workspace arrays for batch processing
        // CRITICAL: For batch processing, we DON'T resize the main tree arrays (nodestatus_, bestvar_, etc.)
        // because they need to persist across batches. Only resize per-sample workspace arrays.
        amat_workspace_.resize(num_trees_batch * config_.mdim * config_.nsample);
        jinbag_workspace_.resize(num_trees_batch * config_.nsample);
        // DON'T resize: tnodewt_, xbestsplit_, nodestatus_, bestvar_, treemap_, catgoleft_, nodeclass_, nnode_
        // These are persistent across batches and were sized in initialize_arrays()!
        jtr_workspace_.resize(num_trees_batch * config_.nsample);
        nodextr_workspace_.resize(num_trees_batch * config_.nsample);
        idmove_workspace_.resize(num_trees_batch * config_.nsample);
        igoleft_workspace_.resize(num_trees_batch * config_.maxcat);
        igl_workspace_.resize(num_trees_batch * config_.maxcat);
        nodestart_workspace_.resize(num_trees_batch * config_.maxnode);
        nodestop_workspace_.resize(num_trees_batch * config_.maxnode);
        itemp_workspace_.resize(num_trees_batch * config_.nsample);
        itmp_workspace_.resize(num_trees_batch * config_.maxcat);
        icat_workspace_.resize(num_trees_batch * config_.maxcat);
        tcat_workspace_.resize(num_trees_batch * config_.maxcat);
        classpop_workspace_.resize(num_trees_batch * config_.nclass * config_.maxnode);
        wr_workspace_.resize(num_trees_batch * config_.nsample);
        wl_workspace_.resize(num_trees_batch * config_.nsample);
        tclasscat_workspace_.resize(num_trees_batch * config_.nclass * config_.maxcat);
        tclasspop_workspace_.resize(num_trees_batch * config_.nclass);
        tmpclass_workspace_.resize(num_trees_batch * config_.nclass);
        
        // Prepare seeds for this batch
        std::vector<integer_t> batch_seeds(num_trees_batch);
        for (integer_t i = 0; i < num_trees_batch; ++i) {
            batch_seeds[i] = config_.iseed + batch_start + i;
        }
        
        // Allocate batch-specific nin workspace (needed for bootstrap results from all trees)
        std::vector<integer_t> nin_batch(num_trees_batch * config_.nsample);
        
        // std::cout << "DEBUG: fit_batch_gpu - About to set proximity_ptr. nsample=" << config_.nsample 
        //           << ", compute_proximity=" << config_.compute_proximity 
        //           << ", use_qlora=" << config_.use_qlora << std::endl;
        // std::cout.flush();
        
        // Cast y to integer_t* for GPU function
        const integer_t* y_data = static_cast<const integer_t*>(y);
        
        // For low-rank mode, DO NOT allocate full proximity matrix (80GB for 100k samples!)
        // Keep proximity in low-rank form (~40MB) and reconstruct on-demand only if needed
        dp_t* proximity_ptr = nullptr;
        bool use_lowrank = false;
        bool use_upper_triangle = false;
        if (config_.compute_proximity && !config_.use_rfgap) {
            // Only compute standard proximity if RF-GAP is not enabled
            // MODIFIED: Always use low-rank mode on GPU, regardless of dataset size
            // This enables low-rank factors for all datasets when using GPU
            // NOTE: Low-rank mode is GPU-only (requires CUDA kernels, cuBLAS, cuSolver)
            // CPU mode will never use low-rank - it always computes full proximity matrix
            // Check if we're actually using GPU (use_gpu flag)
            bool actually_using_gpu = config_.use_gpu;
            if (actually_using_gpu && config_.use_qlora) {
                use_lowrank = true;  // Force low-rank for all GPU datasets
                use_upper_triangle = true;  // Default to true for memory efficiency
            } else {
                // CPU mode or GPU without qlora: use full matrix (low-rank not available on CPU)
                use_lowrank = false;  // CPU cannot use low-rank (CUDA-only feature)
                use_upper_triangle = true;  // Default to true for memory efficiency
            }
            
            // std::cout << "DEBUG: fit_batch_gpu - use_lowrank=" << use_lowrank 
            //           << ", use_upper_triangle=" << use_upper_triangle << std::endl;
            // std::cout.flush();
            
            if (use_lowrank && use_upper_triangle) {
                // For low-rank mode, DO NOT allocate full matrix - keep in low-rank form!
                // This saves ~80GB of memory. Reconstruction can be done on-demand if needed.
                proximity_ptr = nullptr;  // Pass nullptr to indicate low-rank mode
                // std::cout << "DEBUG: fit_batch_gpu - Set proximity_ptr to nullptr for low-rank mode" << std::endl;
                // std::cout.flush();
            } else {
                // Traditional mode: allocate full matrix (for smaller datasets)
                if (proximity_matrix_.empty()) {
                    // std::cout << "DEBUG: fit_batch_gpu - Allocating proximity_matrix_ (nsample=" 
                    //           << config_.nsample << ")" << std::endl;
                    // std::cout.flush();
                    proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
                    // std::cout << "DEBUG: fit_batch_gpu - Allocated proximity_matrix_ size=" 
                    //           << proximity_matrix_.size() << std::endl;
                    // std::cout.flush();
                }
                proximity_ptr = proximity_matrix_.data();
                // std::cout << "DEBUG: fit_batch_gpu - Set proximity_ptr to proximity_matrix_.data()" << std::endl;
                // std::cout.flush();
            }
        } else if (config_.compute_proximity && config_.use_rfgap) {
            // RF-GAP mode: don't compute standard proximity during GPU batching
            // We'll compute RF-GAP proximity after all batches complete
            proximity_ptr = nullptr;
            use_lowrank = false;
            use_upper_triangle = false;
        }
        
        // std::cout << "DEBUG: fit_batch_gpu - proximity_ptr=" << (proximity_ptr != nullptr ? "non-null" : "null") << std::endl;
        // std::cout.flush();
        
        // Copy config values to global config for GPU code BEFORE calling gpu_growtree_batch
        // Set global config values that GPU code needs (before first batch)
        // CRITICAL: Always set use_casewise, not just on first batch
        // This ensures proximity kernels have the correct flag
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (batch_start == 0) {
            // CRITICAL: compute_proximity overrides use_qlora - if proximity is disabled, disable qlora too
            // This prevents low-rank proximity initialization when proximity is not requested
            rf::g_config.use_qlora = config_.compute_proximity ? config_.use_qlora : false;
            rf::g_config.quant_mode = config_.quant_mode;
            // Only compute standard proximity in GPU if RF-GAP is not enabled
            rf::g_config.iprox = (config_.compute_proximity && !config_.use_rfgap) ? 1 : 0;
            rf::g_config.use_rfgap = config_.use_rfgap;
            rf::g_config.lowrank_rank = config_.lowrank_rank;
            // Set impn flag for GPU importance computation (1 = compute local importance, 0 = skip)
            rf::g_config.impn = config_.compute_local_importance ? 1 : 0;
            // std::cerr << "[FIT_BATCH] Set g_config - use_casewise=" << rf::g_config.use_casewise 
            //           << ", use_qlora=" << rf::g_config.use_qlora 
            //           << ", iprox=" << rf::g_config.iprox << std::endl;  // Commented to avoid stream conflicts with Python progress bars
            // std::cerr.flush();
        } else {
            // Ensure use_casewise is set for subsequent batches too
            rf::g_config.use_casewise = config_.use_casewise;
        }
        
        // std::cout << "DEBUG: About to call gpu_growtree_batch - proximity_ptr=" 
        //           << (proximity_ptr != nullptr ? "non-null" : "null") << std::endl;
        // std::cout.flush();
        
        // std::cout << "[FIT_BATCH] About to call gpu_growtree_batch:" << std::endl;
        // std::cout << "  proximity_ptr=" << (proximity_ptr != nullptr ? "non-null" : "null") << std::endl;
        // std::cout << "  avimp=" << (config_.compute_importance ? "enabled" : "disabled") << std::endl;
        // std::cout << "  use_lowrank=" << use_lowrank << ", use_upper_triangle=" << use_upper_triangle << std::endl;
        // std::cout << "  lowrank_ptr=" << ((use_lowrank && use_upper_triangle) ? "will-pass" : "nullptr") << std::endl;
        // std::cout.flush();
        
        // CRITICAL: For low-rank mode (use_qlora=True), we MUST use low-rank mode (no temp buffer)
        // If proximity_ptr is not nullptr and use_qlora is enabled, this would trigger temp buffer allocation in GPU code
        if (proximity_ptr != nullptr && config_.use_qlora) {
            // std::cerr << "ERROR: proximity_ptr is non-null for low-rank mode (use_qlora=True). This would trigger huge temp buffer allocation. Forcing nullptr." << std::endl;  // Commented to avoid stream conflicts
            proximity_ptr = nullptr;  // Force nullptr to prevent temp buffer allocation
        }
        
        gpu_growtree_batch(
            num_trees_batch, X, sample_weight, y_data,
            static_cast<integer_t>(config_.task_type),
            cat_.data(), ties_.data(), nin_batch.data(),
            (config_.task_type == TaskType::REGRESSION) ? y_train_regression_.data() : nullptr,
            config_.mdim, config_.nsample, config_.nclass, config_.maxnode,
            config_.minndsize, config_.mtry, config_.nsample, config_.maxcat,
            batch_seeds.data(), config_.ncsplit, config_.ncmax,
            amat_workspace_.data(), jinbag_workspace_.data(), 
            // CRITICAL: Pass offset pointers for persistent arrays (they contain ALL trees)
            tnodewt_.data() + batch_start * config_.maxnode,
            xbestsplit_.data() + batch_start * config_.maxnode,
            nodestatus_.data() + batch_start * config_.maxnode,
            bestvar_.data() + batch_start * config_.maxnode,
            treemap_.data() + batch_start * 2 * config_.maxnode,
            catgoleft_.data() + batch_start * config_.maxcat * config_.maxnode,
            nodeclass_.data() + batch_start * config_.maxnode,
            nnode_.data() + batch_start,
            jtr_workspace_.data(), nodextr_workspace_.data(),
            idmove_workspace_.data(), igoleft_workspace_.data(), igl_workspace_.data(),
            nodestart_workspace_.data(), nodestop_workspace_.data(), itemp_workspace_.data(), itmp_workspace_.data(),
            icat_workspace_.data(), tcat_workspace_.data(), classpop_workspace_.data(), wr_workspace_.data(),
            wl_workspace_.data(), tclasscat_workspace_.data(), tclasspop_workspace_.data(), tmpclass_workspace_.data(),
            q_.data(), nout_.data(),             config_.compute_importance ? avimp_.data() : nullptr,
            (config_.compute_importance || config_.compute_local_importance) ? qimp_.data() : nullptr,
            config_.compute_local_importance ? qimpm_.data() : nullptr,  // Only pass qimpm if local importance requested
            proximity_ptr,  // Pass proximity matrix pointer (allocated above for low-rank mode)
            (config_.task_type == TaskType::REGRESSION) ? oob_predictions_.data() : nullptr,  // For regression: accumulate OOB predictions
            (use_lowrank && use_upper_triangle) ? &lowrank_proximity_ptr_ : nullptr  // Store LowRankProximityMatrix pointer if low-rank mode
        );
        
        // std::cout << "[FIT_BATCH] gpu_growtree_batch RETURNED for batch " << batch_start << "-" << (batch_end-1) << std::endl;
        // std::cout.flush();
        
        // Sync after batch to ensure GPU ops complete before next batch
        // Use stream sync for better Jupyter compatibility and performance
        #ifdef CUDA_FOUND
        cudaStreamSynchronize(0);
        #endif
        
        // Batch completed successfully - no debug print to avoid potential issues
        
        // Store RF-GAP data if requested (after batch completes, simple memory copy - doesn't slow down GPU)
        if (config_.compute_proximity && config_.use_rfgap) {
            // Ensure storage is allocated - resize to ntree to avoid repeated reallocations
            if (tree_nin_rfgap_.size() < static_cast<size_t>(config_.ntree)) {
                size_t new_size = static_cast<size_t>(config_.ntree);
                tree_nin_rfgap_.resize(new_size);
                tree_nodextr_rfgap_.resize(new_size);
            }
            
            // Copy nin and nodextr for each tree in this batch (already computed by GPU)
            for (integer_t i = 0; i < num_trees_batch; ++i) {
                integer_t tree_id = batch_start + i;
                integer_t nin_offset = i * config_.nsample;
                integer_t nodextr_offset = i * config_.nsample;
                
                // Store bootstrap multiplicities
                tree_nin_rfgap_[tree_id].resize(config_.nsample);
                std::copy(nin_batch.data() + nin_offset,
                         nin_batch.data() + nin_offset + config_.nsample,
                         tree_nin_rfgap_[tree_id].begin());
                
                // Store terminal nodes (OOB samples already have terminal nodes, in-bag will be computed later)
                tree_nodextr_rfgap_[tree_id].resize(config_.nsample, -1);
                std::copy(nodextr_workspace_.data() + nodextr_offset,
                         nodextr_workspace_.data() + nodextr_offset + config_.nsample,
                         tree_nodextr_rfgap_[tree_id].begin());
            }
        }
        
        // Update progress for GPU batch processing (callback)
        if (progress_callback_) {
            progress_callback_(batch_end, config_.ntree);
        }
    }
    
    // After all batches complete, compute terminal nodes for in-bag samples (for RF-GAP)
    // This is done in parallel on CPU and doesn't slow down GPU batching
    // CRITICAL: Store config values in local variables to avoid accessing potentially corrupted config_
    bool compute_proximity_local = config_.compute_proximity;
    bool use_rfgap_local = config_.use_rfgap;
    
    if (compute_proximity_local && use_rfgap_local) {
        // std::cout << "[FIT_BATCH] Computing RF-GAP terminal nodes..." << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        // std::cout.flush();
        // Compute terminal nodes for in-bag samples in parallel
#ifdef _OPENMP
        if (g_config.n_threads_cpu > 0) {
            omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
        }
#pragma omp parallel for
#endif
        for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
            // Get tree-specific data
            integer_t tree_offset = tree_id * config_.maxnode;
            integer_t* nodestatus = nodestatus_.data() + tree_offset;
            integer_t* bestvar = bestvar_.data() + tree_offset;
            real_t* xbestsplit = xbestsplit_.data() + tree_offset;
            integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
            integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
            
            // Compute terminal nodes for in-bag samples
            for (integer_t n = 0; n < config_.nsample; ++n) {
                if (tree_nin_rfgap_[tree_id][n] > 0) {
                    // In-bag sample - compute terminal node
                    integer_t terminal_node = find_terminal_node(
                        X_train_.data() + n * config_.mdim,
                        nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                    tree_nodextr_rfgap_[tree_id][n] = terminal_node;
                }
            }
        }
    }
    
    // Ensure all CUDA operations complete before accessing results
    // CRITICAL: For Jupyter compatibility, skip explicit CUDA sync here
    // The GPU code already synchronizes before returning from each batch
    // Additional sync can cause crashes if CUDA context is in an invalid state
    // This matches the pattern from commit 6a9a52f which avoids aggressive syncs
    // CRITICAL: Store use_gpu in local variable before accessing to avoid potential corruption
    bool use_gpu_local = config_.use_gpu;
    if (use_gpu_local) {
        // Just clear any pending CUDA errors - don't sync (GPU code already synced)
        // This prevents crashes when CUDA context is in an invalid state
        #ifdef CUDA_FOUND
        cudaGetLastError();  // Clear any pending errors
        #endif
    }
    
    // CRITICAL: Add memory barrier before return to ensure all writes are visible
    // This helps prevent issues with stack corruption or invalid memory access during return
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // Note: Variable importance computation is now handled within gpu_growtree_batch
    // to ensure OOB predictions are available for each tree individually
    
    // NOTE: finalize_training() is called in fit_classification() after compute_oob_predictions()
    // Match repo version - don't call it here to avoid double call
}

void RandomForest::finalize_training() {
    // Normalize OOB votes
    // Match repo version - no safety checks, just normalize
    for (integer_t n = 0; n < config_.nsample; ++n) {
        if (nout_[n] > 0) {
            for (integer_t j = 0; j < config_.nclass; ++j) {
                q_[n * config_.nclass + j] /= static_cast<real_t>(nout_[n]);
            }
        }
    }

    // Compute OOB predictions and error based on task type
    // Note: OOB error is computed in compute_classification_error() which is called
    // in fit_classification() after finalize_training(). We don't need to recompute here.
    // The oob_error_ is already set by compute_classification_error().
    if (config_.task_type == TaskType::CLASSIFICATION) {
        // OOB error already computed in compute_classification_error() - nothing to do here
    } else if (config_.task_type == TaskType::REGRESSION) {
        // For regression, compute MSE instead of classification error
        compute_regression_mse();
    }
    
    // Finalize proximity if computed (only once at the end)
    if (config_.compute_proximity && !proximity_normalized_) {
        if (config_.use_rfgap) {
            // RF-GAP proximity mode
            #ifdef CUDA_FOUND
            // Check if RF-GAP was already computed with QLoRA during GPU batching
            if (config_.use_gpu && config_.use_qlora && lowrank_proximity_ptr_ != nullptr) {
                // RF-GAP was already computed incrementally in low-rank form during GPU batching
                // No need to compute full matrix - low-rank factors are already available
                // Set OOB counts for RF-GAP normalization in MDS
                rf::cuda::LowRankProximityMatrix* lowrank_prox = 
                    static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
                lowrank_prox->set_oob_counts_for_rfgap(nout_.data());
                // Just mark as normalized
                proximity_normalized_ = true;
            } else {
                // RF-GAP was not computed with QLoRA during batching, compute full matrix now
                if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                    tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                    throw std::runtime_error("RF-GAP proximity: per-tree data not available. "
                                            "Ensure all trees were grown with use_rfgap=True.");
                }
                
                // Ensure proximity matrix is allocated
                if (proximity_matrix_.empty()) {
                    proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
                }
                
                // Compute RF-GAP proximity (full matrix - expensive but necessary)
                // Use GPU if available for better performance
                if (config_.use_gpu && rf::cuda::cuda_is_available()) {
                    // Flatten per-tree data for GPU
                    std::vector<integer_t> tree_nin_flat(config_.ntree * config_.nsample);
                    std::vector<integer_t> tree_nodextr_flat(config_.ntree * config_.nsample);
                    
                    for (integer_t t = 0; t < config_.ntree; ++t) {
                        for (integer_t s = 0; s < config_.nsample; ++s) {
                            tree_nin_flat[t * config_.nsample + s] = tree_nin_rfgap_[t][s];
                            tree_nodextr_flat[t * config_.nsample + s] = tree_nodextr_rfgap_[t][s];
                        }
                    }
                    
                    // Use GPU RF-GAP for better performance
                    rf::gpu_proximity_rfgap(config_.ntree, config_.nsample,
                                           tree_nin_flat.data(), tree_nodextr_flat.data(),
                                           proximity_matrix_.data());
                } else {
                    // Fall back to CPU RF-GAP
                    cpu_proximity_rfgap(config_.ntree, config_.nsample,
                                       tree_nin_rfgap_, tree_nodextr_rfgap_,
                                       proximity_matrix_.data());
                }
                
                // If QLoRA is enabled, convert RF-GAP full matrix to low-rank factors for memory efficiency
                // This allows GPU MDS to work with RF-GAP proximity
                if (config_.use_gpu && config_.use_qlora) {
                    // Create low-rank proximity matrix and convert RF-GAP to factors
                    rf::cuda::LowRankProximityMatrix* lowrank_rfgap = 
                        new rf::cuda::LowRankProximityMatrix(
                            config_.nsample, 
                            config_.lowrank_rank,
                            static_cast<rf::cuda::QuantizationLevel>(config_.quant_mode),
                            config_.lowrank_rank * 2  // max_rank
                        );
                    
                    if (!lowrank_rfgap->initialize()) {
                        delete lowrank_rfgap;
                        throw std::runtime_error("Failed to initialize low-rank proximity matrix for RF-GAP");
                    }
                    
                    // Convert full RF-GAP matrix to low-rank factors
                    // This uses SVD to approximate P ≈ A × B'
                    lowrank_rfgap->add_tree_contribution(proximity_matrix_.data(), config_.nsample, false);
                    
                    // Store low-rank pointer and free full matrix to save memory
                    // Use exception-safe deletion to prevent double-free
                    if (lowrank_proximity_ptr_ != nullptr) {
                        try {
                            delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
                        } catch (...) {
                            // Ignore errors during deletion (may be called multiple times)
                        }
                        lowrank_proximity_ptr_ = nullptr;
                    }
                    lowrank_proximity_ptr_ = lowrank_rfgap;
                    
                    // Set OOB counts for RF-GAP normalization in MDS
                    lowrank_rfgap->set_oob_counts_for_rfgap(nout_.data());
                    
                    // Free full matrix to save memory (can be reconstructed from factors if needed)
                    proximity_matrix_.clear();
                    proximity_matrix_.shrink_to_fit();
                }
                
                proximity_normalized_ = true;
            }
            #else
            // CUDA not available - use CPU RF-GAP
            if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                throw std::runtime_error("RF-GAP proximity: per-tree data not available. "
                                        "Ensure all trees were grown with use_rfgap=True.");
            }
            
            // Ensure proximity matrix is allocated
            if (proximity_matrix_.empty()) {
                proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            }
            
            // Ensure proximity matrix is properly allocated before RF-GAP computation
            // This prevents memory corruption if the vector was moved or resized
            if (proximity_matrix_.size() != static_cast<size_t>(config_.nsample) * config_.nsample) {
                proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            }
            
            // Verify RF-GAP data vectors are properly sized
            if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                throw std::runtime_error("RF-GAP proximity: per-tree data size mismatch. "
                                        "Expected " + std::to_string(config_.ntree) + " trees, "
                                        "got tree_nin_rfgap_.size()=" + std::to_string(tree_nin_rfgap_.size()) + 
                                        ", tree_nodextr_rfgap_.size()=" + std::to_string(tree_nodextr_rfgap_.size()));
            }
            
            // CRITICAL: Ensure vectors are stable before passing to cpu_proximity_rfgap
            // Reserve capacity to prevent reallocation during function call
            // This prevents memory corruption if vectors are moved
            tree_nin_rfgap_.reserve(config_.ntree);
            tree_nodextr_rfgap_.reserve(config_.ntree);
            for (size_t t = 0; t < static_cast<size_t>(config_.ntree); ++t) {
                if (t < tree_nin_rfgap_.size()) {
                    tree_nin_rfgap_[t].reserve(config_.nsample);
                    tree_nodextr_rfgap_[t].reserve(config_.nsample);
                }
            }
            
            // Ensure proximity matrix is stable - reserve capacity to prevent reallocation
            proximity_matrix_.reserve(static_cast<size_t>(config_.nsample) * config_.nsample);
            
            cpu_proximity_rfgap(config_.ntree, config_.nsample,
                               tree_nin_rfgap_, tree_nodextr_rfgap_,
                               proximity_matrix_.data());
            proximity_normalized_ = true;
            #endif
        } else {
        // For low-rank mode, proximity_matrix_ is empty - don't try to normalize
        if (lowrank_proximity_ptr_ == nullptr && !proximity_matrix_.empty()) {
            // Traditional mode: normalize proximity matrix
            std::vector<dp_t> proxsym(static_cast<size_t>(config_.nsample) * config_.nsample);
            cpu_finishprox(config_.nsample, nout_.data(), proximity_matrix_.data(), proxsym.data());
            proximity_matrix_ = proxsym;  // Use symmetrized version
            proximity_normalized_ = true;
        } else if (lowrank_proximity_ptr_ != nullptr) {
            // Low-rank mode: proximity stored in low-rank factors, no full matrix to normalize
            proximity_normalized_ = true;  // Mark as normalized (no full matrix to normalize)
            // std::cout << "GPU Proximity: Low-rank mode - proximity stored in factors (no full matrix to normalize)" << std::endl;
            }
        }
    }
    
    // For unsupervised, compute outlier scores AFTER proximity is finalized
    if (config_.task_type == TaskType::UNSUPERVISED) {
        if (config_.compute_proximity && proximity_normalized_) {
            // For low-rank mode, outlier scores can't be computed from full matrix
            // Would need to compute from low-rank factors (future enhancement)
            if (lowrank_proximity_ptr_ == nullptr && !proximity_matrix_.empty()) {
                compute_outlier_scores();
            } else if (lowrank_proximity_ptr_ != nullptr) {
                // std::cout << "GPU Proximity: Low-rank mode - skipping outlier scores (requires full matrix)" << std::endl;
            }
        }
    }

    // Finalize feature importance if computed
    if (config_.compute_importance) {
        // Copy accumulated importance values to final feature_importances_ array
        // Note: avimp_ accumulates across all trees - no normalization needed
        // (importance is already normalized per tree by noob in varimp calculation)
        for (integer_t i = 0; i < config_.mdim; i++) {
            feature_importances_[i] = avimp_[i];
        }
        
        // Compute local importance if requested
        if (config_.compute_local_importance) {
            // CLIQUE removed for v1.0 (v2.0 feature)
            if (config_.importance_method == "clique") {
                throw std::runtime_error("CLIQUE importance is not available in v1.0. Use importance_method='local_imp' instead.");
            }
            // Standard local importance (original method)
            localimp(config_.nsample, config_.mdim, config_.ntree, qimp_.data(), qimpm_.data());
        }
    }
    
}

// Task-specific setup methods
void RandomForest::setup_classification_task(const integer_t* y) {
    // For classification, determine number of classes
    std::set<integer_t> unique_classes(y, y + config_.nsample);
    config_.nclass = static_cast<integer_t>(unique_classes.size());
    
    // Store class labels directly (0-based)
    y_train_classification_1based_.resize(config_.nsample);
    for (integer_t i = 0; i < config_.nsample; ++i) {
        y_train_classification_1based_[i] = y[i];  // Keep 0-based
    }
    
    // Ensure mtry is reasonable for classification
    if (config_.mtry == 0) {
        config_.mtry = std::max(1, static_cast<integer_t>(std::sqrt(static_cast<real_t>(config_.mdim))));
    }
}

void RandomForest::setup_regression_task(const real_t* y) {
    // For regression, we use a single "class" but predict continuous values
    config_.nclass = 1;
    
    // Use regression-specific mtry
    config_.mtry = std::max(1, static_cast<integer_t>(config_.mdim / 3));
    
    // Initialize regression-specific arrays
    oob_predictions_.resize(config_.nsample, 0.0f);
    terminal_values_.resize(config_.ntree * config_.maxnode, 0.0f);
}

void RandomForest::setup_unsupervised_task() {
    // For unsupervised, we use proximity-based clustering
    // Use nclass=2 (binary classification style) instead of nclass=1 to avoid issues
    // with tree growing code that expects nclass >= 2 for classification-style splits
    // The synthetic labels are all zeros, which will be treated as class 0
    config_.nclass = 2;  // Changed from 1 to 2 to match classification-style unsupervised mode
    // Don't override compute_proximity - respect user's setting
    // config_.compute_proximity = true;  // REMOVED: Let user control proximity computation
    
    // Set mtry based on unsupervised mode (only if not explicitly set by user, i.e., mtry == 0)
    if (config_.mtry == 0) {
        if (config_.unsupervised_mode == UnsupervisedMode::CLASSIFICATION_STYLE) {
            // Classification-style: use sqrt(mdim)
            config_.mtry = std::max(1, static_cast<integer_t>(std::sqrt(static_cast<real_t>(config_.mdim))));
        } else {
            // Regression-style: use mdim / 3
            config_.mtry = std::max(1, static_cast<integer_t>(config_.mdim / 3));
        }
    }
    
    // Initialize unsupervised-specific arrays
    outlier_scores_.resize(config_.nsample, 0.0f);
    cluster_labels_.resize(config_.nsample, 0);
}

// Unified predict method that dispatches based on task type
void RandomForest::predict(const real_t* X, integer_t nsamples, void* predictions) {
    switch (config_.task_type) {
        case TaskType::CLASSIFICATION:
            predict_classification(X, nsamples, static_cast<integer_t*>(predictions));
            break;
        case TaskType::REGRESSION:
            predict_regression(X, nsamples, static_cast<real_t*>(predictions));
            break;
        case TaskType::UNSUPERVISED:
            predict_unsupervised(X, nsamples, static_cast<integer_t*>(predictions));
            break;
    }
}

// Classification prediction - Use OOB for training data, tree traversal for new data
void RandomForest::predict_classification(const real_t* X, integer_t nsamples, integer_t* predictions) {
    // CRITICAL: Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if we're predicting on training data (same size and potentially same data)
    bool is_training_data = (nsamples == config_.nsample);
    
    if (is_training_data) {
        // Use stored OOB predictions (calculated during fit())
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            if (sample < oob_class_predictions_.size()) {
                predictions[sample] = oob_class_predictions_[sample];
            } else {
                predictions[sample] = 0; // Default to class 0
            }
        }
    } else {
        // Use tree traversal for new/held-out data
        std::vector<integer_t> votes(config_.nclass, 0);
        
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            // Reset votes for this sample
            std::fill(votes.begin(), votes.end(), 0);
            
            // Get sample features
            const real_t* sample_features = X + sample * config_.mdim;
            
            // Vote from each tree
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                integer_t* nodestatus = nodestatus_.data() + tree_id * config_.maxnode;
                integer_t* bestvar = bestvar_.data() + tree_id * config_.maxnode;
                real_t* xbestsplit = xbestsplit_.data() + tree_id * config_.maxnode;
                integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
                integer_t* nodeclass = nodeclass_.data() + tree_id * config_.maxnode;
                integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
                
                // Traverse tree starting from root (node 0) - EXACT match to testreebag.f90
                integer_t kt = 0; // Current node index (0-based, matching Fortran kt=1 converted to 0-based)
                
                // Loop through all possible nodes (matching original algorithm exactly)
                for (integer_t k = 0; k < config_.maxnode; ++k) {
                    if (kt >= config_.maxnode) {
                        // Safety check - node index out of bounds
                        votes[0]++; // Default to class 0
                        break;
                    }
                    
                    if (nodestatus[kt] == -1) {
                        // Terminal node - get class prediction (matching testreebag.f90 line 29)
                        integer_t predicted_class = nodeclass[kt];
                        if (predicted_class >= 0 && predicted_class < config_.nclass) {
                            votes[predicted_class]++; // Use 0-based indexing directly
                        } else {
                            votes[0]++; // Default to class 0 for invalid predictions
                        }
                        break;
                    }
                    
                    // Split node - traverse based on split (matching testreebag.f90 lines 33-48)
                    integer_t m = bestvar[kt]; // Split variable
                    
                    if (m >= 0 && m < config_.mdim) {
                        real_t feature_value = sample_features[m];
                        
                        // Check if variable is quantitative (cat == 1) or categorical
                        if (cat_[m] == 1) {
                            // Quantitative variable (matching testreebag.f90 lines 35-40)
                            if (feature_value <= xbestsplit[kt]) {
                                kt = treemap[kt * 2]; // Left child (0-based indexing)
                            } else {
                                kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                            }
                        } else {
                            // Categorical variable (matching testreebag.f90 lines 42-47)
                            integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                            if (jcat >= 1 && jcat <= cat_[m]) {
                                if (catgoleft[(kt * config_.maxcat) + (jcat - 1)] == 1) {
                                    kt = treemap[kt * 2]; // Left child (0-based indexing)
                                } else {
                                    kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                                }
                            } else {
                                // Invalid category - default to class 0
                                votes[0]++;
                                break;
                            }
                        }
                    } else {
                        // Invalid split variable - default to class 0
                        votes[0]++;
                        break;
                    }
                }
            }
            
            // Find class with most votes (matching predict.f lines 15-28)
            integer_t predicted_class = 0;
            integer_t max_votes = votes[0];
            for (integer_t c = 1; c < config_.nclass; ++c) {
                if (votes[c] > max_votes) {
                    max_votes = votes[c];
                    predicted_class = c;
                }
            }
            
            predictions[sample] = predicted_class;
        }
    }
}

// Regression prediction
void RandomForest::predict_regression(const real_t* X, integer_t nsamples, real_t* predictions) {
    // CRITICAL: Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Implement regression prediction matching original Breiman-Cutler algorithm
    // For regression, prediction is the average of terminal node values from all trees
    
    for (integer_t sample = 0; sample < nsamples; ++sample) {
        real_t sum_predictions = 0.0;
        integer_t tree_count = 0;
        
        // Get sample features
        const real_t* sample_features = X + sample * config_.mdim;
        
        // Sum predictions from each tree
        for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
            // Get pointers to this tree's storage
            integer_t* nodestatus = nodestatus_.data() + tree_id * config_.maxnode;
            integer_t* bestvar = bestvar_.data() + tree_id * config_.maxnode;
            real_t* xbestsplit = xbestsplit_.data() + tree_id * config_.maxnode;
            integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
            real_t* tnodewt = tnodewt_.data() + tree_id * config_.maxnode;
            integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
            
            // Traverse tree starting from root (node 0) - matching testreebag.f
            integer_t current_node = 0;
            
            // Loop through all possible nodes (matching original algorithm)
            for (integer_t k = 0; k < config_.maxnode; ++k) {
                if (nodestatus[current_node] == -1) {
                    // Terminal node - get prediction value (tnodewt for regression)
                    sum_predictions += tnodewt[current_node];
                    tree_count++;
                    break;
                }
                
                // Split node - traverse based on split (matching testreebag.f lines 31-46)
                integer_t split_var = bestvar[current_node];
                real_t split_value = xbestsplit[current_node];
                
                if (split_var >= 0 && split_var < config_.mdim) {
                    real_t feature_value = sample_features[split_var];
                    
                    // Check if variable is quantitative (cat == 1) or categorical
                    if (cat_[split_var] == 1) {
                        // Quantitative variable (matching testreebag.f lines 33-38)
                        if (feature_value <= split_value) {
                            current_node = treemap[current_node * 2]; // Left child
                        } else {
                            current_node = treemap[current_node * 2 + 1]; // Right child
                        }
                    } else {
                        // Categorical variable (matching testreebag.f lines 40-45)
                        integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                        if (jcat >= 1 && jcat <= cat_[split_var]) {
                            if (catgoleft[(current_node * config_.maxcat) + (jcat - 1)] == 1) {
                                current_node = treemap[current_node * 2]; // Left child
                            } else {
                                current_node = treemap[current_node * 2 + 1]; // Right child
                            }
                        } else {
                            // Invalid category - use root terminal value
                            sum_predictions += tnodewt[0];
                            tree_count++;
                            break;
                        }
                    }
                } else {
                    // Invalid split variable - use root terminal value
                    sum_predictions += tnodewt[0];
                    tree_count++;
                    break;
                }
            }
        }
        
        // Average prediction across all trees
        if (tree_count > 0) {
            predictions[sample] = sum_predictions / static_cast<real_t>(tree_count);
        } else {
            predictions[sample] = 0.0;
        }
    }
}

// Unsupervised prediction (clustering)
void RandomForest::predict_unsupervised(const real_t* X, integer_t nsamples, integer_t* cluster_labels) {
    // CRITICAL: Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Implement unsupervised prediction using proximity-based clustering
    // This is a simplified implementation - in practice, you might want to use
    // more sophisticated clustering algorithms on the proximity matrix
    
    // For now, assign cluster labels based on proximity to training samples
    // This is a basic implementation that could be enhanced
    
    for (integer_t sample = 0; sample < nsamples; ++sample) {
        // Find the training sample with highest proximity
        integer_t best_match = 0;
        real_t max_proximity = 0.0;
        
        // Get sample features
        const real_t* sample_features = X + sample * config_.mdim;
        
        // Compute proximity to each training sample by traversing trees
        for (integer_t train_sample = 0; train_sample < config_.nsample; ++train_sample) {
            real_t proximity_sum = 0.0;
            
            // For each tree, check if both samples end up in same terminal node
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                integer_t* nodestatus = nodestatus_.data() + tree_id * config_.maxnode;
                integer_t* bestvar = bestvar_.data() + tree_id * config_.maxnode;
                real_t* xbestsplit = xbestsplit_.data() + tree_id * config_.maxnode;
                integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
                integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
                
                // Find terminal node for test sample
                integer_t test_node = find_terminal_node(sample_features, nodestatus, bestvar, 
                                                       xbestsplit, treemap, catgoleft);
                
                // Find terminal node for training sample
                const real_t* train_features = X_train_.data() + train_sample * config_.mdim;
                integer_t train_node = find_terminal_node(train_features, nodestatus, bestvar,
                                                        xbestsplit, treemap, catgoleft);
                
                // If both samples end up in same terminal node, add to proximity
                if (test_node == train_node) {
                    proximity_sum += 1.0;
                }
            }
            
            real_t avg_proximity = proximity_sum / static_cast<real_t>(config_.ntree);
            if (avg_proximity > max_proximity) {
                max_proximity = avg_proximity;
                best_match = train_sample;
            }
        }
        
        // Assign cluster label based on best match
        // For simplicity, use the cluster label of the best matching training sample
        if (cluster_labels_.size() > best_match) {
            cluster_labels[sample] = cluster_labels_[best_match];
        } else {
            cluster_labels[sample] = 0; // Default cluster
        }
    }
}

// Helper function to find terminal node for a sample
integer_t RandomForest::find_terminal_node(const real_t* sample_features,
                                          integer_t* nodestatus, integer_t* bestvar,
                                          real_t* xbestsplit, integer_t* treemap,
                                          integer_t* catgoleft) {
    integer_t current_node = 0;
    
    // Loop through all possible nodes (matching original algorithm)
    for (integer_t k = 0; k < config_.maxnode; ++k) {
        if (nodestatus[current_node] == -1) {
            // Terminal node found
            return current_node;
        }
        
        // Split node - traverse based on split
        integer_t split_var = bestvar[current_node];
        real_t split_value = xbestsplit[current_node];
        
        if (split_var >= 0 && split_var < config_.mdim) {
            real_t feature_value = sample_features[split_var];
            
            // Check if variable is quantitative (cat == 1) or categorical
            if (cat_[split_var] == 1) {
                // Quantitative variable
                if (feature_value <= split_value) {
                    current_node = treemap[current_node * 2]; // Left child
                } else {
                    current_node = treemap[current_node * 2 + 1]; // Right child
                }
            } else {
                // Categorical variable
                integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                if (jcat >= 1 && jcat <= cat_[split_var]) {
                    if (catgoleft[(current_node * config_.maxcat) + (jcat - 1)] == 1) {
                        current_node = treemap[current_node * 2]; // Left child
                    } else {
                        current_node = treemap[current_node * 2 + 1]; // Right child
                    }
                } else {
                    // Invalid category - return root node
                    return 0;
                }
            }
        } else {
            // Invalid split variable - return root node
            return 0;
        }
    }
    
    // Should not reach here, but return root node as fallback
    return 0;
}

void RandomForest::compute_oob_predictions() {
    // Compute OOB class predictions and probabilities from the q_ array (raw OOB votes)
    // This must be called BEFORE finalize_training() which normalizes q_
    oob_class_predictions_.resize(config_.nsample);
    oob_probabilities_.resize(config_.nsample * config_.nclass);
    
    for (integer_t sample = 0; sample < config_.nsample; ++sample) {
        if (nout_[sample] > 0) {  // Sample was out-of-bag
            // Find the class with maximum raw OOB votes for this sample
            // In case of ties, prefer the class with the higher index (more conservative)
            integer_t best_class = 0;
            real_t max_votes = q_[sample * config_.nclass];
            real_t total_votes = 0.0;
            
            // Calculate total votes and find best class using raw vote counts
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                integer_t idx = sample * config_.nclass + class_idx;
                if (idx >= 0 && idx < q_.size()) {
                    real_t votes = q_[idx];  // Raw vote count
                    total_votes += votes;
                    // Use >= instead of > to break ties in favor of higher class index
                    // This ensures class 2 gets preference over class 0/1 when votes are equal
                    if (votes >= max_votes) {
                        max_votes = votes;
                        best_class = class_idx;
                    }
                }
            }
            
            oob_class_predictions_[sample] = best_class;
            
            // Calculate probabilities from raw votes
            if (total_votes > 0.0) {
                for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                    integer_t idx = sample * config_.nclass + class_idx;
                    if (idx >= 0 && idx < q_.size()) {
                        oob_probabilities_[sample * config_.nclass + class_idx] = q_[idx] / total_votes;
                    } else {
                        oob_probabilities_[sample * config_.nclass + class_idx] = 1.0 / config_.nclass;
                    }
                }
            } else {
                // No OOB votes - use uniform distribution
                for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                    oob_probabilities_[sample * config_.nclass + class_idx] = 1.0 / config_.nclass;
                }
            }
        } else {
            // Sample was not out-of-bag - no prediction
            oob_class_predictions_[sample] = 0;
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                oob_probabilities_[sample * config_.nclass + class_idx] = 1.0 / config_.nclass;
            }
        }
    }
}

void RandomForest::predict_proba(const real_t* X, integer_t nsamples, real_t* probabilities) {
    // CRITICAL: Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if we're predicting on training data (same size and potentially same data)
    bool is_training_data = (nsamples == config_.nsample);
    
    if (is_training_data) {
        // Use stored OOB probabilities (calculated during fit())
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                integer_t prob_idx = sample * config_.nclass + class_idx;
                if (prob_idx < oob_probabilities_.size()) {
                    probabilities[prob_idx] = oob_probabilities_[prob_idx];
                } else {
                    probabilities[prob_idx] = 1.0 / config_.nclass;
                }
            }
        }
    } else {
        // Use tree traversal for new/held-out data
        std::vector<integer_t> votes(config_.nclass, 0);
        
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            // Reset votes for this sample
            std::fill(votes.begin(), votes.end(), 0);
            
            // Get sample features
            const real_t* sample_features = X + sample * config_.mdim;
            
            // Vote from each tree
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                integer_t* nodestatus = nodestatus_.data() + tree_id * config_.maxnode;
                integer_t* bestvar = bestvar_.data() + tree_id * config_.maxnode;
                real_t* xbestsplit = xbestsplit_.data() + tree_id * config_.maxnode;
                integer_t* treemap = treemap_.data() + tree_id * 2 * config_.maxnode;
                integer_t* nodeclass = nodeclass_.data() + tree_id * config_.maxnode;
                integer_t* catgoleft = catgoleft_.data() + tree_id * config_.maxcat * config_.maxnode;
                
                // Traverse tree starting from root (node 0) - EXACT match to testreebag.f90
                integer_t kt = 0; // Current node index (0-based, matching Fortran kt=1 converted to 0-based)
                
                // Loop through all possible nodes (matching original algorithm exactly)
                for (integer_t k = 0; k < config_.maxnode; ++k) {
                    if (kt >= config_.maxnode) {
                        // Safety check - node index out of bounds
                        votes[0]++; // Default to class 0
                        break;
                    }
                    
                    if (nodestatus[kt] == -1) {
                        // Terminal node - get class prediction (matching testreebag.f90 line 29)
                        integer_t predicted_class = nodeclass[kt];
                        if (predicted_class >= 0 && predicted_class < config_.nclass) {
                            votes[predicted_class]++; // Use 0-based indexing directly
                        } else {
                            votes[0]++; // Default to class 0 for invalid predictions
                        }
                        break;
                    }
                    
                    // Split node - traverse based on split (matching testreebag.f90 lines 33-48)
                    integer_t m = bestvar[kt]; // Split variable
                    
                    if (m >= 0 && m < config_.mdim) {
                        real_t feature_value = sample_features[m];
                        
                        // Check if variable is quantitative (cat == 1) or categorical
                        if (cat_[m] == 1) {
                            // Quantitative variable (matching testreebag.f90 lines 35-40)
                            if (feature_value <= xbestsplit[kt]) {
                                kt = treemap[kt * 2]; // Left child (0-based indexing)
                            } else {
                                kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                            }
                        } else {
                            // Categorical variable (matching testreebag.f90 lines 42-47)
                            integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                            if (jcat >= 1 && jcat <= cat_[m]) {
                                if (catgoleft[(kt * config_.maxcat) + (jcat - 1)] == 1) {
                                    kt = treemap[kt * 2]; // Left child (0-based indexing)
                                } else {
                                    kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                                }
                            } else {
                                // Invalid category - default to class 0
                                votes[0]++;
                                break;
                            }
                        }
                    } else {
                        // Invalid split variable - default to class 0
                        votes[0]++;
                        break;
                    }
                }
            }
            
            // Convert votes to probabilities (matching predict.f logic)
            real_t total_votes = static_cast<real_t>(config_.ntree);
            for (integer_t c = 0; c < config_.nclass; ++c) {
                probabilities[sample * config_.nclass + c] = static_cast<real_t>(votes[c]) / total_votes;
            }
        }
    }
}

void RandomForest::compute_classification_error() {
    // Compute OOB error for classification using raw vote counts in q_ array
    // This must be called BEFORE finalize_training() which normalizes q_
    // Use the same logic as compute_oob_predictions() to ensure consistency
    
    // CRITICAL: Validate object state before accessing member variables
    if (config_.nsample <= 0 || config_.nclass <= 0) {
        throw std::runtime_error("Invalid config in compute_classification_error");
    }
    if (nout_.size() < static_cast<size_t>(config_.nsample) || 
        q_.size() < static_cast<size_t>(config_.nsample * config_.nclass)) {
        throw std::runtime_error("Invalid member arrays in compute_classification_error");
    }
    
    integer_t correct = 0;
    integer_t total_oob = 0;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (nout_[i] > 0) {  // Sample was out-of-bag
            total_oob++;
            
            // Find predicted class with maximum raw votes (same logic as compute_oob_predictions)
            // In case of ties, prefer the class with the higher index (more conservative)
            integer_t predicted_class = 0;
            real_t max_votes = q_[i * config_.nclass];
            for (integer_t j = 1; j < config_.nclass; ++j) {
                integer_t idx = i * config_.nclass + j;
                if (idx >= 0 && idx < q_.size()) {
                    real_t votes = q_[idx];
                    // Use >= instead of > to break ties in favor of higher class index
                    if (votes >= max_votes) {
                        max_votes = votes;
                        predicted_class = j;
                    }
                }
            }
            
            // Check if prediction is correct
            if (predicted_class == y_train_classification_[i]) {
                correct++;
            }
        }
    }
    
    if (total_oob > 0) {
        oob_error_ = 1.0f - static_cast<real_t>(correct) / static_cast<real_t>(total_oob);
    } else {
        oob_error_ = 0.0f;  // No OOB samples
    }
}

void RandomForest::compute_regression_mse() {
    // Compute OOB MSE for regression
    // oob_predictions_ contains the SUM of predictions across all trees
    // We need to divide by nout_[i] (number of trees that had this sample OOB) to get the average
    real_t sum_squared_error = 0.0f;
    integer_t total_oob = 0;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (nout_[i] > 0) {  // Sample was out-of-bag
            total_oob++;
            // Normalize: divide sum by number of trees to get average prediction
            real_t prediction = oob_predictions_[i] / static_cast<real_t>(nout_[i]);
            real_t error = y_train_regression_[i] - prediction;
            sum_squared_error += error * error;
            // Update oob_predictions_ to store the final average (for get_oob_predictions())
            oob_predictions_[i] = prediction;
        }
    }
    
    if (total_oob > 0) {
        oob_mse_ = sum_squared_error / static_cast<real_t>(total_oob);
        oob_error_ = oob_mse_;  // Also set oob_error_ for consistency with get_oob_error()
    }
}

void RandomForest::compute_outlier_scores() {
    // Compute outlier scores based on proximity matrix
    if (proximity_matrix_.empty()) {
        return;
    }
    
    // Ensure GPU operations are complete before accessing proximity matrix
    // GPU synchronization is handled by GPU code itself
    // Avoid direct CUDA calls in CPU compilation path
    
    outlier_scores_.resize(config_.nsample, 0.0f);
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        real_t prox_sum = 0.0f;
        for (integer_t j = 0; j < config_.nsample; ++j) {
            prox_sum += proximity_matrix_[i * config_.nsample + j] * proximity_matrix_[i * config_.nsample + j];
        }
        if (prox_sum > 0) {
            outlier_scores_[i] = 1.0f / prox_sum;
        }
    }
    
    // Normalize by median
    std::vector<real_t> sorted_scores = outlier_scores_;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    real_t median = sorted_scores[config_.nsample / 2];
    if (median > 0) {
        for (integer_t i = 0; i < config_.nsample; ++i) {
            outlier_scores_[i] /= median;
        }
    }
}

void RandomForest::compute_cluster_labels() {
    // Simple clustering based on proximity matrix
    cluster_labels_.resize(config_.nsample, 0);
    
    if (proximity_matrix_.empty()) {
        return;
    }
    
    // Use a simple threshold-based clustering
    std::vector<bool> assigned(config_.nsample, false);
    integer_t current_cluster = 0;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (!assigned[i]) {
            cluster_labels_[i] = current_cluster;
            assigned[i] = true;
            
            // Assign nearby samples to the same cluster
            for (integer_t j = i + 1; j < config_.nsample; ++j) {
                if (!assigned[j] && proximity_matrix_[i * config_.nsample + j] > 0.5f) {
                    cluster_labels_[j] = current_cluster;
                    assigned[j] = true;
                }
            }
            current_cluster++;
        }
    }
}

void RandomForest::initialize_workspace_arrays() {
    win_workspace_.resize(config_.nsample);
    nin_workspace_.resize(config_.nsample);
    jinbag_workspace_.resize(config_.nsample);
    joobag_workspace_.resize(config_.nsample);
    amat_workspace_.resize(config_.mdim * config_.nsample);
    idmove_workspace_.resize(config_.nsample);
    igoleft_workspace_.resize(config_.maxcat);
    igl_workspace_.resize(config_.maxcat);
    nodestart_workspace_.resize(config_.maxnode);
    nodestop_workspace_.resize(config_.maxnode);
    itemp_workspace_.resize(config_.nsample);
    itmp_workspace_.resize(config_.maxcat);
    icat_workspace_.resize(config_.maxcat);
    tcat_workspace_.resize(config_.maxcat);
    classpop_workspace_.resize(config_.nclass * config_.maxnode);
    wr_workspace_.resize(config_.nclass);
    wl_workspace_.resize(config_.nclass);
    tclasscat_workspace_.resize(config_.nclass * config_.maxcat);
    tclasspop_workspace_.resize(config_.nclass);
    tmpclass_workspace_.resize(config_.nclass);
    jtr_workspace_.resize(config_.nsample);
    nodextr_workspace_.resize(config_.nsample);
}

// Synchronized getters for GPU memory safety
const dp_t* RandomForest::get_proximity_matrix() const {
    // CRITICAL: Before get_proximity, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // For low-rank mode, DO NOT auto-reconstruct - return nullptr
    // Users should call get_lowrank_factors() or compute_mds_3d_from_factors() instead
    // Auto-reconstruction was causing performance issues and memory problems
    if (lowrank_proximity_ptr_ != nullptr) {
        // Low-rank mode active - do not auto-reconstruct full matrix
        // Use get_lowrank_factors() to get A, B factors directly
        // Use compute_mds_3d_from_factors() for MDS visualization
        return nullptr;
    }
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    // Return pointer only if vector is not empty and properly sized
    // This prevents returning invalid pointers if the vector was moved or resized
    if (proximity_matrix_.empty()) {
        return nullptr;
    }
    // Verify the vector size matches expected size to prevent out-of-bounds access
    size_t expected_size = static_cast<size_t>(config_.nsample) * config_.nsample;
    if (proximity_matrix_.size() != expected_size) {
        // std::cerr << "WARNING: proximity_matrix_ size mismatch. Expected " << expected_size 
        //           << ", got " << proximity_matrix_.size() << std::endl;  // Commented to avoid stream conflicts
        return nullptr;
    }
    
    // CRITICAL: Ensure the vector's data pointer is valid
    // Check if the vector's capacity is sufficient (prevents issues if vector was moved)
    if (proximity_matrix_.capacity() < expected_size) {
        // std::cerr << "WARNING: proximity_matrix_ capacity insufficient. Expected at least "  // Commented to avoid stream conflicts 
                //   << expected_size << ", got " << proximity_matrix_.capacity() << std::endl;
        return nullptr;
    }
    
    // CRITICAL: Verify the data pointer is actually valid
    // Try to access the first element to ensure the memory is valid
    // This helps catch cases where the vector was moved or the memory was freed
    try {
        volatile dp_t test = proximity_matrix_[0];
        (void)test;  // Suppress unused variable warning
    } catch (...) {
        // std::cerr << "WARNING: proximity_matrix_ data pointer is invalid (vector may have been moved)" << std::endl;  // Commented to avoid stream conflicts
        return nullptr;
    }
    
    // Return pointer - vector should remain stable as long as model exists
    // The Python binding will copy the data immediately, so this is safe
    // However, we must ensure the vector is not moved or cleared while Python holds the pointer
    return proximity_matrix_.data();
}

const real_t* RandomForest::get_qimpm() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return qimpm_.data();
}

const real_t* RandomForest::get_avimp() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return avimp_.data();
}

const real_t* RandomForest::get_sqsd() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return sqsd_.data();
}

bool RandomForest::get_lowrank_factors(dp_t** A_host, dp_t** B_host, integer_t* r) const {
    // CRITICAL: Before get_lowrank_factors, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if low-rank mode is active
    if (lowrank_proximity_ptr_ == nullptr) {
        return false;  // Low-rank mode not active
    }
    
    #ifdef CUDA_FOUND
    try {
        // Use helper function from CUDA file to avoid CUDA header inclusion in CPU compilation
        bool result = rf::cuda::get_lowrank_factors_host(lowrank_proximity_ptr_, A_host, B_host, r, get_n_samples());
        
        // CRITICAL: After GPU operation, finalize to ensure all kernels complete
        if (config_.use_gpu && result) {
            rf::cuda::cuda_finalize_operations();
        }
        
        return result;
    } catch (...) {
        return false;  // Error occurred
    }
    #else
    // CPU-only compilation - low-rank mode requires CUDA
    return false;
    #endif
}

std::vector<double> RandomForest::compute_mds_from_factors(rf::integer_t k) const {
    // CRITICAL: Before compute_mds, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if low-rank mode is active
    if (lowrank_proximity_ptr_ == nullptr) {
        return std::vector<double>();  // Low-rank mode not active
    }
    
    #ifdef CUDA_FOUND
    try {
        // Use helper function from CUDA file to avoid CUDA header inclusion in CPU compilation
        std::vector<double> result = rf::cuda::compute_mds_from_factors_host(lowrank_proximity_ptr_, k);
        
        // CRITICAL: After GPU operation, finalize to ensure all kernels complete
        if (config_.use_gpu && !result.empty()) {
            rf::cuda::cuda_finalize_operations();
        }
        
        return result;
    } catch (...) {
        return std::vector<double>();  // Error occurred
    }
    #else
    // CPU-only compilation - low-rank mode requires CUDA
    return std::vector<double>();
    #endif
}

} // namespace rf
