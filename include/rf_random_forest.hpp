#ifndef RF_RANDOM_FOREST_HPP
#define RF_RANDOM_FOREST_HPP

#include "rf_types.hpp"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iostream>
#include <functional>

namespace rf {
namespace cuda {
    // Forward declaration for LowRankProximityMatrix
    class LowRankProximityMatrix;
}

// Standalone train_test_split function (doesn't require RandomForest instance)
void train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                     real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                     integer_t& n_train, integer_t& n_test, 
                     real_t test_size, integer_t random_state, bool stratify);

// Random Forest task types
enum class TaskType {
    CLASSIFICATION,
    REGRESSION,
    UNSUPERVISED
};

// Unsupervised learning mode (classification-style vs regression-style splitting)
enum class UnsupervisedMode {
    CLASSIFICATION_STYLE,  // Use classification-style mtry (sqrt(mdim)) and Gini-like splitting
    REGRESSION_STYLE       // Use regression-style mtry (mdim/3) and MSE-like splitting
};

// Random Forest configuration
struct RandomForestConfig {
    // Core parameters
    integer_t nsample;
    integer_t ntree;
    integer_t mdim;
    integer_t nclass;
    integer_t maxcat;

    // Derived parameters
    integer_t mtry;
    integer_t maxnode;
    integer_t minndsize;

    // Algorithm parameters
    integer_t ncsplit;  // Number of random splits for big categoricals
    integer_t ncmax;    // Threshold for "big" categorical (> ncmax levels)
    integer_t iseed;    // Random seed

    // Task type
    TaskType task_type;  // Classification, regression, or unsupervised

    // Feature flags
    bool compute_proximity;  // iprox
    bool compute_importance;  // imp
    bool compute_local_importance;  // impn
    bool use_rfgap;  // DEPRECATED: Use proximity_type="rfgap" instead. Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity instead of standard proximity
    bool use_casewise;  // Use case-wise calculations (per-sample) vs non-case-wise (aggregated across all samples)
                        // Non-case-wise follows UC Berkeley standard: simple averaging/aggregation
                        // Case-wise follows Breiman-Cutler: per-sample tracking with bootstrap weights
    
    // Proximity type selection
    std::string proximity_type;  // "standard" or "rfgap"
    
    // Local importance method selection
    std::string local_importance_method;  // "local_imp" or "clique" (for LOCAL importance only)
    std::string importance_method;  // DEPRECATED: Use local_importance_method instead. "local_imp" or "clique"
    integer_t clique_M;  // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)

    // Execution mode
    std::string execution_mode;  // "auto", "gpu", "cpu"
    bool use_gpu;

    // GPU-specific
    bool use_qlora;      // QLORA-Proximity compression
    integer_t quant_mode;  // 0=FP32, 1=FP16, 2=INT8, 3=NF4
    integer_t lowrank_rank;  // Low-rank proximity matrix rank (A and B factors: n_samples × rank)
                            // Default: 100. Higher rank = better approximation but more memory.
                            // Memory usage: 2 × n_samples × rank × bytes_per_element
                            // For 100k samples: rank=100 → ~40MB (FP16), rank=200 → ~80MB (FP16)
    bool use_sparse;     // Block-sparse mode
    real_t sparsity_threshold;
    integer_t batch_size;  // GPU batch size for tree processing (0=auto)

    // Regression-specific parameters
    integer_t nodesize;  // Minimum terminal node size for regression
    real_t cutoff;       // Early stopping threshold

    // Unsupervised-specific parameters
    bool use_unsupervised;  // Enable unsupervised mode
    real_t outlier_threshold;  // Threshold for outlier detection
    UnsupervisedMode unsupervised_mode;  // Classification-style or regression-style splitting for unsupervised

    // GPU-only advanced features
    // Loss function type for GPU tree growing
    // 0=Gini (classification only)
    integer_t gpu_loss_function;  // Default: 0 (Gini)
    real_t min_gain_to_split;  // Minimum gain threshold for early stopping (GPU only)
    bool gpu_parallel_mode0;  // True if parallel mode is enabled
    real_t l1_regularization;  // L1 regularization strength (lambda) - penalizes |leaf_value|
    real_t l2_regularization;  // L2 regularization strength (lambda) - penalizes leaf_value^2
    integer_t n_threads_cpu;  // Number of CPU threads (0 = auto-detect, use all available cores)

    // Constructor with defaults
    RandomForestConfig() :
        nsample(1000), ntree(100), mdim(10), nclass(2), maxcat(10),
        mtry(3), maxnode(2001), minndsize(1),
        ncsplit(25), ncmax(25), iseed(12345),
        task_type(TaskType::CLASSIFICATION),
        compute_proximity(false), compute_importance(true), compute_local_importance(false),
        use_rfgap(false), use_casewise(false),
        proximity_type("standard"),
        local_importance_method("local_imp"),
        importance_method("local_imp"), clique_M(-1),
        execution_mode("auto"), use_gpu(false),
        use_qlora(false), quant_mode(1), lowrank_rank(100), use_sparse(false), sparsity_threshold(1e-6f), batch_size(0),
        nodesize(5), cutoff(0.01),
        use_unsupervised(false), outlier_threshold(2.0),
        unsupervised_mode(UnsupervisedMode::CLASSIFICATION_STYLE),
        gpu_loss_function(0), min_gain_to_split(0.0f),
        gpu_parallel_mode0(false),
        l1_regularization(0.0f), l2_regularization(0.0f), n_threads_cpu(0) {

        // Set derived parameters
        mtry = static_cast<integer_t>(std::sqrt(static_cast<real_t>(mdim)));
        maxnode = 2 * nsample + 1;
    }
};

// Random Forest model class
class RandomForest {
public:
    RandomForest(const RandomForestConfig& config);
    ~RandomForest();

    // Training methods for different task types
    void fit_classification(const real_t* X, const integer_t* y, const real_t* sample_weight = nullptr);
    void fit_regression(const real_t* X, const real_t* y, const real_t* sample_weight = nullptr);
    void fit_unsupervised(const real_t* X, const real_t* sample_weight = nullptr);
    
    // Unified fit method that dispatches based on task type
    void fit(const real_t* X, const void* y, const real_t* sample_weight = nullptr);

    // Train-test split method (like XGBoost)
    void train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                         real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                         integer_t& n_train, integer_t& n_test, 
                         real_t test_size = 0.2, integer_t random_state = 42,
                         bool stratify = true);

    // Prediction methods
    void predict_classification(const real_t* X, integer_t nsamples, integer_t* predictions);
    void predict_regression(const real_t* X, integer_t nsamples, real_t* predictions);
    void predict_unsupervised(const real_t* X, integer_t nsamples, integer_t* cluster_labels);
    
    // Unified predict method
    void predict(const real_t* X, integer_t nsamples, void* predictions);
    
    // Probability prediction (classification only)
    void predict_proba(const real_t* X, integer_t nsamples, real_t* probabilities);
    void predict_classification_proba(const real_t* X, integer_t nsamples, real_t* probabilities) { 
        predict_proba(X, nsamples, probabilities); 
    }

    // Getters
    real_t get_oob_error() const { return oob_error_; }
    const real_t* get_feature_importances() const { return feature_importances_.data(); }
    const dp_t* get_proximity_matrix() const;
    integer_t get_n_trees() const { return config_.ntree; }
    TaskType get_task_type() const { return config_.task_type; }
    
    // Dimension getters
    integer_t get_n_samples() const { return config_.nsample; }
    integer_t get_n_features() const { return config_.mdim; }
    integer_t get_n_classes() const { return config_.nclass; }
    
    // Configuration getters
    bool get_compute_proximity() const { return config_.compute_proximity; }
    bool get_compute_importance() const { return config_.compute_importance; }
    bool get_compute_local_importance() const { return config_.compute_local_importance; }
    // std::string get_execution_mode() const { return config_.execution_mode; }
    
    // Variable importance getters
    const real_t* get_qimp() const { return qimp_.data(); }           // Overall importance per sample
    const real_t* get_qimpm() const;         // Local importance per sample per feature
    const real_t* get_avimp() const;         // Average importance per feature
    const real_t* get_sqsd() const;           // Standard deviation per feature
    
    // Regression-specific getters
    real_t get_oob_mse() const { return oob_mse_; }
    const real_t* get_oob_predictions() const { 
        return oob_predictions_.empty() ? nullptr : oob_predictions_.data(); 
    }
    
    // Classification-specific getters
    const integer_t* get_oob_class_predictions() const { 
        return oob_class_predictions_.empty() ? nullptr : oob_class_predictions_.data(); 
    }
    const real_t* get_oob_probabilities() const { 
        return oob_probabilities_.empty() ? nullptr : oob_probabilities_.data(); 
    }
    
    // Unsupervised-specific getters
    const real_t* get_outlier_scores() const { return outlier_scores_.data(); }
    const integer_t* get_cluster_labels() const { return cluster_labels_.data(); }
    
    // Low-rank proximity getters (for QLoRA mode)
    // Returns A, B factors and rank as output parameters
    // Returns true if low-rank factors are available, false otherwise
    bool get_lowrank_factors(dp_t** A_host, dp_t** B_host, integer_t* r) const;
    
    // Compute 3D MDS coordinates from low-rank factors (memory efficient)
    // Returns empty vector if low-rank factors are not available
    std::vector<double> compute_mds_from_factors(integer_t k = 3) const;
    std::vector<double> compute_mds_3d_from_factors() const {
        return compute_mds_from_factors(3);
    }
    
    // Get OOB counts for RF-GAP normalization (|Si| for each sample i)
    const integer_t* get_nout_ptr() const { return nout_.data(); }
    
    // Check if RF-GAP is enabled
    bool get_use_rfgap() const { return config_.use_rfgap; }

    // Batch tree growing (GPU-optimized)
    void fit_batch_gpu(const real_t* X, const void* y, const real_t* sample_weight,
                      integer_t batch_size = 10);

    // Progress callback mechanism
    using ProgressCallback = std::function<void(integer_t current, integer_t total)>;
    void set_progress_callback(ProgressCallback callback) { progress_callback_ = callback; }
    
    // Categorical feature handling
    void set_categorical_features(const integer_t* cat_array, integer_t mdim);

private:
    RandomForestConfig config_;

    // Training data (stored for reference)
    std::vector<real_t> X_train_;
    std::vector<integer_t> y_train_classification_;  // For classification
    std::vector<integer_t> y_train_classification_1based_;  // 1-based class labels for Fortran compatibility
    std::vector<real_t> y_train_regression_;         // For regression
    std::vector<integer_t> y_train_regression_int_;  // Converted regression data for growtree
    std::vector<integer_t> y_train_unsupervised_;    // Synthetic labels for unsupervised
    std::vector<real_t> sample_weight_;

    // Model outputs
    real_t oob_error_;      // Classification error rate
    real_t oob_mse_;        // Regression MSE
    std::vector<real_t> feature_importances_;
    mutable std::vector<dp_t> proximity_matrix_;  // Mutable for on-demand reconstruction from low-rank factors
    
    // Low-rank proximity storage (for GPU mode with QLoRA)
    void* lowrank_proximity_ptr_;  // Pointer to rf::cuda::LowRankProximityMatrix (opaque)
    
    // Variable importance arrays (accumulated across trees)
    std::vector<real_t> qimp_;      // Overall importance per sample
    std::vector<real_t> qimpm_;     // Local importance per sample per feature (nsample * mdim)
    std::vector<real_t> avimp_;     // Average importance per feature
    std::vector<real_t> sqsd_;      // Standard deviation of importance per feature
    
    // Regression-specific outputs
    std::vector<real_t> oob_predictions_;
    std::vector<integer_t> oob_class_predictions_;  // Store OOB class predictions from fit()
    std::vector<real_t> oob_probabilities_;  // Store OOB class probabilities from fit()
    std::vector<real_t> terminal_values_;  // Regression values for terminal nodes
    
    // Unsupervised-specific outputs
    std::vector<real_t> outlier_scores_;
    std::vector<integer_t> cluster_labels_;
    
    // Class label mapping (for non-0-based class labels)
    // Maps original labels (e.g., 3,4,5,6,7,8) to 0-based indices (0,1,2,3,4,5)
    std::map<integer_t, integer_t> original_to_0based_label_map_;  // original -> 0-based
    std::vector<integer_t> zero_based_to_original_label_map_;  // Reverse mapping: 0-based index -> original label
    
    // Workspace arrays for tree growing (to avoid repeated allocation/deallocation)
    std::vector<real_t> win_workspace_;
    std::vector<integer_t> nin_workspace_;
    std::vector<integer_t> jinbag_workspace_;
    std::vector<integer_t> joobag_workspace_;
    std::vector<integer_t> amat_workspace_;
    std::vector<integer_t> idmove_workspace_;
    std::vector<integer_t> igoleft_workspace_;
    std::vector<integer_t> igl_workspace_;
    std::vector<integer_t> nodestart_workspace_;
    std::vector<integer_t> nodestop_workspace_;
    std::vector<integer_t> itemp_workspace_;
    std::vector<integer_t> itmp_workspace_;
    std::vector<integer_t> icat_workspace_;
    std::vector<real_t> tcat_workspace_;
    std::vector<real_t> classpop_workspace_;
    std::vector<real_t> wr_workspace_;
    std::vector<real_t> wl_workspace_;
    std::vector<real_t> tclasscat_workspace_;
    std::vector<real_t> tclasspop_workspace_;
    std::vector<real_t> tmpclass_workspace_;
    std::vector<integer_t> jtr_workspace_;
    std::vector<integer_t> nodextr_workspace_;

    // Trees (compact storage)
    std::vector<real_t> tnodewt_;  // Node weights (tw/tn) for classification, weights for regression
    std::vector<real_t> node_predictions_regression_;  // Regression predictions (mean of y) for terminal nodes
    std::vector<real_t> xbestsplit_;
    std::vector<integer_t> nodestatus_;
    std::vector<integer_t> bestvar_;
    std::vector<integer_t> treemap_;
    std::vector<integer_t> catgoleft_;
    std::vector<integer_t> nodeclass_;
    std::vector<integer_t> nnode_;  // Nodes per tree
    
    // RF-GAP proximity data (per-tree bootstrap multiplicities and terminal nodes)
    std::vector<std::vector<integer_t>> tree_nin_rfgap_;  // [tree][sample] = bootstrap multiplicity
    std::vector<std::vector<integer_t>> tree_nodextr_rfgap_;  // [tree][sample] = terminal node index

    // Workspace arrays
    std::vector<real_t> q_;           // Vote accumulation
    std::vector<integer_t> nout_;     // OOB count per sample
    std::vector<integer_t> cat_;      // Variable types (1=quantitative, >1=categorical)
    std::vector<integer_t> ties_;     // Tie indicators
    std::vector<integer_t> asave_;    // Sorted indices
    
    // State flags
    bool proximity_normalized_;       // Track if proximity matrix has been normalized
    
    // Progress callback
    ProgressCallback progress_callback_;

    // Helper methods
    void initialize_arrays();
    void detect_categorical_features(const real_t* X, integer_t nsample, integer_t mdim, integer_t maxcat);
    void prepare_data();
    void grow_tree_single(integer_t tree_id, integer_t seed);
    void accumulate_oob_votes(integer_t tree_id);
    void finalize_training();
    
    // Task-specific helper methods
    void setup_classification_task(const integer_t* y);
    void setup_regression_task(const real_t* y);
    void setup_unsupervised_task();
    
    void compute_classification_error();
    void compute_regression_mse();
    void compute_outlier_scores();
    void compute_cluster_labels();
    void compute_oob_predictions();
    void initialize_workspace_arrays();
    
    // Split criterion methods
    real_t compute_gini_impurity(const std::vector<real_t>& class_counts, real_t total_weight);
    real_t compute_mse(const std::vector<real_t>& values, const std::vector<real_t>& weights);
    real_t compute_proximity_criterion(const std::vector<real_t>& proximities);
    
    // Helper methods
    integer_t find_terminal_node(const real_t* sample_features,
                                integer_t* nodestatus, integer_t* bestvar,
                                real_t* xbestsplit, integer_t* treemap,
                                integer_t* catgoleft);
};

} // namespace rf

#endif // RF_RANDOM_FOREST_HPP
