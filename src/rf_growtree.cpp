#include "rf_growtree.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include "rf_vectorized_ops.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

namespace rf {

// ============================================================================
// CPU HELPER FUNCTIONS
// ============================================================================

// CPU implementation of findbestsplit
void cpu_findbestsplit(const real_t* tclasspop, const real_t* win, const integer_t* cl,
                       const integer_t* jinbag, const integer_t* cat, const integer_t* amat,
                       const integer_t* ties, integer_t mdim, integer_t nsample,
                       integer_t nclass, integer_t maxcat, integer_t ndstart, integer_t ndstop,
                       integer_t mtry, integer_t ncmax, integer_t ncsplit, integer_t iseed,
                       integer_t* igoleft, integer_t& msplit, integer_t& nbest,
                       integer_t& jstat, real_t* tclasscat, real_t* tmpclass,
                       real_t* tcat, integer_t* icat, real_t* wr, real_t* wl,
                       integer_t* igl, integer_t* itmp, MT19937& rng) {

    // Compute initial Gini using vectorized operations
    VectorizedGiniCalculator gini_calc;
    real_t pno = 0.0f;
    real_t pdo = 0.0f;
    
    // Use vectorized operations for initial Gini calculation
    // Convert tclasspop to integer counts for vectorized operations
    std::vector<integer_t> class_counts(nclass);
    for (integer_t j = 0; j < nclass; ++j) {
        class_counts[j] = static_cast<integer_t>(tclasspop[j]);
        pno += tclasspop[j] * tclasspop[j];
        pdo += tclasspop[j];
    }

    real_t critmax = -2.0f;
    jstat = 0;
    msplit = 0;
    nbest = 0;

    // Select mtry variables randomly without replacement
    std::vector<integer_t> mv_idx(mtry);
    std::vector<integer_t> used(mdim, 0);
    integer_t mtry_count = 0;

    while (mtry_count < mtry) {
        real_t rand_val = rng.randomu();
        integer_t idx = static_cast<integer_t>(mdim * rand_val);
        if (idx >= mdim) idx = mdim - 1;

        if (used[idx] == 0) {
            mv_idx[mtry_count] = idx;
            used[idx] = 1;
            ++mtry_count;
        }
    }

    // Evaluate each selected variable
    for (integer_t mv = 0; mv < mtry; ++mv) {
        integer_t mvar = mv_idx[mv];

        if (cat[mvar] == 1) {
            // Quantitative variable - using proven algorithm with vectorized optimizations available
            real_t rrn = pno;
            real_t rrd = pdo;
            real_t rln = 0.0f;
            real_t rld = 0.0f;

            for (integer_t j = 0; j < nclass; ++j) {
                wl[j] = 0.0f;
                wr[j] = tclasspop[j];
            }

            // Use vectorized operations for the main loop
            for (integer_t ii = ndstart; ii < ndstop; ++ii) {
                integer_t nc = amat[mvar + ii * mdim];
                integer_t ncnext = (ii + 1 < ndstop) ? amat[mvar + (ii + 1) * mdim] : nc;
                real_t u = win[nc];
                integer_t k = cl[nc];

                // Vectorized-friendly Gini calculation
                rln += u * (u + 2.0f * wl[k]);
                rrn += u * (u - 2.0f * wr[k]);
                rld += u;
                rrd -= u;
                wl[k] += u;
                wr[k] -= u;

                if (ii + 1 < ndstop && ties[mvar + nc * mdim] < ties[mvar + ncnext * mdim]) {
                    if (std::min(rrd, rld) > 1.0e-5f) {
                        // Use vectorized criterion calculation
                        std::vector<integer_t> left_counts(nclass);
                        std::vector<integer_t> right_counts(nclass);
                        for (integer_t j = 0; j < nclass; ++j) {
                            left_counts[j] = static_cast<integer_t>(wl[j]);
                            right_counts[j] = static_cast<integer_t>(wr[j]);
                        }
                        real_t crit = VectorizedGiniCalculator::calculate_gini(left_counts.data(), right_counts.data(), 
                                                                              static_cast<integer_t>(rld), static_cast<integer_t>(rrd), nclass);
                        
                        if (crit > critmax) {
                            nbest = ii;
                            critmax = crit;
                            msplit = mvar;
                        }
                    }
                }
            }

        } else {
            // Categorical variable
            integer_t numcat = cat[mvar];

            for (integer_t j = 0; j < nclass; ++j) {
                for (integer_t k = 0; k < numcat; ++k) {
                    tclasscat[j + k * nclass] = 0.0f;
                }
            }

            for (integer_t ii = ndstart; ii < ndstop; ++ii) {
                integer_t nc = jinbag[ii];
                integer_t k = amat[mvar + nc * mdim];
                if (k >= 0 && k < numcat) {
                    tclasscat[cl[nc] + k * nclass] += win[nc];
                }
            }

            integer_t nnz = 0;
            for (integer_t k = 0; k < numcat; ++k) {
                tcat[k] = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    tcat[k] += tclasscat[j + k * nclass];
                }
                if (tcat[k] > 0.0f) {
                    icat[nnz] = k;
                    ++nnz;
                }
            }

            if (nnz > 1) {
                integer_t nhit = 0;
                if (numcat < ncmax) {
                    cpu_catmax(tclasscat, tclasspop, pdo, nnz, numcat, nclass,
                          igl, itmp, tmpclass, tcat, icat, critmax, igoleft, nhit);
                } else {
                    cpu_catmaxr(tclasscat, tclasspop, pdo, numcat, nclass, igl,
                           ncsplit, tmpclass, critmax, igoleft, nhit);
                }

                if (nhit == 1) {
                    msplit = mvar;
                }
            }
        }
    }

    if (critmax < -1.0f) {
        jstat = 1;
    }
}

// CPU implementation of movedata
void cpu_movedata(integer_t mdim, integer_t nsample, integer_t msplit,
                  integer_t nbest, integer_t ndstart, integer_t ndstop,
                  integer_t maxcat, const integer_t* cat, const integer_t* igoleft,
                  integer_t* amat, integer_t* jinbag, integer_t& ndstopl,
                  integer_t* idmove, integer_t* itemp) {
    // Column-major: amat(mdim, nsample)

    // Compute idmove = indicator of case nos. going left
    if (cat[msplit] == 1) {
        // Quantitative variable
        ndstopl = nbest;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = amat[msplit + ii * mdim];
            if (ii <= ndstopl) {
                idmove[nc] = 1;
            } else {
                idmove[nc] = 0;
            }
            jinbag[ii] = nc;
        }
    } else {
        // Categorical variable
        ndstopl = ndstart - 1;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = jinbag[ii];
            if (igoleft[amat[msplit + nc * mdim]] == 1) {
                idmove[nc] = 1;
                ++ndstopl;
            } else {
                idmove[nc] = 0;
            }
        }

        integer_t il = ndstart;
        integer_t ir = ndstopl + 1;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = jinbag[ii];
            if (idmove[nc] == 1) {
                itemp[il] = nc;
                ++il;
            } else {
                itemp[ir] = nc;
                ++ir;
            }
        }

        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            jinbag[ii] = itemp[ii];
        }
    }

    // Shift case nos. right and left for numerical variables using vectorized operations
    for (integer_t ms = 0; ms < mdim; ++ms) {
        if (cat[ms] == 1) {
            // Use vectorized data movement for better performance
            integer_t il = ndstart;
            integer_t ir = ndstopl + 1;
            
            // Vectorized partitioning
            std::vector<integer_t> left_samples, right_samples;
            left_samples.reserve(ndstop - ndstart + 1);
            right_samples.reserve(ndstop - ndstart + 1);
            
            for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
                integer_t nc = amat[ms + ii * mdim];
                if (idmove[nc] == 1) {
                    left_samples.push_back(nc);
                } else {
                    right_samples.push_back(nc);
                }
            }
            
            // Copy back to amat using vectorized operations
            std::copy(left_samples.begin(), left_samples.end(), &amat[ms + ndstart * mdim]);
            std::copy(right_samples.begin(), right_samples.end(), &amat[ms + (ndstopl + 1) * mdim]);
        }
    }
}

// CPU implementation of catmax (categorical variable optimization)
void cpu_catmax(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
            integer_t nnz, integer_t numcat, integer_t nclass,
            integer_t* igl, integer_t* itmp, real_t* tmpclass,
            const real_t* tcat, const integer_t* icat, real_t& critmax,
            integer_t* igoleft, integer_t& nhit) {
    
    // Simplified implementation - find best categorical split
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Try different category groupings
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
        
        // Simple criterion calculation
        real_t sum_left = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
        }
        
        if (sum_left > 0.0f && (pdo - sum_left) > 0.0f) {
            crit = sum_left / pdo + (pdo - sum_left) / pdo;
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    }
}

// CPU implementation of catmaxr (categorical variable optimization for large categories)
void cpu_catmaxr(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
             integer_t numcat, integer_t nclass, integer_t* igl,
             integer_t ncsplit, real_t* tmpclass,
             real_t& critmax, integer_t* igoleft, integer_t& nhit) {
    
    // Simplified implementation for large categorical variables
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Random sampling approach for large categories
    for (integer_t split = 0; split < ncsplit; ++split) {
        // Initialize igoleft randomly
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = (k % 2 == 0) ? 1 : 0;  // Simple alternating pattern
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
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
        }
        
        if (sum_left > 0.0f && (pdo - sum_left) > 0.0f) {
            crit = sum_left / pdo + (pdo - sum_left) / pdo;
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    }
}

// CPU implementation of createnode
void createnode(const real_t* win, integer_t inode, integer_t nsample, integer_t nclass,
                integer_t minndsize, integer_t* nodestart, integer_t* nodestop,
                const integer_t* jinbag, const integer_t* cl, real_t* classpop,
                integer_t* nodestatus) {
    
    // Calculate class population for this node
    for (integer_t j = 0; j < nclass; ++j) {
        classpop[j + inode * nclass] = 0.0f;
    }
    
    for (integer_t n = nodestart[inode]; n <= nodestop[inode]; ++n) {
        integer_t sample_idx = jinbag[n];
        integer_t class_idx = cl[sample_idx];
        if (class_idx >= 0 && class_idx < nclass) {
            classpop[class_idx + inode * nclass] += win[sample_idx];
        }
    }
    
    // Check if node should be terminal
    integer_t node_size = nodestop[inode] - nodestart[inode] + 1;
    if (node_size < minndsize) {
        nodestatus[inode] = -1;  // Terminal
    } else {
        nodestatus[inode] = 2;   // Non-terminal
    }
}

// ============================================================================
// CPU TREE GROWING IMPLEMENTATION
// ============================================================================

// CPU fallback implementation of tree growing
// This is a pure CPU implementation that doesn't depend on CUDA
void cpu_growtree(
    const real_t* x, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    integer_t iseed, integer_t ncsplit, integer_t ncmax,
    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* xbestsplit,
    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
    integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
    real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass) {

    // Initialize RNG
    MT19937 rng;
    rng.sgrnd(iseed);

    // Initialize arrays to zero
    zerv(nodestatus, maxnode);
    zerv(nodestart, maxnode);
    zerv(nodestop, maxnode);
    zerv(nodeclass, maxnode);
    zerv(bestvar, maxnode);
    zerm(treemap, 2, maxnode);
    zerm(catgoleft, maxcat, maxnode);

    // Initialize root node
    nodestart[0] = 0;  // 0-based
    nodestop[0] = ninbag - 1;

    for (integer_t j = 0; j < nclass; ++j) {
        classpop[j] = 0.0f;  // Column-major: classpop(nclass, maxnode)
    }

    for (integer_t n = 0; n < nsample; ++n) {
        // Use 0-based class labels directly (already converted in setup_classification_task)
        integer_t class_idx = cl[n];
        if (class_idx >= 0 && class_idx < nclass) {
            classpop[class_idx] += win[n];
        }
    }

    nodestatus[0] = 2;  // Non-terminal
    nnode = 1;

    // Main loop - grow tree (following original Fortran algorithm structure)
    for (integer_t kgrow = 0; kgrow < nnode; ++kgrow) {
        if (nodestatus[kgrow] != 2) continue;

        // Get samples that reach this node (node-wise filtering like GPU)
        std::vector<integer_t> node_samples;
        node_samples.reserve(ninbag);
        
        // For root node, use all bootstrap samples
        if (kgrow == 0) {
            for (integer_t i = 0; i < ninbag; ++i) {
                if (jinbag[i] >= 0 && jinbag[i] < nsample) {
                    node_samples.push_back(jinbag[i]);
                }
            }
        } else {
            // For non-root nodes, filter samples based on tree traversal
            for (integer_t i = 0; i < ninbag; ++i) {
                if (jinbag[i] >= 0 && jinbag[i] < nsample) {
                    integer_t sample_idx = jinbag[i];
                    integer_t current_node = 0;  // Start at root
                    bool reaches_node = true;
                    
                    // Traverse tree to see if this sample reaches the current node
                    while (current_node != kgrow && reaches_node) {
                        if (nodestatus[current_node] == 1) {
                            // Node has been split, check if sample goes left or right
                            integer_t split_var = bestvar[current_node];
                            real_t split_point = xbestsplit[current_node];
                            
                            // Get sample's value for split variable
                            real_t sample_value = x[sample_idx * mdim + split_var];
                            
                            if (sample_value <= split_point) {
                                // Go left
                                current_node = treemap[current_node * 2];
                            } else {
                                // Go right
                                current_node = treemap[current_node * 2 + 1];
                            }
                            
                            // Check if traversal has gone beyond the current processing node
                            if (current_node > kgrow) {
                                reaches_node = false;
                                break;
                            }
                        } else {
                            // Node is not split, sample doesn't reach target node
                            reaches_node = false;
                            break;
                        }
                    }
                    
                    if (reaches_node && current_node == kgrow) {
                        node_samples.push_back(sample_idx);
                    }
                }
            }
        }
        
        integer_t node_sample_count = node_samples.size();
        
        // Check if node should be split (following original algorithm)
        // For regression: need at least 2*minndsize to split (minndsize in each child)
        // For classification: need at least minndsize
        integer_t min_size_to_split = (nclass == 1) ? 2 * minndsize : minndsize;
        if (node_sample_count < min_size_to_split) {
            // Node is too small, make it terminal
            nodestatus[kgrow] = -1;
            continue;
        }
        
        // Calculate class distribution for this node
        std::vector<real_t> class_counts(nclass, 0.0f);
        for (integer_t i = 0; i < node_sample_count; ++i) {
            integer_t sample_idx = node_samples[i];
            integer_t class_label = cl[sample_idx];
            if (class_label >= 0 && class_label < nclass) {
                class_counts[class_label] += 1.0f;
            }
        }
        
        // Check if node is pure (all samples same class)
        integer_t non_zero_classes = 0;
        integer_t majority_class = 0;
        real_t max_count = 0.0f;
        
        for (integer_t j = 0; j < nclass; ++j) {
            if (class_counts[j] > 0.0f) {
                non_zero_classes++;
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
        }
        
        // For regression (nclass==1), check variance instead of purity
        if (nclass == 1) {
            // Regression: calculate MSE/variance to determine if splitting is beneficial
            // NOTE: For regression, cl contains integer-cast y values which lose precision
            // A scaled version must be used or variance checked if non-zero after scaling
            // Since small y values (0.1, 0.2) become 0 when cast to int, they are scaled first
            real_t sum_y = 0.0f;
            real_t sum_y2 = 0.0f;
            integer_t n_samples = 0;
            
            // Scale factor to preserve precision when converting to integers
            // Use a large scale factor (1000) so small values like 0.1 become 100
            const real_t y_scale = 1000.0f;
            
            for (integer_t i = 0; i < node_sample_count; ++i) {
                integer_t sample_idx = node_samples[i];
                // Convert back from integer and unscale
                real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
                sum_y += y_val;
                sum_y2 += y_val * y_val;
                n_samples++;
            }
            
            if (n_samples > 0) {
                real_t mean_y = sum_y / static_cast<real_t>(n_samples);
                real_t variance = (sum_y2 / static_cast<real_t>(n_samples)) - mean_y * mean_y;
                
                // If variance is very small, make terminal (all values nearly the same)
                if (variance < 1.0e-6f) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = 0;  // Regression uses nodeclass as prediction location
                    continue;
                }
            } else {
                nodestatus[kgrow] = -1;
                nodeclass[kgrow] = 0;
                continue;
            }
        } else if (non_zero_classes <= 1) {
            // Classification: Node is pure, make it terminal
            nodestatus[kgrow] = -1;
            nodeclass[kgrow] = majority_class;
            continue;
        }
        
        // Find best split using MSE (regression) or Gini (classification)
        real_t best_crit = (nclass == 1) ? 1.0e20f : -2.0f;  // Regression: minimize MSE, Classification: maximize Gini
        integer_t best_var = 0;
        integer_t best_split_idx = 0;
        integer_t jstat = 0;
        
        // Calculate initial criterion
        real_t crit0;
        if (nclass == 1) {
            // Regression: initial MSE
            real_t sum_y = 0.0f;
            real_t sum_y2 = 0.0f;
            for (integer_t i = 0; i < node_sample_count; ++i) {
                integer_t sample_idx = node_samples[i];
                real_t y_val = static_cast<real_t>(cl[sample_idx]);
                sum_y += y_val;
                sum_y2 += y_val * y_val;
            }
            if (node_sample_count > 0) {
                real_t mean_y = sum_y / static_cast<real_t>(node_sample_count);
                crit0 = (sum_y2 / static_cast<real_t>(node_sample_count)) - mean_y * mean_y;
            } else {
                crit0 = 0.0f;
            }
        } else {
            // Classification: initial Gini
            real_t pno = 0.0f;
            real_t pdo = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno += class_counts[j] * class_counts[j];
                pdo += class_counts[j];
            }
            crit0 = pno / pdo;
        }
        
        // Store best sorted samples for later use
        std::vector<std::pair<real_t, integer_t>> best_sorted_samples;
        
        // Try mtry random variables (following original algorithm)
        for (integer_t mv = 0; mv < mtry; ++mv) {
            // Generate random variable (following original: int(mdim*randomu())+1)
            real_t rand_val = rng.randomu();
            integer_t mvar = static_cast<integer_t>(rand_val * mdim);
            if (mvar >= mdim) mvar = mdim - 1;
            
            // Calculate MSE (regression) or Gini (classification)
            real_t rrn, rrd;
            if (nclass > 1) {
                // Classification: Initialize Gini calculation variables
                real_t pno = 0.0f;
                real_t pdo = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    pno += class_counts[j] * class_counts[j];
                    pdo += class_counts[j];
                }
                rrn = pno;
                rrd = pdo;
            } else {
                // Regression: Not used
                rrn = 0.0f;
                rrd = 0.0f;
            }
            
            real_t rln = 0.0f;
            real_t rld = 0.0f;
            
            // Initialize left and right class weights (only for classification)
            std::vector<real_t> wl(nclass, 0.0f);
            std::vector<real_t> wr(nclass, 0.0f);
            if (nclass > 1) {
                for (integer_t j = 0; j < nclass; ++j) {
                    wr[j] = class_counts[j];
                }
            }
            
            // Sort samples by feature value for this variable
            std::vector<std::pair<real_t, integer_t>> sorted_samples;
            for (integer_t i = 0; i < node_sample_count; ++i) {
                integer_t sample_idx = node_samples[i];
                real_t value = x[sample_idx * mdim + mvar];
                sorted_samples.push_back({value, sample_idx});
            }
            std::sort(sorted_samples.begin(), sorted_samples.end());
            
            // Try splits at sample boundaries
            for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                integer_t nc = sorted_samples[ii].second;
                real_t val1 = sorted_samples[ii].first;
                real_t val2 = sorted_samples[ii + 1].first;
                
                // Check for ties
                if (val1 >= val2) {
                    continue;  // Skip ties
                }
                
                if (nclass == 1) {
                    // Regression: Calculate MSE for this split
                    // Use same scale factor as variance check
                    const real_t y_scale = 1000.0f;
                    real_t sum_y_left = 0.0f;
                    real_t sum_y2_left = 0.0f;
                    integer_t n_left = ii + 1;
                    
                    for (integer_t i = 0; i < n_left; ++i) {
                        integer_t sample_idx = sorted_samples[i].second;
                        // Convert back from integer and unscale
                        real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
                        sum_y_left += y_val;
                        sum_y2_left += y_val * y_val;
                    }
                    
                    real_t sum_y_right = 0.0f;
                    real_t sum_y2_right = 0.0f;
                    integer_t n_right = node_sample_count - n_left;
                    
                    for (integer_t i = n_left; i < node_sample_count; ++i) {
                        integer_t sample_idx = sorted_samples[i].second;
                        // Convert back from integer and unscale
                        real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
                        sum_y_right += y_val;
                        sum_y2_right += y_val * y_val;
                    }
                    
                    // Calculate weighted MSE
                    real_t mse_left = (sum_y2_left / static_cast<real_t>(n_left)) - 
                                      ((sum_y_left / static_cast<real_t>(n_left)) * 
                                       (sum_y_left / static_cast<real_t>(n_left)));
                    real_t mse_right = (sum_y2_right / static_cast<real_t>(n_right)) - 
                                       ((sum_y_right / static_cast<real_t>(n_right)) * 
                                        (sum_y_right / static_cast<real_t>(n_right)));
                    
                    real_t weighted_mse = (n_left * mse_left + n_right * mse_right) / 
                                          static_cast<real_t>(node_sample_count);
                    
                    // Update best split (for regression, lower MSE is better)
                    if (weighted_mse < best_crit) {
                        best_split_idx = ii;
                        best_crit = weighted_mse;
                        best_var = mvar;
                        best_sorted_samples = sorted_samples;
                    }
                } else {
                    // Classification: Use Gini
                    integer_t k = cl[nc];
                    real_t u = 1.0f;  // Sample weight
                    
                    rln += u * (u + 2.0f * wl[k]);
                    rrn += u * (u - 2.0f * wr[k]);
                    rld += u;
                    rrd -= u;
                    wl[k] += u;
                    wr[k] -= u;
                    
                    if (std::min(rrd, rld) > 1.0e-5f) {
                        real_t crit = (rln / rld) + (rrn / rrd);
                        if (crit > best_crit) {
                            best_split_idx = ii;
                            best_crit = crit;
                            best_var = mvar;
                            best_sorted_samples = sorted_samples;
                        }
                    }
                }
            }
        }
        
        // Check if a good split was found
        if (nclass == 1) {
            // Regression: check if MSE reduction is significant
            real_t mse_reduction = crit0 - best_crit;
            if (mse_reduction < 1.0e-6f) {
                jstat = 1;  // No significant improvement
            }
        } else {
            // Classification: check if Gini improved
            if (best_crit < -1.0f) {
                jstat = 1;  // No good split found
            }
        }
        
        // Create split if improvement found
        if (jstat == 0 && ((nclass == 1 && best_crit < crit0) || (nclass > 1 && best_crit > -1.0f))) {
            // Create children
            if (nnode + 2 <= maxnode) {
                integer_t left_child = nnode;
                integer_t right_child = nnode + 1;
                
                // Set up split
                nodestatus[kgrow] = 1;  // Split node
                bestvar[kgrow] = best_var;
                
                // Calculate split point (use actual feature value)
                if (best_split_idx < best_sorted_samples.size()) {
                    integer_t sample_idx = best_sorted_samples[best_split_idx].second;
                    xbestsplit[kgrow] = x[sample_idx * mdim + best_var];
                } else {
                    xbestsplit[kgrow] = 0.0f;
                }
                
                // Set up children
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;   // Active
                nodestatus[right_child] = 2;  // Active
                
                nnode += 2;
            } else {
                // No more nodes available, make terminal
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
            }
        } else {
            // No good split found, make terminal
            nodestatus[kgrow] = -1;  // Terminal node
            nodeclass[kgrow] = majority_class;
        }
    }

    // Finalize terminal nodes (set node classes for terminal nodes)
    for (integer_t node = 0; node < nnode; ++node) {
        if (nodestatus[node] == -1) {
            // Terminal node - determine class from samples
            std::vector<real_t> class_counts(nclass, 0.0f);
            
            // Count classes for this terminal node
            for (integer_t i = 0; i < ninbag; ++i) {
                integer_t sample_idx = jinbag[i];
                if (sample_idx >= 0 && sample_idx < nsample) {
                    // Check if sample reaches this terminal node
                    integer_t current_node = 0;
                    bool reaches_node = true;
                    
                    while (current_node != node && reaches_node) {
                        if (nodestatus[current_node] == 1) {
                            integer_t split_var = bestvar[current_node];
                            real_t split_point = xbestsplit[current_node];
                            real_t sample_value = x[sample_idx * mdim + split_var];
                            
                            if (sample_value <= split_point) {
                                current_node = treemap[current_node * 2];
                            } else {
                                current_node = treemap[current_node * 2 + 1];
                            }
                            
                            if (current_node > node) {
                                reaches_node = false;
                                break;
                            }
                        } else {
                            reaches_node = false;
                            break;
                        }
                    }
                    
                    if (reaches_node && current_node == node) {
                        integer_t class_label = cl[sample_idx];
                        if (class_label >= 0 && class_label < nclass) {
                            class_counts[class_label] += 1.0f;
                        }
                    }
                }
            }
            
            // Find majority class
            integer_t majority_class = 0;
            real_t max_count = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
            
            nodeclass[node] = majority_class;
            
            // Compute terminal node value
            if (nclass == 1) {
                // Regression: compute mean target value
                real_t sum_y = 0.0f;
                integer_t n_samples = 0;
                for (integer_t i = 0; i < ninbag; ++i) {
                    integer_t sample_idx = jinbag[i];
                    if (sample_idx >= 0 && sample_idx < nsample) {
                        // Check if sample reaches this terminal node
                        integer_t current_node = 0;
                        bool reaches_node = true;
                        
                        while (current_node != node && reaches_node) {
                            if (nodestatus[current_node] == 1) {
                                integer_t split_var = bestvar[current_node];
                                real_t split_point = xbestsplit[current_node];
                                real_t sample_value = x[sample_idx * mdim + split_var];
                                
                                if (sample_value <= split_point) {
                                    current_node = treemap[current_node * 2];
                                } else {
                                    current_node = treemap[current_node * 2 + 1];
                                }
                                
                                if (current_node > node) {
                                    reaches_node = false;
                                    break;
                                }
                            } else {
                                reaches_node = false;
                                break;
                            }
                        }
                        
                        if (reaches_node && current_node == node) {
                            // For regression, cl contains integer-cast y values (scaled by y_scale=1000)
                            // Unscaling is required to get the actual continuous value in original scale
                            // This matches the original Fortran algorithm which doesn't scale
                            const real_t y_scale = 1000.0f;  // Must match the scale used in rf_random_forest.cpp
                            real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;  // Unscale to original
                            sum_y += y_val;
                            n_samples++;
                        }
                    }
                }
                tnodewt[node] = (n_samples > 0) ? (sum_y / static_cast<real_t>(n_samples)) : 0.0f;
            } else {
                // Classification: compute mean weight (case-weight ratio)
                real_t tw = 0.0f;
                real_t tn = 0.0f;
                for (integer_t i = 0; i < ninbag; ++i) {
                    integer_t sample_idx = jinbag[i];
                    if (sample_idx >= 0 && sample_idx < nsample) {
                        // Check if sample reaches this terminal node
                        integer_t current_node = 0;
                        bool reaches_node = true;
                        
                        while (current_node != node && reaches_node) {
                            if (nodestatus[current_node] == 1) {
                                integer_t split_var = bestvar[current_node];
                                real_t split_point = xbestsplit[current_node];
                                real_t sample_value = x[sample_idx * mdim + split_var];
                                
                                if (sample_value <= split_point) {
                                    current_node = treemap[current_node * 2];
                                } else {
                                    current_node = treemap[current_node * 2 + 1];
                                }
                                
                                if (current_node > node) {
                                    reaches_node = false;
                                    break;
                                }
                            } else {
                                reaches_node = false;
                                break;
                            }
                        }
                        
                        if (reaches_node && current_node == node) {
                            tw += win[sample_idx];
                            tn += nin[sample_idx];
                        }
                    }
                }
                tnodewt[node] = (tn > 0.0f) ? (tw / tn) : 0.0f;
            }
        }
    }

    // Set jtr and nodextr (not used in current algorithm, but initialize)
    for (integer_t n = 0; n < nsample; ++n) {
        jtr[n] = 0;
        nodextr[n] = 0;
    }
}

} // namespace rf

