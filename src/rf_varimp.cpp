#include "rf_varimp.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ============================================================================
// CPU VARIABLE IMPORTANCE IMPLEMENTATION
// ============================================================================

void cpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob) {
    // Create a copy of joob for permutation
    for (integer_t i = 0; i < noob; ++i) {
        pjoob[i] = joob[i];
    }
    
    // Simple permutation - swap elements randomly
    MT19937 rng;
    rng.sgrnd(42);  // Fixed seed for reproducibility
    
    for (integer_t i = 0; i < noob; ++i) {
        integer_t j = static_cast<integer_t>(rng.randomu() * noob);
        std::swap(pjoob[i], pjoob[j]);
    }
}

void cpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* joob, integer_t noob, integer_t mr,
                   const integer_t* treemap, const integer_t* nodestatus,
                   const real_t* xbestsplit, const integer_t* bestvar,
                   const integer_t* nodeclass, integer_t nnode,
                   const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                   integer_t* jvr, integer_t* nodexvr) {
    
    // Create permuted OOB samples
    std::vector<integer_t> pjoob(noob);
    cpu_permobmr(joob, pjoob.data(), noob);
    
    // For each OOB sample
    for (integer_t n = 0; n < noob; ++n) {
        integer_t kt = 0;  // Start at root node
        
        // Traverse tree
        for (integer_t k = 0; k < nnode; ++k) {
            if (nodestatus[kt] == -1) {  // Terminal node
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                break;
            }
            
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) break;  // Invalid variable index
            real_t xmn;
            
            if (m == mr) {
                // Use permuted value for variable mr
                // Column-major indexing: column m, row pjoob[n] (matches GPU)
                if (pjoob[n] < 0 || pjoob[n] >= nsample) break;  // Invalid sample index
                integer_t idx = m + pjoob[n] * mdim;
                if (idx >= 0 && idx < nsample * mdim) {
                    xmn = x[idx];
                } else {
                    break;  // Out of bounds
                }
            } else {
                // Use original value
                // Column-major indexing: column m, row joob[n] (matches GPU)
                if (joob[n] < 0 || joob[n] >= nsample) break;  // Invalid sample index
                integer_t idx = m + joob[n] * mdim;
                if (idx >= 0 && idx < nsample * mdim) {
                    xmn = x[idx];
                } else {
                    break;  // Out of bounds
                }
            }
            
            // Determine if case goes left or right
            if (cat == nullptr || cat[m] == 1) {  // Quantitative variable (or cat is nullptr)
                if (xmn <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];  // Left child
                } else {
                    kt = treemap[1 + kt * 2];  // Right child
                }
            } else {  // Categorical variable
                integer_t jcat = static_cast<integer_t>(xmn + 0.5f);  // Round to nearest integer
                if (catgoleft != nullptr && jcat >= 0 && jcat < maxcat) {
                    if (catgoleft[jcat + kt * maxcat] == 1) {
                        kt = treemap[0 + kt * 2];  // Left child
                    } else {
                        kt = treemap[1 + kt * 2];  // Right child
                    }
                } else {
                    // Invalid categorical value, cannot traverse further
                    break;
                }
            }
        }
    }
}

void cpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
               const integer_t* cl, const integer_t* nin, const integer_t* jtr,
               integer_t impn, real_t* qimp, real_t* qimpm,
               real_t* avimp, real_t* sqsd,
               const integer_t* treemap, const integer_t* nodestatus,
               const real_t* xbestsplit, const integer_t* bestvar,
               const integer_t* nodeclass, integer_t nnode,
               const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
               integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr) {
    
    // Set OpenMP thread count from config
#ifdef _OPENMP
    if (g_config.n_threads_cpu > 0) {
        omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
    }
#endif
    
    // printf("DEBUG: cpu_varimp called with nsample=%d, mdim=%d, nnode=%d\n", nsample, mdim, nnode);
    
    // Initialize arrays
    zervr(qimp, nsample);
    zervr(qimpm, nsample * mdim);
    zervr(avimp, mdim);
    zervr(sqsd, mdim);
    
    // Step 1: Find OOB samples and calculate original accuracy
    integer_t noob = 0;
    real_t right = 0.0f;
    std::vector<integer_t> joob(nsample);
    
    // Check if we should use case-wise (bootstrap frequency weighted) or non-case-wise (simple averaging)
    bool use_casewise = g_config.use_casewise;
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] == 0) {  // OOB sample
            // Update count of correct OOB classifications
            if (jtr[n] == cl[n]) {
                // Case-wise: use bootstrap frequency weighted (tnodewt)
                // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                if (use_casewise && nodextr[n] >= 0 && nodextr[n] < nnode) {
                    right += tnodewt[nodextr[n]];
                } else {
                    right += 1.0f;
                }
            }
            joob[noob] = n;  // Store OOB sample index
            noob++;
        }
    }
    
    // printf("DEBUG: Found %d OOB samples, original accuracy=%f\n", noob, right);
    
    // Step 2: Update qimp for local importance (if impn == 1)
    // Case-wise: Match Fortran varimp.f exactly: qimp(nn) = qimp(nn) + tnodewt(nodextr(nn))/noob
    // Non-case-wise: Simple averaging: qimp(nn) = qimp(nn) + 1.0/noob
    if (impn == 1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (integer_t n = 0; n < noob; ++n) {
            integer_t nn = joob[n];
            if (jtr[nn] == cl[nn]) {
                real_t weight = 1.0f;
                if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                    weight = tnodewt[nodextr[nn]];
                }
                qimp[nn] += weight / static_cast<real_t>(noob);
            }
        }
    }
    
    // Step 3: Mark which variables were used in splits
    std::vector<integer_t> iv(mdim, 0);
    for (integer_t jj = 0; jj < nnode; ++jj) {
        if (nodestatus[jj] != -1) {  // Non-terminal node
            iv[bestvar[jj]] = 1;
        }
    }
    
    // Step 4: Calculate importance for each variable (parallelize across variables)
    // Use thread-local arrays to avoid race conditions
#ifdef _OPENMP
#pragma omp parallel
    {
        // Thread-local arrays for each thread
        std::vector<integer_t> jvr_local(noob, 0);
        std::vector<integer_t> nodexvr_local(noob, 0);
        
#pragma omp for nowait
        for (integer_t k = 0; k < mdim; ++k) {
            if (iv[k] == 1) {  // Variable was used in splits
                // Test tree with permuted variable k (use thread-local arrays)
                cpu_testreeimp(x, nsample, mdim, joob.data(), noob, k,
                              treemap, nodestatus, xbestsplit, bestvar, nodeclass,
                              nnode, cat, maxcat, catgoleft, jvr_local.data(), nodexvr_local.data());
                
                // Calculate permuted accuracy
                real_t rightimp = 0.0f;
                for (integer_t n = 0; n < noob; ++n) {
                    integer_t nn = joob[n];  // Original case index
                    if (impn == 1) {
                        if (jvr_local[n] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                                weight = tnodewt[nodexvr_local[n]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
                    if (jvr_local[n] == cl[nn]) {
                        real_t weight = 1.0f;
                        if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                            weight = tnodewt[nodexvr_local[n]];
                        }
                        rightimp += weight;
                    }
                }
                
                // Calculate importance (use atomic operations for thread safety)
                real_t rr = (right - rightimp) / static_cast<real_t>(noob);
                avimp[k] += rr;
                sqsd[k] += rr * rr;
            } else {
                // Variable not used in splits - use original predictions (permuting has no effect)
                for (integer_t n = 0; n < noob; ++n) {
                    integer_t nn = joob[n];
                    if (impn == 1) {
                        if (jtr[nn] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                                weight = tnodewt[nodextr[nn]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
                }
            }
        }
    }
#else
    // Sequential version (no OpenMP)
    for (integer_t k = 0; k < mdim; ++k) {
        if (iv[k] == 1) {  // Variable was used in splits
            // Test tree with permuted variable k
            cpu_testreeimp(x, nsample, mdim, joob.data(), noob, k,
                          treemap, nodestatus, xbestsplit, bestvar, nodeclass,
                          nnode, cat, maxcat, catgoleft, jvr, nodexvr);
            
            // Calculate permuted accuracy
            real_t rightimp = 0.0f;
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];  // Original case index
                if (impn == 1) {
                    if (jvr[n] == cl[nn]) {
                        // Case-wise: use bootstrap frequency weighted (tnodewt)
                        // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                        real_t weight = 1.0f;
                        if (use_casewise && nodexvr[n] >= 0 && nodexvr[n] < nnode) {
                            weight = tnodewt[nodexvr[n]];
                        }
                        qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                    }
                }
                if (jvr[n] == cl[nn]) {
                    real_t weight = 1.0f;
                    if (use_casewise && nodexvr[n] >= 0 && nodexvr[n] < nnode) {
                        weight = tnodewt[nodexvr[n]];
                    }
                    rightimp += weight;
                }
            }
            
            // Calculate importance
            real_t rr = (right - rightimp) / static_cast<real_t>(noob);
            avimp[k] += rr;
            sqsd[k] += rr * rr;
        } else {
            // Variable not used in splits - use original predictions (permuting has no effect)
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];
                    if (impn == 1) {
                        if (jtr[nn] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                                weight = tnodewt[nodextr[nn]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
            }
        }
    }
#endif
    
    // printf("DEBUG: Final avimp values: [%f, %f, %f, %f]\n", avimp[0], avimp[1], avimp[2], avimp[3]);
}

} // namespace rf
