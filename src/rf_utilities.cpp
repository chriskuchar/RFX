#include "rf_utilities.hpp"
#include "rf_utils.hpp"
#include "rf_arrays.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>

namespace rf {

// ----------------------------------------------------------------------
// Compute error rate (exact port of comperr.f90)
// ----------------------------------------------------------------------
void comperr(const integer_t* jest, const integer_t* cl, integer_t nsample, dp_t& errtr) {
    errtr = 0.0;
    for (integer_t n = 0; n < nsample; ++n) {
        if (jest[n] != cl[n]) {
            errtr += 1.0;
        }
    }
    errtr = errtr / static_cast<dp_t>(nsample);
}

// ----------------------------------------------------------------------
// Local importance calculation (exact port of localimp.f90)
// ----------------------------------------------------------------------
void localimp(integer_t nsample, integer_t mdim, integer_t ntree,
              const real_t* qimp, real_t* qimpm) {
    // Array layout: column-major format (Fortran-compatible)
    // qimpm indexing: qimpm[feature * nsample + sample] = qimpm[m * nsample + n]
    // This matches GPU accumulation and Fortran array layout expectations
    // Column-major: qimpm[feature * nsample + sample] = qimpm[m * nsample + n]
    for (integer_t n = 0; n < nsample; ++n) {
        for (integer_t m = 0; m < mdim; ++m) {
            // Fortran: qimpm(n, m) = 100.0 * (qimp(n) - qimpm(n, m)) / real(ntree)
            // Column-major: qimpm[m * nsample + n] (feature-major, matches GPU)
            integer_t idx = m * nsample + n;  // Column-major: feature * nsample + sample
            real_t old_value = qimpm[idx];
            qimpm[idx] = 100.0f * (qimp[n] - qimpm[idx]) / static_cast<real_t>(ntree);
        }
    }
}

// ----------------------------------------------------------------------
// Prepare data - sort and handle ties (exact port of prepdata.f90)
// ----------------------------------------------------------------------
void prepdata(const real_t* x, integer_t mdim, integer_t nsample,
              const integer_t* cat, integer_t* isort, real_t* v,
              integer_t* asave, integer_t* ties) {

    // x is (nsample, mdim) row-major
    // asave is (mdim, nsample) column-major (keep for compatibility)
    // ties is (mdim, nsample) column-major (keep for compatibility)

    for (integer_t mvar = 0; mvar < mdim; ++mvar) {
        if (cat[mvar] == 1) {
            // Quantitative variable - need to sort and handle ties

            // Copy column to v for sorting
            for (integer_t n = 0; n < nsample; ++n) {
                v[n] = x[n * mdim + mvar];  // Row-major access: x[sample, feature]
                isort[n] = n;  // 0-based indices
            }

            // Sort v and isort together
            // std::cout << "DEBUG: About to call quicksort for variable " << mvar << std::endl;
            // std::cout.flush();
            
            // Temporary: Use std::sort instead of quicksort to test
            std::vector<std::pair<real_t, integer_t>> pairs;
            for (integer_t n = 0; n < nsample; ++n) {
                pairs.push_back({v[n], isort[n]});
            }
            std::sort(pairs.begin(), pairs.end());
            for (integer_t n = 0; n < nsample; ++n) {
                v[n] = pairs[n].first;
                isort[n] = pairs[n].second;
            }
            
            // std::cout << "DEBUG: Completed sorting for variable " << mvar << std::endl;
            // std::cout.flush();

            // Copy sorted indices to asave
            for (integer_t n = 0; n < nsample; ++n) {
                integer_t idx = mvar + n * mdim;
                if (idx >= 0 && idx < mdim * nsample) {
                    asave[idx] = isort[n];
                } else {
                    // std::cout << "ERROR: asave index out of bounds: " << idx << " >= " << (mdim * nsample) << std::endl;
                    // std::cout.flush();
                }
            }

            // Handle ties - mark tied values
            for (integer_t n = 0; n < nsample; ++n) {
                integer_t idx = mvar + n * mdim;
                if (idx >= 0 && idx < mdim * nsample) {
                    ties[idx] = 0;
                } else {
                    // std::cout << "ERROR: ties index out of bounds: " << idx << " >= " << (mdim * nsample) << std::endl;
                    // std::cout.flush();
                }
            }

            integer_t n = 0;
            while (n < nsample - 1) {
                // Check if current and next values are tied
                if (std::abs(v[n] - v[n + 1]) < 1e-6f) {
                    // Found a tie - mark all tied values
                    integer_t tie_start = n;
                    while (n < nsample - 1 && std::abs(v[n] - v[n + 1]) < 1e-6f) {
                        integer_t idx = mvar + n * mdim;
                        if (idx >= 0 && idx < mdim * nsample) {
                            ties[idx] = 1;
                        } else {
                            // std::cout << "ERROR: ties index out of bounds in tie loop: " << idx << " >= " << (mdim * nsample) << std::endl;
                            // std::cout.flush();
                        }
                        ++n;
                    }
                    integer_t idx = mvar + n * mdim;
                    if (idx >= 0 && idx < mdim * nsample) {
                        ties[idx] = 1;  // Mark last in tie group
                    } else {
                        // std::cout << "ERROR: ties index out of bounds for last tie: " << idx << " >= " << (mdim * nsample) << std::endl;
                        // std::cout.flush();
                    }
                }
                ++n;
            }

        } else {
            // Categorical variable - just copy rounded values
            for (integer_t n = 0; n < nsample; ++n) {
                asave[mvar + n * mdim] = static_cast<integer_t>(std::round(x[mvar + n * mdim]));
                ties[mvar + n * mdim] = 0;  // No ties for categorical
            }
        }
    }
}

} // namespace rf
