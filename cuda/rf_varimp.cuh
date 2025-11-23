#ifndef RF_VARIMP_CUH
#define RF_VARIMP_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches varimp_cuda in Fortran)
void varimp(const real_t* x, integer_t nsample, integer_t mdim,
            const integer_t* cl, const integer_t* nin, const integer_t* jtr,
            integer_t impn, real_t* qimp, real_t* qimpm,
            real_t* avimp, real_t* sqsd,
            const integer_t* treemap, const integer_t* nodestatus,
            const real_t* xbestsplit, const integer_t* bestvar,
            const integer_t* nodeclass, integer_t nnode,
            const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
            integer_t maxcat, const integer_t* catgoleft,
            const real_t* tnodewt, const integer_t* nodextr,
            integer_t* joob, integer_t* pjoob, integer_t* iv);

// CPU fallback - exact copy of original algorithm (matches varimp_cpu_fallback in Fortran)
void varimp_cpu_fallback(const real_t* x, integer_t nsample, integer_t mdim,
                         const integer_t* cl, const integer_t* nin, const integer_t* jtr,
                         integer_t impn, real_t* qimp, real_t* qimpm,
                         real_t* avimp, real_t* sqsd,
                         const integer_t* treemap, const integer_t* nodestatus,
                         const real_t* xbestsplit, const integer_t* bestvar,
                         const integer_t* nodeclass, integer_t nnode,
                         const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                         integer_t maxcat, const integer_t* catgoleft,
                         const real_t* tnodewt, const integer_t* nodextr,
                         integer_t* joob, integer_t* pjoob, integer_t* iv);

// Helper functions for CPU fallback
void testreeimp_cpu_fallback(const real_t* x, integer_t nsample, integer_t mdim,
                            const integer_t* cl, const integer_t* joob, integer_t* pjoob, integer_t noob,
                            integer_t mr, const integer_t* treemap, const integer_t* nodestatus,
                            const real_t* xbestsplit, const integer_t* bestvar,
                            const integer_t* nodeclass, integer_t nnode,
                            const integer_t* cat, integer_t maxcat,
                            const integer_t* catgoleft, integer_t* jvr, integer_t* nodexvr);

void permobmr_cpu_fallback(const integer_t* joob, integer_t* pjoob, integer_t noob);

} // namespace rf

// CUDA kernel declarations (must be outside namespace)
// GPU kernel to compute tnodewt (node weights) in parallel
// Each thread block handles one node, threads within block parallelize over samples
// For classification: tnodewt = sum(win) / sum(nin) for in-bag samples (matches CPU: tw/tn)
// For regression: tnodewt = mean(y) for samples in the node
// Matches CPU implementation: iterates through jinbag (in-bag samples) only
__global__ void gpu_compute_tnodewt_kernel(
    const rf::real_t* x, rf::integer_t nsample, rf::integer_t mdim, rf::integer_t nnode,
    const rf::integer_t* treemap, const rf::integer_t* nodestatus,
    const rf::real_t* xbestsplit, const rf::integer_t* bestvar,
    const rf::integer_t* cl, const rf::real_t* y_regression, const rf::real_t* win,
    const rf::integer_t* nin,  // Bootstrap frequency - needed for correct tnodewt computation
    const rf::integer_t* jinbag,  // In-bag sample indices (like CPU)
    rf::integer_t ninbag,  // Number of in-bag samples (like CPU)
    rf::integer_t nclass, rf::integer_t task_type,
    rf::real_t* tnodewt);

#endif // RF_VARIMP_CUH
