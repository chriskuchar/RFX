#ifndef RF_VARIMP_HPP
#define RF_VARIMP_HPP

#include "rf_types.hpp"

namespace rf {

// CPU variable importance implementations
void cpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob);

void cpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* joob, integer_t noob, integer_t mr,
                   const integer_t* treemap, const integer_t* nodestatus,
                   const real_t* xbestsplit, const integer_t* bestvar,
                   const integer_t* nodeclass, integer_t nnode,
                   const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                   integer_t* jvr, integer_t* nodexvr);

void cpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
               const integer_t* cl, const integer_t* nin, const integer_t* jtr,
               integer_t impn, real_t* qimp, real_t* qimpm,
               real_t* avimp, real_t* sqsd,
               const integer_t* treemap, const integer_t* nodestatus,
               const real_t* xbestsplit, const integer_t* bestvar,
               const integer_t* nodeclass, integer_t nnode,
               const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
               integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr);

// Regression-specific variable importance
void cpu_varimp_regression(const real_t* x, integer_t nsample, integer_t mdim,
                           const real_t* y_true, const integer_t* nin, const integer_t* jtr_predictions,
                           integer_t impn, real_t* qimp, real_t* qimpm,
                           real_t* avimp, real_t* sqsd,
                           const integer_t* treemap, const integer_t* nodestatus,
                           const real_t* xbestsplit, const integer_t* bestvar,
                           const integer_t* nodeclass, integer_t nnode,
                           const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                           integer_t maxcat, const integer_t* catgoleft,
                           const real_t* tnodewt, const integer_t* nodextr);

// GPU variable importance implementations with parallel permutation testing
void gpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob);

void gpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* treemap, const integer_t* bestvar,
                   const integer_t* nodeclass, const integer_t* cat,
                   const integer_t* nodestatus, const integer_t* catgoleft,
                   const real_t* xbestsplit, integer_t nnode, integer_t maxcat,
                   const integer_t* cl, integer_t nclass, integer_t* nodextr,
                   real_t* qimpm);

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
               integer_t nclass, integer_t task_type);

} // namespace rf

#endif // RF_VARIMP_HPP
