#ifndef RF_TYPES_HPP
#define RF_TYPES_HPP

#include <cstdint>

namespace rf {

// Precision types
using float32 = float;
using float64 = double;
using int32 = int32_t;

// Default precision (matches Fortran REAL and INTEGER)
using real_t = float32;
using integer_t = int32;

// Double precision (matches Fortran DOUBLE PRECISION)
using dp_t = float64;

// FP16 type for CUDA (will be __half in CUDA code)
#ifdef __CUDACC__
#include <cuda_fp16.h>
using fp16_t = __half;
#else
using fp16_t = uint16_t;  // Placeholder for non-CUDA builds
#endif

} // namespace rf

#endif // RF_TYPES_HPP
