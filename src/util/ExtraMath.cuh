#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

__device__ double3 homogenousToCartesian(double4 P);

__device__ double _2Norm(double3 p);

__device__ double3 TruncateHomogenous(double4 R);