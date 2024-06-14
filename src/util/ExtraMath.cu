#include "ExtraMath.cuh"

__device__ double3 homogenousToCartesian(double4 P) {
	return { P.x / P.w,P.y / P.w,P.z / P.w };
}

__device__ double _2Norm(double3 p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ double3 TruncateHomogenous(double4 R) {
	return { R.x,R.y,R.z };
} 