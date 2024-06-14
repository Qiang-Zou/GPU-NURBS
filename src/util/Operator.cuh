#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

__device__ double4 operator*(double w, double4 R);

__device__ double4 operator*(double4 R, double w);

__device__ double3 operator*(double w, double3 R);

__device__ double3 operator*(double3 R, double w);

__device__ double4 operator/(double4 R, double w);

__device__ double3 operator/(double3 R, double w);

__device__ double4 operator+(double4 a, double4 b);

__device__ double4 operator-(double4 a, double4 b);

__device__ double3 operator-(double3 a, double3 b);

__device__ double2 operator-(double2 a, double2 b);

__device__ double3 operator+(double3 a, double3 b);

__device__ void print(double4 a);

__device__ void print(double3 a);

__device__ void print(double2 a);

__device__ void print(double a);

__device__ void print(double* a, int k1, int k2);
