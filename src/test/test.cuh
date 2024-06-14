#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "Evaluate.cuh"
#include "ExtraMath.cuh"
#include "math.h"
#include "Operator.cuh"
#include "cublas_v2.h"
#include "curand_kernel.h"

__global__ void NormalTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double2 Current);

__global__ void PointTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double u, double v);

__global__ void print(half *a, int k1, int k2);

__global__ void print(half **a, int s, int e, int k1, int k2);

__global__ void kernel_print(double* a, int k1, int k2);

__global__ void kernel_print(double4* a, int k);

// print diagonal of a k*k matrix
__global__ void print_diag(half *a, int k);

__global__ void PerfTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double u, double v, double4* out);

__device__ float generate(curandState *globalState, int ind);

__global__ void setup_kernel(curandState *state, unsigned long seed);

__global__ void RandomAccess(curandState *globalState,int nx,int ny,int* arr,int size);

__global__ void UniformAccess(curandState *globalState,int nx,int ny,int* arr,int size);