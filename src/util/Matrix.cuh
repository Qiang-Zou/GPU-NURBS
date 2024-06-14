#include <stdio.h>
#include <typeinfo>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "math.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"

__global__ void gpu_matrix_mult(double *a, double *b, double *c, int m, int n, int k);

__global__ void gpu_matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols);

__device__ double norm(const double3 &P);

__device__ void matrix_mult(double *a, double *b, double *c, int m, int n, int k);

__device__ void matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols);

__device__ double3 cross_product(double3 a, double3 b);

__device__ double dot_product(double3 a, double3 b);

// Called by host, copy device matrix src with a*b to des matrix with c*d 
// src and des are both row-major
// type double to half
__global__ void MatrixFit(double* src, int a, int b, half* des, int c, int d);

// A and B are entry pointers of a array of col-major matrix
// Ex. A[0] is a entry pointer of a m*n col-major matrix
// result stores the result of the multiply of jth row of A[i] and jth col of B[i]
__global__ void VectorMultiply(half** A, half** B, int m, int n, half* result);


