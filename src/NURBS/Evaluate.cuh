#include <stdio.h>
#include <time.h>
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "Matrix.cuh"
#include "Operator.cuh"
#include "ExtraMath.cuh"

#define BLOCK_SIZE 2
//GTX 1660
#define GPU_clock_rate 1785000
//max degree for NURBS (k+1)
#define MAX_DEGREE 8

__global__ void GetMatrixN(double *MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights);

__device__ double Nrc(double* knot, int m, int r, int c, int j, int i, int d);

__device__ double Nabla(double* knot, int d, int i, int k);

__host__ __device__ int MatrixNPosition(int i, int j, int degree_u, int degree_v, int cp_u, int cp_v, int index);

// output R in 4D
__device__ double4 MatrixSurfacePoint(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v);

__device__ double4 MatrixSurfacePoint(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectorV);

__device__ double4 MatrixSurfaceRdU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectorV, double spanU);

__device__ double4 MatrixSurfaceRdV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordV, double spanV);

__device__ double4 MatrixSurfaceRdUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUT, double* vectorV, double spanU);

__device__ double4 MatrixSurfaceRdUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectordV, double spanU, double spanV);

__device__ double4 MatrixSurfaceRdVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordVV, double spanV);

__device__ double4 MatrixSurfaceRdUUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUUT, double* vectorV, double spanU);

__device__ double4 MatrixSurfaceRdUUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUT, double* vectordV, double spanU, double spanV);

__device__ double4 MatrixSurfaceRdUVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectordVV, double spanU, double spanV);

__device__ double4 MatrixSurfaceRdVVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordVVV, double spanV);

//Input:输入的uv是原参数域uv从[knots[m],knots[m + 1])线性映射到[0,1)上的值 span是映射的放大倍数
//Output:对原参数域uv的导数
//dU 
__device__ double3 MatrixSurfaceDerivativeU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU);

//dV
__device__ double3 MatrixSurfaceDerivativeV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanV);

// old version of surface der
//dU2
__device__ double3 MatrixSurfaceDerivativeUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV);

//dUV
__device__ double3 MatrixSurfaceDerivativeUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV);

//dV2
__device__ double3 MatrixSurfaceDerivativeVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV);

//new version of surface der
__device__ double3 MatrixSurfaceDerivativeU(double4 RdU, double3 P, double h);

__device__ double3 MatrixSurfaceDerivativeV(double4 RdV, double3 P, double h);


__device__ double3 MatrixSurfaceDerivativeUU(double4 RdUU, double3 PdU, double3 P, double h, double hdU);

__device__ double3 MatrixSurfaceDerivativeVV(double4 RdVV, double3 PdV, double3 P, double h, double hdV);

__device__ double3 MatrixSurfaceDerivativeUV(double4 RdUV, double3 PdU, double3 PdV, double3 P, double h, double hdU, double hdV);

//dU3
__device__ double3 MatrixSurfaceDerivativeUUU(double4 RdUUU, double3 PdUU, double3 PdU, double3 P, double h, double hdU, double hdUU);

//dU2V
__device__ double3 MatrixSurfaceDerivativeUUV(double4 RdUUV, double3 PdUU, double3 PdUV, double3 PdU, double3 PdV, double3 P,
	double h, double hdU, double hdV, double hdUU, double hdUV);

//dUV2
__device__ double3 MatrixSurfaceDerivativeUVV(double4 RdUVV, double3 PdUV, double3 PdVV, double3 PdU, double3 PdV, double3 P,
	double h, double hdU, double hdV, double hdUV, double hdVV);

//dV3
__device__ double3 MatrixSurfaceDerivativeVVV(double4 RdVVV, double3 PdVV, double3 PdV, double3 P, double h, double hdV, double hdVV);

__host__ __device__ int findSpan(unsigned int degree, int cp, double *knots, double u);

__global__ void check(double* MatrixN, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v);

__global__ void EvaluateRandom(double2* random, double3* random3,
	double* MatrixN, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v);

__device__ double4 matrix_test(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v);

// true normal
__device__ double3 TrueNormal(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v, double spanU, double spanV);

// normal calculated by matrix representation
__device__ double3 Normal(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v, double spanU, double spanV);

// the last argument has no name and will never be used
__device__ double3 SurfaceNormalDerivativeUU(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdUUU, double3 PdUUV); 

__device__ double3 SurfaceNormalDerivativeUV(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdVV, double3 PdUUV, double3 PdUVV);

__device__ double3 SurfaceNormalDerivativeVV(double3 PdU, double3 PdV, double3 PdUV, double3 PdVV, double3 PdUVV, double3 PdVVV);