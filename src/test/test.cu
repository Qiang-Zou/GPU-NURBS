#include "test.cuh"

__global__ void NormalTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double2 Current) {
	int m = findSpan(degree_u, cp_u, knots_u, Current.x);
	int n = findSpan(degree_v, cp_v, knots_v, Current.y);
	double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double offset = 1e-11;
	double u_offset = (Current.x + offset - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	double v_offset = (Current.y + offset - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectorUT[MAX_DEGREE], vectorV[MAX_DEGREE];
	double now = 1;
	for (int i = 0; i < k1; i++) {
		vectorUT[i] = now;
		now *= u;
	}
	now = 1;
	for (int i = 0; i < k2; i++) {
		vectorV[i] = now;
		now *= v;
	}

	double vectordV[MAX_DEGREE];
	now = 1;
	vectordV[0] = 0;
	for (int i = 0; i < k2 - 1; i++) {
		vectordV[i + 1] = (i + 1) * now;
		now *= v;
	}

	double vectordVV[MAX_DEGREE];
	now = 1;
	vectordVV[0] = vectordVV[1] = 0;
	for (int i = 0; i < k2 - 2; i++) {
		vectordVV[i + 2] = (i + 2)*(i + 1)*now;
		now *= v;
	}

	double vectordVVV[MAX_DEGREE];
	now = 1;
	vectordVVV[0] = vectordVVV[1] = vectordVVV[2] = 0;
	for (int i = 0; i < k2 - 3; i++) {
		vectordVVV[i + 3] = (i + 3)*(i + 2)*(i + 1)*now;
		now *= v;
	}

	double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	double4 RdV = MatrixSurfaceRdV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordV, spanV);
	double4 RdVV = MatrixSurfaceRdVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordVV, spanV);
	double4 RdVVV = MatrixSurfaceRdVVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordVVV, spanV);
	now = 1;
	vectorUT[0] = 0;
	for (int i = 0; i < k1 - 1; i++) {
		vectorUT[i + 1] = (i + 1) * now;
		now *= u;
	}
	// for (int i = 0; i < k1; i++){
	// 	printf("%lf ", vectorUT[i]);
	// }
	// printf("\n");
	double4 RdU = MatrixSurfaceRdU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectorV, spanU);
	double4 RdUV = MatrixSurfaceRdUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordV, spanU, spanV);
	double4 RdUVV = MatrixSurfaceRdUVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordVV, spanU, spanV);
	now = 1;
	vectorUT[0] = vectorUT[1] = 0;
	for (int i = 0; i < k1 - 2; i++) {
		vectorUT[i + 2] = (i + 2)*(i + 1) * now;
		now *= u;
	}
	double4 RdUU = MatrixSurfaceRdUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectorV, spanU);
	double4 RdUUV = MatrixSurfaceRdUUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordV, spanU, spanV);
	now = 1;
	vectorUT[0] = vectorUT[1] = vectorUT[2] = 0;
	for (int i = 0; i < k1 - 3; i++) {
		vectorUT[i + 3] = (i + 3)*(i + 2)*(i + 1) * now;
		now *= u;
	}
	double4 RdUUU = MatrixSurfaceRdUUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectorV, spanU);

	double3 P = homogenousToCartesian(R);
	double3 PdU = MatrixSurfaceDerivativeU(RdU, P, R.w);
	printf("RdU: ");
	print(RdU);
	printf("P: ");
	print(P);
	printf("R.w: ");
	print(R.w);
	double3 PdV = MatrixSurfaceDerivativeV(RdV, P, R.w);
	double3 PdUU = MatrixSurfaceDerivativeUU(RdUU, PdU, P, R.w, RdU.w);
	double3 PdUV = MatrixSurfaceDerivativeUV(RdUV, PdU, PdV, P, R.w, RdU.w, RdV.w);
	double3 PdVV = MatrixSurfaceDerivativeVV(RdVV, PdV, P, R.w, RdV.w);
	double3 PdUUU = MatrixSurfaceDerivativeUUU(RdUUU, PdUU, PdU, P, R.w, RdU.w, RdUU.w);
	double3 PdUUV = MatrixSurfaceDerivativeUUV(RdUUV, PdUU, PdUV, PdU, PdV, P, R.w, RdU.w, RdV.w, RdUU.w, RdUV.w);
	double3 PdUVV = MatrixSurfaceDerivativeUVV(RdUVV, PdUV, PdVV, PdU, PdV, P, R.w, RdU.w, RdV.w, RdUV.w, RdVV.w);
	double3 PdVVV = MatrixSurfaceDerivativeVVV(RdVVV, PdVV, PdV, P, R.w, RdV.w, RdVV.w);

	/*print(R);
	print(RdU);
	print(RdV);
	print(RdUU);
	print(RdUV);
	print(RdVV);
	print(RdUUU);
	print(RdUUV);
	print(RdUVV);
	print(RdVVV);

	printf("\n");

	print(PdU);
	print(PdV);
	print(PdUU);
	print(PdUV);
	print(PdVV);
	print(PdUUU);
	print(PdUUV);
	print(PdUVV);
	print(PdVVV);

	printf("\n");

	print(MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanU));
	print(MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanV));
	print(TrueNormal(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, spanU, spanV));

	printf("\n");*/

	double3 PdUU_u_offset = MatrixSurfaceDerivativeUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u_offset, v, R, spanU, spanV);
	double3 PdUU_v_offset = MatrixSurfaceDerivativeUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v_offset, R, spanU, spanV);
	double3 PdVV_u_offset = MatrixSurfaceDerivativeVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u_offset, v, R, spanU, spanV);
	double3 PdVV_v_offset = MatrixSurfaceDerivativeVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v_offset, R, spanU, spanV);

	printf("PdU: ");
	print(PdU);
	printf("PdV: ");
	print(PdV);

	printf("PdUU: ");
	print(PdUU);
	printf("PdUV: ");
	print(PdUV);
	printf("PdVV: ");
	print(PdVV);

	printf("Calculated PdUUU: ");
	print(PdUUU);
	/*printf("Estimated PdUUU: ");
	print((PdUU_u_offset - PdUU) / offset);*/
	printf("Calculated PdUUV: ");
	print(PdUUV);
	/*printf("Estimated PdUUV: ");
	print((PdUU_v_offset - PdUU) / offset);*/
	printf("Calculated PdUVV: ");
	print(PdUVV);
	/*printf("Estimated PdUVV: ");
	print((PdVV_u_offset - PdVV) / offset);*/
	printf("Calculated PdVVV: ");
	print(PdVVV);
	/*printf("Estimated PdVVV: ");
	print((PdVV_v_offset - PdVV) / offset);*/
	printf("Normal: ");
	print(TrueNormal(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, spanU, spanV));
	printf("NdUU: ");
	print(SurfaceNormalDerivativeUU(PdU, PdV, PdUU, PdUV, PdUUU, PdUUV));
	printf("NdUV: ");
	print(SurfaceNormalDerivativeUV(PdU, PdV, PdUU, PdUV, PdVV, PdUUV, PdUVV));
	printf("NdVV: ");
	print(SurfaceNormalDerivativeVV(PdU, PdV, PdUV, PdVV, PdUVV, PdVVV));

	double3 tmp = TruncateHomogenous(RdVVV);
	print(tmp);
	tmp = tmp - 3 * RdV.w *PdVV;
	print(tmp);
	tmp = tmp - 3 * RdVV.w*PdV;
	print(tmp);
	tmp = tmp - RdVVV.w*P;
	print(tmp);
	tmp = tmp / R.w;
	print(tmp);

	printf("\n");
	print(R);
	print(RdV);
	print(RdVV);
	print(RdVVV);
}

__global__ void print(half *a, int k1, int k2) {
	for (int i = 0; i < k1; i++) {
		for (int j = 0; j < k2; j++) {
			float t = __half2float(a[i*k2 + j]);
			printf("%f ", t);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void print(half **a, int s, int e, int k1, int k2){
	for(int k=s;k<e;k++){
		for (int i = 0; i < k1; i++) {
			for (int j = 0; j < k2; j++) {
				float t = __half2float(a[k][i*k2 + j]);
				printf("%f ", t);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__global__ void kernel_print(double* a, int k1, int k2) {
	for (int i = 0; i < k1; i++) {
		for (int j = 0; j < k2; j++) {
			printf("%lf ", a[i*k2 + j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void kernel_print(double4* a, int k){
	for (int i = 0; i < k; i++) {
		print(a[i]);
	}
}

__global__ void PointTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double u, double v){
	int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

	double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);

	print(R);
}

__global__ void print_diag(half *a, int k){
	for (int i = 0; i < k; i++) {
		float t = __half2float(a[i*k * i]);
		printf("%f ", t);
	}
	printf("\n");
}

__global__ void PerfTest(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double u, double v, double4* out){
	int index = gridDim.x * blockIdx.x + threadIdx.x;
	int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

	out[index] = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	// if(index == 0)
	// 	print(out[index]);
}

//-------------------generate random numbers-------//
__device__ float generate(curandState *globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);// uniform distribution
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = iy * blockDim.x*gridDim.x + ix;
	curand_init(seed, idx, 0, &state[idx]);// initialize the state
}

__global__ void RandomAccess(curandState *globalState,int nx,int ny,int* arr,int size)
{
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = iy * blockDim.x*gridDim.x + ix;

	int t = 0, k;

	if (ix < nx&&iy < ny)
	{
		for(int i=0;i<1000;i++){
			k = generate(globalState, idx) * size;
			t += arr[k];
		}
	}
	// arr[idx] = t;
}

__global__ void UniformAccess(curandState *globalState,int nx,int ny,int* arr,int size){
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = iy * blockDim.x*gridDim.x + ix;

	int t = 0, k;

	if (ix < nx&&iy < ny)
	{
		for(int i=0;i<1000;i++){
			k = generate(globalState, idx) * size;
			t += arr[0];
		}
	}
	// arr[idx] = t;
}