#include "Evaluate.cuh"

__global__ void GetMatrixN(double *MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights) {
	//get Nu and Nv
	int i = blockIdx.x, j = threadIdx.x;
	//int i = 4, j = 2;

	//printf("%f\n", knots_u[5]);
	if (knots_u[i] == knots_u[i + 1] || knots_v[j] == knots_v[j + 1])
		return;

	//printf("%d %d\n", i, j);

	double* Nu = new double[(degree_u + 1)*(degree_u + 1)];
	for (int m = 0; m < degree_u + 1; m++) {
		for (int n = 0; n < degree_u + 1; n++) {
			Nu[m*(degree_u + 1) + n] = Nrc(knots_u, degree_u, m + 1, n + 1, i - (degree_u - 1), i - (degree_u - 1), degree_u);
		} 
	}

	//transpose in place
	double* NvT = new double[(degree_v + 1)*(degree_v + 1)];
	for (int m = 0; m < degree_v + 1; m++) {
		for (int n = 0; n < degree_v + 1; n++) {
			NvT[n*(degree_v + 1) + m] = Nrc(knots_v, degree_v, m + 1, n + 1, j - (degree_v - 1), j - (degree_v - 1), degree_v);
		}
	}

	double* MiddleX = new double[(degree_u + 1)*(degree_v + 1)];
	double* MiddleY = new double[(degree_u + 1)*(degree_v + 1)];
	double* MiddleZ = new double[(degree_u + 1)*(degree_v + 1)];
	double* MiddleW = new double[(degree_u + 1)*(degree_v + 1)];

	for (int s = 0; s <= degree_u; s++) {
		for (int l = 0; l <= degree_v; l++) {
			double w = weights[(s + i - degree_u)*cp_v + (l + j - degree_v)];
			MiddleX[s*(degree_v + 1) + l] = control_points[(s + i - degree_u)*cp_v + (l + j - degree_v)].x * w;
			MiddleY[s*(degree_v + 1) + l] = control_points[(s + i - degree_u)*cp_v + (l + j - degree_v)].y * w;
			MiddleZ[s*(degree_v + 1) + l] = control_points[(s + i - degree_u)*cp_v + (l + j - degree_v)].z * w;
			MiddleW[s*(degree_v + 1) + l] = w;
		}
	}

	unsigned int grid_u = ((degree_u + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = ((degree_v + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGriduv(grid_u, grid_v);
	dim3 dimGridvv(grid_v, grid_v);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	//double* NvT = new double[(degree_v + 1)*(degree_v + 1)];
	//gpu_matrix_transpose<<<dimGridvv,dimBlock>>>(Nv, NvT, degree_v + 1, degree_v + 1);
	double* tempX = new double[(degree_u + 1)*(degree_v + 1)];
	double* tempY = new double[(degree_u + 1)*(degree_v + 1)];
	double* tempZ = new double[(degree_u + 1)*(degree_v + 1)];
	double* tempW = new double[(degree_u + 1)*(degree_v + 1)];

	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(MiddleX, NvT, tempX, degree_u + 1, degree_v + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(MiddleY, NvT, tempY, degree_u + 1, degree_v + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(MiddleZ, NvT, tempZ, degree_u + 1, degree_v + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(MiddleW, NvT, tempW, degree_u + 1, degree_v + 1, degree_v + 1);

	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(Nu, tempX, MatrixN + MatrixNPosition(i, j, degree_u, degree_v, cp_u, cp_v, 0), degree_u + 1, degree_u + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(Nu, tempY, MatrixN + MatrixNPosition(i, j, degree_u, degree_v, cp_u, cp_v, 1), degree_u + 1, degree_u + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(Nu, tempZ, MatrixN + MatrixNPosition(i, j, degree_u, degree_v, cp_u, cp_v, 2), degree_u + 1, degree_u + 1, degree_v + 1);
	gpu_matrix_mult<<<dimGriduv,dimBlock>>>(Nu, tempW, MatrixN + MatrixNPosition(i, j, degree_u, degree_v, cp_u, cp_v, 3), degree_u + 1, degree_u + 1, degree_v + 1);
	// cudaDeviceSynchronize();
	//test
	/*if (i == 4 && j == 2) {
		for (int a = 0; a < degree_u + 1; a++) {
			for (int b = 0; b < degree_v + 1; b++) {
				printf("%f ", MatrixN[MatrixNPosition(i, j, degree_u, degree_v, cp_u, cp_v, 0) + a * (degree_v + 1) + b]);
			}
			printf("\n");
		}
	}*/
	//free memory
	delete[] Nu;
	delete[] NvT;
	delete[] MiddleX;
	delete[] MiddleY;
	delete[] MiddleZ;
	delete[] MiddleW;
	delete[] tempX;
	delete[] tempY;
	delete[] tempZ;
	delete[] tempW;
}

__device__ double Nrc(double* knot, int m, int r, int c, int j, int i, int d) {
	if (m == 0) {
		if (((i + c - 1) == (j + d)) && (r == 1))
			return 1;
		else
			return 0;
	}
	else {
		return Nabla(knot, d, i, 1) / Nabla(knot, d, j, d - m + 1)*Nrc(knot, m - 1, r - 1, c, j, i, d)
			+ Nabla(knot, d, j, i - j) / Nabla(knot, d, j, d - m + 1)*Nrc(knot, m - 1, r, c, j, i, d)
			+ (1 - Nabla(knot, d, j, i - j) / Nabla(knot, d, j, d - m + 1))*Nrc(knot, m - 1, r, c, j - 1, i, d)
			+ (-Nabla(knot, d, i, 1) / Nabla(knot, d, j, d - m + 1))*Nrc(knot, m - 1, r - 1, c, j - 1, i, d);
	}
}

__device__ double Nabla(double* knot, int d, int i, int k) {
	int index = i + d - 1;
	return knot[index + k] - knot[index];
}

__host__ __device__ int MatrixNPosition(int i, int j, int degree_u, int degree_v, int cp_u, int cp_v,int index) {
	return i * (degree_v + cp_v) * 4 * (degree_u + 1)*(degree_v + 1) + j * 4 * (degree_u + 1)*(degree_v + 1) + index * (degree_u + 1)*(degree_v + 1);
}

__global__ void check(double* MatrixN, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v) {
	int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

	clock_t s, e;
	double time;

	s = clock();
	double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	e = clock();
	time = (double)(e - s) / GPU_clock_rate;
	//printf("Point time cost: %f\n", time);
	printf("%f %f %f %f\n", R.x, R.y, R.z, R.w);

	/*s = clock();
	double4 R1 = matrix_test(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	e = clock();
	time = (double)(e - s) / GPU_clock_rate;
	printf("Point time cost: %f\n", time);
	printf("%f %f %f %f\n", R1.x, R1.y, R1.z, R1.w);*/

	s = clock();
	//double3 derU = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v,
		//u, v, R, spanU);
	e = clock();
	time = (double)(e - s) / GPU_clock_rate;
	//printf("derU time cost: %f\n", time);

	s = clock();
	//double3 derV = MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v,
		//u, v, R, spanV);
	e = clock();
	time = (double)(e - s) / GPU_clock_rate;
	//printf("derV time cost: %f\n", time);
	//printf("%f %f %f\n", derU.x, derU.y, derU.z);
	//printf("%f %f %f\n", derV.x, derV.y, derV.z);
}

//k1 = degree_u+1, k2 = degree_v+1
__device__ double4 MatrixSurfacePoint(double* middle, int degree_u, int degree_v, 
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v) {

	int k1 = degree_u + 1, k2 = degree_v + 1;
	//get matrix middle
	//commented because middle matrix should be found before this function is called
	/*int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);*/

	double4 result;
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

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];
	
	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();

	result.x = x[0];
	result.y = y[0];
	result.z = z[0];
	result.w = w[0];*/

	matrix_mult(vectorUT, temp0, &(result.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(result.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(result.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(result.w), 1, k1, 1);

	return result;
}

__device__ double4 MatrixSurfacePoint(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectorV){
	int k1 = degree_u + 1, k2 = degree_v + 1;
	//get matrix middle
	//commented because middle matrix should be found before this function is called
	/*int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);*/

	double4 result;

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];
	
	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();

	result.x = x[0];
	result.y = y[0];
	result.z = z[0];
	result.w = w[0];*/

	matrix_mult(vectorUT, temp0, &(result.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(result.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(result.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(result.w), 1, k1, 1);

	return result;
}

__device__ double4 MatrixSurfaceRdU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectorV, double spanU) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	double4 RdU;
	matrix_mult(vectordUT, temp0, &(RdU.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdU.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdU.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdU.w), 1, k1, 1);

	return RdU * spanU;
}

__device__ double4 MatrixSurfaceRdV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordV, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);

	double4 RdV;
	matrix_mult(vectorUT, temp0, &(RdV.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(RdV.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(RdV.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(RdV.w), 1, k1, 1);

	return RdV * spanV;
}

__device__ double4 MatrixSurfaceRdUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUT, double* vectorV, double spanU) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	double4 RdUU;
	matrix_mult(vectordUUT, temp0, &(RdUU.x), 1, k1, 1);
	matrix_mult(vectordUUT, temp1, &(RdUU.y), 1, k1, 1);
	matrix_mult(vectordUUT, temp2, &(RdUU.z), 1, k1, 1);
	matrix_mult(vectordUUT, temp3, &(RdUU.w), 1, k1, 1);

	return RdUU * spanU * spanU;
}

__device__ double4 MatrixSurfaceRdUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectordV, double spanU, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);

	double4 RdUV;
	matrix_mult(vectordUT, temp0, &(RdUV.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdUV.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdUV.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdUV.w), 1, k1, 1);

	return RdUV * spanU * spanV;
}

__device__ double4 MatrixSurfaceRdVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordVV, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(vectorUT, middle + 0 * k1*k2, temp0, 1, k1, k2);
	matrix_mult(vectorUT, middle + 1 * k1*k2, temp1, 1, k1, k2);
	matrix_mult(vectorUT, middle + 2 * k1*k2, temp2, 1, k1, k2);
	matrix_mult(vectorUT, middle + 3 * k1*k2, temp3, 1, k1, k2);

	double4 RdVV;
	matrix_mult(temp0, vectordVV, &(RdVV.x), 1, k2, 1);
	matrix_mult(temp1, vectordVV, &(RdVV.y), 1, k2, 1);
	matrix_mult(temp2, vectordVV, &(RdVV.z), 1, k2, 1);
	matrix_mult(temp3, vectordVV, &(RdVV.w), 1, k2, 1);

	return RdVV * spanV * spanV;
}

__device__ double4 MatrixSurfaceRdUUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUUT, double* vectorV, double spanU) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	double4 RdUUU;
	matrix_mult(vectordUUUT, temp0, &(RdUUU.x), 1, k1, 1);
	matrix_mult(vectordUUUT, temp1, &(RdUUU.y), 1, k1, 1);
	matrix_mult(vectordUUUT, temp2, &(RdUUU.z), 1, k1, 1);
	matrix_mult(vectordUUUT, temp3, &(RdUUU.w), 1, k1, 1);

	return RdUUU * spanU * spanU *spanU;
}

__device__ double4 MatrixSurfaceRdUUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUUT, double* vectordV, double spanU, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);

	double4 RdUUV;
	matrix_mult(vectordUUT, temp0, &(RdUUV.x), 1, k1, 1);
	matrix_mult(vectordUUT, temp1, &(RdUUV.y), 1, k1, 1);
	matrix_mult(vectordUUT, temp2, &(RdUUV.z), 1, k1, 1);
	matrix_mult(vectordUUT, temp3, &(RdUUV.w), 1, k1, 1);

	return RdUUV * spanU * spanU * spanV;
}

__device__ double4 MatrixSurfaceRdUVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectordUT, double* vectordVV, double spanU, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectordVV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordVV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordVV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordVV, temp3, k1, k2, 1);

	double4 RdUVV;
	matrix_mult(vectordUT, temp0, &(RdUVV.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdUVV.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdUVV.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdUVV.w), 1, k1, 1);

	return RdUVV * spanU * spanV * spanV;
}

__device__ double4 MatrixSurfaceRdVVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double* vectorUT, double* vectordVVV, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	matrix_mult(middle + 0 * k1*k2, vectordVVV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordVVV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordVVV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordVVV, temp3, k1, k2, 1);

	double4 RdVVV;
	matrix_mult(vectorUT, temp0, &(RdVVV.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(RdVVV.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(RdVVV.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(RdVVV.w), 1, k1, 1);

	return RdVVV * spanV * spanV * spanV;
}

__host__ __device__ int findSpan(unsigned int degree, int cp, double *knots, double u) {
	int n = cp - 1;
	// For values of u that lies outside the domain
	if (u >= knots[n + 1])
	{
		return n;
	}
	if (u <= knots[degree])
	{
		return degree;
	}
	// Binary search
	// TODO: Replace this with std::lower_bound
	int low = degree;
	int high = n + 1;
	int mid = (int)((low + high) / 2.0);
	while (u < knots[mid] || u >= knots[mid + 1])
	{
		if (u < knots[mid])
		{
			high = mid;
		}
		else
		{
			low = mid;
		}
		mid = (int)((low + high) / 2.0);
	}
	return mid;
}

__device__ double3 MatrixSurfaceDerivativeU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectordUT[MAX_DEGREE];
	double vectorV[MAX_DEGREE];
	double now = 1;
	vectordUT[0] = 0;
	for (int i = 0; i < k1-1; i++) {
		vectordUT[i + 1] = (i + 1) * now;
		now *= u;
	}
	now = 1;
	for (int i = 0; i < k2; i++) {
		vectorV[i] = now;
		now *= v;
	}

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();
	double4 RdU = { x[0],y[0],z[0],w[0] };*/

	double4 RdU;
	matrix_mult(vectordUT, temp0, &(RdU.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdU.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdU.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdU.w), 1, k1, 1);

	double h = R.w;
	double hdU = RdU.w;

	double4 temp = R * (-hdU / pow(h, 2)) + RdU / h;

	return { temp.x * spanU,temp.y * spanU,temp.z * spanU };
}

__device__ double3 MatrixSurfaceDerivativeV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanV) {
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectorUT[MAX_DEGREE];
	double vectordV[MAX_DEGREE];
	double now = 1;
	for (int i = 0; i < k1; i++) {
		vectorUT[i] = now;
		now *= u;
	}
	now = 1;
	vectordV[0] = 0;
	for (int i = 0; i < k2 - 1; i++) {
		vectordV[i + 1] = (i + 1) * now;
		now *= v;
	}

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();
	double4 RdV = { x[0],y[0],z[0],w[0] };*/

	double4 RdV;
	matrix_mult(vectorUT, temp0, &(RdV.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(RdV.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(RdV.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(RdV.w), 1, k1, 1);

	double h = R.w;
	double hdV = RdV.w;

	double4 temp = R * (-hdV / pow(h, 2)) + RdV / h;

	return { temp.x * spanV,temp.y * spanV,temp.z * spanV };
}

__device__ double3 MatrixSurfaceDerivativeUU(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV) {
	//printf("0\n");
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectordUT[MAX_DEGREE];
	double vectorV[MAX_DEGREE];
	double vectordUUT[MAX_DEGREE];
	double now = 1;
	vectordUT[0] = 0;
	for (int i = 0; i < k1 - 1; i++) {
		vectordUT[i + 1] = (i + 1) * now;
		now *= u;
	}
	vectordUUT[0] = vectordUUT[1] = 0;
	now = 1;
	for (int i = 0; i < k1 - 2; i++) {
		vectordUUT[i + 2] = (i + 2)*(i + 1) * now;
		now *= u;
	}
	now = 1;
	for (int i = 0; i < k2; i++) {
		vectorV[i] = now;
		now *= v;
	}

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();
	double4 RdU = { x[0],y[0],z[0],w[0] };*/

	double4 RdU;
	matrix_mult(vectordUT, temp0, &(RdU.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdU.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdU.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdU.w), 1, k1, 1);

	double4 RdUU;
	matrix_mult(vectordUUT, temp0, &(RdUU.x), 1, k1, 1);
	matrix_mult(vectordUUT, temp1, &(RdUU.y), 1, k1, 1);
	matrix_mult(vectordUUT, temp2, &(RdUU.z), 1, k1, 1);
	matrix_mult(vectordUUT, temp3, &(RdUU.w), 1, k1, 1);

	double h = R.w;
	double hdU = RdU.w;
	double hdUU = RdUU.w;

	double4 temp = R * ((2 * pow(hdU, 2) / pow(h, 3)) - (hdUU / pow(h, 2))) +
		2 * RdU*(-hdU / pow(h, 2)) + RdUU / h;

	return { temp.x  * spanU * spanU, temp.y * spanU * spanU, temp.z * spanU * spanU };
}

__device__ double3 MatrixSurfaceDerivativeUV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV) {
	//printf("1\n");
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectorUT[MAX_DEGREE];
	double vectordUT[MAX_DEGREE];
	double vectorV[MAX_DEGREE];
	double vectordV[MAX_DEGREE];

	double now = 1;
	for (int i = 0; i < k1; i++) {
		vectorUT[i] = now;
		now *= u;
	}
	now = 1;
	vectordUT[0] = 0;
	for (int i = 0; i < k1 - 1; i++) {
		vectordUT[i + 1] = (i + 1) * now;
		now *= u;
	}
	now = 1;
	for (int i = 0; i < k2; i++) {
		vectorV[i] = now;
		now *= v;
	}
	now = 1;
	vectordV[0] = 0;
	for (int i = 0; i < k2 - 1; i++) {
		vectordV[i + 1] = (i + 1) * now;
		now *= v;
	}

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11, dimBlock>>>(vectordUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();
	double4 RdU = { x[0],y[0],z[0],w[0] };*/

	double4 RdU;
	matrix_mult(vectordUT, temp0, &(RdU.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdU.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdU.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdU.w), 1, k1, 1);

	double4 RdV;
	matrix_mult(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);

	matrix_mult(vectorUT, temp0, &(RdV.x), 1, k1, 1);
	matrix_mult(vectorUT, temp1, &(RdV.y), 1, k1, 1);
	matrix_mult(vectorUT, temp2, &(RdV.z), 1, k1, 1);
	matrix_mult(vectorUT, temp3, &(RdV.w), 1, k1, 1);

	double4 RdUV;
	matrix_mult(vectordUT, temp0, &(RdUV.x), 1, k1, 1);
	matrix_mult(vectordUT, temp1, &(RdUV.y), 1, k1, 1);
	matrix_mult(vectordUT, temp2, &(RdUV.z), 1, k1, 1);
	matrix_mult(vectordUT, temp3, &(RdUV.w), 1, k1, 1);

	double h = R.w;
	double hdU = RdU.w;
	double hdV = RdV.w;
	double hdUV = RdUV.w;

	double4 temp = (RdUV*h - hdV * RdU - RdV * hdU - R * hdUV) / pow(h, 2) + (2 * R*hdU*hdV) / pow(h, 3);

	return { temp.x * spanU*spanV,temp.y * spanU * spanV,temp.z * spanU * spanV };
}

__device__ double3 MatrixSurfaceDerivativeVV(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v,
	double4 R, double spanU, double spanV) {
	//printf("2\n");
	int k1 = degree_u + 1, k2 = degree_v + 1;

	double vectorUT[MAX_DEGREE];
	double vectordV[MAX_DEGREE];
	double vectordVV[MAX_DEGREE];
	double now = 1;
	for (int i = 0; i < k1; i++) {
		vectorUT[i] = now;
		now *= u;
	}
	now = 1;
	vectordV[0] = 0;
	for (int i = 0; i < k2 - 1; i++) {
		vectordV[i + 1] = (i + 1) * now;
		now *= v;
	}
	now = 1;
	vectordVV[0] = vectordVV[1] = 0;
	for (int i = 0; i < k2 - 2; i++) {
		vectordVV[i + 2] = (i + 2)*(i + 1)*now;
		now *= v;
	}

	/*unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);*/

	double temp0[MAX_DEGREE];
	double temp1[MAX_DEGREE];
	double temp2[MAX_DEGREE];
	double temp3[MAX_DEGREE];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectordV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectordV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectordV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectordV, temp3, k1, k2, 1);*/

	matrix_mult(vectorUT, middle + 0 * k1*k2, temp0, 1, k1, k2);
	matrix_mult(vectorUT, middle + 1 * k1*k2, temp1, 1, k1, k2);
	matrix_mult(vectorUT, middle + 2 * k1*k2, temp2, 1, k1, k2);
	matrix_mult(vectorUT, middle + 3 * k1*k2, temp3, 1, k1, k2);

	/*double x[1];
	double y[1];
	double z[1];
	double w[1];

	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();
	double4 RdV = { x[0],y[0],z[0],w[0] };*/

	double4 RdV;
	matrix_mult(temp0, vectordV, &(RdV.x), 1, k2, 1);
	matrix_mult(temp1, vectordV, &(RdV.y), 1, k2, 1);
	matrix_mult(temp2, vectordV, &(RdV.z), 1, k2, 1);
	matrix_mult(temp3, vectordV, &(RdV.w), 1, k2, 1);

	double4 RdVV;
	matrix_mult(temp0, vectordVV, &(RdVV.x), 1, k2, 1);
	matrix_mult(temp1, vectordVV, &(RdVV.y), 1, k2, 1);
	matrix_mult(temp2, vectordVV, &(RdVV.z), 1, k2, 1);
	matrix_mult(temp3, vectordVV, &(RdVV.w), 1, k2, 1);

	double h = R.w;
	double hdV = RdV.w;
	double hdVV = RdVV.w;

	double4 temp = R * ((2 * pow(hdV, 2) / pow(h, 3)) - (hdVV / pow(h, 2))) +
		2 * RdV*(-hdV / pow(h, 2)) + RdVV / h;

	return { temp.x * spanV * spanV,temp.y * spanV * spanV,temp.z * spanV * spanV };
}

__device__ double3 MatrixSurfaceDerivativeU(double4 RdU, double3 P, double h) {
	double hdU = RdU.w;
	double3 PdU = (TruncateHomogenous(RdU) - hdU * P) / h;
	return PdU;
}

__device__ double3 MatrixSurfaceDerivativeV(double4 RdV, double3 P, double h) {
	double hdV = RdV.w;
	double3 PdV = (TruncateHomogenous(RdV) - hdV * P) / h;
	return PdV;
}

__device__ double3 MatrixSurfaceDerivativeUU(double4 RdUU, double3 PdU, double3 P, double h, double hdU) {
	double hdUU = RdUU.w;
	double3 PdUU = (TruncateHomogenous(RdUU) - 2 * hdU*PdU - hdUU * P) / h;
	return PdUU;
}

__device__ double3 MatrixSurfaceDerivativeVV(double4 RdVV, double3 PdV, double3 P, double h, double hdV) {
	double hdVV = RdVV.w;
	double3 PdVV = (TruncateHomogenous(RdVV) - 2 * hdV*PdV - hdVV * P) / h;
	return PdVV;
}

__device__ double3 MatrixSurfaceDerivativeUV(double4 RdUV, double3 PdU, double3 PdV, double3 P, double h, double hdU, double hdV) {
	double hdUV = RdUV.w;
	double3 PdUV = (TruncateHomogenous(RdUV) - hdV * PdU - hdU * PdV + hdUV * P) / h;
	return PdUV;
}

//dU2V
__device__ double3 MatrixSurfaceDerivativeUUV(double4 RdUUV, double3 PdUU, double3 PdUV, double3 PdU, double3 PdV, double3 P,
	double h, double hdU, double hdV, double hdUU, double hdUV) {
	double hdUUV = RdUUV.w;
	double3 PdUUV = (TruncateHomogenous(RdUUV) - hdV * PdUU - 2 * hdU*PdUV + 2 * hdUV*PdU - hdUU * PdV + hdUUV * P) / h;

	//double4 ret = R * (4 * hdU*hdUV / pow(h, 6) - 6 * hdV*pow(hdU, 2) / pow(h, 4) - hdUUV / pow(h, 2) + 2 * hdV*hdUU / pow(h, 3))
	//	+ RdV * (2 * pow(hdU, 2) / pow(h, 3) - hdUU / pow(h, 2))
	//	+ 2 * RdU*(-hdUV / pow(h, 2) + 2 * hdV*hdU / pow(h, 3))
	//	+ 2 * RdUV*(-hdU / pow(h, 2))
	//	+ RdUU * (-hdV / pow(h, 2))
	//	+ RdUUV * (1.0 / h);

	return PdUUV;
}

//dUV2
__device__ double3 MatrixSurfaceDerivativeUVV(double4 RdUVV, double3 PdUV, double3 PdVV, double3 PdU, double3 PdV, double3 P,
	double h, double hdU, double hdV, double hdUV, double hdVV) {
	double hdUVV = RdUVV.w;
	double3 PdUVV = (TruncateHomogenous(RdUVV) - hdU * PdVV - 2 * hdV*PdUV + 2 * hdUV*PdV - hdVV * PdU + hdUVV * P) / h;

	//double4 ret = R * (4 * hdV*hdUV / pow(h, 6) - 6 * hdU*pow(hdV, 2) / pow(h, 4) - hdUVV / pow(h, 2) + 2 * hdU*hdVV / pow(h, 3))
	//	+ RdU * (2 * pow(hdV, 2) / pow(h, 3) - hdVV / pow(h, 2))
	//	+ 2 * RdV*(-hdUV / pow(h, 2) + 2 * hdV*hdU / pow(h, 3))
	//	+ 2 * RdUV*(-hdV / pow(h, 2))
	//	+ RdVV * (-hdU / pow(h, 2))
	//	+ RdUVV * (1.0 / h);

	return PdUVV;
}

//dU3
__device__ double3 MatrixSurfaceDerivativeUUU(double4 RdUUU, double3 PdUU, double3 PdU, double3 P, double h, double hdU, double hdUU) {

	double hdUUU = RdUUU.w;
	double3 PdUUU = (TruncateHomogenous(RdUUU) - 3 * hdU*PdUU - 3 * hdUU*PdU - hdUUU * P) / h;

	//double4 temp = R * (-hdUUU / pow(h, 2) + 6 * hdU * hdUU / pow(h, 3) - 6 * pow(hdU, 3) / pow(h, 4))
	//	+ 3 * RdU*(-hdUU / pow(h, 2) + 2 * pow(hdU, 2)*pow(h, 3))
	//	+ 3 * RdUU*(-hdU / pow(h, 2))
	//	+ RdUUU * (1.0 / h);

	return PdUUU;
}

//dV3
__device__ double3 MatrixSurfaceDerivativeVVV(double4 RdVVV, double3 PdVV, double3 PdV, double3 P, double h, double hdV, double hdVV) {

	double hdVVV = RdVVV.w;
	double3 PdVVV = (TruncateHomogenous(RdVVV) - 3 * hdV*PdVV - 3 * hdVV*PdV - hdVVV * P) / h;

	//double4 temp = R * (-hdVVV / pow(h, 2) + 6 * hdV * hdVV / pow(h, 3) - 6 * pow(hdV, 3) / pow(h, 4))
	//	+ 3 * RdV*(-hdVV / pow(h, 2) + 2 * pow(hdV, 2)*pow(h, 3))
	//	+ 3 * RdVV*(-hdV / pow(h, 2))
	//	+ RdVVV * (1.0 / h);

	return PdVVV;
}


__global__ void EvaluateRandom(double2* random, double3* random3,
	double* MatrixN, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v) {

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	double u = random[index].x, v = random[index].y;
	int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	//printf("%d %d\n", m, n);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

	random3[index] = homogenousToCartesian(MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v));
	//printf("%f %f %f\n", random3[index].x, random3[index].y, random3[index].z);
}

__device__ double4 matrix_test(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v) {
	int k1 = 4, k2 = 3;
	//get matrix middle
	//commented because middle matrix should be found before this function is called
	/*int m = findSpan(degree_u, cp_u, knots_u, u);
	int n = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	v = (v - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);*/

	double4 result;
	double vectorUT[4], vectorV[3];
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

	unsigned int grid_u = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_v = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGridu1(grid_u, grid_1);
	dim3 dimGrid11(grid_1, grid_1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	double temp0[4];
	double temp1[4];
	double temp2[4];
	double temp3[4];

	/*gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	gpu_matrix_mult<<<dimGridu1,dimBlock>>>(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);*/
	matrix_mult(middle + 0 * k1*k2, vectorV, temp0, k1, k2, 1);
	matrix_mult(middle + 1 * k1*k2, vectorV, temp1, k1, k2, 1);
	matrix_mult(middle + 2 * k1*k2, vectorV, temp2, k1, k2, 1);
	matrix_mult(middle + 3 * k1*k2, vectorV, temp3, k1, k2, 1);

	//if (u == 0.5 && v == 0) {
		//print(temp3, 1, k1);
	//}

	double x[1];
	double y[1];
	double z[1];
	double w[1];

	/*gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp0, x, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp1, y, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp2, z, 1, k1, 1);
	gpu_matrix_mult<<<dimGrid11,dimBlock>>>(vectorUT, temp3, w, 1, k1, 1);
	cudaDeviceSynchronize();*/
	matrix_mult(vectorUT, temp0, x, 1, k1, 1);
	matrix_mult(vectorUT, temp1, y, 1, k1, 1);
	matrix_mult(vectorUT, temp2, z, 1, k1, 1);
	matrix_mult(vectorUT, temp3, w, 1, k1, 1);

	result.x = x[0];
	result.y = y[0];
	result.z = z[0];
	result.w = w[0];

	return result;
}

__device__ double3 TrueNormal(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v, double spanU, double spanV) {
	double4 SurfCurrent4d = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	double3 derU = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU);
	double3 derV = MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanV);
	return cross_product(derU, derV);
}

__device__ double3 Normal(double* middle, int degree_u, int degree_v,
	int cp_u, int cp_v, double* knots_u, double* knots_v, double u, double v, double spanU, double spanV) {
	return TrueNormal(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, spanU, spanV);
}

__device__ double3 SurfaceNormalDerivativeUU(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdUUU, double3 PdUUV) {
	return cross_product(PdUUU, PdV) + 2*cross_product(PdUU, PdUV) + cross_product(PdU, PdUUV);
}

__device__ double3 SurfaceNormalDerivativeUV(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdVV, double3 PdUUV, double3 PdUVV) {
	return cross_product(PdUUV, PdV) + cross_product(PdUU, PdVV) + cross_product(PdUV, PdUV) + cross_product(PdU, PdUVV);
}

__device__ double3 SurfaceNormalDerivativeVV(double3 PdU, double3 PdV, double3 PdUV, double3 PdVV, double3 PdUVV, double3 PdVVV) {
	return cross_product(PdUVV, PdV) + 2*cross_product(PdUV, PdVV) + cross_product(PdU, PdVVV);
}

// __device__ double3 SurfaceNormalDerivativeUU(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdUUU, double3 PdUUV) {
// 	double3 ret;

// 	ret.x = (PdUUU.y*PdV.z + 2 * PdUU.y*PdUV.z + PdU.y*PdUUV.z) - (PdUUU.z*PdV.y + 2 * PdUU.z*PdUV.y + PdU.z*PdUUV.y);
// 	ret.y = (PdUUU.z*PdV.x + 2 * PdUU.z*PdUV.x + PdU.z*PdUUV.x) - (PdUUU.x*PdV.z + 2 * PdUU.x*PdUV.z + PdU.x*PdUUV.z);
// 	ret.z = (PdUUU.x*PdV.y + 2 * PdUU.x*PdUV.y + PdU.x*PdUUV.y) - (PdUUU.y*PdV.x + 2 * PdUU.y*PdUV.x + PdU.y*PdUUV.x);

// 	return ret;
// }

// __device__ double3 SurfaceNormalDerivativeUV(double3 PdU, double3 PdV, double3 PdUU, double3 PdUV, double3 PdVV, double3 PdUUV, double3 PdUVV) {
// 	double3 ret;

// 	ret.x = (PdUUV.y*PdV.z + PdUU.y*PdVV.z + PdUV.y*PdUV.z + PdU.y*PdUVV.z) - (PdUUV.z*PdV.y + PdUU.z*PdVV.y + PdUV.z*PdUV.y + PdU.z*PdUVV.y);
// 	ret.y = (PdUUV.z*PdV.x + PdUU.z*PdVV.x + PdUV.z*PdUV.x + PdU.z*PdUVV.x) - (PdUUV.x*PdV.z + PdUU.x*PdVV.z + PdUV.x*PdUV.z + PdU.x*PdUVV.z);
// 	ret.z = (PdUUV.x*PdV.y + PdUU.x*PdVV.y + PdUV.x*PdUV.y + PdU.x*PdUVV.y) - (PdUUV.y*PdV.x + PdUU.y*PdVV.x + PdUV.y*PdUV.x + PdU.y*PdUVV.x);

// 	return ret;
// }

// __device__ double3 SurfaceNormalDerivativeVV(double3 PdU, double3 PdV, double3 PdUV, double3 PdVV, double3 PdUVV, double3 PdVVV) {
// 	double3 ret;

// 	ret.x = (PdUVV.y*PdV.z + 2 * PdUV.y*PdVV.z + PdU.y*PdVVV.z) - (PdUVV.z*PdV.y + 2 * PdUV.z*PdVV.y + PdU.z*PdVVV.y);
// 	ret.y = (PdUVV.z*PdV.x + 2 * PdUV.z*PdVV.x + PdU.z*PdVVV.x) - (PdUVV.x*PdV.z + 2 * PdUV.x*PdVV.z + PdU.x*PdVVV.z);
// 	ret.z = (PdUVV.x*PdV.y + 2 * PdUV.x*PdVV.y + PdU.x*PdVVV.y) - (PdUVV.y*PdV.x + 2 * PdUV.y*PdVV.x + PdU.y*PdVVV.x);

// 	return ret;
// }