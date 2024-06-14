#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "kernel.cuh"
#include "Evaluate.cuh"
#include "test.cuh"
#include "Matrix.cuh"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// #define FUNC_TEST
// #define PROJECTION_TEST
// #define INVERSION_TEST
// #define CUBLAS_TEST 
// #define CUBLAS_NURBS_TEST
#define CUBLAS_PERF_TEST
// #define RANDOM_ACCESS
// #define GPU_PRO


#define T_ELEM_IN half
#define T_ELEM_OUT half

using namespace std;

//HANDLE_ERROR
static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void readNu(string filename, int &degree_u, int &degree_v, int &cp_u, int &cp_v, double* &knots_u, double* &knots_v, double3* &control_points, double* &weights)
{
	ifstream infile(filename, ios::in);
	if (!infile)
	{
		cout << "Error when read NURBS file!\n";
	}

	string line, flag;
	istringstream linestream;

	//NURBS node size
	do {
		getline(infile, line);
		linestream = istringstream(line);
		linestream >> flag;
	} while (flag != "NU");

	linestream >> degree_u >> degree_v;

	// control points number
	do {
		getline(infile, line);
		linestream = istringstream(line);
		linestream >> flag;
	} while (flag != "CP");

	linestream >> cp_u >> cp_v;

	//knot vector
	//reserve space
	knots_u = new double[degree_u + 1 + cp_u];
	knots_v = new double[degree_v + 1 + cp_v];
	do {
		getline(infile, line);
		linestream = istringstream(line);
		linestream >> flag;
	} while (flag != "KNu");
	for (int i = 0; i < degree_u + cp_u + 1; i++)linestream >> knots_u[i];

	do {
		getline(infile, line);
		linestream = istringstream(line);
		linestream >> flag;
	} while (flag != "KNv");
	for (int i = 0; i < degree_v + cp_v + 1; i++)linestream >> knots_v[i];


	//NURBS points and weights
	double pos_x, pos_y, pos_z, w;
	control_points = new double3[cp_u*cp_v];
	weights = new double[cp_u*cp_v];
	for (int i = 0; i < cp_u; i++)
	{
		for (int j = 0; j < cp_v; j++)
		{
			do {
				getline(infile, line);
				linestream = istringstream(line);
				linestream >> flag;
			} while (flag != "DP");
			flag = "";

			linestream >> pos_x >> pos_y >> pos_z >> w;
			control_points[i*cp_v + j] = { pos_x, pos_y, pos_z };
			weights[i*cp_v + j] = w;
		}
	}
}

int get_GPU_Rate()
{
	cudaDeviceProp deviceProp;//CUDA定义的存储GPU属性的结构体
	cudaGetDeviceProperties(&deviceProp, 0);//CUDA定义函数
	return deviceProp.clockRate;
}

typedef double3(*Der)(double*, int, int, int, int, double*, double*, double, double, double4, double, double);

__device__ Der StaticDerUU = MatrixSurfaceDerivativeUU;
__device__ Der StaticDerUV = MatrixSurfaceDerivativeUV;
__device__ Der StaticDerVV = MatrixSurfaceDerivativeVV;

int main() {
	double *weights;
	double3 *control_points;
	double *knots_u, *knots_v;
	int degree_u, degree_v, cp_u, cp_v;

	string file_name("model/center_bk.nu");
	// cout << "Input file name: " << endl;
	//cin >> file_name;

	readNu(file_name, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, control_points, weights);

	//double max_length;
	//cout << "Input max edge length: " << endl;
	//cin >> max_length;

	double2* random = new double2[DataSize];
	for (int i = 0; i < DataSize; i++) {
		//random[i].x = rand() / double(RAND_MAX);
		//random[i].y = rand() / double(RAND_MAX);
		random[i].x = 0.2;
		random[i].y = 0.2;
	}

	bool* Divided = (bool*)malloc(TreeArraySize * sizeof(bool));
	memset(Divided, false, TreeArraySize);
	Divided[0] = true;

	double *Gweights;
	double3 *Gcontrol_points;
	double *Gknots_u, *Gknots_v;
	double2 *Grandom;
	double3 *Grandom3;
	double *MatrixN;
	bool *Gflag;
	double2 *Gresult;
	bool* GDivided;
	double *GdUU, *GdVV, *GdUV;
	double4* GdU, *GdV;
	double4* GRPoints;
	TreeNode* GTree, *GNormalBox;
	NormalTreeNode* GNormalTree;
	double* M1, *M2, *M3, *N_M1, *N_M2, *N_M3;
	double4* R, *RdU, *RdV, *RdUU, *RdUV, *RdVV, *RdUUU, *RdUUV, *RdUVV, *RdVVV;
	double3 *P, *PdU, *PdV, *PdUU, *PdUV, *PdVV, *PdUUU, *PdUUV, *PdUVV, *PdVVV, *Normal;
	double* NdUU, *NdUV, *NdVV;
	double* vectorUT, *vectorV, *vectordUT, *vectordV, *vectordUUT, *vectordVV, *vectordUUUT, *vectordVVV;

	HANDLE_ERROR(cudaMalloc((void**)&Gweights, sizeof(double) * cp_u * cp_v));
	HANDLE_ERROR(cudaMalloc((void**)&Gcontrol_points, sizeof(double3) * cp_u*cp_v));
	HANDLE_ERROR(cudaMalloc((void**)&Gknots_u, sizeof(double) * (degree_u + 1 + cp_u)));
	HANDLE_ERROR(cudaMalloc((void**)&Gknots_v, sizeof(double) * (degree_v + 1 + cp_v)));
	HANDLE_ERROR(cudaMalloc((void**)&Grandom, sizeof(double2) * DataSize));
	HANDLE_ERROR(cudaMalloc((void**)&Grandom3, sizeof(double3) * DataSize));
	HANDLE_ERROR(cudaMalloc((void**)&MatrixN, sizeof(double) * (degree_u + cp_u)*(degree_v + cp_v) * 4 * (degree_u + 1)*(degree_v + 1)));
	HANDLE_ERROR(cudaMalloc((void**)&Gflag, sizeof(bool) * DataSize));
	HANDLE_ERROR(cudaMalloc((void**)&Gresult, sizeof(double2) * DataSize));
	HANDLE_ERROR(cudaMalloc((void**)&GDivided, sizeof(bool) * TreeArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GdUU, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GdVV, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GdUV, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GdU, sizeof(double4) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GdV, sizeof(double4) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GRPoints, sizeof(double4) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GTree, sizeof(TreeNode) * TreeArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GNormalBox, sizeof(TreeNode) * TreeArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&GNormalTree, sizeof(NormalTreeNode) * TreeArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&M1, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&M2, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&M3, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&N_M1, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&N_M2, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&N_M3, sizeof(double) * PointsArraySize));
	HANDLE_ERROR(cudaMalloc((void**)&R, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdU, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdV, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdUU, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdUV, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdVV, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdUUU, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdUUV, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdUVV, sizeof(double4) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&RdVVV, sizeof(double4) * PointsArraySize));
	// // HANDLE_ERROR(cudaMalloc((void**)&P, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdU, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdUU, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdUV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdVV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdUUU, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdUUV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdUVV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&PdVVV, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&Normal, sizeof(double3) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&NdUU, sizeof(double) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&NdUV, sizeof(double) * PointsArraySize));
	// HANDLE_ERROR(cudaMalloc((void**)&NdVV, sizeof(double) * PointsArraySize));

	// HANDLE_ERROR(cudaMalloc((void**)&vectorUT, sizeof(double) * PointsArraySize * (degree_u + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectorV, sizeof(double) * PointsArraySize * (degree_v + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordUT, sizeof(double) * PointsArraySize* (degree_u + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordV, sizeof(double) * PointsArraySize * (degree_v + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordUUT, sizeof(double) * PointsArraySize* (degree_u + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordVV, sizeof(double) * PointsArraySize * (degree_v + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordUUUT, sizeof(double) * PointsArraySize* (degree_u + 1)));
	// HANDLE_ERROR(cudaMalloc((void**)&vectordVVV, sizeof(double) * PointsArraySize * (degree_v + 1)));

	HANDLE_ERROR(cudaMemcpy(Gweights, weights, sizeof(double) * cp_u*cp_v, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Gcontrol_points, control_points, sizeof(double3) * cp_u*cp_v, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Gknots_u, knots_u, sizeof(double) * (degree_u + 1 + cp_u), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Gknots_v, knots_v, sizeof(double) * (degree_v + 1 + cp_v), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Grandom, random, sizeof(double2) * DataSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(GDivided, Divided, sizeof(bool) * TreeArraySize, cudaMemcpyHostToDevice));

	// 传递函数指针
	Der HostDerUU, HostDerUV, HostDerVV;
	HANDLE_ERROR(cudaMemcpyFromSymbol(&HostDerUU, StaticDerUU, sizeof(Der)));
	HANDLE_ERROR(cudaMemcpyFromSymbol(&HostDerUV, StaticDerUV, sizeof(Der)));
	HANDLE_ERROR(cudaMemcpyFromSymbol(&HostDerVV, StaticDerVV, sizeof(Der)));
	
	//some cuda events
	dim3 grids(10, 10);
	dim3 blocks(16, 10);
	
	GetMatrixN<<<(degree_u+cp_u),(degree_v+cp_v)>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights);
	//EvaluateRandom<<<2500,600>>>(Grandom, Grandom3, MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v);
	cudaThreadSynchronize();
	printf("0: %s\n", cudaGetErrorString(cudaGetLastError()));
	/*printf("GetMatrixN Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));*/
	// time counting terminate
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	clock_t s, e;
	s = clock();
	//check<<<500,200>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, 0.2, 0.2);

	// First version of inversion
	//Inverse<<<1000,100>>>(0.0001, MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, Grandom3, Gflag, Gresult);

	// Test second derivative
	//TestSecondDerivative<<<1, 1>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, { 0.95, 0.95 });
	//cudaThreadSynchronize();
	//printf("test Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
	
#ifdef INVERSION_TEST

	// Second version of inversion
	int base = 0;
	blocks = { PointsPerEdge ,PointsPerEdge };
	grids = { 1, 1 };
	int level = 2;
	for (int i = 0; i < level; i++) {
		GetPoints<<< grids, blocks >>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, GRPoints);
		cudaThreadSynchronize();
		//printf("1: %s\n", cudaGetErrorString(cudaGetLastError()));
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, GRPoints, GdUU, HostDerUU);
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, GRPoints, GdUV, HostDerUV);
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, GRPoints, GdVV, HostDerVV);
		cudaThreadSynchronize();
		//printf("2: %s\n", cudaGetErrorString(cudaGetLastError()));

		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdUU, PointsPerEdge * PointsPerEdge, M1);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdUV, PointsPerEdge * PointsPerEdge, M2);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdVV, PointsPerEdge * PointsPerEdge, M3);
		cudaThreadSynchronize();
		//printf("3: %s\n", cudaGetErrorString(cudaGetLastError()));
		TightBoundingBox << < grids, PointsPerEdge * PointsPerEdge >> > (GRPoints, M1, M2, M3, base, GTree, GDivided);
		cudaThreadSynchronize();
		//printf("4: %s\n", cudaGetErrorString(cudaGetLastError()));
		
		//OutputBoolData << <1, 1 >> > (GDivided, base);
		base += grids.x*grids.y;
		grids.x *= GapPerEdge;
		grids.y *= GapPerEdge;
		//OutputTree << <1, 1 >> > (GTree, base);
		cudaThreadSynchronize();
	}
	NewInverse<<<65536,512>>>(0.01, MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, Grandom3, GTree, Gflag, Gresult);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	printf("Inverse Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
	e = clock();
	float gpu_elapsed_time_ms;
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Time elapsed total: %f ms.\n\n", gpu_elapsed_time_ms);
	//OutputTree << <1, 1 >> > (GTree, 65);
	//OutputRPoints << <1, 1 >> > (GRPoints, 5*81);
	//OutputData << <1, 1 >> > (GdUU, 5 * 81);
	//cudaThreadSynchronize();
	//OutputData << <1, 1 >> > (M1, 64);
	double time = (double)(e - s) / CLOCKS_PER_SEC;
	//cout << time*1000 << endl;
	//CheckAnswer<<<1,1>>>(Gflag, Gresult);
#endif

#ifdef PROJECTION_TEST

	int base = 0;
	blocks = { PointsPerEdge ,PointsPerEdge };
	grids = { 1, 1 };
	int level = 2;
	for (int i = 0; i < level; i++) {
		GetVector <<< grids, blocks >>>(degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectorUT, vectorV, vectordUT, vectordV, vectordUUT, vectordVV, vectordUUUT, vectordVVV);
		cudaThreadSynchronize();
		// printf("1: %s\n", cudaGetErrorString(cudaGetLastError()));
		getRs <<<grids, blocks>>> (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectorUT, vectorV, R);
		getRdUs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUT, vectorV, RdU);
		getRdVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectorUT, vectordV, RdV);
		getRdUUs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUUT, vectorV, RdUU);
		getRdUVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUT, vectordV, RdUV);
		getRdVVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectorUT, vectordVV, RdVV);
		getRdUUUs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUUUT, vectorV, RdUUU);
		getRdUUVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUUT, vectordV, RdUUV);
		getRdUVVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectordUT, vectordVV, RdUVV);
		getRdVVVs << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, GDivided, base, vectorUT, vectordVVV, RdVVV);
		cudaThreadSynchronize();
		// kernel_print<<<1,1>>>(R, 2);
		// cudaThreadSynchronize();
		// printf("2: %s\n", cudaGetErrorString(cudaGetLastError()));
		// GetDerivatives << <grids, blocks >> > (GDivided, base, R, RdU, RdV, RdUU, RdUV, RdUV, RdUUU, RdUUV, RdUVV, RdVVV, P, PdU, PdV, PdUU, PdUV, PdVV, PdUUU, PdUUV, PdUVV, PdVVV);
		//cudaThreadSynchronize();
		getNormals<< <grids, blocks >> >(GDivided, base, R, RdU, RdV, Normal);
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, R, GdUU, HostDerUU);
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, R, GdUV, HostDerUV);
		GetSecondDerivatives << <grids, blocks >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, GDivided, base, R, GdVV, HostDerVV);
		GetNdUUs << <grids, blocks >> > (GDivided, base, R, RdU, RdV, RdUU, RdUV, RdUUU, RdUUV, NdUU);
		GetNdUVs << <grids, blocks >> > (GDivided, base, R, RdU, RdV, RdUU, RdUV, RdVV, RdUUV, RdUVV, NdUV);
		GetNdVVs << <grids, blocks >> > (GDivided, base, R, RdU, RdV, RdUV, RdVV, RdUVV, RdVVV, NdVV);
		cudaThreadSynchronize();
		// kernel_print<<<1,1>>>(NdUU, 1, 1);
		// cudaThreadSynchronize();

		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdUU, PointsPerEdge * PointsPerEdge, M1);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdUV, PointsPerEdge * PointsPerEdge, M2);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (GdVV, PointsPerEdge * PointsPerEdge, M3);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (NdUU, PointsPerEdge * PointsPerEdge, N_M1);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (NdUV, PointsPerEdge * PointsPerEdge, N_M2);
		Max_Sequential_Addressing_Shared << < grids, PointsPerEdge * PointsPerEdge >> > (NdUV, PointsPerEdge * PointsPerEdge, N_M3);
		cudaThreadSynchronize();
		kernel_print<<<1,1>>>(N_M1, 1, 1);
		kernel_print<<<1,1>>>(N_M2, 1, 1);
		kernel_print<<<1,1>>>(N_M3, 1, 1);
		cudaThreadSynchronize();
		// printf("3: %s\n", cudaGetErrorString(cudaGetLastError()));
		TightBoundingBox << < grids, PointsPerEdge * PointsPerEdge >> > (R, M1, M2, M3, base, GTree, GDivided);
		NormalBoundingBox << < grids, PointsPerEdge * PointsPerEdge >> > (Normal, N_M1, N_M2, N_M3, base, GNormalBox, GDivided); 
		NormalBoxToNormalTreeNode<<<grids.x, grids.y>>>(base, GNormalBox, GNormalTree);
		cudaThreadSynchronize();
		// printf("4: %s\n", cudaGetErrorString(cudaGetLastError()));

		//OutputBoolData << <1, 1 >> > (GDivided, base);
		base += grids.x*grids.y;
		grids.x *= GapPerEdge;
		grids.y *= GapPerEdge;
		OutputTree << <1, 1 >> > (GTree, base);
		// OutputTree << <1, 1 >> > (GNormalBox, base);
		// OutputNormalTree<<< 1, 1>>>(GNormalTree, base);
		cudaThreadSynchronize();
	}
	Project<<<1,1>>>(0.001, MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, Grandom3, GTree, GNormalTree, Gflag, Gresult);
	cudaThreadSynchronize();
	printf("Project Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif

#ifdef FUNC_TEST

	NormalTest << <1, 1 >> > (MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, { 0.5, 0.5 });
	cudaThreadSynchronize();
	printf("func test Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));

#endif // FUNC_TEST

#ifdef CUBLAS_TEST
	
	// First, create a cuBLAS handle:
	int m = 8, k = 8, n = 8;

	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	// Set the math mode to allow cuBLAS to use Tensor Cores:
	cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	// Allocate and initialize your matrices (only the A matrix is shown):
	int rowsA = m, colsA = k;
	size_t matrixSizeA = (size_t)rowsA * colsA;
	T_ELEM_IN *devPtrA = 0;
	cudaMalloc((void**)&devPtrA, matrixSizeA * sizeof(T_ELEM_IN));
	cudaDeviceSynchronize();
	T_ELEM_IN* A = (T_ELEM_IN *)malloc(matrixSizeA * sizeof(T_ELEM_IN));
	memset(A, 0, matrixSizeA * sizeof(T_ELEM_IN));
	A[17] = __float2half(1.0);
	cublasStatus_t status1 = cublasSetMatrix(rowsA, colsA, sizeof(T_ELEM_IN), A, rowsA, devPtrA, rowsA); // ... allocate and initialize B and C matrices (not shown) ... // Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8, // and m is a multiple of 4:
	cudaDeviceSynchronize();
	print << <1, 1 >> > (devPtrA, 8, 8);
	cudaDeviceSynchronize();

	int rowsB = k, colsB = n;
	size_t matrixSizeB = (size_t)rowsB * colsB;
	T_ELEM_IN *devPtrB = 0;
	cudaMalloc((void**)&devPtrB, matrixSizeB * sizeof(T_ELEM_IN));
	cudaDeviceSynchronize();
	T_ELEM_IN* B = (T_ELEM_IN *)malloc(matrixSizeB * sizeof(T_ELEM_IN));
	memset(B, 0, matrixSizeB * sizeof(T_ELEM_IN));
	B[11] = __float2half(1.0);
	cublasStatus_t status2 = cublasSetMatrix(rowsB, colsB, sizeof(T_ELEM_IN), B, rowsB, devPtrB, rowsB);
	cudaDeviceSynchronize();
	print << <1, 1 >> > (devPtrB, 8, 8);
	cudaDeviceSynchronize();

	int rowsC = m, colsC = n;
	size_t matrixSizeC = (size_t)rowsC * colsC;
	T_ELEM_OUT *devPtrC = 0;
	cudaMalloc((void**)&devPtrC, matrixSizeC * sizeof(T_ELEM_OUT));
	cudaDeviceSynchronize();

	// A * B = C
	// get C in transpose
	float h_alpha = 1.0, h_beta = 0;
	int lda = m, ldb = k, ldc = m;
	cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &h_alpha, devPtrA, CUDA_R_16F, lda, devPtrB, CUDA_R_16F, ldb, &h_beta, devPtrC, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();
	cout << cublasStat << endl;
	print << <1, 1 >> > (devPtrC, m, n);
	cudaDeviceSynchronize();

#endif // CUBLAS_TEST

#ifdef CUBLAS_NURBS_TEST

	double u = 0.8, v = 0.8;
	int a = findSpan(degree_u, cp_u, knots_u, u);
	int b = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[a]) / (knots_u[a + 1] - knots_u[a]);
	v = (v - knots_v[b]) / (knots_v[b + 1] - knots_v[b]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[a + 1] - knots_u[a]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[b + 1] - knots_v[b]);
	double* middle = MatrixN + MatrixNPosition(a, b, degree_u, degree_v, cp_u, cp_v, 0);
	
	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	// Set the math mode to allow cuBLAS to use Tensor Cores:
	cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

	int m = 8, n = 8, k = 8;
	int k1 = degree_u + 1, k2 = degree_v + 1;
	size_t sizeU = (size_t)m*k, sizeMiddle = (size_t)k*n, sizeV = (size_t)n*m, sizeResult = (size_t)m*m;
	T_ELEM_IN *devU = 0, *devV = 0, *result = 0, *temp = 0, *new_middle_x = 0, *new_middle_y = 0,*new_middle_z = 0, *new_middle_w = 0;
	HANDLE_ERROR(cudaMalloc((void**)&devU, sizeU * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&devV, sizeV * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_x, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_y, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_z, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_w, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&result, sizeResult * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&temp, sizeV * sizeof(T_ELEM_IN)));
	cudaDeviceSynchronize();
	T_ELEM_IN *U = new T_ELEM_IN[sizeU]{__float2half(0.0)}, *V = new T_ELEM_IN[sizeV]{__float2half(0.0)};
	float now = 1;
	for (int i = 0; i < k1; i++) {
		U[i] = U[1*k+i] = U[2*k+i] = U[3*k+i] = __float2half(now);
		now *= u;
	}
	now = 1;
	for (int i = 0; i < k2; i++) {
		V[i*m] = V[i*m+1] = V[i*m+2] = V[i*m+3] = __float2half(now);
		now *= v;
	}
	cublasStatus_t status1 = cublasSetMatrix(m, k, sizeof(T_ELEM_IN), U, m, devU, m);
	cublasStatus_t status2 = cublasSetMatrix(n, m, sizeof(T_ELEM_IN), V, n, devV, n);
	cudaDeviceSynchronize();
	MatrixFit<<<k1, k2>>>(middle, k1, k2, new_middle_x, k, n);
	MatrixFit<<<k1, k2>>>(middle + 1*k1*k2, k1, k2, new_middle_y, k, n);
	MatrixFit<<<k1, k2>>>(middle + 2*k1*k2, k1, k2, new_middle_z, k, n);
	MatrixFit<<<k1, k2>>>(middle + 3*k1*k2, k1, k2, new_middle_w, k, n);
	cudaDeviceSynchronize();
	print<<<1,1>>>(devU, m, k);
	cudaDeviceSynchronize();
	print<<<1,1>>>(devV, n, m);
	cudaDeviceSynchronize();
	print<<<1,1>>>(new_middle_x, k, n);
	cudaDeviceSynchronize();
	kernel_print<<<1,1>>>(middle, k1, k2);
	cudaDeviceSynchronize();

	float h_alpha = 1.0, h_beta = 0; // C = αA*B + βC
	int lda = m, ldb = k, ldc = m; // ABC的行数
	cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &h_alpha, devU, CUDA_R_16F, lda, new_middle_w, CUDA_R_16F, ldb, &h_beta, temp, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();
	print<<<1,1>>>(temp, m, n);
	cudaDeviceSynchronize();
	lda = m, ldb = n, ldc = m;
	cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, &h_alpha, temp, CUDA_R_16F, lda, devV, CUDA_R_16F, ldb, &h_beta, result, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();
	print<<<1,1>>>(result, m, m);
	cudaDeviceSynchronize();
	PointTest<<<1,1>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, 0.8,0.8);
	cudaDeviceSynchronize();

#endif // CUBLAS_NURBS_TEST

#ifdef CUBLAS_PERF_TEST

	double u = 0.8, v = 0.8;
	int a = findSpan(degree_u, cp_u, knots_u, u);
	int b = findSpan(degree_v, cp_v, knots_v, v);
	u = (u - knots_u[a]) / (knots_u[a + 1] - knots_u[a]);
	v = (v - knots_v[b]) / (knots_v[b + 1] - knots_v[b]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[a + 1] - knots_u[a]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[b + 1] - knots_v[b]);
	double* middle = MatrixN + MatrixNPosition(a, b, degree_u, degree_v, cp_u, cp_v, 0);

	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	// Set the math mode to allow cuBLAS to use Tensor Cores:
	cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	T_ELEM_IN *devUT = 0, *devV = 0, *result = 0, *temp = 0, *new_middle_x = 0, *new_middle_y = 0,*new_middle_z = 0, *new_middle_w = 0;
	int m = 288*288, k = 8, n = 8;
	// U: m*k H:k*n V:n*m 
	size_t sizeU = (size_t)m*k, sizeMiddle = (size_t)k*n, sizeV = (size_t)n*m, sizeResult = (size_t)m*m;
	int k1 = degree_u + 1, k2 = degree_v + 1;

	//warm up
	warm<<<256,256>>>();

	HANDLE_ERROR(cudaMalloc((void**)&devUT, sizeof(half) * m * k));
	HANDLE_ERROR(cudaMalloc((void**)&devV, sizeof(half) * n * m));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_x, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_y, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_z, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_w, sizeMiddle * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&result, sizeResult * sizeof(T_ELEM_IN)));
	HANDLE_ERROR(cudaMalloc((void**)&temp, sizeV * sizeof(T_ELEM_IN)));

	PrepareHalfVectorTransform<<<sqrt(m), sqrt(m)>>>(devUT, degree_u, cp_u, Gknots_u, m, k);
	PrepareHalfVector<<<sqrt(m), sqrt(m)>>>(devV, degree_v, cp_v, Gknots_v, n, m);
	MatrixFit<<<k1, k2>>>(middle, k1, k2, new_middle_x, k, n);
	MatrixFit<<<k1, k2>>>(middle + 1*k1*k2, k1, k2, new_middle_y, k, n);
	MatrixFit<<<k1, k2>>>(middle + 2*k1*k2, k1, k2, new_middle_z, k, n);
	MatrixFit<<<k1, k2>>>(middle + 3*k1*k2, k1, k2, new_middle_w, k, n);

	// print<<<1,1>>>(devV, n, m);
	// cudaDeviceSynchronize();
	// compute
	// C = αA*B + βC
	float h_alpha = 1.0, h_beta = 0; 
	// ABC的行数 
	// 若 transa 或 transb 参数 为 CUBLAS_OP_T, 则lda和ldb均为转置前的行数
	// 如transa == CUBLAS_OP_T, A转置后应为m*k, 则 A转置前应为k*m, 即lda = k;
	int lda = k, ldb = n, ldc = m; 
	cudaEventRecord(start);
	cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &h_alpha, devUT, CUDA_R_16F, lda, new_middle_x, CUDA_R_16F, ldb, &h_beta, temp, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = m, ldb = m, ldc = m;
	cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, &h_alpha, temp, CUDA_R_16F, lda, devV, CUDA_R_16F, ldb, &h_beta, result, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	// cout << cublasStat << endl;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << "ms" << endl;
	// print_diag<<<1,1>>>(result, m);
	// cudaDeviceSynchronize();
	// PointTest<<<1,1>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, Gcontrol_points, Gweights, 0.8,0.8);
	cudaEventRecord(start);
	PerfTest<<<sqrt(m), sqrt(m)>>>(MatrixN, degree_u, degree_v, cp_u, cp_v, Gknots_u, Gknots_v, 0.8, 0.8, R);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << "ms" << endl;

	PrepareHalfVectorTransform<<<sqrt(m), sqrt(m)>>>(devV, degree_v, cp_v, Gknots_v, m, n);
	T_ELEM_IN** Uarray = 0, **new_middle_x_array = 0, **Varray = 0, **temp_array = 0, **result_array = 0;
	HANDLE_ERROR(cudaMalloc((void**)&Uarray, sizeof(T_ELEM_IN *) * m));
	HANDLE_ERROR(cudaMalloc((void**)&new_middle_x_array, sizeof(T_ELEM_IN *) * m));
	HANDLE_ERROR(cudaMalloc((void**)&Varray, sizeof(T_ELEM_IN *) * m));
	HANDLE_ERROR(cudaMalloc((void**)&temp_array, sizeof(T_ELEM_IN *) * m));
	HANDLE_ERROR(cudaMalloc((void**)&result_array, sizeof(T_ELEM_IN *) * m));
	cudaEventRecord(start);
	SetArray<<<sqrt(m), sqrt(m)>>>(Uarray, devUT, 8*8);
	SetArray<<<sqrt(m), sqrt(m)>>>(new_middle_x_array, new_middle_x);
	SetArray<<<sqrt(m), sqrt(m)>>>(Varray, devV, 8*8);
	SetArray<<<sqrt(m), sqrt(m)>>>(temp_array, temp, 8*8);
	SetArray<<<sqrt(m), sqrt(m)>>>(result_array, result, 8*8);

	int bm = 8, bk = 8, bn = 8;
	lda = bk, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, bm, bn, bk, &h_alpha, (void**)Uarray, CUDA_R_16F, lda, (void**)new_middle_x_array, CUDA_R_16F, ldb, &h_beta, (void**)temp_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	// print<<<1,1>>>(Uarray, 0, 1, bm, bk);
	// cudaDeviceSynchronize();
	// print<<<1,1>>>(new_middle_x_array, 0, 1, bk, bn);
	// cudaDeviceSynchronize();
	// print<<<1,1>>>(temp_array, 0, 1, bm, bn);
	// cudaDeviceSynchronize();
	lda = bm, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, bm, bm, bn, &h_alpha, (void**)temp_array, CUDA_R_16F, lda, (void**)Varray, CUDA_R_16F, ldb, &h_beta, (void**)result_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
	lda = bk, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, bm, bn, bk, &h_alpha, (void**)Uarray, CUDA_R_16F, lda, (void**)new_middle_x_array, CUDA_R_16F, ldb, &h_beta, (void**)temp_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = bm, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, bm, bm, bn, &h_alpha, (void**)temp_array, CUDA_R_16F, lda, (void**)Varray, CUDA_R_16F, ldb, &h_beta, (void**)result_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = bk, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, bm, bn, bk, &h_alpha, (void**)Uarray, CUDA_R_16F, lda, (void**)new_middle_x_array, CUDA_R_16F, ldb, &h_beta, (void**)temp_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = bm, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, bm, bm, bn, &h_alpha, (void**)temp_array, CUDA_R_16F, lda, (void**)Varray, CUDA_R_16F, ldb, &h_beta, (void**)result_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = bk, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, bm, bn, bk, &h_alpha, (void**)Uarray, CUDA_R_16F, lda, (void**)new_middle_x_array, CUDA_R_16F, ldb, &h_beta, (void**)temp_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	lda = bm, ldb = bn, ldc = bm;
	cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, bm, bm, bn, &h_alpha, (void**)temp_array, CUDA_R_16F, lda, (void**)Varray, CUDA_R_16F, ldb, &h_beta, (void**)result_array, CUDA_R_16F, ldc, bm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << "ms" << endl;
	print<<<1,1>>>(result_array, 1, 2, bm, bm);
	cudaDeviceSynchronize();

	printf("3: %s\n", cudaGetErrorString(cudaGetLastError()));

	// test temp*vectorV without cublas
	// cudaEventRecord(start);
	// lda = bk, ldb = bn, ldc = bm;
	// cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, bm, bn, bk, &h_alpha, (void**)Uarray, CUDA_R_16F, lda, (void**)new_middle_x_array, CUDA_R_16F, ldb, &h_beta, (void**)temp_array, CUDA_R_16F, ldc, m/8, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	// lda = bm, ldb = bn, ldc = bm;
	// cublasStat = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, bm, bm, bn, &h_alpha, (void**)temp_array, CUDA_R_16F, lda, (void**)Varray, CUDA_R_16F, ldb, &h_beta, (void**)result_array, CUDA_R_16F, ldc, m/8, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	// // VectorMultiply<<<sqrt(m/8), sqrt(m/8)>>>(temp_array, Varray, 8, 8, result);
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	// cudaEventElapsedTime(&elapsedTime, start, stop);
	// cout << elapsedTime << "ms" << endl;
	// print<<<1,1>>>(result, 1, 1);
	// cudaDeviceSynchronize();

#endif // CUBLAS_PERF_TEST

#ifdef RANDOM_ACCESS

	int nx = 1024;
	int ny = 1024;// generate nx*ny random numbers

	int blockx = 32;
	int blocky = 32;
	dim3 block(blockx, blocky);//(32,1)

	int gridx = (nx + block.x - 1) / block.x;
	int gridy = (ny + block.y - 1) / block.y;
	dim3 grid(gridx,gridy); //(1,10)

	int N = gridx*gridy*blockx*blocky;// the number of states
//--------------------//
	curandState* devStates;

	cudaMalloc(&devStates, N * sizeof(curandState));

	int *arr;
	int size = 1024*1024;

	cudaMalloc(&arr, size * sizeof(int));

	srand(time(0));
	int seed = rand();

	//  Initialize the states
	setup_kernel <<<grid, block>>> (devStates, seed);

	float elapsedTime;

	cudaEventRecord(start);
	UniformAccess << <grid, block >> > (devStates,nx,ny,arr,size);
	// cudaThreadSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << "ms" << endl;

	cudaEventRecord(start);
	RandomAccess << <grid, block >> > (devStates,nx,ny,arr,size);
	// cudaThreadSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << "ms" << endl;

	cudaDeviceReset();
#endif 

#ifdef GPU_PRO
	cudaDeviceProp prop;
HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);

#endif

	return 0;
}