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

//max level 4, max size 4^4
#define NODE_MAX_SIZE 64
#define STEP 32
//
#define GapPerEdge 8
#define PointsPerEdge (GapPerEdge + 1)
#define ChildrenNumber (GapPerEdge * GapPerEdge)
#define PointsPerBlock (PointsPerEdge * PointsPerEdge)
// moved from main.cu
#define DataSize 65536*512
#define BLOCK_SIZE 16
#define TreeArraySize (1+ChildrenNumber+ChildrenNumber*ChildrenNumber+ChildrenNumber*ChildrenNumber*ChildrenNumber)  // 4 layers
#define TwoLayerTree (1+ChildrenNumber)
#define ThreeLayerTree (1+ChildrenNumber+ChildrenNumber*ChildrenNumber)
#define PointsArraySize (ChildrenNumber*ChildrenNumber*ChildrenNumber * PointsPerBlock) //最底一层的点的个数
// 判断是否需要细分的门槛值
#define Threshold 0.5

struct Node {
	double2 left_bottom;
	double2 right_top;
};

struct TreeNode {
	Node uv;
	double3 left_bottom;
	double3 right_top;
};

struct NormalTreeNode{
	double3 end_point; // start point is {0, 0, 0}
	double angle;
	bool contain_zero;
};

enum SplitType
{
	NOT_SPLIT,
	SPLIT_V,
	SPLIT_U,
	SPLIT_UV
};

__global__ void Inverse(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, bool* flag, double2* result);

__device__ bool BoundingBoxCheck(Node node, double3 P, double max_edge,
	int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points);

__device__ bool GaussNewton(Node node, double3 P, double max_edge, double* MatrixN,
	int degree_u, int degree_v, int cp_u, int cp_v, double* knots_u, double *knots_v,
	double3* control_points, double* weights, bool* flag, double2* result, int index);

__device__ SplitType getSplitType(Node node, double max_edge, double* MatrixN,
	int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights);

//transfer nodes from s2 to s1
__device__ void NodeTransfer(Node* s1, Node* s2, int size);

__global__ void CheckAnswer(bool* flag, double2* result);

__global__ void TestSecondDerivative(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double2 Current);

__global__ void BuildTree(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Tree, int base, double unit);

__global__ void GetPoints(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints);

__global__ void GetFirstDerivatives(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Tree, int base, double unit);

// 只存储二阶导三个坐标中最大的一个在result中
__global__ void GetSecondDerivatives(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints, double *result, double3(*Der)(double*, int, int, int, int, double*, double*, double, double, double4, double, double));

__global__ void Max_Sequential_Addressing_Shared(double* data, int data_size, double* result);

__global__ void MaxValuePoints(double3* data, int data_size, double3* result);

__global__ void MinValuePoints(double3* data, int data_size, double3* result);

__global__ void TightBoundingBox(double4* RPoints, double *M1, double *M2, double *M3, int base, TreeNode* Tree, bool* Divided);

// ToDo
__global__ void NormalBoundingBox(double3* Normal, double *N_M1, double *N_M2, double *N_M3, int base, TreeNode* NormalTree, bool* Divided);

__device__ bool NeedDivide(TreeNode n);

__device__ bool NormalNeedDivide(TreeNode n);

__global__ void NewInverse(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, TreeNode* Tree, bool* flag, double2* result);

__device__ bool NewBoundingBoxCheck(TreeNode T, double3 P);

//ToDo: 
__global__ void Project(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, TreeNode* Tree, NormalTreeNode* NormalTree, bool* flag, double2* result);

__device__ bool NormalBoundingBoxCheck(TreeNode T, NormalTreeNode Nt, double3 P);

// Debug function
__global__ void OutputTree(TreeNode* Tree, int size);

__global__ void OutputRPoints(double4* Points, int size);

__global__ void OutputPoints(double3* Points, int size);

__global__ void OutputData(double* data, int size);

__global__ void OutputBoolData(bool* data, int size);

__global__ void OutputNormalTree(NormalTreeNode* Tree, int size);

__device__ TreeNode min(const TreeNode& a, const TreeNode& b);

__device__ TreeNode max(const TreeNode& a, const TreeNode& b);

// projection kernels

__global__ void GetNormalSecondDerivativeUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints, double *result);

__global__ void BuildNormalBVH();

__global__ void getNormals(bool* Divided, int base, double4 *R, double4 *RdU, double4 *RdV, double3* Normal);

__global__ void getRs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectorUT, double* vectorV, double4 *R);

__global__ void getRdUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v,bool* Divided, int base, 
	double* vectordUT, double* vectorV, double4 *RdU);

__global__ void getRdVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectorUT, double* vectordV, double4 *RdV);

__global__ void getRdUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectordUUT, double* vectorV, double4 *RdUU);

__global__ void getRdUVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectordUT, double* vectordV, double4 *RdUV);

__global__ void getRdVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectorUT, double* vectordVV, double4 *RdVV);

__global__ void getRdUUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectordUUUT, double* vectorV, double4 *RdUUU);

__global__ void getRdUUVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectordUUT, double* vectordV, double4 *RdUUV);

__global__ void getRdUVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUT, double* vectordVV, double4 *RdUVV);

__global__ void getRdVVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectorUT, double* vectordVVV, double4 *RdVVV);

__global__ void GetDerivatives(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdVV, double4 *RdUUU, double4 *RdUUV, double4 *RdUVV, double4 *RdVVV,
	double3 *P, double3 *PdU, double3 *PdV, double3 *PdUU, double3 *PdUV, double3 *PdVV,
	double3 *PdUUU, double3 *PdUUV, double3 *PdUVV, double3 *PdVVV);

__global__ void GetVector(int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double *vectorUT, double *vectorV, double *vectordUT, double *vectordV,
	double *vectordUUT, double *vectordVV, double *vectordUUUT, double *vectordVVV);

__global__ void GetNdUUs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdUUU, double4 *RdUUV, double *NdUU);

__global__ void GetNdUVs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdVV, double4 *RdUUV, double4 *RdUVV, double *NdUV);

__global__ void GetNdVVs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUV, double4 *RdVV, double4 *RdUVV, double4 *RdVVV, double *NdVV);

__global__ void PrepareHalfVectorTransform(half* vector, int degree, int cp, double* knots, int row,  int col);

__global__ void PrepareHalfVector(half* vector, int degree, int cp, double* knots, int row, int col);

__global__ void warm();

__global__ void SetArray(half** array, half* src, int size = 0);

__device__ bool ContainZero(NormalTreeNode node);

__global__ void NormalBoxToNormalTreeNode(int base, TreeNode* NormalBox, NormalTreeNode* NormalTree);

// get the angle between line ab and ac
__device__ double GetAngle(double3 a, double3 b, double3 c);