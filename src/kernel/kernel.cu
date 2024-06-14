#include "kernel.cuh"
#include <cublas_v2.h>

__global__ void Inverse(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, bool* flag, double2* result) {
	clock_t s, e;
	s = clock();
	//double2 origin = random[blockIdx.x*blockDim.x + threadIdx.x];
	//printf("%f %f", origin.x, origin.y);
	//get P locally
	double2 Current = { 0.25,0.25 };
	int m = findSpan(degree_u, cp_u, knots_u, Current.x);
	int n = findSpan(degree_v, cp_v, knots_v, Current.y);
	double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
	double3 P = homogenousToCartesian(MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v));
	//get P from array
	//double3 P = random[blockIdx.x*blockDim.x + threadIdx.x];
	//hard code P
	//double3 P = { -0.058757,0.191172,0.611200 }; 
	int size = 1, next_size = 0;
	//Node *current_layer = new Node[1];
	//Node *next_layer = new Node[4];
	bool current = true;
	Node current_layer[NODE_MAX_SIZE];
	Node next_layer[NODE_MAX_SIZE];
	current_layer[0] = { {0,0},{1,1} };
	while (size > 0) {
		for (int i = 0; i < size; i++) {
			//printf("%d: %f %f %f %f\n", i, current_layer[i].left_bottom.x, current_layer[i].left_bottom.y, current_layer[i].right_top.x, current_layer[i].right_top.y);
			if (BoundingBoxCheck(current_layer[i], P, max_edge, degree_u, degree_v,
				cp_u, cp_v, knots_u, knots_v, control_points)) {
				if (GaussNewton(current_layer[i], P, max_edge, MatrixN, degree_u, degree_v,
					cp_u, cp_v, knots_u, knots_v, control_points, weights, flag, result, blockIdx.x*blockDim.x + threadIdx.x)) {
					e = clock();
					double time = (double)(e - s) / GPU_clock_rate;
					//printf("Inverse time cost: %f\n", time);
					return;
				}
			}
		}
		for (int i = 0; i < size; i++) {
			SplitType type = getSplitType(current_layer[i], max_edge, MatrixN,
				degree_u, degree_v, cp_u, cp_v, knots_u, knots_v,
				control_points, weights);
			//printf("%d: %d\n", i, type);
			double lbx = current_layer[i].left_bottom.x, lby = current_layer[i].left_bottom.y;
			double rtx = current_layer[i].right_top.x, rty = current_layer[i].right_top.y;
			if (type == SPLIT_UV) {
				next_size += 4;
				if (next_size > NODE_MAX_SIZE)
					break;

				next_layer[next_size - 4] = { {lbx,lby},{(lbx + rtx) / 2,(lby + rty) / 2} };
				next_layer[next_size - 3] = { {lbx,(lby + rty) / 2},{(lbx + rtx) / 2,rty} };
				next_layer[next_size - 2] = { {(lbx + rtx) / 2,(lby + rty) / 2},{rtx,rty} };
				next_layer[next_size - 1] = { {(lbx + rtx) / 2,lby},{rtx,(lby + rty) / 2} };
			}
			else if (type == SPLIT_U) {
				next_size += 2;
				if (next_size > NODE_MAX_SIZE)
					break;

				next_layer[next_size - 2] = { {lbx,lby} ,{(lbx + rtx) / 2,rty} };
				next_layer[next_size - 1] = { {(lbx + rtx) / 2,lby},{rtx,rty} };
			}
			else if (type == SPLIT_V) {
				next_size += 2;
				if (next_size > NODE_MAX_SIZE)
					break;

				next_layer[next_size - 2] = { {lbx,lby} ,{rtx,(lby + rty) / 2} };
				next_layer[next_size - 1] = { {lbx,(lby + rty) / 2},{rtx,rty} };
			}
		}
		//size = next_size;
		/*delete[] current_layer;
		current_layer = next_layer;
		size = next_size;
		next_size = 0;
		next_layer = new Node[size * 4];*/
		if (next_size > NODE_MAX_SIZE)
			break;

		for (int i = 0; i < next_size; i++){
			current_layer[i] = next_layer[i];
		}
		size = next_size;
		next_size = 0;
	}
	flag[blockIdx.x*blockDim.x + threadIdx.x] = false;
	e = clock();
	double time = (double)(e - s) / GPU_clock_rate;
	//printf("%d: Inverse time cost: %f\n", blockIdx.x*blockDim.x + threadIdx.x, time);
	printf("%d: Inverse failed\n", blockIdx.x*blockDim.x + threadIdx.x);

	//free memory
	/*delete[] current_layer;
	delete[] next_layer;*/
}

__device__ bool BoundingBoxCheck(Node node, double3 P, double max_edge,
	int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points) {
	clock_t s, e;
	s = clock();
	int minspan_u = findSpan(degree_u, cp_u, knots_u, node.left_bottom.x) - degree_u;
	int minspan_v = findSpan(degree_v, cp_v, knots_v, node.left_bottom.y) - degree_v;
	int maxspan_u = findSpan(degree_u, cp_u, knots_u, node.right_top.x);
	int maxspan_v = findSpan(degree_v, cp_v, knots_v, node.right_top.y);
	//printf("%d %d\n%d %d\n", minspan_u, minspan_v, maxspan_u, maxspan_v);

	double3 surf_min_bound = control_points[minspan_u*cp_v + minspan_v];
	double3 surf_max_bound = control_points[minspan_u*cp_v + minspan_v];
	
	for (int i = minspan_u; i <= maxspan_u; i++) {
		for (int j = minspan_v; j <= maxspan_v; j++) {
			if ((surf_min_bound.x > control_points[i*cp_v + j].x))
				surf_min_bound.x = control_points[i*cp_v + j].x;
			if ((surf_min_bound.y > control_points[i*cp_v + j].y))
				surf_min_bound.y = control_points[i*cp_v + j].y;
			if ((surf_min_bound.z > control_points[i*cp_v + j].z))
				surf_min_bound.z = control_points[i*cp_v + j].z;
			if ((surf_max_bound.x < control_points[i*cp_v + j].x))
				surf_max_bound.x = control_points[i*cp_v + j].x;
			if ((surf_max_bound.y < control_points[i*cp_v + j].y))
				surf_max_bound.y = control_points[i*cp_v + j].y;
			if ((surf_max_bound.z < control_points[i*cp_v + j].z))
				surf_max_bound.z = control_points[i*cp_v + j].z;
		}
	}
	//printf("%f %f %f\n", surf_min_bound.x, surf_min_bound.y, surf_min_bound.z);
	//printf("%f %f %f\n", surf_max_bound.x, surf_max_bound.y, surf_max_bound.z);

	if (P.x >= surf_min_bound.x - max_edge && P.x <= surf_max_bound.x + max_edge
		&& P.y >= surf_min_bound.y - max_edge && P.y <= surf_max_bound.y + max_edge
		&& P.z >= surf_min_bound.z - max_edge && P.z <= surf_max_bound.z + max_edge) {
		e = clock();
		double time = (double)(e - s) / GPU_clock_rate;
		//printf("BoudingBox time cost: %f\n", time);
		return true;
	}

	return false;
}

__device__ bool GaussNewton(Node node, double3 P, double max_edge,double* MatrixN,
	int degree_u, int degree_v, int cp_u, int cp_v, double* knots_u, double *knots_v, 
	double3* control_points, double* weights, bool* flag, double2* result, int index) {
	clock_t s, e;
	double time;
	double2 Current = { (node.left_bottom.x + node.right_top.x) / 2, (node.left_bottom.y + node.right_top.y) / 2 };
	for (int i = 0; i < 10; i++) {
		//printf("%d\n", i);
		s = clock();
		//printf("%d: %f %f\n", i, Current.x, Current.y);
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		//if (i == 0)
			//printf("%d %d\n", m, n);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
		double4 SurfCurrent4d = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
		double3 SurfCurrent = homogenousToCartesian(SurfCurrent4d);
		e = clock();
		time = (double)(e - s) / GPU_clock_rate;
		//printf("P1 time cost: %f\n", time);
		s = clock();

		//distance between SurfCurrent and P is less than tolerance
		if (norm(SurfCurrent - P) < max_edge) {
			result[index] = Current;
			flag[index] = true;
			// printf("%d: %f %f\n", index, result[index].x, result[index].y);
			e = clock();
			double time = (double)(e - s) / GPU_clock_rate;
			//printf("Gauss time cost: %f\n", time);
			return true;
		}
		double3 derU = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU);
		double3 derV = MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanV);

		e = clock();
		time = (double)(e - s) / GPU_clock_rate;
		//printf("P2 time cost: %f\n", time);
		s = clock();

		double Jacob[6];
		double JacobT[6];
		Jacob[0] = derU.x;
		Jacob[1] = derV.x;
		Jacob[2] = derU.y;
		Jacob[3] = derV.y;
		Jacob[4] = derU.z;
		Jacob[5] = derV.z;

		double temp[4];

		unsigned int grid_1 = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
		unsigned int grid_2 = (2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
		unsigned int grid_3 = (3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 dimGrid23(grid_2, grid_3);
		dim3 dimGrid32(grid_3, grid_2);
		dim3 dimGrid22(grid_2, grid_2);
		dim3 dimGrid21(grid_2, grid_1);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		/*gpu_matrix_transpose<<<3,2>>>(Jacob, JacobT, 3, 2);
		gpu_matrix_mult<<<dimGrid22,dimBlock>>>(JacobT, Jacob, temp, 2, 3, 2);
		cudaDeviceSynchronize();*/
		matrix_transpose(Jacob, JacobT, 3, 2);
		matrix_mult(JacobT, Jacob, temp, 2, 3, 2);
		
		double tempInverse[4];
		double det = temp[0] * temp[3] - temp[1] * temp[2];
		tempInverse[0] = temp[3] / det;
		tempInverse[1] = -temp[1] / det;
		tempInverse[2] = -temp[2] / det;
		tempInverse[3] = temp[0] / det;

		double InverseJacob[6];
		//gpu_matrix_mult<<<dimGrid23,dimBlock>>>(tempInverse, JacobT, InverseJacob, 2, 2, 3);
		matrix_mult(tempInverse, JacobT, InverseJacob, 2, 2, 3);

		double dif[3];
		dif[0] = (SurfCurrent - P).x;
		dif[1] = (SurfCurrent - P).y;
		dif[2] = (SurfCurrent - P).z;

		double steplist[2];
		//gpu_matrix_mult<<<dimGrid21,dimBlock>>>(InverseJacob, dif, steplist, 2, 3, 1);
		//cudaDeviceSynchronize();
		matrix_mult(InverseJacob, dif, steplist, 2, 3, 1);

		double2 step = { steplist[0],steplist[1] };
		Current = Current - step;
		e = clock();
		time = (double)(e - s) / GPU_clock_rate;
		//printf("P3 time cost: %f\n", time);
	}
	return false;
}

__device__ SplitType getSplitType(Node node, double max_edge, double* MatrixN,
	int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights) {

	double min_bound[2] = { node.left_bottom.x,node.left_bottom.y }, max_bound[2] = { node.right_top.x,node.right_top.y };
	int integral_num = 4;
	double integral_u = (node.right_top.x - node.left_bottom.x) / integral_num;
	double integral_v = (node.right_top.y - node.left_bottom.y) / integral_num;
	double u_length = 0.0, v_length = 0.0;

	for (int i = 0; i < integral_num; i++)
	{
		//derivatives
		double2 point = { node.left_bottom.x + i * integral_u, node.left_bottom.y + i * integral_v };
		int m = findSpan(degree_u, cp_u, knots_u, point.x);
		int n = findSpan(degree_v, cp_v, knots_v, point.y);
		double u = (point.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (point.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
		double3 derU = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanU);
		double3 derV = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanV);

		u_length += norm(derU);
		v_length += norm(derV);
	}
	u_length *= (max_bound[0] - min_bound[0])*0.25;
	v_length *= (max_bound[1] - min_bound[1])*0.25;

	//get split type
	if (u_length < max_edge && v_length < max_edge)
	{
		return SplitType::NOT_SPLIT;
	}
	else if (max_bound[0] - min_bound[0] > 0.25 || max_bound[1] - min_bound[1] > 0.25)
	{
		return SplitType::SPLIT_UV;
	}
	else if (u_length < max_edge) 
		return SplitType::SPLIT_V;
	else if (v_length < max_edge) 
		return SplitType::SPLIT_U;
	else 
		return SplitType::SPLIT_UV;
}

__global__ void CheckAnswer(bool* flag, double2* result) {
	printf("%lf %lf\n", result[0].x, result[0].y);
}

__global__ void TestSecondDerivative(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, double2 Current) {
	int m = findSpan(degree_u, cp_u, knots_u, Current.x);
	int n = findSpan(degree_v, cp_v, knots_v, Current.y);
	double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double offset = 1e-10;
	double u_offset = (Current.x + offset - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	double v_offset = (Current.y + offset - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
	double4 SurfCurrent4d = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
	double3 SurfCurrent = homogenousToCartesian(SurfCurrent4d);
	double3 SurfOffsetU = homogenousToCartesian(MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u_offset, v));
	double3 SurfOffsetV = homogenousToCartesian(MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v_offset));


	double3 derU = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU);
	double3 derV = MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanV);
	double3 derU_offset = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u_offset, v, SurfCurrent4d, spanU);
	double3 derV_offset = MatrixSurfaceDerivativeV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v_offset, SurfCurrent4d, spanV);
	double3 derU_offset_v = MatrixSurfaceDerivativeU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v_offset, SurfCurrent4d, spanU);

	double3 derUU = MatrixSurfaceDerivativeUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU, spanV);
	double3 derVV = MatrixSurfaceDerivativeVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU, spanV);
	double3 derUV = MatrixSurfaceDerivativeUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, SurfCurrent4d, spanU, spanV);

	int k1 = degree_u + 1, k2 = degree_v + 1;
	printf("middle: \n");
	print(middle + 2 * k1*k2, k1, k2);

	printf("SurfCurrent4d: ");
	print(SurfCurrent4d);
	printf("SurfCurrent: ");
	print(SurfCurrent);
	printf("Estimated derU: ");
	print((SurfOffsetU - SurfCurrent) / offset);
	printf("Estimated derV: ");
	print((SurfOffsetV - SurfCurrent) / offset);

	printf("derU: ");
	print(derU);
	printf("derV: ");
	print(derV);
	printf("derU_offset: ");
	print(derU_offset);
	printf("derV_offset: ");
	print(derV_offset);
	printf("derU_offset_v: ");
	print(derU_offset_v);
	printf("offset: %.7lf\n", offset);

	printf("Estimated derUU: ");
	print((derU_offset - derU) / offset);
	printf("Calculated derUU: ");
	print(derUU);

	printf("Estimated derVV: ");
	print((derV_offset - derV) / offset);
	printf("Calculated derVV: ");
	print(derVV);

	printf("Estimated derUV: ");
	print((derU_offset_v - derU) / offset);
	printf("Calculated derUV: ");
	print(derUV);

	
}

__global__ void GetSecondDerivatives(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints, double *result, double3(*Der)(double*, int, int, int, int, double*, double*, double, double, double4, double, double)) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
		//double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);

		double3 der = Der(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, RPoints[index], spanU, spanV);
		result[index] = fmax(abs(der.x), fmax(abs(der.y), abs(der.z)));
	}
}

__global__ void GetPoints(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints) {
	int index = (blockIdx.x * gridDim.y  + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		//double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		//double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		RPoints[index] = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);
		//printf("%d: \n%lf %lf %lf %lf\n", index, 
			//RPoints[index].x, RPoints[index].y, RPoints[index].z, RPoints[index].w);
	}
}

__global__ void NewInverse(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, TreeNode* Tree, bool* flag, double2* result) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//double3 origin = random[index];
	//printf("%lf %lf", origin.x, origin.y);
	//double2 Current = { 0.25,0.25 };
	//int m = findSpan(degree_u, cp_u, knots_u, Current.x);
	//int n = findSpan(degree_v, cp_v, knots_v, Current.y);
	//double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
	//double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
	//double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
	//double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
	//double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
	//double3 P = homogenousToCartesian(MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v));
	//print(P);
	//get P from array
	//double3 P = random[blockIdx.x*blockDim.x + threadIdx.x];
	//hard code P for test_face2.nu with (0.25, 0.25)
	//double3 P = { 0.700000,0.455556,0.055556 };
	//hard code P for center_bk.nu with (0.25, 0.25)
	double3 P = { 0.000000,0.200049,0.616000 };
	
	for (int i = 0; i < TreeArraySize; i++) {
		if (NewBoundingBoxCheck(Tree[i], P)) {
			if (GaussNewton(Tree[i].uv, P, max_edge, MatrixN, degree_u, degree_v,
				cp_u, cp_v, knots_u, knots_v, control_points, weights, flag, result, index))
				//printf("%lf %lf\n", result[index].x, result[index].y);
				break;
		}
	}
}

__device__ bool NewBoundingBoxCheck(TreeNode T, double3 P) {
	return (P.x >= T.left_bottom.x) && (P.y >= T.left_bottom.y) && (P.z >= T.left_bottom.z) &&
		(P.x <= T.right_top.x) && (P.y <= T.right_top.y) && (P.z <= T.right_top.z);
}

__global__ void Project(double max_edge, double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	double3* random, TreeNode* Tree, NormalTreeNode* NormalTree, bool* flag, double2* result){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	// test input from uv = {0.25, 0.25} for center_bk.nu
	// real input is random array
	double3 P = { 0.000000,0.200049,0.616000 };
	
	for (int i = 0; i < TwoLayerTree; i++) {
		if (NormalBoundingBoxCheck(Tree[i], NormalTree[i], P)) {
			if (GaussNewton(Tree[i].uv, P, max_edge, MatrixN, degree_u, degree_v,
				cp_u, cp_v, knots_u, knots_v, control_points, weights, flag, result, index))
				printf("project answer: %lf %lf\n", result[index].x, result[index].y);
				break;
		}
	}
}

__device__ bool NormalBoundingBoxCheck(TreeNode T, NormalTreeNode Nt, double3 P){
	if((P.x >= T.left_bottom.x) && (P.y >= T.left_bottom.y) && (P.z >= T.left_bottom.z) &&
		(P.x <= T.right_top.x) && (P.y <= T.right_top.y) && (P.z <= T.right_top.z)){
			return true;
	}

	double x[2]{T.left_bottom.x, T.right_top.x};
	double y[2]{T.left_bottom.y, T.right_top.y};
	double z[2]{T.left_bottom.z, T.right_top.z};

	double now_angle;
	double3 vertex;
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				vertex = {x[i], y[j], z[k]}; // point on bbox
				now_angle = GetAngle(vertex, vertex + Nt.end_point, P);
				if(now_angle <= Nt.angle){
					return true;
				}
			}
		}
	}
	return false;
}

__global__ void BuildTree(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights, bool* Tree, int offset) {
	int total_node = offset * (ChildrenNumber - 1) + 1;
	int tree_index = blockIdx.x + offset;
	int index = threadIdx.x;
	int row = index / STEP, col = index % STEP;
	int node_row = blockIdx.x / sqrt((double)total_node), node_col = blockIdx.x % (int)(sqrt((double)total_node));
	double size = 1.0 / total_node;
	double2 node_low = { node_row * size, node_col * size }; // node左下角的参数值

	__shared__ double M1[STEP * STEP], M2[STEP * STEP], M3[STEP * STEP];

	if (offset == 0 || Tree[(index - 1) / 64]) {
		double2 Current = { node_low.x + size / STEP * row, node_low.y + size / STEP * col };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);
		double4 R = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v);

		M1[row * STEP + col] = _2Norm(MatrixSurfaceDerivativeUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanU, spanV));
		M2[row * STEP + col] = _2Norm(MatrixSurfaceDerivativeUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanU, spanV));
		M3[row * STEP + col] = _2Norm(MatrixSurfaceDerivativeVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, R, spanU, spanV));
	}
}

__global__ void TightBoundingBox(double4* RPoints, double *M1, double *M2, double *M3, int base, TreeNode* Tree, bool* Divided) {
	int index = blockIdx.x * gridDim.y + blockIdx.y;
	//printf("%d\n", index);
	int sub_index = index * PointsPerEdge * PointsPerEdge;
	int tree_index = base + index;
	int data_size = blockDim.x;

	RPoints = RPoints + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
	int idx = threadIdx.x;
	__shared__ double3 sdata[PointsPerBlock];
	if (idx < data_size) {

		/*copy to shared memory*/
		sdata[idx] = homogenousToCartesian(RPoints[idx]);
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x < rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y < rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z < rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		double K = (M1[index] + 2 * M2[index] + M3[index]) * __dsqrt_rn(3) / (8 * GapPerEdge * gridDim.x * GapPerEdge * gridDim.x);
		// printf("K: %lf\n", K);
		Tree[tree_index].right_top = { sdata[0].x + K, sdata[0].y + K, sdata[0].z + K };
		// printf("%lf %lf %lf\n", sdata[0].x, sdata[0].y, sdata[0].z);
	}

	if (idx < data_size) {
		/*copy to shared memory*/
		sdata[idx] = homogenousToCartesian(RPoints[idx]);
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x > rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y > rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z > rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		int parent_index = (tree_index - 1) / ChildrenNumber;
		if (parent_index < 0 || Divided[parent_index]) {
			double K = (M1[index] + 2*M2[index] + M3[index]) * __dsqrt_rn(3) / (8 * GapPerEdge * gridDim.x * GapPerEdge * gridDim.x);
			Tree[tree_index].left_bottom = { sdata[0].x - K, sdata[0].y - K, sdata[0].z - K };

			double unit = 1.0 / gridDim.x;
			Tree[tree_index].uv = { { blockIdx.x * unit ,blockIdx.y * unit }, { (blockIdx.x + 1) * unit , (blockIdx.y + 1) * unit } };
			if (parent_index >= 0) {
				Tree[tree_index] = min(Tree[tree_index], Tree[parent_index]);
				Tree[tree_index] = max(Tree[tree_index], Tree[parent_index]);
			}
			//判断是否需要继续细分
			Divided[tree_index] = NeedDivide(Tree[tree_index]);
		}

		/*printf("block: %d  K: %lf\n%lf %lf %lf\n%lf %lf %lf\n", index, K, 
			Tree[tree_index].right_top.x, Tree[tree_index].right_top.y, Tree[tree_index].right_top.z,
			Tree[tree_index].left_bottom.x, Tree[tree_index].left_bottom.y, Tree[tree_index].left_bottom.z);*/
		//print(Tree[tree_index].right_top);
		//print(Tree[tree_index].left_bottom);
		//printf("%lf", data[0]);
	}
}

__global__ void NormalBoundingBox(double3* Normal, double *N_M1, double *N_M2, double *N_M3, int base, TreeNode* NormalTree, bool* Divided){
	int index = blockIdx.x * gridDim.y + blockIdx.y;
	//printf("%d\n", index);
	int sub_index = index * PointsPerEdge * PointsPerEdge;
	int tree_index = base + index;
	int data_size = blockDim.x;

	Normal = Normal + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
	int idx = threadIdx.x;
	__shared__ double3 sdata[PointsPerBlock];
	if (idx < data_size) {

		/*copy to shared memory*/
		sdata[idx] = Normal[idx];
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x < rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y < rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z < rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		double K = (N_M1[index] + 2 * N_M2[index] + N_M3[index]) * __dsqrt_rn(3) / (8 * GapPerEdge * gridDim.x * GapPerEdge * gridDim.x);
		printf("NormalK: %lf\n", K);
		NormalTree[tree_index].right_top = { sdata[0].x + K, sdata[0].y + K, sdata[0].z + K };
		// printf("%lf %lf %lf\n", sdata[0].x, sdata[0].y, sdata[0].z);
	}

	if (idx < data_size) {
		/*copy to shared memory*/
		sdata[idx] = Normal[idx];
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x > rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y > rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z > rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		int parent_index = (tree_index - 1) / ChildrenNumber;
		// ToDo: remove this condition
		if(parent_index >= 0){
			Divided[parent_index] = true;
		}
		if (parent_index < 0 || Divided[parent_index]) {
			double K = (N_M1[index] + 2*N_M2[index] + N_M3[index]) * __dsqrt_rn(3) / (8 * GapPerEdge * gridDim.x * GapPerEdge * gridDim.x);
			NormalTree[tree_index].left_bottom = { sdata[0].x - K, sdata[0].y - K, sdata[0].z - K };

			double unit = 1.0 / gridDim.x;
			NormalTree[tree_index].uv = { { blockIdx.x * unit ,blockIdx.y * unit }, { (blockIdx.x + 1) * unit , (blockIdx.y + 1) * unit } };
			if (parent_index >= 0) {
				NormalTree[tree_index] = min(NormalTree[tree_index], NormalTree[parent_index]);
				NormalTree[tree_index] = max(NormalTree[tree_index], NormalTree[parent_index]);
			}
			//判断是否需要继续细分
			Divided[tree_index] = NormalNeedDivide(NormalTree[tree_index]);
		}

		/*printf("block: %d  K: %lf\n%lf %lf %lf\n%lf %lf %lf\n", index, K, 
			Tree[tree_index].right_top.x, Tree[tree_index].right_top.y, Tree[tree_index].right_top.z,
			Tree[tree_index].left_bottom.x, Tree[tree_index].left_bottom.y, Tree[tree_index].left_bottom.z);*/
		//print(Tree[tree_index].right_top);
		//print(Tree[tree_index].left_bottom);
		//printf("%lf", data[0]);
	}
}

// double version MAX value
__global__ void Max_Sequential_Addressing_Shared(double* data, int data_size, double* result) {
	data = data + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
	int idx = threadIdx.x;
	__shared__ double sdata[PointsPerBlock];
	if (idx < data_size) {

		/*copy to shared memory*/
		sdata[idx] = data[idx];
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double lhs = sdata[idx];
				double rhs = sdata[idx + stride];
				sdata[idx] = lhs < rhs ? rhs : lhs;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		result += blockIdx.x * gridDim.y + blockIdx.y;
		*result = sdata[0];
		// printf("%lf\n", data[0]);
	}
}

//double3 Version MAX Value
__global__ void MaxValuePoints(double3* data, int data_size, double3* result) {
	data = data + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
	int idx = threadIdx.x;
	__shared__ double3 sdata[PointsPerBlock];
	if (idx < data_size) {

		/*copy to shared memory*/
		sdata[idx] = data[idx];
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x < rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y < rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z < rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		result += blockIdx.x * gridDim.y + blockIdx.y;
		*result = sdata[0];
		//printf("%lf", data[0]);
	}
}

__global__ void MinValuePoints(double3* data, int data_size, double3* result) {
	data = data + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
	int idx = threadIdx.x;
	__shared__ double3 sdata[PointsPerBlock];
	if (idx < data_size) {

		/*copy to shared memory*/
		sdata[idx] = data[idx];
		__syncthreads();

		for (int stride = (blockDim.x + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
			if ((idx + stride) < data_size) {
				double3 lhs = sdata[idx];
				double3 rhs = sdata[idx + stride];
				sdata[idx].x = lhs.x > rhs.x ? rhs.x : lhs.x;
				sdata[idx].y = lhs.y > rhs.y ? rhs.y : lhs.y;
				sdata[idx].z = lhs.z > rhs.z ? rhs.z : lhs.z;
			}
			__syncthreads();
			if (stride == 1)
				break;
		}
	}
	if (idx == 0) {
		result += blockIdx.x * gridDim.y + blockIdx.y;
		*result = sdata[0];
		//printf("%lf", data[0]);
	}
}

__global__ void OutputTree(TreeNode* Tree, int size) {
	printf("Tree: \n");
	for (int i = 0; i < size; i++) {
		printf("%d :\n", i);
		printf("uv: \n");
		print(Tree[i].uv.left_bottom);
		print(Tree[i].uv.right_top);
		print(Tree[i].left_bottom);
		print(Tree[i].right_top);
	}
	printf("================\n");
}

__global__ void OutputNormalTree(NormalTreeNode* Tree, int size){
	printf("NormalTree: \n");
	for (int i = 0; i < size; i++) {
		printf("%d :\n", i);
		printf("uv: \n");
		print(Tree[i].end_point);
		print(Tree[i].angle);
		print(Tree[i].contain_zero);
	}
	printf("================\n");
}

__global__ void OutputRPoints(double4* Points, int size) {
	printf("RPoints: \n");
	for (int i = 0; i < size; i++) {
		printf("%d :\n", i);
		print(Points[i]);
	}
}

__global__ void OutputPoints(double3* Points, int size) {
	printf("Points: \n");
	for (int i = 0; i < size; i++) {
		printf("%d :\n", i);
		print(Points[i]);
	}
}

__global__ void OutputData(double* data, int size) {
	printf("Data: \n");
	for (int i = 0; i < size; i++) {
		printf("%d :\n", i);
		print(data[i]);
	}
}

__global__ void OutputBoolData(bool* data, int size) {
	printf("BoolData: \n");
	for (int i = 0; i < size; i++) {
		if (data[i]) {
			//printf("True\n");
		}
		else {
			printf("%d :\n", i);
			printf("False\n");
		}
	}
	printf("================\n");
}

__device__ bool NeedDivide(TreeNode n) {
	double delta_x = n.right_top.x - n.left_bottom.x;
	double delta_y = n.right_top.y - n.left_bottom.y;
	double delta_z = n.right_top.z - n.left_bottom.z;

	if (delta_x > delta_y) {
		double temp = delta_x;
		delta_x = delta_y;
		delta_y = temp;
	}
	if (delta_x > delta_z) {
		double temp = delta_x;
		delta_x = delta_z;
		delta_z = temp;
	}
	if ((delta_x < Threshold*delta_y) && (delta_x < Threshold*delta_z)) {
		return false;
	}
	return true;
}

// ToDo
__device__ bool NormalNeedDivide(TreeNode n){
	return true;
}

__device__ TreeNode min(const TreeNode& a, const TreeNode& b) {
	// keep a.left_bottom and a.uv
	TreeNode res = a;
	res.right_top.x = min(a.right_top.x, b.right_top.x);
	res.right_top.y = min(a.right_top.y, b.right_top.y);
	res.right_top.z = min(a.right_top.z, b.right_top.z);
	return res;
}

__device__ TreeNode max(const TreeNode& a, const TreeNode& b) {
	// keep a.right_top and a.uv
	TreeNode res = a;
	res.left_bottom.x = max(a.left_bottom.x, b.left_bottom.x);
	res.left_bottom.y = max(a.left_bottom.y, b.left_bottom.y);
	res.left_bottom.z = max(a.left_bottom.z, b.left_bottom.z);
	return res;
}

__global__ void GetNormals(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double3* Normals) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		Normals[index] = Normal(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, spanU, spanV);
		//printf("%d: \n%lf %lf %lf %lf\n", index, 
			//RPoints[index].x, RPoints[index].y, RPoints[index].z, RPoints[index].w);
	}
}

__global__ void GetNormalSecondDerivativeUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, double3* control_points, double* weights,
	bool* Divided, int base, double4* RPoints, double *result) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
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
		double3 PdV = MatrixSurfaceDerivativeV(RdV, P, R.w);
		double3 PdUU = MatrixSurfaceDerivativeUU(RdUU, PdU, P, R.w, RdU.w);
		double3 PdUV = MatrixSurfaceDerivativeUV(RdUV, PdU, PdV, P, R.w, RdU.w, RdV.w);
		double3 PdVV = MatrixSurfaceDerivativeVV(RdVV, PdV, P, R.w, RdV.w);
		double3 PdUUU = MatrixSurfaceDerivativeUUU(RdUUU, PdUU, PdU, P, R.w, RdU.w, RdUU.w);
		double3 PdUUV = MatrixSurfaceDerivativeUUV(RdUUV, PdUU, PdUV, PdU, PdV, P, R.w, RdU.w, RdV.w, RdUU.w, RdUV.w);
		double3 PdUVV = MatrixSurfaceDerivativeUVV(RdUVV, PdUV, PdVV, PdU, PdV, P, R.w, RdU.w, RdV.w, RdUV.w, RdVV.w);
		double3 PdVVV = MatrixSurfaceDerivativeVVV(RdVVV, PdVV, PdV, P, R.w, RdV.w, RdVV.w);

		double3 der = SurfaceNormalDerivativeUU(PdU, PdV, PdUU, PdUV, PdUUU, PdUUV);

		// result[index] = fmax(abs(der.x), fmax(abs(der.y), abs(der.z)));
	}
}

__global__ void BuildNormalBVH() {

}

__global__ void getNormals(bool* Divided, int base, double4 *R, double4 *RdU, double4 *RdV, double3* Normal){
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]){
		double3 P = homogenousToCartesian(R[index]);
		double3 PdU = MatrixSurfaceDerivativeU(RdU[index], P, R[index].w);
		double3 PdV = MatrixSurfaceDerivativeV(RdV[index], P, R[index].w);
		Normal[index] = cross_product(PdU, PdV);
		// if(index == 40){
		// 	printf("Normal: ");
		// 	print(Normal[index]);
		// }
	}
}

__global__ void getRs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectorUT, double* vectorV, double4 *R) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;

		vectorUT += index * k1;
		vectorV += index * k2;

		R[index] = MatrixSurfacePoint(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectorV);
		// if(index == 40){
		// 	printf("R: ");
		// 	print(Current);
		// 	print(R[index]);
		// 	for (int i = 0; i < k1; i++){
		// 		printf("%lf ", vectorUT[i]);
		// 	}
		// 	printf("\n");
		// }
	}
}

__global__ void getRdUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, double* vectordUT, 
	double* vectorV, double4 *RdU) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		// double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;

		vectordUT += index * k1;
		vectorV += index * k2;

		RdU[index] = MatrixSurfaceRdU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUT, vectorV, spanU);
		// if(index == 40){
		// 	print(Current);
		// 	printf("RdU: ");
		// 	print(RdU[index]);
		// }
	}
}

__global__ void getRdVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base, 
	double* vectorUT, double* vectordV, double4 *RdV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		// double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectorUT += index * k1;
		vectordV += index * k2;

		RdV[index] = MatrixSurfaceRdV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordV, spanV);
	}
}

__global__ void getRdUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUUT, double* vectorV, double4 *RdUU) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		// double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectordUUT += index * k1;
		vectorV += index * k2;

		RdUU[index] = MatrixSurfaceRdUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUUT, vectorV, spanU);
	}
}

__global__ void getRdUVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUT, double* vectordV, double4 *RdUV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectordUT += index * k1;
		vectordV += index * k2;

		RdUV[index] = MatrixSurfaceRdUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUT, vectordV, spanU, spanV);
	}
}

__global__ void getRdVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectorUT, double* vectordVV, double4 *RdVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		// double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectorUT += index * k1;
		vectordVV += index * k2;

		RdVV[index] = MatrixSurfaceRdVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordVV, spanV);
	}
}

__global__ void getRdUUUs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUUUT, double* vectorV, double4 *RdUUU) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		// double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectordUUUT += index * k1;
		vectorV += index * k2;

		RdUUU[index] = MatrixSurfaceRdUUU(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUUUT, vectorV, spanU);
	}
}

__global__ void getRdUUVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUUT, double* vectordV, double4 *RdUUV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectordUUT += index * k1;
		vectordV += index * k2;

		RdUUV[index] = MatrixSurfaceRdUUV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUUT, vectordV, spanU, spanV);
	}
}

__global__ void getRdUVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectordUT, double* vectordVV, double4 *RdUVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectordUT += index * k1;
		vectordVV += index * k2;

		RdUVV[index] = MatrixSurfaceRdUVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectordUT, vectordVV, spanU, spanV);
	}
}

__global__ void getRdVVVs(double* MatrixN, int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double* vectorUT, double* vectordVVV, double4 *RdVVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		// double spanU = (knots_u[degree_u + cp_u] - knots_u[0]) / (knots_u[m + 1] - knots_u[m]);
		double spanV = (knots_v[degree_v + cp_v] - knots_v[0]) / (knots_v[n + 1] - knots_v[n]);
		double* middle = MatrixN + MatrixNPosition(m, n, degree_u, degree_v, cp_u, cp_v, 0);

		int k1 = degree_u + 1, k2 = degree_v + 1;
		vectorUT += index * k1;
		vectordVVV += index * k2;

		RdVVV[index] = MatrixSurfaceRdVVV(middle, degree_u, degree_v, cp_u, cp_v, knots_u, knots_v, u, v, vectorUT, vectordVVV, spanV);
	}
}

__global__ void GetVector(int degree_u, int degree_v, int cp_u, int cp_v,
	double* knots_u, double *knots_v, bool* Divided, int base,
	double *vectorUT, double *vectorV, double *vectordUT, double *vectordV, 
	double *vectordUUT, double *vectordVV, double *vectordUUUT, double *vectordVVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double base_unit = 1.0 / gridDim.x;
		double unit = base_unit / GapPerEdge;

		double base_u = base_unit * blockIdx.x, base_v = base_unit * blockIdx.y;
		double p1 = base_u + threadIdx.x * unit, p2 = base_v + threadIdx.y * unit;

		int k1 = degree_u + 1, k2 = degree_v + 1;

		double2 Current = { p1,p2 };
		int m = findSpan(degree_u, cp_u, knots_u, Current.x);
		int n = findSpan(degree_v, cp_v, knots_v, Current.y);
		double u = (Current.x - knots_u[m]) / (knots_u[m + 1] - knots_u[m]);
		double v = (Current.y - knots_v[n]) / (knots_v[n + 1] - knots_v[n]);
		
		vectorUT += index * k1;
		vectordUT += index * k1;
		vectordUUT += index * k1;
		vectordUUUT += index * k1;

		vectorV += index * k2;
		vectordV += index * k2;
		vectordVV += index * k2;
		vectordVVV += index * k2;

		double now = 1;
		for (int i = 0; i < k1; i++) {
			vectorUT[i] = now;
			now *= u;
		}
		vectordUT[0] = 0;
		for (int i = 0; i < k1 - 1; i++) {
			vectordUT[i + 1] = (i + 1) * vectorUT[i];
		}
		vectordUUT[0] = vectordUUT[1] = 0;
		for (int i = 0; i < k1 - 2; i++) {
			vectordUUT[i + 2] = (i + 2)*(i + 1) * vectorUT[i];
		}
		vectordUUUT[0] = vectordUUUT[1] = vectordUUUT[2] = 0;
		for (int i = 0; i < k1 - 3; i++) {
			vectordUUUT[i + 3] = (i + 3)*(i + 2)*(i + 1) * vectorUT[i];
		}

		now = 1;
		for (int i = 0; i < k2; i++) {
			vectorV[i] = now;
			now *= v;
		}
		vectordV[0] = 0;
		for (int i = 0; i < k2 - 1; i++) {
			vectordV[i + 1] = (i + 1) * vectorV[i];
		}
		vectordVV[0] = vectordVV[1] = 0;
		for (int i = 0; i < k2 - 2; i++) {
			vectordVV[i + 2] = (i + 2)*(i + 1)*vectorV[i];
		}
		vectordVVV[0] = vectordVVV[1] = vectordVVV[2] = 0;
		for (int i = 0; i < k2 - 3; i++) {
			vectordVVV[i + 3] = (i + 3)*(i + 2)*(i + 1)*vectorV[i];
		}
		// if(index == 40){
		// 	printf("vectorUT: ");
		// 	for (int i = 0; i < k1; i++){
		// 		printf("%lf ", vectorUT[i]);
		// 	}
		// 	printf("\n");
		// }
	}
}

__global__ void GetDerivatives(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdVV, double4 *RdUUU, double4 *RdUUV, double4 *RdUVV, double4 *RdVVV,
	double3 *P, double3 *PdU, double3 *PdV, double3 *PdUU, double3 *PdUV, double3 *PdVV,
	double3 *PdUUU, double3 *PdUUV, double3 *PdUVV, double3 *PdVVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		P[index] = homogenousToCartesian(R[index]);
		PdU[index] = MatrixSurfaceDerivativeU(RdU[index], P[index], R[index].w);
		PdV[index] = MatrixSurfaceDerivativeV(RdV[index], P[index], R[index].w);
		PdUU[index] = MatrixSurfaceDerivativeUU(RdUU[index], PdU[index], P[index], R[index].w, RdU[index].w);
		PdUV[index] = MatrixSurfaceDerivativeUV(RdUV[index], PdU[index], PdV[index], P[index], R[index].w, RdU[index].w, RdV[index].w);
		PdVV[index] = MatrixSurfaceDerivativeVV(RdVV[index], PdV[index], P[index], R[index].w, RdV[index].w);
		PdUUU[index] = MatrixSurfaceDerivativeUUU(RdUUU[index], PdUU[index], PdU[index], P[index], R[index].w, RdU[index].w, RdUU[index].w);
		PdUUV[index] = MatrixSurfaceDerivativeUUV(RdUUV[index], PdUU[index], PdUV[index], PdU[index], PdV[index], P[index], R[index].w, RdU[index].w, RdV[index].w, RdUU[index].w, RdUV[index].w);
		PdUVV[index] = MatrixSurfaceDerivativeUVV(RdUVV[index], PdUV[index], PdVV[index], PdU[index], PdV[index], P[index], R[index].w, RdU[index].w, RdV[index].w, RdUV[index].w, RdVV[index].w);
		PdVVV[index] = MatrixSurfaceDerivativeVVV(RdVVV[index], PdVV[index], PdV[index], P[index], R[index].w, RdV[index].w, RdVV[index].w);
	}
}

__global__ void GetNdUUs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdUUU, double4 *RdUUV, double *NdUU) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double3 P = homogenousToCartesian(R[index]);
		double3 PdU = MatrixSurfaceDerivativeU(RdU[index], P, R[index].w);
		double3 PdV = MatrixSurfaceDerivativeV(RdV[index], P, R[index].w);
		double3 PdUU = MatrixSurfaceDerivativeUU(RdUU[index], PdU, P, R[index].w, RdU[index].w);
		double3 PdUV = MatrixSurfaceDerivativeUV(RdUV[index], PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w);
		double3 PdUUU = MatrixSurfaceDerivativeUUU(RdUUU[index], PdUU, PdU, P, R[index].w, RdU[index].w, RdUU[index].w);
		double3 PdUUV = MatrixSurfaceDerivativeUUV(RdUUV[index], PdUU, PdUV, PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w, RdUU[index].w, RdUV[index].w);

		double3 der = SurfaceNormalDerivativeUU(PdU, PdV, PdUU, PdUV, PdUUU, PdUUV);
		NdUU[index] = fmax(abs(der.x), fmax(abs(der.y), abs(der.z)));
		if(index == 40){
			printf("NdUU: ");
			print(NdUU[index]);
		}
	}
}

__global__ void GetNdUVs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUU, double4 *RdUV, double4 *RdVV, double4 *RdUUV, double4 *RdUVV, double *NdUV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;

	if (parent_index < 0 || Divided[parent_index]) {
		double3 P = homogenousToCartesian(R[index]);
		double3 PdU = MatrixSurfaceDerivativeU(RdU[index], P, R[index].w);
		double3 PdV = MatrixSurfaceDerivativeV(RdV[index], P, R[index].w);
		double3 PdUU = MatrixSurfaceDerivativeUU(RdUU[index], PdU, P, R[index].w, RdU[index].w);
		double3 PdUV = MatrixSurfaceDerivativeUV(RdUV[index], PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w);
		double3 PdVV = MatrixSurfaceDerivativeVV(RdVV[index], PdV, P, R[index].w, RdV[index].w);
		double3 PdUUV = MatrixSurfaceDerivativeUUV(RdUUV[index], PdUU, PdUV, PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w, RdUU[index].w, RdUV[index].w);
		double3 PdUVV = MatrixSurfaceDerivativeUVV(RdUVV[index], PdUV, PdVV, PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w, RdUV[index].w, RdVV[index].w);

		double3 der = SurfaceNormalDerivativeUV(PdU, PdV, PdUU, PdUV, PdVV, PdUUV, PdUVV);
		NdUV[index] = fmax(abs(der.x), fmax(abs(der.y), abs(der.z)));
		if(index == 40){
			printf("NdUV: ");
			print(NdUV[index]);
		}
	}
}

__global__ void GetNdVVs(bool* Divided, int base, double4* R, double4 *RdU, double4 *RdV,
	double4 *RdUV, double4 *RdVV, double4 *RdUVV, double4 *RdVVV, double *NdVV) {
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x *  blockDim.y + threadIdx.y;
	// blockIdx.x等均为uint类型，-1可能会造成溢出
	int parent_index = (base + (int)(blockIdx.x * gridDim.y + blockIdx.y) - 1) / ChildrenNumber;
	if (parent_index < 0 || Divided[parent_index]) {
		double3 P = homogenousToCartesian(R[index]);
		double3 PdU = MatrixSurfaceDerivativeU(RdU[index], P, R[index].w);
		double3 PdV = MatrixSurfaceDerivativeV(RdV[index], P, R[index].w);
		double3 PdUV = MatrixSurfaceDerivativeUV(RdUV[index], PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w);
		double3 PdVV = MatrixSurfaceDerivativeVV(RdVV[index], PdV, P, R[index].w, RdV[index].w);
		double3 PdUVV = MatrixSurfaceDerivativeUVV(RdUVV[index], PdUV, PdVV, PdU, PdV, P, R[index].w, RdU[index].w, RdV[index].w, RdUV[index].w, RdVV[index].w);
		double3 PdVVV = MatrixSurfaceDerivativeVVV(RdVVV[index], PdVV, PdV, P, R[index].w, RdV[index].w, RdVV[index].w);

		double3 der = SurfaceNormalDerivativeVV(PdU, PdV, PdUV, PdVV, PdUVV, PdVVV);
		NdVV[index] = fmax(abs(der.x), fmax(abs(der.y), abs(der.z)));
		if(index == 40){
			printf("NdVV: ");
			print(NdVV[index]);
		}
	}
}

__global__ void PrepareHalfVectorTransform(half* vector, int degree, int cp, double* knots, int row, int col){
	int index = gridDim.x * blockIdx.x + threadIdx.x;
	vector = vector + index * col;
	double u = 0.8; // test data
	
	int a = findSpan(degree, cp, knots, u);
	u = (u - knots[a]) / (knots[a + 1] - knots[a]);

	double now  = 1.0;
	int k = degree + 1;
	for(int i=0;i<k;i++){
		vector[i] = now;
		now *= u;
	}
	for(int i=k;i<col;i++){
		vector[i] = __float2half(0.0);
	}
}

__global__ void PrepareHalfVector(half* vector, int degree, int cp, double* knots, int row, int col){
	int index = gridDim.x * blockIdx.x + threadIdx.x;
	vector = vector + index;

	double u = 0.8; // test data
	
	int a = findSpan(degree, cp, knots, u);
	u = (u - knots[a]) / (knots[a + 1] - knots[a]);

	double now  = 1.0;
	int k = degree + 1;
	for(int i=0;i<k*col;i+=col){
		vector[i] = now;
		now *= u;
	}
	for(int i=k*col;i<row*col;i+=col){
		vector[i] = __float2half(0.0);
	}
}

__global__ void warm(){

}

__global__ void SetArray(half** array, half* src, int size){
	int index = gridDim.x * blockIdx.x + threadIdx.x;
	array[index] = src + size * index;
}

__device__ bool ContainZero(TreeNode node){
	return node.left_bottom.x <= 0 && node.right_top.x >= 0 &&
	node.left_bottom.y <= 0 && node.right_top.y >= 0 && 
	node.left_bottom.z <= 0 && node.right_top.z >= 0;
}


__global__ void NormalBoxToNormalTreeNode(int base, TreeNode* NormalBox, NormalTreeNode* NormalTree){
	int index = base + gridDim.x * blockIdx.x + threadIdx.x;
	if(ContainZero(NormalBox[index])){
		NormalTree[index].contain_zero = true;
		return;
	}
	NormalTree[index].contain_zero = false;
	double3 mid = (NormalBox[index].left_bottom + NormalBox[index].right_top) / 2;
	double now = 0;

	double x[2]{NormalBox[index].left_bottom.x, NormalBox[index].right_top.x};
	double y[2]{NormalBox[index].left_bottom.y, NormalBox[index].right_top.y};
	double z[2]{NormalBox[index].left_bottom.z, NormalBox[index].right_top.z};

	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				now = fmax(now, GetAngle({0,0,0}, mid, {x[i], y[j], z[k]}));
			}
		}
	}
	NormalTree[index].angle = now;
	NormalTree[index].end_point = mid;
}

__device__ double GetAngle(double3 a, double3 b, double3 c){
	double3 ab = b-a, ac = c-a;
	return acos(dot_product(ab,ac) / (_2Norm(ab)*_2Norm(ac)));
}