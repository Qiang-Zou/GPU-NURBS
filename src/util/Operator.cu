#include "Operator.cuh"

__device__ double4 operator*(double w, double4 R) {
	return { R.x*w,R.y*w,R.z*w,R.w*w };
}

__device__ double4 operator*(double4 R, double w) {
	return { R.x*w,R.y*w,R.z*w,R.w*w };
}

__device__ double3 operator*(double w, double3 R) {
	return { R.x*w,R.y*w,R.z*w };
}
 
__device__ double3 operator*(double3 R, double w) {
	return { R.x*w,R.y*w,R.z*w };
}

__device__ double4 operator/(double4 R, double w) {
	return { R.x / w,R.y / w,R.z / w,R.w / w };
}

__device__ double3 operator/(double3 R, double w) {
	return { R.x / w,R.y / w,R.z / w };
}

__device__ double4 operator+(double4 a, double4 b) {
	return { a.x + b.x,a.y + b.y,a.z + b.z,a.w + b.w };
}

__device__ double3 operator+(double3 a, double3 b) {
	return { a.x + b.x,a.y + b.y,a.z + b.z };
}

__device__ double4 operator-(double4 a, double4 b) {
	return { a.x - b.x,a.y - b.y,a.z - b.z,a.w - b.w };
}

__device__ double3 operator-(double3 a, double3 b) {
	return { a.x - b.x,a.y - b.y,a.z - b.z };
}

__device__ double2 operator-(double2 a, double2 b) {
	return { a.x - b.x,a.y - b.y };
}

__device__ void print(double4 a) {
	printf("%lf %lf %lf %lf\n", a.x, a.y, a.z, a.w);
}

__device__ void print(double3 a) {
	printf("%lf %lf %lf\n", a.x, a.y, a.z);
}

__device__ void print(double2 a) {
	printf("%lf %lf\n", a.x, a.y);
}

__device__ void print(double a) {
	printf("%lf\n", a);
}

__device__ void print(double* a, int k1, int k2) {
	for (int i = 0; i < k1; i++) {
		for (int j = 0; j < k2; j++) {
			printf("%f ", a[i*k2 + j]);
		}
		printf("\n");
	}
	printf("\n");
}



