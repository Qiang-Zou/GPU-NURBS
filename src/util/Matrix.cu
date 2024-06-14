#include "Matrix.cuh"

// HANDLE_ERROR
static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void gpu_matrix_mult(double* a, double* b, double* c, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
        // if (k == 1)
        // printf("%d %d\n", row, col);
        // if (k == 1 && m == 1) 
        // printf("%f\n", c[row * k + col]);
    }
}

__global__ void gpu_matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols) {
    unsigned int idx = blockIdx.x;
    unsigned int idy = threadIdx.x;

    if (idx < rows && idy < cols) {
        unsigned int pos = idx * cols + idy;
        unsigned int trans_pos = idy * rows + idx;
        mat_out[trans_pos] = mat_in[pos];
    }
}

__device__ double norm(const double3& P) {
    return sqrt(P.x * P.x + P.y * P.y + P.z * P.z);
}

__device__ void matrix_mult(double* a, double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = 0;
            for (int l = 0; l < n; l++) {
                c[i * k + j] += a[i * n + l] * b[l * k + j];
            }
        }
    }
}

__device__ void matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat_out[j * rows + i] = mat_in[i * cols + j];
        }
    }
}

__device__ double3 cross_product(double3 a, double3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ double dot_product(double3 a, double3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__global__ void MatrixFit(double* src, int a, int b, half* des, int c, int d) {
    if (a > c || b > d) {
        return;
    }

    unsigned int i = blockIdx.x, j = threadIdx.x;
    des[i*d + j] = __double2half(src[i*b+j]);
    
}

__global__ void VectorMultiply(half** A, half** B, int m, int n, half* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    half res = __double2half(0.0);

    int matrix_index = index/m, vector_index = index%m;
    for(int i=0;i<n;i++){
        res = __hadd(res, __hmul(A[matrix_index][i*m+vector_index], B[matrix_index][vector_index * n + i]));
    }
    
    result[index] = res;
}
