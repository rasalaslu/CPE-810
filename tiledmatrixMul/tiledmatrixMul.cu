#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#define BLOCK_SIZE 64

/* if the matrix size is intergal multiple of the block size
 * we can implemnt the algorithm without handdling boundary condition
 */
__global__ void matrixMul(float* a, float* b, float* c, int m, int n, int k)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (Col < k && Row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[Row * n + i] * b[i * k + Col];
        }
        c[Row * k + Col] = sum;
    }
}

// when need to handle the boundary condition
__global__ void matrixMulKernel(float* d_a, float* d_b, float* d_result, int n)
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float Pvalue = 0;
    int idx;

    for (int k = 0; k < gridDim.x; ++k)
    {
        idx = Row * n + k * BLOCK_SIZE + threadIdx.x;
        // fill the rest threads in the block located at boundary with 0
        if (idx >= n * n)
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (k * BLOCK_SIZE + threadIdx.y) * n + Col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Pvalue += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (Row < n && Col < n)
    {
        d_result[Row * n + Col] = Pvalue;
    }
}


int main(int argc, char const* argv[])
{
    int m, n, k;
    srand(3333);
    printf("Typin in the size of matrix A (MxN) and matrix B (NxK) in order of N M K:\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host memory
    int* h_a, * h_b, * h_c;
    cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
    cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
    cudaMallocHost((void**)&h_c, sizeof(int) * m * k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = 0 + 1.0 * (rand() % RAND_MAX) / RAND_MAX * (1024 - 0);
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = 0 + 1.0 * (rand() % RAND_MAX) / RAND_MAX * (1024 - 0);
        }
    }

    float gpu_elapsed_time_ms;

    // count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time
    cudaEventRecord(start, 0);
    // allocate memory space on the device
    float * d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(int) * m * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n * k);
    cudaMalloc((void**)&d_c, sizeof(int) * m * k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel 
    if (m == n && n == k)
    {
        matrixMulKernel << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
    }
    else
    {
        matrixMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);
    }
    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Compute time on matrix multiplication of %dx%d, %dx%d with block size of %d: %f ms.\n", m, n, n, k,BLOCK_SIZE, gpu_elapsed_time_ms);
    double flopsPerMatrixMul = 2.0 * static_cast<double>(n) *
        static_cast<double>(m) *
        static_cast<double>(k);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
        (gpu_elapsed_time_ms / 1000.0f);
    printf(
        "The compute throughput is %.2f GFlop/s.\n", gigaFlops);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}