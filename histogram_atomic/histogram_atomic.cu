#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#define BLOCK_SIZE 256

// histogram kernel with atomic addition and privitization
__global__ void hist_GPU(int* d_vec, int* d_hist, int BinNum, int VecDim) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // get the bin size
    int BinSize = 1024 / BinNum;
    // allocates a shared memory array
    extern __shared__ int histo_s[];
    for (unsigned int binIdx = threadIdx.x; binIdx < BinNum; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    __syncthreads();
    // implement atomic addition
    for (unsigned int i = tid; i < VecDim; i += blockDim.x * gridDim.x) {
        atomicAdd(&(histo_s[d_vec[i] / BinSize]), 1);
    }
    __syncthreads();
    // commit to global memory
    for (unsigned int binIdx = threadIdx.x; binIdx < BinNum; binIdx += blockDim.x) {
        atomicAdd(&(d_hist[binIdx]), histo_s[binIdx]);
    }
}

// histogram in CPU version, to comfirm whether the result is correct
void hist_CPU(int* vector, int* hist_cpu, int BinNum, int VecDim) {
    int BinSize = 1024 / BinNum;
    for (int i = 0; i < VecDim; ++i) {
        ++hist_cpu[vector[i] / BinSize];
    }
    return;
}

// check whether the input parameters are in the proper range
int check_input(int BinNum, int VecDim) {
    if ((BinNum & (BinNum - 1)) != 0) {
        printf("Invalid <BinNum> \n");
        printf("<BinNum> must be 2 ^ n, and 2 < n < 8 \n");
        return -1;
    }

    if (VecDim < 0) {
        printf("Invalid <VecDim> \n");
        printf("<VecDim> must >= 0 \n");
        return -1;
    }

    return 1;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Input error! \n");
        printf("Please input <BinNum> and <VecDim> \n");
        return 0;
    }
    if (argc == 4 && (strcmp(argv[1], "-i") == 0)) {
        printf("Inputing Data...\n");
    }
    else {
        printf("Correct input Format: ./histogram_atomic -i binNum vecNum\n");
        return -1;
    }

    int BinNum = atoi(argv[2]);
    int VecDim = atoi(argv[3]);

    if (check_input(BinNum, VecDim) == 1) {
        printf("BinNum = %d, VecDim = %d.\n", BinNum, VecDim);
    }
    else {
        return -1;
    }

    // initialize vector
    int* vector;
    cudaMallocHost((void**)&vector, sizeof(int) * VecDim);
    // generate input vector
    srand((unsigned)time(NULL));
    for (int i = 0; i < VecDim; i++) {
        vector[i] = rand() % 1024;
    }

    // allocate memory on host for saving results
    int* hist_cpu = (int*)calloc(VecDim, sizeof(int));
    int* hist_gpu = (int*)calloc(VecDim, sizeof(int));

    // allocate memory on device
    int* d_vec, * d_hist;
    cudaMalloc((void**)&d_vec, sizeof(int) * VecDim);
    cudaMalloc((void**)&d_hist, sizeof(int) * BinNum);

    // transfer vector from host to device
    cudaMemcpy(d_vec, vector, sizeof(int) * VecDim, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BinNum);

    // prepare for recording the run time
    float gpu_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(128);

    // implement GPU version histogram and record the run time
    hist_GPU << < dimGrid, dimBlock, sizeof(int)* BinNum >> > (d_vec, d_hist, BinNum, VecDim);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    // copy the result to the host
    cudaMemcpy(hist_gpu, d_hist, sizeof(int) * BinNum, cudaMemcpyDeviceToHost);

    // implement the CPU version histogram
    hist_CPU(vector, hist_cpu, BinNum, VecDim);


    // validate results computed by GPU with shared memory
    int all_ok = 1;
    for (int i = 0; i < BinNum; i++)
    {
        if (hist_gpu[i] != hist_cpu[i])
        {
            all_ok = 0;
        }
    }
    if (all_ok == 1) {
        printf("Results from GPU and CPU are matched.\n");
    }
    else {
        printf("The result is incorrect!\n");
    }

    // performance analysis
    printf("Performance: %f ms. Throughput = %.4f MB/s.\n", gpu_time_ms, 1.0e-3 * (double)VecDim / gpu_time_ms);

    // free memory
    cudaFree(d_vec);
    cudaFree(d_hist);
    cudaFreeHost(vector);
    cudaFreeHost(hist_cpu);
    cudaFreeHost(hist_gpu);

    return 0;
}