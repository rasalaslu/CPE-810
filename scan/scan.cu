#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>

typedef unsigned int uint;
#define BLOCK_SIZE 1024

__global__ void scanGPU(uint* d_list, uint* flags, uint* AuxArray, uint* AuxScannedArray, int dim) {
    extern __shared__ uint I;
    if (threadIdx.x == 0) {
        I = atomicAdd(&AuxScannedArray[0], 1);
    }
    __syncthreads();

    extern __shared__ uint scanBlockSum[2 * BLOCK_SIZE];
    uint t = threadIdx.x;
    uint s = 2 * I * blockDim.x;

    if (s + t < dim) scanBlockSum[t] = d_list[s + t];
    if (s + t + blockDim.x < dim) scanBlockSum[blockDim.x + t] = d_list[s + blockDim.x + t];
    __syncthreads();

    for (uint stride = 1; stride <= blockDim.x; stride *= 2) {
        int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) scanBlockSum[idx] += scanBlockSum[idx - stride];
        __syncthreads();
    }

    for (uint stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < 2 * blockDim.x) {
            scanBlockSum[idx + stride] += scanBlockSum[idx];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (I == 0) {
            AuxArray[I] = scanBlockSum[2 * blockDim.x - 1];
            atomicAdd(&flags[I], 1);
        }
        else {
            while (atomicAdd(&flags[I - 1], 0) == 0) { ; }
            AuxArray[I] = AuxArray[I - 1] + scanBlockSum[2 * blockDim.x - 1];
            __threadfence();
            atomicAdd(&flags[I], 1);
        }
    }
    __syncthreads();

    if (I > 0) {
        scanBlockSum[t] += AuxArray[I - 1];
        scanBlockSum[t + blockDim.x] += AuxArray[I - 1];
    }
    __syncthreads();

    if (s + t < dim)  d_list[s + t] = scanBlockSum[t];
    if (s + t + blockDim.x < dim) d_list[s + blockDim.x + t] = scanBlockSum[blockDim.x + t];
}

void scanCPU(uint* list, uint* sum, int dim) {
    uint res = 0;
    for (int i = 0; i < dim; ++i) {
        res += list[i];
        sum[i] = res;
    }
    return;
}


int check_input(int dim) {
    if (dim < 0) {
        printf("Invalid list size \n");
        printf("list size must be >= 0 \n");
        return -1;
    }
    return 1;
}

int main(int argc, char** argv) {
    // Input and check
    if (argc != 3) {
        printf("Error input Parameter \n");
        printf("Please input dim for input list \n");
        printf("Example: ./execute_file -i dim \n");
        return 0;
    }
    if (argc == 3 && (strcmp(argv[1], "-i") == 0)) {
        printf("Input Data\n");
    }
    else {
        printf("Please Follow Format to Run Program: ./execute_file -i dim\n");
        return -1;
    }
    const int Dim = atoi(argv[2]);
    if (check_input(Dim) == 1) {
        printf("Input is Valid \n");
    }
    else {
        return -1;
    }
    printf("InputSize = %d, Block size = %d.\n\n", Dim, BLOCK_SIZE);

    //timer
    float gpu_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //clock_t start, end;


    // Initialize list data
    uint* list = (uint*)malloc(Dim * sizeof(uint));
    srand((unsigned)time(NULL));
    for (uint i = 0; i < Dim; i++) {
        list[i] = rand();
    }

    // Allocate memory on host for saving results
    uint* scan_CPU = (uint*)malloc(Dim * sizeof(uint));
    uint* scan_GPU = (uint*)malloc(Dim * sizeof(uint));

    // Allocate memory on device for variable
    uint* d_list, * d_flags, * d_AuxArray, * d_AuxScannedArray;
    cudaMalloc((uint**)&d_list, Dim * sizeof(uint));
    cudaMalloc((uint**)&d_AuxScannedArray, sizeof(uint));
    cudaMalloc((uint**)&d_flags, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(uint));
    cudaMalloc((uint**)&d_AuxArray, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(uint));
    cudaMemset(d_flags, 0, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(uint));
    cudaMemset(d_AuxScannedArray, 0, sizeof(uint));
    cudaMemcpy(d_list, list, Dim * sizeof(uint), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    // Initialize and Invoke the kernel
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil(1.0 * Dim / dimBlock.x));
    scanGPU << <dimGrid, dimBlock, (2 * BLOCK_SIZE + 1) * sizeof(uint) >> > (d_list, d_flags, d_AuxArray, d_AuxScannedArray, Dim);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Copy the result to host
    cudaMemcpy(scan_GPU, d_list, Dim * sizeof(uint), cudaMemcpyDeviceToHost);


    //start = clock();
    // Invoke the CPU scan algorithm
    scanCPU(list, scan_CPU, Dim);
    //end = clock();
    //double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;

    // Check the result
    int check = 1;
    for (int i = 0; i < Dim; i++) {
        if (scan_CPU[i] != scan_GPU[i]) {
            check = 0;
        }
    }
    if (check == 1) {
        printf("Results match.\n");
    }
    else {
        printf("Wrong Result.\n");
    }

    // Performance calculation
    printf("GPU executing time: %.4f ms, throughput = %.4f MElements/s\n", gpu_time, Dim / gpu_time / 1000);
    //printf("CPU executing time: %.4f ms, throughput = %.4f MElements/s\n", cpu_time, Dim / cpu_time / 1000);

    // Deallocate memory
    cudaFree(d_list);
    cudaFree(d_flags);
    cudaFree(d_AuxArray);
    cudaFree(d_AuxScannedArray);
    cudaFreeHost(scan_CPU);
    cudaFreeHost(scan_GPU);

    return 0;
}