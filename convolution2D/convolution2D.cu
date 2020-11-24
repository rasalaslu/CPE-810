#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <ctime>
clock_t start, row, col;
// TILE_SIZE can be 8, 16, 32 and 64...
#define TILE_SIZE 16

// Define constant memory for Mask with max width of 10
#define MAX_MASK_WIDTH 64
__device__ __constant__ int M[MAX_MASK_WIDTH];


// Kernel to implement row convolution
__global__ void convolution_Row(int* d_P, int* d_N, int Width, int Height, int Mask_Width)
{
    __shared__ int N_ds[TILE_SIZE * MAX_MASK_WIDTH];
    // index for N and P
    const int index = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * Width + blockIdx.y * blockDim.y * Width;
    int x_i;
    int n = Mask_Width / 2;
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const int shift = threadIdx.y * (TILE_SIZE + Mask_Width - 1);

    // Go right
    x_i = tx + n;
    if (x_i > Width - 1) {
        N_ds[threadIdx.x + blockDim.x + shift] = 0;
    }
    else {
        N_ds[threadIdx.x + blockDim.x + shift] = d_N[index + n];
    }

    __syncthreads();

    // Convolution
    int sum = 0;
    for (int i = -n; i <= n; i++) {
        sum += N_ds[x_i + i + shift] * M[n + i];
    }
    d_P[index] = sum;
}

__global__ void convolution_Col(int* d_P, int* d_N, int Width, int Height, int Mask_Width)
{
    __shared__ int N_ds[TILE_SIZE * MAX_MASK_WIDTH];
    // index for N and P
    const int index = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * Width + blockIdx.y * blockDim.y * Width;
    int y_i;
    int n = Mask_Width / 2;
    const int ty = threadIdx.y + blockIdx.y * blockDim.y;
    const int shift = threadIdx.y * (TILE_SIZE + Mask_Width - 1);

    // Go down
    y_i = ty + n;
    const int shift_col = shift + blockDim.y * TILE_SIZE;
    if (y_i > Height - 1) {
        N_ds[threadIdx.x + shift_col] = 0;
    }
    else {
        N_ds[threadIdx.x + shift_col] = d_N[index + Width * n];
    }

    __syncthreads();

    // Convolution
    int sum = 0;
    for (int i = 0; i <= Mask_Width - 1; i++) {
        sum += N_ds[threadIdx.x + (threadIdx.y + i) * TILE_SIZE] * M[i];
    }
    d_P[index] = sum;
}


int main(int argc, char* argv[])
{
    // Check input format
    if (argc == 4) {
        printf("Inputing...\n");
    }
    else {
        printf("Error input Parameter \n");
        printf("Please Follow Format to Run Program: ./convolution2D <dimX> <dimY> <dimK>\n");
        printf("Please input <dimX>, <dimY>, <dimK> \n");
        printf("dimX is image width\ndimY is image height\ndimK is Mask size\n");
        return 0;
    }

    int dimX = atoi(argv[1]);
    int dimY = atoi(argv[2]);
    int dimK = atoi(argv[3]);


    // Check Mask sie
    if (dimK < 3 || dimK > MAX_MASK_WIDTH || dimK % 2 == 0) {
        printf("Mask size should be odd number and smaller than %d.\nWrong Input\n", MAX_MASK_WIDTH);
        return -1;
    }

    printf("Computing...\n");

    // Initialize image size and kernel size
    unsigned int img_size = dimX * dimY;
    const int kernel_Size = dimK;

    // Allocate space on host
    int* h_Mask, * h_Input, * h_Output1, * h_Output2;
    h_Mask = (int*)malloc(kernel_Size * sizeof(int));
    h_Input = (int*)malloc(img_size * sizeof(int));
    //h_Output1 = (int*)malloc(img_size * sizeof(int));
    h_Output2 = (int*)malloc(img_size * sizeof(int));

    // Initialize and generate Mask and Image. 
    srand((unsigned)time(NULL));
    for (unsigned int i = 0; i < kernel_Size; i++)
    {
        h_Mask[i] = rand() % 16;
    }

    for (unsigned i = 0; i < img_size; i++)
    {
        h_Input[i] = rand() % 16;
    }



    start = clock();
    // Allocate space on device
    int* d_Input, * d_Output1, * d_Output2;
    cudaMalloc((void**)&d_Input, img_size * sizeof(int));
    //cudaMalloc((void**)&d_Output1, img_size * sizeof(int));
    cudaMalloc((void**)&d_Output2, img_size * sizeof(int));
    cudaMemcpy(d_Input, h_Input, img_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_Mask, kernel_Size);
    dim3 blocks(TILE_SIZE, TILE_SIZE);
    dim3 grids(dimX / TILE_SIZE, dimY / TILE_SIZE);

    /*
    // Implement row convolution, deactivated when implementing col convolution
    convolution_Row << <grids, blocks >> > (d_Output1, d_Input, dimX, dimY, dimK);
    cudaDeviceSynchronize();
    // copy the result to the host
    cudaMemcpy(h_Output1, d_Output1, img_size * sizeof(int), cudaMemcpyDeviceToHost);

    row = clock();
    */


    // Implement col convolution, deactivated when implementing row convolution
    convolution_Col << <grids, blocks >> > (d_Output2, d_Input, dimX, dimY, dimK);
    cudaDeviceSynchronize();
    // copy the result to the host
    cudaMemcpy(h_Output2, d_Output2, img_size * sizeof(int), cudaMemcpyDeviceToHost);

    col = clock();
    


    // performance analysis
    /*
    double rowtime = (double)(row - start) / CLOCKS_PER_SEC;
    double rm = 2.0 * static_cast<double>(dimX) * static_cast<double>(dimY) * static_cast<double>(dimK);
    double rf = (rm * 1.0e-9f) / (rowtime / 1000.0f);
    */
    
    double coltime = (double)(col - start) / CLOCKS_PER_SEC;
    double cm = static_cast<double>(dimX) * static_cast<double>(dimY) * static_cast<double>(dimK);
    double cf = (cm * 1.0e-9f) / (coltime / 1000.0f);
    
    printf("The size of 2D input is: %d * %d\nThe Mask_Size = %d, TILE_SIZE = %d\n", dimX, dimY, dimK, TILE_SIZE);
    //printf("Performance for row convolution: %f ms, throughput: %f GFLOPS/s.\n", rowtime * 1000, rf);
    printf("Performance for col convolution: %f ms, throughput: %f GFLOPS/s.\n", coltime * 1000, cf);

    cudaFree(d_Input);
    //cudaFree(d_Output1);
    cudaFree(d_Output2);
    cudaFreeHost(h_Mask);
    cudaFreeHost(h_Input);
    //cudaFreeHost(h_Output1);
    cudaFreeHost(h_Output2);

    return 0;
}
