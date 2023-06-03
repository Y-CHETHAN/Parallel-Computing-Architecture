#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>

#define N 1024
#define BLOCK_SIZE 16

__global__ void reduceUnrolling8(int* input, int* output)
{
    __shared__ int sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = input[i] + input[i + blockDim.x] +
        input[i + 2 * blockDim.x] + input[i + 3 * blockDim.x] +
        input[i + 4 * blockDim.x] + input[i + 5 * blockDim.x] +
        input[i + 6 * blockDim.x] + input[i + 7 * blockDim.x];

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void reduceUnrolling16(int* input, int* output)
{
    __shared__ int sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = input[i] + input[i + blockDim.x] +
        input[i + 2 * blockDim.x] + input[i + 3 * blockDim.x] +
        input[i + 4 * blockDim.x] + input[i + 5 * blockDim.x] +
        input[i + 6 * blockDim.x] + input[i + 7 * blockDim.x] +
        input[i + 8 * blockDim.x] + input[i + 9 * blockDim.x] +
        input[i + 10 * blockDim.x] + input[i + 11 * blockDim.x] +
        input[i + 12 * blockDim.x] + input[i + 13 * blockDim.x] +
        input[i + 14 * blockDim.x] + input[i + 15 * blockDim.x];

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

int main()
{
    int input[N];
    int output;

    // Initialize input data
    for (int i = 0; i < N; i++)
    {
        input[i] = i;
    }

    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch reduceUnrolling8 kernel and measure execution time with nvprof
    cudaEvent_t start8, stop8;
    cudaEventCreate(&start8);
    cudaEventCreate(&stop8);

    cudaEventRecord(start8);
    reduceUnrolling8 << <N / BLOCK_SIZE, BLOCK_SIZE >> > (d_input, d_output);
    cudaEventRecord(stop8);
    cudaEventSynchronize(stop8);

    // Copy result from device to host
    cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("reduceUnrolling8 result: %d\n", output);

    float milliseconds8 = 0;
    cudaEventElapsedTime(&milliseconds8, start8, stop8);
    printf("reduceUnrolling8 execution time: %.3f ms\n", milliseconds8);

    // Launch reduceUnrolling16 kernel and measure execution time with nvprof
    cudaEvent_t start16, stop16;
    cudaEventCreate(&start16);
    cudaEventCreate(&stop16);

    cudaEventRecord(start16);
    reduceUnrolling16 << <N / (BLOCK_SIZE * 16), BLOCK_SIZE >> > (d_input, d_output);
    cudaEventRecord(stop16);
    cudaEventSynchronize(stop16);

    // Copy result from device to host
    cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("reduceUnrolling16 result: %d\n", output);

    float milliseconds16 = 0;
    cudaEventElapsedTime(&milliseconds16, start16, stop16);
    printf("reduceUnrolling16 execution time: %.3f ms\n", milliseconds16);

    // Determine the kernel with the least execution time
    if (milliseconds8 < milliseconds16)
    {
        printf("reduceUnrolling8 has the least execution time.\n");
    }
    else if (milliseconds16 < milliseconds8)
    {
        printf("reduceUnrolling16 has the least execution time.\n");
    }
    else
    {
        printf("reduceUnrolling8 and reduceUnrolling16 have the same execution time.\n");
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
