# Matrix Summation using 2D grids and 2D blocks

## Aim:
To perform matrix summation using 2D grids and 2D blocks.

## Procedure:
1. Allocate and initialize memory for the input matrices (float and integer) on the host.
2. Allocate memory for the output matrices (float and integer) on the host.
3. Allocate memory for the input matrices (float and integer) on the device (GPU).
4. Allocate memory for the output matrices (float and integer) on the device (GPU).
5. Transfer the input matrices (float and integer) from the host to the device.
6. Define the dimensions of the grid and block for launching the GPU kernel.
7. Launch the kernel for float calculations on the GPU.
8. Synchronize the device to ensure the kernel execution is complete.
9. Copy the output matrix for float calculations from the device to the host.
10. Compare the GPU results for float calculations with the host results.
11. Launch the kernel for integer calculations on the GPU.
12. Synchronize the device to ensure the kernel execution is complete.
13. Copy the output matrix for integer calculations from the device to the host.
14. Compare the GPU results for integer calculations with the host results.
15. Free the memory allocated for the input and output matrices on the device.
16. Free the memory allocated for the input and output matrices on the host.
17. Reset the device.

<br><br><br><br><br><br>

## Program:
```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <windows.h>

inline double seconds()
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

void initialData(float* ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void initialData(int* ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = rand() & 0xFF;
    }
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
        }
    }
}

void sumMatrixOnHost(int* A, int* B, int* C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
        }
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Float Arrays match.\n");
    else
        printf("Float Arrays do not match.\n");
}

void checkResult(int* hostRef, int* gpuRef, const int N)
{
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = false;
            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Integer Arrays match.\n");
    else
        printf("Integer Arrays do not match.\n");
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        unsigned int idx = iy * nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(int* MatA, int* MatB, int* MatC, int nx, int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        unsigned int idx = iy * nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting float and integer matrix addition at ", deviceProp.name);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 10;
    int ny = 1 << 10;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    int nIntBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float* h_A, * h_B, * hostRef_float, * gpuRef_float;
    int* h_A_int, * h_B_int, * hostRef_int, * gpuRef_int;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef_float = (float*)malloc(nBytes);
    gpuRef_float = (float*)malloc(nBytes);
    h_A_int = (int*)malloc(nIntBytes);
    h_B_int = (int*)malloc(nIntBytes);
    hostRef_int = (int*)malloc(nIntBytes);
    gpuRef_int = (int*)malloc(nIntBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    initialData(h_A_int, nxy);
    initialData(h_B_int, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef_float, 0, nBytes);
    memset(gpuRef_float, 0, nBytes);
    memset(hostRef_int, 0, nIntBytes);
    memset(gpuRef_int, 0, nIntBytes);

    // add matrix at host side for float result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef_float, nx, ny);
    sumMatrixOnHost(h_A_int, h_B_int, hostRef_int, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    float* d_MatA_float, * d_MatB_float, * d_MatC_float;
    int* d_MatA_int, * d_MatB_int, * d_MatC_int;
    CHECK(cudaMalloc((void**)&d_MatA_float, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB_float, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC_float, nBytes));
    CHECK(cudaMalloc((void**)&d_MatA_int, nIntBytes));
    CHECK(cudaMalloc((void**)&d_MatB_int, nIntBytes));
    CHECK(cudaMalloc((void**)&d_MatC_int, nIntBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA_float, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB_float, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatA_int, h_A_int, nIntBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB_int, h_B_int, nIntBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warmup kernel
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA_float, d_MatB_float, d_MatC_float, nx, ny);
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA_int, d_MatB_int, d_MatC_int, nx, ny);
    CHECK(cudaDeviceSynchronize());

    // execute the float kernel
    iStart = seconds();
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA_float, d_MatB_float, d_MatC_float, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D float <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef_float, d_MatC_float, nBytes, cudaMemcpyDeviceToHost));

    // check device results with host results for float
    checkResult(hostRef_float, gpuRef_float, nxy);

    // execute the integer kernel
    iStart = seconds();
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA_int, d_MatB_int, d_MatC_int, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D integer <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef_int, d_MatC_int, nIntBytes, cudaMemcpyDeviceToHost));

    // check device results with host results for integer
    checkResult(hostRef_int, gpuRef_int, nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA_float));
    CHECK(cudaFree(d_MatB_float));
    CHECK(cudaFree(d_MatC_float));
    CHECK(cudaFree(d_MatA_int));
    CHECK(cudaFree(d_MatB_int));
    CHECK(cudaFree(d_MatC_int));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef_float);
    free(gpuRef_float);
    free(hostRef_int);
    free(gpuRef_int);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}
```
## Output:

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/dee5a19d-b22f-4b3b-a35b-3d959d22cc1a)

## Result:
The host took 0.0038 seconds to complete it's computation, while the GPU outperforms the host and completes the computation in 0.000197 seconds for float variables and 0.000212 seconds for integer variables. Therefore, float variables in the GPU will result in the best possible result.
Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
