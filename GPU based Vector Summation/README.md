# GPU based Vector Summation
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.
<br>
<br>ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.

## Aim:
To explore the differences between the execution configurations of PCA-GPU-based vector summation.

## Procedure:
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile the program and execute it. Then, set the execution configuration of block.x = 1024 and recompile the program. Finally, compare the results obtained from the two execution configurations.
<br>
<br>
ii) Refer to sumArraysOnGPU-timer.cu, and set block.x = 256. Create a new kernel that allows each thread to handle two elements of the vector. Execute the program and compare the results obtained from this execution configuration with those obtained from other execution configurations.
```c
__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
        if (i + blockDim.x < N)
        {
            C[i + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
        }
    }
}

```
## Output:
### 1-
### Block size = 1023
![image](https://user-images.githubusercontent.com/65499285/235474476-543c8153-67c7-4488-b117-efaaecf4a71e.png)
### Block size = 1024
![image](https://user-images.githubusercontent.com/65499285/235474535-547c521d-50e1-4625-94d2-ce04598cc622.png)
### 2- 
### Block size = 256. Two Threads.
![image](https://user-images.githubusercontent.com/65499285/235474812-97ac4808-6fd8-4b47-a0b0-656e6d1c94f3.png)
## Result:
The result of the experiment will be a comparison of the execution times and results obtained from different execution configurations. This comparison will help determine the most efficient execution configuration for PCA-GPU-based vector summation.
