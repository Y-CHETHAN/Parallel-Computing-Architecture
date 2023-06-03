# Matrix Addition With Unified Memory

## Aim
The aim of this experiment is to demonstrate matrix addition using CUDA programming with unified memory.

## Procedure
1. Allocate unified memory for matrices A, B, and C.
2. Initialize matrices A and B with appropriate values.
3. Define the grid and block dimensions for the CUDA kernel.
4. Launch the CUDA kernel to perform matrix addition.
5. Synchronize the device to ensure all CUDA operations are completed.
6. Print the result matrix C.
7. Free the allocated device memory.

## Output

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/07a8472f-09d2-4601-a9c3-baefbec34321)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/f125fd3b-d021-41c3-96e6-007b1b78ea9f)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/eb8a8c46-1e57-4b60-be1f-85b0aa6f39c9)

## Result
The `memset` function calls are not necessary in this program. They were originally used to set the memory blocks for matrices A, B, and C to zero. However, the subsequent initialization loops already assign specific values to each element of the matrices, overwriting the previous values set by `memset`. Removing the `memset` calls does not affect the correctness of the program and has no significant impact on its performance. It is good practice to remove unnecessary code to improve code readability and maintainability.
<br>
The result printed to the console, showing the elasped time in the Host and the GPU to compare their performance.
