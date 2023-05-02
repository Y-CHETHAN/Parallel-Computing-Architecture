# Matrix Multiplication on Host and GPU

## Aim

The aim of this project is to demonstrate how to perform matrix multiplication using CUDA and compare the performance of the CPU and GPU implementations.

## Procedure

1. Allocate memory for matrices `h_a`, `h_b`, and `h_c` on the host.

2. Initialize matrices `h_a` and `h_b` with random values between 0 and 1.

3. Allocate memory for matrices `d_a`, `d_b`, and `d_c` on the device.

4. Copy matrices `h_a` and `h_b` from the host to the device.

5. Launch the kernel `matrixMulGPU` with `numBlocks` blocks of `threadsPerBlock` threads.

6. Measure the time taken by the CPU and GPU implementations using CUDA events.

7. Print the elapsed time for each implementation.

8. Free the memory allocated on both the host and the device.


## Output:
![image](https://user-images.githubusercontent.com/65499285/235492282-632e4a61-1194-48f7-9b0a-86d7d28c4904.png)

## Result

The result of this project is a comparison of the performance of the CPU and GPU implementations of matrix multiplication using CUDA. The elapsed time for each implementation is printed to the console.
