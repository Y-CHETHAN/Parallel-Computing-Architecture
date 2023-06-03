# Simple Warp Divergence

## Aim:
The aim of this experiment is to compare the performance of two CUDA kernels, `reduceUnrolling8` and `reduceUnrolling16`, which handle 8 and 16 data blocks per thread, respectively.

## Procedure:
The experiment follows the following steps:
1. Initialize an input array of size 1024.
2. Launch the `reduceUnrolling8` kernel, which performs reduction using 8 data blocks per thread.
3. Launch the `reduceUnrolling16` kernel, which performs reduction using 16 data blocks per thread.
4. Compare the results obtained from both kernels.

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

## Output:
The program outputs the results of the reduction performed by each kernel. Specifically, it displays the final reduced value obtained from the `reduceUnrolling8` kernel and the `reduceUnrolling16` kernel.

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/597ff299-dae1-45b7-8e8c-561cfe9efe34)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/441c1ebd-2dd6-4f14-b049-d039c6795b37)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/2d862570-2d14-4774-b3b6-a914dfc56252)

<br><br><br><br><br><br><br><br><br><br><br><br><br><br>

## Result:
The performance of the two kernels can be compared based on the reduction results. A higher reduction result indicates a more efficient reduction algorithm.
The comparison between the two results can provide insights into the performance difference between the kernels.
