# Matrix Summation using 2D grids and 2D blocks

## Aim:
To perform matrix addition on CPU and GPU and compare their execution time.


## Procedure:
- Initialize variables and allocate memory for the host and device.
- Initialize data on the host side.
- Calculate the sum of matrices on the host side and measure the execution time.
- Transfer the data from the host to the device.
- Calculate the sum of matrices on the device side and measure the execution time.
- Transfer the result from the device to the host.
- Compare the results from the host and the device.
- Free memory on the host and the device.
## Output:
![image](https://user-images.githubusercontent.com/65499285/235485535-aca4aa6e-c704-4580-bbc9-e328d9c7c768.png)

## Result:
The program prints the device information, matrix size, and the execution time for matrix initialization, matrix addition on the host, matrix addition on the device, and memory transfer. Finally, the program prints whether the results from the host and the device match or not. The output of the program provides insights into the performance of matrix addition on the host and the device, which can be used for further optimization.
