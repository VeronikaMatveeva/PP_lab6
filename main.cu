#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

//лучший результат при соотношении 1 к 25
#define N 2000
#define div 80

//алгоритм Флойда-Уоршелла
__global__ void floyd(int* b, int i) {
	int k = blockIdx.x*(N / div) + threadIdx.x;
	int j = blockIdx.y*(N / div) + threadIdx.y;
	if (b[j * N + k] > b[j * N + i] + b[i * N + k]) {
		b[j * N + k] = b[j * N + i] + b[i * N + k];
	}
}

int main()
{
	//заполнение матрицы смежности
	int* G;
	G = new int[N * N];
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			if (i == j) {
				G[i * N + j] = 0;
			}
			else {
				G[i * N + j] = G[j * N + i] = rand() % 10;
			}
		}
	}

	printf("N = %d \n", N);
	
	/*printf("\n INPUT: \n");

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d ", G[i * N + j]);
		}
		printf("\n");
	}*/

	int * dev;
	cudaMalloc((void**)&dev, N * N * sizeof(int));

	cudaError_t error;
	error = cudaMemcpy(dev, G, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
	}

	dim3 grid(div, div);
	dim3 blocks(N/div, N/div);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//запускаем алгоритм
	for (int i = 0; i < N; ++i) {
		floyd << <grid, blocks >> > (dev, i);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	//записываем время работы
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaDeviceSynchronize();

	error = cudaMemcpy(G, dev, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
	}

	/*printf("\n RESULT: \n");

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d ", G[i * N + j]);
		}
		printf("\n");
	}*/

	printf("\nTIME: \n");
	printf("%f ms\n", time);

	delete G;
	cudaFree(dev);
	return 0;
}