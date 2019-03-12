#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "ped_agent.h"

#define THREADS_PER_BLOCK 1024
#define SIZE 1024
#define DIVIDE_AGENTS 8
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE
#define WEIGHTSUM 273
#define TILESIZE 32

__global__ void heatFade(int* heatmap) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	heatmap[index] = (int)round(heatmap[index] * 0.80);
}

__global__ void countAgents(int* heatmap, int* x_desired, int* y_desired)
{
	int	index = threadIdx.x + blockIdx.x * blockDim.x;
	int x = x_desired[index];
	int y = y_desired[index];

	if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
	{
		return;
	}

	atomicAdd(&heatmap[(y*SIZE) + x], 40);

}

__global__ void rgbLimit(int* heatmap) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (heatmap[index] >= 255)
	{
		heatmap[index] = 255;
	}
}

__global__ void scaleData(int* heatmap, int* scaled_heatmap) {
	int Y = blockIdx.x;
	int X = threadIdx.x;
	int index = Y*SIZE + X;
	int value = heatmap[index];
	
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (X * CELLSIZE) + 0] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (X * CELLSIZE) + 1] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (X * CELLSIZE) + 2] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (X * CELLSIZE) + 3] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (X * CELLSIZE) + 4] = value;

	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 1) + (X * CELLSIZE) + 0] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 1) + (X * CELLSIZE) + 1] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 1) + (X * CELLSIZE) + 2] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 1) + (X * CELLSIZE) + 3] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 1) + (X * CELLSIZE) + 4] = value;

	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 2) + (X * CELLSIZE) + 0] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 2) + (X * CELLSIZE) + 1] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 2) + (X * CELLSIZE) + 2] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 2) + (X * CELLSIZE) + 3] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 2) + (X * CELLSIZE) + 4] = value;

	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 3) + (X * CELLSIZE) + 0] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 3) + (X * CELLSIZE) + 1] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 3) + (X * CELLSIZE) + 2] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 3) + (X * CELLSIZE) + 3] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 3) + (X * CELLSIZE) + 4] = value;

	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 4) + (X * CELLSIZE) + 0] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 4) + (X * CELLSIZE) + 1] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 4) + (X * CELLSIZE) + 2] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 4) + (X * CELLSIZE) + 3] = value;
	scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * 4) + (X * CELLSIZE) + 4] = value;
	
}

__global__ void blurredFilter(int *d_scaled_heatmap, int *d_blurred_heatmap)
{
	//int const w[25] = {1, 4, 7, 4, 1,      4, 16, 26, 16, 4,       7, 26, 41, 26, 7,       4, 16, 26, 16, 4,        1, 4, 7, 4, 1};

	int X = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * 4 ;
	int Y = blockIdx.y * blockDim.y + threadIdx.y - blockIdx.y * 4 ;
	int index = Y * SCALED_SIZE + X;
	int offsetX = (X + blockIdx.x * 4 + 2) % TILESIZE;
	int offsetY = (Y + blockIdx.y * 4 + 2) % TILESIZE;
	int offsetIndex = offsetY * TILESIZE + offsetX;
	
	__shared__ int scaled_heatmap_shared[SIZE];
	scaled_heatmap_shared[offsetIndex] = d_scaled_heatmap[index];

	if (offsetX < 2 || offsetY < 2 || offsetX >= TILESIZE-2 || offsetY >= TILESIZE - 2)
	{
		return;
	}

	__syncthreads();

	int sum = 0;
	
	sum += 1 * scaled_heatmap_shared[offsetIndex -2*TILESIZE -2];
	sum += 4 * scaled_heatmap_shared[offsetIndex - 2 * TILESIZE - 1];
	sum += 7 * scaled_heatmap_shared[offsetIndex];
	sum += 4 * scaled_heatmap_shared[offsetIndex - 2 * TILESIZE + 1];
	sum += 1 * scaled_heatmap_shared[offsetIndex - 2 * TILESIZE + 2];

	sum += 4 * scaled_heatmap_shared[offsetIndex -1 * TILESIZE - 2];
	sum += 16 * scaled_heatmap_shared[offsetIndex - 1 * TILESIZE * 1 - 1];
	sum += 26 * scaled_heatmap_shared[offsetIndex - 1 * TILESIZE * 1];
	sum += 16 * scaled_heatmap_shared[offsetIndex - 1 * TILESIZE * 1 + 1];
	sum += 4 * scaled_heatmap_shared[offsetIndex - 1 * TILESIZE * 1 + 2];

	sum += 7 * scaled_heatmap_shared[offsetIndex - 2];
	sum += 26 * scaled_heatmap_shared[offsetIndex  - 1];
	sum += 41 * scaled_heatmap_shared[offsetIndex];
	sum += 26 * scaled_heatmap_shared[offsetIndex + 1];
	sum += 7 * scaled_heatmap_shared[offsetIndex + 2];

	sum += 4 * scaled_heatmap_shared[offsetIndex + TILESIZE - 2];
	sum += 16 * scaled_heatmap_shared[offsetIndex + TILESIZE - 1];
	sum += 26 * scaled_heatmap_shared[offsetIndex + TILESIZE];
	sum += 16 * scaled_heatmap_shared[offsetIndex + TILESIZE + 1];
	sum += 4 * scaled_heatmap_shared[offsetIndex + TILESIZE + 2];

	sum += 1 * scaled_heatmap_shared[offsetIndex + 2 * TILESIZE - 2];
	sum += 4 * scaled_heatmap_shared[offsetIndex + 2 * TILESIZE - 1];
	sum += 7 * scaled_heatmap_shared[offsetIndex + 2 * TILESIZE];
	sum += 4 * scaled_heatmap_shared[offsetIndex + 2 * TILESIZE + 1];
	sum += 1 * scaled_heatmap_shared[offsetIndex + 2 * TILESIZE + 2];

	int value = sum / WEIGHTSUM;
	d_blurred_heatmap[index] = 0x00FF0000 | value << 24;
	
}


void updateheatmapCUDA(int *d_blurred_heatmap, int *d_heatmap, int *d_scaled_heatmap, int* heatmap, int* scaled_heatmap, int* blurred_heatmap, int *xDesired, int *yDesired, int number_of_agents) {
	int size = SIZE * SIZE * sizeof(int);

	heatFade<<<SIZE, THREADS_PER_BLOCK>>>(d_heatmap);
	
	//#############################

	int size2 = number_of_agents * sizeof(int);
	int *d_xDesired;
	int *d_yDesired;

	
	cudaMalloc(&d_xDesired, size2);
	cudaMalloc(&d_yDesired, size2);

	cudaMemcpy(d_xDesired, xDesired, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_yDesired, yDesired, size2, cudaMemcpyHostToDevice);
	countAgents <<<DIVIDE_AGENTS, number_of_agents/ DIVIDE_AGENTS >>> (d_heatmap, d_xDesired, d_yDesired);
	
	cudaFree(d_xDesired);
	cudaFree(d_yDesired);
	//############################# 
	
	rgbLimit<<<SIZE, THREADS_PER_BLOCK >>>(d_heatmap);

	//#############################
	
	
	int size3 = SCALED_SIZE * SCALED_SIZE * sizeof(int);
	scaleData<<<SIZE, THREADS_PER_BLOCK>>>(d_heatmap, d_scaled_heatmap);

	//#############################

	
	dim3 grid(182, 182, 1);
	dim3 block(32, 32, 1);
	blurredFilter<<<grid, block>>>(d_scaled_heatmap, d_blurred_heatmap);
	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, size3, cudaMemcpyDeviceToHost);
}