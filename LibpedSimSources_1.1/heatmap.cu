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
	heatmap[(y*SIZE)+x] += 40;
}

__global__ void rgbLimit(int* heatmap) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (heatmap[index] >= 255)
	{
		heatmap[index] = 255;
	}
}

__global__ void scaleData(int* heatmap, int* scaled_heatmap) {
	int Y = blockIdx.x; // *blockDim.x;
	int X = threadIdx.x;
	int index = Y*SIZE + X;
	int value = heatmap[index];
	for (int cellY = 0; cellY < CELLSIZE; cellY++)
	{
		for (int cellX = 0; cellX < CELLSIZE; cellX++) 
		{
			scaled_heatmap[(Y * SCALED_SIZE * CELLSIZE) + (SCALED_SIZE * cellY) + (X * CELLSIZE) + cellX] = value;
		}
	}
}


/*
// Weights for blur filter
const int w[5][5] = {
	{ 1, 4, 7, 4, 1 },
{ 4, 16, 26, 16, 4 },
{ 7, 26, 41, 26, 7 },
{ 4, 16, 26, 16, 4 },
{ 1, 4, 7, 4, 1 }
};

#define WEIGHTSUM 273
// Apply gaussian blurfilter		       
for (int i = 2; i < SCALED_SIZE - 2; i++)
{
	for (int j = 2; j < SCALED_SIZE - 2; j++)
	{
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
			}
		}
		int value = sum / WEIGHTSUM;
		blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
	}
}
}
*/

__global__ void blurredFilter(int *d_w, int size_of_filter, int *d_scaled_heatmap, int *d_blurred_heatmap)
{
	//__shared__ int *blur_filter;  TODO

	int X = threadIdx.y * blockDim.x + threadIdx.x;
	int Y = gridDim.x * blockIdx.y + blockIdx.x;
	int index = Y * SCALED_SIZE + X;
	if (X < 2 || Y < 2 || X < SCALED_SIZE - 2 || Y < SCALED_SIZE - 2)
	{
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				sum += d_w[(k+2)*5 + (l+2)] * d_scaled_heatmap[index + SCALED_SIZE * k + l];
			}
		}
		int value = sum / WEIGHTSUM;
		d_blurred_heatmap[index] = 0x00FF0000 | value << 24;
	}

	/*
	__shared__ int temp[BLOCK_SIZE + 2 * CELLSIZE*CELLSIZE];

	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + CELLSIZE * CELLSIZE;

	temp[lindex] = in[gindex];
	if (threadIdx.x < CELLSIZE*CELLSIZE)
	{
		temp[lindex - CELLSIZE * CELLSIZE] = in[gindex - CELLSIZE * CELLSIZE];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}

	void __syncthreads();

	int result = 0;
	for (int OFFSET = -CELLSIZE * CELLSIZE; OFFSET <= CELLSIZE * CELLSIZE; OFFSET++)
	{
		result += temp[lindex + OFFSET];
	}
	out[gindex] = result;
	*/
}


void updateheatmapCUDA(int *d_w, int *d_blurred_heatmap, int *d_heatmap, int *d_scaled_heatmap, int* heatmap, int* scaled_heatmap, int* blurred_heatmap, int *xDesired, int *yDesired, int number_of_agents) {
	int size = SIZE * SIZE * sizeof(int);

	heatFade<<<SIZE, THREADS_PER_BLOCK>>>(d_heatmap);
	//cudaMemcpy(heatmap, d_heatmap, size, cudaMemcpyDeviceToHost);

	//cudaFree(d_heatmap);

	//#############################

	int size2 = number_of_agents * sizeof(int);
	int *d_xDesired;
	int *d_yDesired;

	//cudaMalloc(&d_heatmap, size);
	cudaMalloc(&d_xDesired, size2);
	cudaMalloc(&d_yDesired, size2);

	//cudaMemcpy(d_heatmap, heatmap, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xDesired, xDesired, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_yDesired, yDesired, size2, cudaMemcpyHostToDevice);
	countAgents <<<DIVIDE_AGENTS, number_of_agents/ DIVIDE_AGENTS >>> (d_heatmap, d_xDesired, d_yDesired);
	//cudaMemcpy(heatmap, d_heatmap, size, cudaMemcpyDeviceToHost);

	
	//cudaFree(d_heatmap);
	cudaFree(d_xDesired);
	cudaFree(d_yDesired);
	//############################# 
	
	//cudaMalloc(&d_heatmap, size);
	//cudaMemcpy(d_heatmap, heatmap, size, cudaMemcpyHostToDevice);
	rgbLimit<<<SIZE, THREADS_PER_BLOCK >>>(d_heatmap);
	//cudaMemcpy(heatmap, d_heatmap, size, cudaMemcpyDeviceToHost);

	//cudaFree(d_heatmap);
	//#############################
	
	
	int size3 = SCALED_SIZE * SCALED_SIZE * sizeof(int);
	//cudaMalloc(&d_heatmap, size);
	//cudaMemcpy(d_heatmap, heatmap, size, cudaMemcpyHostToDevice);
	scaleData<<<SIZE, THREADS_PER_BLOCK>>>(d_heatmap, d_scaled_heatmap);
	
	//#############################

	blurredFilter<<<(1024, 5), (1024, 5)>>>(d_w, 25, d_scaled_heatmap, d_blurred_heatmap);
	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, size3, cudaMemcpyDeviceToHost);

}
/*

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int cuda_test()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		//fprintf(stderr, "Cuda launch succeeded! \n");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
*/