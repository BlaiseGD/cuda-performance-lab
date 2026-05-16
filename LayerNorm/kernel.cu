
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#define BLOCK_SIZE 256

#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;


/*
typical transformer activaton shape[batch_size][sequence_length][hidden_dimension]
batch: how big of a chunk are we computing on
sequence_length: what length of data are we computing over e.g. for each sequence
e.g. batch_size sequences at once of size N (sequence length)
hidden_dimension: for each token in the sequence we operate on N-dimensional vector

Layer Norm operates across the hidden dimension so we compute mean, variance, and normalize 
for each token vector

Reason we do this: training will vary wildly bc of activation values if not normalized
this LayerNorm stabilizes this and makes training much more predictable and accurate

steps: calc mean, then variance, then normalize

ex: [1,2,3,4] -> mean = 2.5 var = 1.67 -> normalized arr of size 4 with same dimensions
*/

//just operate directly on the token vectors in this, no batch sizing
//input and output array is [sequence_length][hidden_dimension]
//NOTE: doing it in 1D representation bc CUDA flattens into 1D
void layernorm_cpu(float* inputArr, float *outputArr, int tokenNum, int tokenVecLen) {
	float epsilon = 1e-5f;

	for (int i = 0; i < tokenNum; i++) {
		float mean = 0.0f;
		float var = 0.0f;
		//getting mean
		for (int j = 0; j < tokenVecLen; j++) {
			mean += inputArr[i * tokenVecLen + j];
		}
		mean /= (float)tokenVecLen;
		//getting var
		for (int j = 0; j < tokenVecLen; j++) {
			float diff = inputArr[i * tokenVecLen + j] - mean;
			var += diff * diff;
		}
		//ensure float return type
		var /= (float)tokenVecLen;

		//getting output
		for (int j = 0; j < tokenVecLen; j++) {
			outputArr[i * tokenVecLen + j] = (inputArr[i * tokenVecLen + j] - mean) / sqrt(var + epsilon);
		}
	}
		
}

__global__ void naive_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float mean = 0.0f;
	float var = 0.0f;
	float epsilon = 1e-5f;

	//our index basically says for each token do X
	if (idx < tokenNum) {
		//getting mean
		for (int i = 0; i < tokenVecLen; i++) {
			mean += inputArr[idx * tokenVecLen + i];
		}
		mean /= (float)tokenVecLen;

		//getting var
		for (int i = 0; i < tokenVecLen; i++) {
			float diff = inputArr[idx * tokenVecLen + i] - mean;
			var += diff * diff;
		}
		var /= (float)tokenVecLen;

		//getting output
		//saying go through the length of the token at token[index][] (if it was 2D) and computer the normalization
		for (int i = 0; i < tokenVecLen; i++) {
			outputArr[idx * tokenVecLen + i] = (inputArr[idx * tokenVecLen + i] - mean) / sqrtf(var + epsilon);
		}
	}

}

//1 Block is one token, and 1 thread computes one hidden dimension element so grid, block shoudl be sequenceLen, hiddenSize
__global__ void block_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	int token = blockIdx.x;
	int hiddenThread = threadIdx.x;
	float epsilon = 1e-5f;
	float xi = (hiddenThread < tokenVecLen) ? inputArr[token * tokenVecLen + hiddenThread] : 0.0f;

	//declaring shared memory that is specified at runtime
	extern __shared__ float shared[];
	float* mean = shared;
	//starts at shared mem address + size of the block
	float* var = shared + blockDim.x;
	
	//load mean into shared
	if (hiddenThread < tokenVecLen) {
		mean[hiddenThread] = xi;
	}
	else mean[hiddenThread] = 0.0f;
	__syncthreads();

	//reduction for the mean (assumes blockDim.x is in powers of two)
	for (int i = blockDim.x >> 1; i > 0; i>>=1) {
		if (hiddenThread < i) mean[hiddenThread] += mean[hiddenThread + i];
		__syncthreads();
	}
	if (hiddenThread == 0) mean[0] /= (float)tokenVecLen;
	__syncthreads();

	//load into shared with values to add for reduction
	if (hiddenThread < tokenVecLen) {
		float diff = xi - mean[0];
		var[hiddenThread] = diff * diff;
	}
	else var[hiddenThread] = 0.0f;
	__syncthreads();

	//reduction for the var
	for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
		if (hiddenThread < i) var[hiddenThread] += var[hiddenThread + i];
		__syncthreads();
	}
	//get reduced var value
	if (hiddenThread == 0) var[0] /= (float)tokenVecLen;
	__syncthreads();

	if(token < tokenNum && hiddenThread < tokenVecLen) outputArr[token * tokenVecLen + hiddenThread] = (xi - mean[0]) / sqrtf(var[0] + epsilon);
}
	
__global__ void warp_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	float epsilon = 1e-5f;
}

__global__ void fused_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	float epsilon = 1e-5f;
}



int main() {
	//small for testing purposes
	//for profiling hiddensize = 256, 512, etc. SequenceLen = 1024, 2048, etc
	const int sequenceLen = 4;
	const int hiddenSize = 16;
	const int totalElements = sequenceLen * hiddenSize;

	float h_input[totalElements];
	float h_output_cpu[totalElements] = { 0 };
	float h_output_naive[totalElements] = { 0 };
	float h_output_block[totalElements] = { 0 };
	float* d_input, float* d_output_naive, float* d_output_block;

	//filling input with random floats from 0-9
	for (int i = 0; i < sequenceLen * hiddenSize; i++) {
		h_input[i] = (float)(rand() % 10);
	}

	dim3 grid(((sequenceLen * hiddenSize) +  BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);

	//allocate, copy, and set memory for device variables
	cudaMalloc(&d_input, totalElements * sizeof(float));
	cudaMalloc(&d_output_naive, totalElements * sizeof(float));
	cudaMalloc(&d_output_block, totalElements * sizeof(float));
	cudaMemcpy(d_input, h_input, totalElements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_output_naive, 0, totalElements * sizeof(float));
	cudaMemset(d_output_block, 0, totalElements * sizeof(float));
	
	auto start = chrono::steady_clock::now();

	//calling function and kernels
	layernorm_cpu(h_input, h_output_cpu, sequenceLen, hiddenSize);
	auto end = chrono::steady_clock::now();
	chrono::duration<double> cpuTime = end - start;
	//1 thread per token
	naive_layernorm_gpu << <sequenceLen, block >> > (d_input, d_output_naive, sequenceLen, hiddenSize);
	//1 block per token, 1 thread per hidden dimension element
	block_layernorm_gpu << <sequenceLen, hiddenSize, 2*hiddenSize*sizeof(float) >> > (d_input, d_output_block, sequenceLen, hiddenSize);

	//copy back to host
	cudaMemcpy(h_output_naive, d_output_naive, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_block, d_output_naive, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

	//COMPUTE ERROR & CHECKING OUTPUT
	//NOTE: get rid of all the printing for profiling
	float maxError = 0.0f;
	for (int i = 0; i < totalElements; i++) {
		cout << "CPU: " << i << h_output_cpu[i] << '\n';
	}
	cout << "\nCPU Runtime: " << cpuTime.count() << " ms\n\n";
	for (int i = 0; i < totalElements; i++) {
		cout << "GPU Naive: " << i << h_output_naive[i] << '\n';
		maxError = fmax(maxError, fabs(h_output_cpu[i] - h_output_naive[i]));
	}
	cout << "\nMax Error (Naive): " << maxError << '\n\n';
	maxError = 0.0f;
	for (int i = 0; i < totalElements; i++) {
		cout << "GPU Block & Shared: " << i << h_output_block[i] << '\n';
		maxError = fmax(maxError, fabs(h_output_cpu[i] - h_output_block[i]));
	}
	cout << "\nMax Error (Block): " << maxError;


	//free the device variables from DRAM
	cudaFree(d_input);
	cudaFree(d_output_naive);
	cudaFree(d_output_block);

	return 0;
}
