
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#define BLOCK_SIZE 256
#define WARP_SIZE 32

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
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
//warp reduce the mean and var and then normalize to outputArr
__global__ void warp_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	const int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
	//declaring shared warpSum
	__shared__ float s_sum[WARP_NUM];
	int token = blockIdx.x;
	int hiddenThread = threadIdx.x;
	float xi = (hiddenThread < tokenVecLen) ? inputArr[token * tokenVecLen + hiddenThread] : 0.0f;
	//making sure hidden elements of same token are what's being summed
	float sum = xi;
	float mean = 0.0f;
	float var = 0.0f;
	float epsilon = 1e-5f;
	//gets what warp we're in
	int warpID = hiddenThread / WARP_SIZE;
	//gets position of thread in warp
	int laneID = hiddenThread % WARP_SIZE;

	//MEAN REDUCTION
	//get sum for each warp
	#pragma unroll
	for(int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
		sum += __shfl_down_sync(0xffffffff, sum, i);
	}
	if (laneID == 0) s_sum[warpID] = sum;
	__syncthreads();
	//sum for each warp acquired, reduce across the WarpSums stored in s_sum
	if (warpID == 0) {

		if (laneID < WARP_NUM) sum = s_sum[laneID];
		else sum = 0.0f;
			#pragma unroll
			for (int i = WARP_NUM >> 1; i > 0; i >>= 1) {
				sum += __shfl_down_sync(0xffffffff, sum, i);
			}
			if(laneID == 0) s_sum[0] = sum;
	}
	__syncthreads();
	mean = s_sum[0] / (float)tokenVecLen;

	//VAR REDUCTION
	float diff = xi - mean;
	var = diff * diff;
	#pragma unroll
	for (int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
		 var += __shfl_down_sync(0xffffffff, var, i);
	}
	if (laneID == 0) s_sum[warpID] = var;
	__syncthreads();
	//sum for each warp acquired, reduce across the WarpSums stored in s_sum
	//only lanes 0 - (WARP_NUM-1) are active in this
	if (warpID == 0) {

		if (laneID < WARP_NUM) var = s_sum[laneID];
		else var = 0.0f;
		#pragma unroll
		for (int i = WARP_NUM >> 1; i > 0; i >>= 1) {
			var += __shfl_down_sync(0xffffffff, var, i);
		}
		if(laneID == 0) s_sum[0] = var;
	}
	__syncthreads();
	var = s_sum[0] / (float)tokenVecLen;


	if (token < tokenNum && hiddenThread < tokenVecLen) outputArr[token * tokenVecLen + hiddenThread] = (xi - mean) / sqrtf(var + epsilon);
}

__global__ void fused_layernorm_gpu(float* inputArr, float* outputArr, int tokenNum, int tokenVecLen) {
	const int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
	//declaring shared warpSum
	__shared__ float sdata[WARP_NUM];
	int token = blockIdx.x;
	int hiddenThread = threadIdx.x;
	float xi = (hiddenThread < tokenVecLen) ? inputArr[token * tokenVecLen + hiddenThread] : 0.0f;
	//making sure hidden elements of same token are what's being summed
	float sum = xi;
	float mean = 0.0f;
	float var = 0.0f;
	float epsilon = 1e-5f;
	//gets what warp we're in
	int warpID = hiddenThread / WARP_SIZE;
	//gets position of thread in warp
	int laneID = hiddenThread % WARP_SIZE;

	//MEAN REDUCTION
	//get sum for each warp
	#pragma unroll
	for (int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
		sum += __shfl_down_sync(0xffffffff, sum, i);
	}
	//sdata has each reduced partial sum from each warp now
	if (laneID == 0) sdata[warpID] = sum;
	__syncthreads();

	//sum for each warp acquired, reduce across the WarpSums stored in s_sum
	//SECOND REDUCTION PASS
	
	#pragma unroll
	for (int i = WARP_NUM >> 1; i > 0; i >>= 1) {
		if (hiddenThread < i) sdata[hiddenThread] += sdata[hiddenThread + i];
		__syncthreads();
	}
	__syncthreads();
	mean = sdata[0] / (float)tokenVecLen;
	
	//VAR REDUCTION
	float diff = xi - mean;
	var = diff * diff;
	#pragma unroll
	for (int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
		var += __shfl_down_sync(0xffffffff, var, i);
	}
	//sdata has each reduced partial variance from each warp now
	if (laneID == 0) sdata[warpID] = var;
	__syncthreads();
	//SECOND REDUCTION PASS
	#pragma unroll
	for (int i = WARP_NUM >> 1; i > 0; i >>= 1) {
		if(hiddenThread < i) sdata[hiddenThread] += sdata[hiddenThread + i];
		__syncthreads();
	}
	__syncthreads();
	var = sdata[0] / (float)tokenVecLen;
	if (token < tokenNum && hiddenThread < tokenVecLen) outputArr[token * tokenVecLen + hiddenThread] = (xi - mean) / sqrtf(var + epsilon);
}



int main() {
	//small for testing purposes
	//for profiling hiddensize = 256, 512, etc. SequenceLen = 1024, 2048, etc
	const int sequenceLen = 1024;
	const int hiddenSize = BLOCK_SIZE;
	const int totalElements = sequenceLen * hiddenSize;
	//float totalNaive = 0.0f;
	//float totalBlock = 0.0f;

	vector<float> h_input(totalElements);
	vector<float> h_output_cpu(totalElements, 0.0f);
	vector<float> h_output_naive(totalElements, 0.0f);
	vector<float> h_output_block(totalElements, 0.0f);
	vector<float> h_output_warp(totalElements, 0.0f);
	vector<float> h_output_fused(totalElements, 0.0f);

	float* d_input, *d_output_naive, *d_output_block, *d_output_warp, *d_output_fused;

	//filling input with random floats from 0-9
	for (int i = 0; i < sequenceLen * hiddenSize; i++) {
		h_input[i] = ((float)rand() / RAND_MAX) * 1000.0f - 500.0f;
	}

	dim3 block(BLOCK_SIZE);

	//allocate, copy, and set memory for device variables
	cudaMalloc(&d_input, totalElements * sizeof(float));
	cudaMalloc(&d_output_naive, totalElements * sizeof(float));
	cudaMalloc(&d_output_block, totalElements * sizeof(float));
	cudaMalloc(&d_output_warp, totalElements * sizeof(float));
	cudaMalloc(&d_output_fused, totalElements * sizeof(float));
	cudaMemcpy(d_input, h_input.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_output_naive, 0, totalElements * sizeof(float));
	cudaMemset(d_output_block, 0, totalElements * sizeof(float));
	cudaMemset(d_output_warp, 0, totalElements * sizeof(float));
	cudaMemset(d_output_fused, 0, totalElements * sizeof(float));

	layernorm_cpu(h_input.data(), h_output_cpu.data(), sequenceLen, hiddenSize);
	//warmup
	for (int i = 0; i < 10; i++) {
		naive_layernorm_gpu << <sequenceLen, block >> > (d_input, d_output_naive, sequenceLen, hiddenSize);
		block_layernorm_gpu << <sequenceLen, hiddenSize, 2 * hiddenSize * sizeof(float) >> > (d_input, d_output_block, sequenceLen, hiddenSize);
		warp_layernorm_gpu << <sequenceLen, hiddenSize >> > (d_input, d_output_warp, sequenceLen, hiddenSize);
		fused_layernorm_gpu << <sequenceLen, hiddenSize >> > (d_input, d_output_fused, sequenceLen, hiddenSize);

	}
	cudaDeviceSynchronize();

	for (int i = 0; i < 20; i++) {
		naive_layernorm_gpu << <sequenceLen, block >> > (d_input, d_output_naive, sequenceLen, hiddenSize);
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 20; i++) {
		block_layernorm_gpu << <sequenceLen, hiddenSize, 2 * hiddenSize * sizeof(float) >> > (d_input, d_output_block, sequenceLen, hiddenSize);
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 20; i++) {
		warp_layernorm_gpu << <sequenceLen, hiddenSize>> > (d_input, d_output_warp, sequenceLen, hiddenSize);
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 20; i++) {
		fused_layernorm_gpu << <sequenceLen, hiddenSize >> > (d_input, d_output_fused, sequenceLen, hiddenSize);
	}
	cudaDeviceSynchronize();

	/*
	TIMING CODE
	auto startCPU = chrono::steady_clock::now();

	//calling function and kernels
	layernorm_cpu(h_input.data(), h_output_cpu.data(), sequenceLen, hiddenSize);
	auto stopCPU = chrono::steady_clock::now();
	chrono::duration<double> cpuTime = stopCPU - startCPU;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//warmup
	cudaDeviceSynchronize();
	for (int i = 0; i < 10; i++) naive_layernorm_gpu << <sequenceLen, block >> > (d_input, d_output_naive, sequenceLen, hiddenSize);
	cudaDeviceSynchronize();
	//avg time
	for (int i = 0; i < 100; i++) {

		cudaEventRecord(start);
		//1 thread per token
		naive_layernorm_gpu << <sequenceLen, block >> > (d_input, d_output_naive, sequenceLen, hiddenSize);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		totalNaive += ms;
	}
	cudaDeviceSynchronize();
	//1 block per token, 1 thread per hidden dimension element
	//warmup
	for (int i = 0; i < 10; i++) block_layernorm_gpu << <sequenceLen, hiddenSize, 2 * hiddenSize * sizeof(float) >> > (d_input, d_output_block, sequenceLen, hiddenSize);
	cudaDeviceSynchronize();
	//avg time
	for (int i = 0; i < 100; i++) {

		cudaEventRecord(start);
		//1 thread per token
		block_layernorm_gpu << <sequenceLen, hiddenSize, 2 * hiddenSize * sizeof(float) >> > (d_input, d_output_block, sequenceLen, hiddenSize);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		totalBlock += ms;
	}
	std::cout << "CPU: " << cpuTime.count() << '\n';
	std::cout << "Naive avg: " << totalNaive / 100 << " ms\n";
	std::cout << "Block avg: " << totalBlock / 100 << " ms\n";
	*/

	//copy back to host
	cudaMemcpy(h_output_naive.data(), d_output_naive, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_block.data(), d_output_block, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_warp.data(), d_output_warp, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_fused.data(), d_output_fused, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

	/*
	//COMPUTE ERROR & CHECKING OUTPUT
	//NOTE: get rid of all the printing for profiling
	float maxError = 0.0f;
	for (int i = 0; i < totalElements; i++) {
		//cout << "CPU element at " << i << ": " << h_output_cpu[i] << '\n';
	}
	cout << "\nCPU Runtime: " << cpuTime.count() << " ms\n\n";
	for (int i = 0; i < totalElements; i++) {
		//cout << "GPU element at " << i << ": " << h_output_naive[i] << '\n';
		maxError = fmax(maxError, fabs(h_output_cpu[i] - h_output_naive[i]));
	}
	cout << "\nMax Error (Naive): " << maxError << "\n\n";
	maxError = 0.0f;
	for (int i = 0; i < totalElements; i++) {
		//cout << "GPU Block & Shared at " << i << ": " << h_output_block[i] << '\n';
		maxError = fmax(maxError, fabs(h_output_cpu[i] - h_output_block[i]));
	}
	cout << "\nMax Error (Block): " << maxError;
	*/
	

	//free the device variables from DRAM
	cudaFree(d_input);
	cudaFree(d_output_naive);
	cudaFree(d_output_block);
	cudaFree(d_output_warp);
	cudaFree(d_output_fused);

	return 0;
}
