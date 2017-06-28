#include "head.h"

float *h_a;
float *h_b;

float *d_a;

void CPU_malloc(){

	size_t size = N*sizeof(float);

	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
}

void GPU_malloc(){

	size_t size = N*sizeof(float);

	cudaError_t Error;

	Error = cudaMalloc((void**)&d_a,size);
	printf("CUDA error(malloc d_a) = %s\n", cudaGetErrorString(Error));
}

void Free(){

	free(h_a);
	free(h_b);

	cudaFree(d_a);
}
