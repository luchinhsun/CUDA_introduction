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

void Init(){

	int i;
	for(i=0;i<N;i++){
		h_a[i] = i;
	}
}

void print(float *a){

	int i;
	for(i=0;i<N;i++){
		printf("%f ", a[i]);
	}
	printf("\n");

}

void Sent_to_device(){

	size_t size = N*sizeof(float);
	cudaError_t Error;

	Error = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	printf("CUDA error(copy h_a) = %s\n", cudaGetErrorString(Error));
}

void Sent_to_host(){

        size_t size = N*sizeof(float);
        cudaError_t Error;

        Error = cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost);
	printf("CUDA error(copy d_a) = %s\n", cudaGetErrorString(Error));
}

