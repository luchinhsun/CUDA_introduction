#include "head.h"

float *h_a;
float *h_b;
float *h_c;

float *d_a;
float *d_b;
float *d_c;

void CPU_malloc(){

	size_t size = N*sizeof(float);

	h_a = (float *)malloc(N*size);
	h_b = (float *)malloc(size);
	h_c = (float *)malloc(size);
}

void GPU_malloc(){

	size_t size = N*sizeof(float);

	cudaError_t Error;

	Error = cudaMalloc((void**)&d_a,N*size);
	printf("CUDA error(malloc d_a) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_b,size);
        printf("CUDA error(malloc d_b) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_c,size);
        printf("CUDA error(malloc d_c) = %s\n", cudaGetErrorString(Error));
}

void Free(){

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void Init(){

	int i;
	for(i=0;i<N*N;i++){
		h_a[i] = i;
	}
	for(i=0;i<N;i++){
		h_b[i] = 2*i;
	}
}

void print(float *a){

	int i;
	for(i=0;i<N;i++){
		printf("%f ", a[i]);
	}
	printf("\n");

}

void print_matrix(float *a){

        int i, j;
        for(i=0;i<N;i++){
		for(j=0;j<N;j++){
                	printf("%f ", a[i*N+j]);
		}
        	printf("\n");
	}

}


void Sent_to_device(){

	size_t size = N*sizeof(float);
	cudaError_t Error;

	Error = cudaMemcpy(d_a, h_a, N*size, cudaMemcpyHostToDevice);
	printf("CUDA error(copy h_a) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
        printf("CUDA error(copy h_b) = %s\n", cudaGetErrorString(Error));
}

void Sent_to_host(){

        size_t size = N*sizeof(float);
        cudaError_t Error;

        Error = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	printf("CUDA error(copy d_c) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPU_product(float *x, float *y, float *z){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j;

	if(i<N){
		z[i] = 0.0;
		for(j=0;j<N;j++){
			z[i] += x[i*N+j]*y[j];
		}
	}
}

void product1(){
	int i, j;

	for(i=0;i<N;i++){
		h_c[i] = 0.0;
		for(j=0;j<N;j++){
                	h_c[i] += h_a[i*N+j]*h_b[j];
		}
        }
}

void product2(){
	int tpb = 256;
	int bpg = (N+tpb-1)/tpb;

	GPU_product<<<bpg, tpb>>>(d_a, d_b, d_c);
}

