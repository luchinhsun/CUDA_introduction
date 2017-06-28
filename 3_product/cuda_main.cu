#include "head.h"

float *h_a;
float *h_b;
float *h_c;

float *d_a;
float *d_b;
float *d_c;

void CPU_malloc(){

	size_t size = N*sizeof(float);

	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
	h_c = (float *)malloc(size);
}

void GPU_malloc(){

	size_t size = N*sizeof(float);

	cudaError_t Error;

	Error = cudaMalloc((void**)&d_a,size);
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
	for(i=0;i<N;i++){
		h_a[i] = i;
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

void Sent_to_device(){

	size_t size = N*sizeof(float);
	cudaError_t Error;

	Error = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
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

__global__ void GPU_product(float *d_a, float *d_b, float *d_c){
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if(i<N)	d_c[i] = d_a[i]*sqrt(d_b[i]);
}

__global__ void GPU_product_serial(float *d_a, float *d_b, float *d_c){
        int i = blockDim.x*blockIdx.x + threadIdx.x;

        for(i=0;i<N;i++){
		d_c[i] = d_a[i]*sqrt(d_b[i]);
	}
}

void product1(){
	int i;

	for(i=0;i<N;i++){
                h_c[i] = h_a[i]*sqrt(h_b[i]);
        }
}

void product2(){
        int bpg = 1;
        int tpb = 1;

        GPU_product_serial<<<bpg, tpb>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();
}

void product3(){
	int bpg = 256;
	int tpb = (N+bpg-1)/bpg;

	GPU_product<<<bpg, tpb>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();
}

