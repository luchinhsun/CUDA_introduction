#include "head.h"

float *h_a;
float *h_b;
float *h_c;

float *d_a;
float *d_b;
float *d_c;

float *d_temp;

void CPU_malloc(){

	size_t size = N*sizeof(float);

	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
	h_c = (float *)malloc(sizeof(float));
}

void GPU_malloc(){

	size_t size = N*sizeof(float);

	cudaError_t Error;

	Error = cudaMalloc((void**)&d_a,size);
	printf("CUDA error(malloc d_a) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_b,size);
        printf("CUDA error(malloc d_b) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_c,sizeof(float));
        printf("CUDA error(malloc d_c) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_temp,(N+256-1)/256*sizeof(float));
        printf("CUDA error(malloc d_temp) = %s\n", cudaGetErrorString(Error));

}

void Free(){

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_temp);
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

        size_t size = sizeof(float);
        cudaError_t Error;

        Error = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	printf("CUDA error(copy d_c) = %s\n", cudaGetErrorString(Error));
}

void product(){
	int i;

	for(i=0;i<N;i++){
                h_c[0] += h_a[i]*h_b[i];
        }
}

__global__ void GPU_DotProduct(float *x, float *y, float *z, int n){
        int I = threadIdx.x;
        int i = blockDim.x * blockIdx.x + I;

        __shared__ float temp[256];
        temp[I] = 0.0;
        if(i<n){
                temp[I] = x[i] * y[i];
        }

        __syncthreads();

        for (int stride = blockDim.x/2; stride>0; stride = stride/2){
                if(I<stride){
                        temp[I] = temp[I] + temp[I+stride];
                }
                __syncthreads();
        }

        if(I==0){
                z[blockIdx.x] = temp[0];
        }

}

__global__ void GPU_DotProduct_Sum(float *x, float *y, int y_location, int n){

        int i;
        float sum = 0.0;
        for(i=0; i<n; i++){
                sum += x[i];
        }
        y[y_location] = sum;
}

__global__ void GPU_DotProduct_Sum_reduction(float *x, float *y, int y_location){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int stride = blockDim.x/2; stride>0; stride = stride/2){
		if(i<stride){
			x[i] = x[i] + x[i+stride];
		}
		__syncthreads();
	}

	if(i==0){
		y[y_location] = x[0];
	}
}

void DotFunction(){
        int tpb = 256;
        int bpg = (N+tpb-1)/tpb;

        GPU_DotProduct<<<bpg, tpb>>>(d_a, d_b, d_temp, N);
        //GPU_DotProduct_Sum<<<1, 1>>>(d_temp, d_c, 0, bpg);
	GPU_DotProduct_Sum_reduction<<<1, bpg>>>(d_temp, d_c, 0);
}
