#include "head.h"

// Define our variables of interest
float *h_A;     // A matrix
float *h_B;     // B vector
float *h_x;     // x (solution) vector
float *h_R;     // Residual vector
float *h_P;
float *h_AP;
float *h_scalars;  // Scalars

// Define GPU Variables
float *d_A;
float *d_B;
float *d_x;
float *d_R;
float *d_P;
float *d_AP;
float *d_scalars;
float *d_temp;

clock_t start_cu;
clock_t end_cu;
float time_used_cu;

float gpu_timer;

void Allocate_Memory() {

        size_t size;

        // Our N*N variable (A)
        size = N*N*sizeof(float);
        h_A = (float*)malloc(size);

        // Our 1D (N) variables
        size = N*sizeof(float);
        h_B = (float*)malloc(size);
        h_x = (float*)malloc(size);
        h_R = (float*)malloc(size);
        h_P = (float*)malloc(size);
        h_AP = (float*)malloc(size);

        // Small array holding scalars
        size = 5*sizeof(float);
        h_scalars = (float*)malloc(size);

	// GPU variables
	size = N*N*sizeof(float);
	cudaError_t Error;
	d_A = (float*)malloc(size);
	Error = cudaMalloc((void**)&d_A, size);
	if (Error != cudaSuccess)
	printf("CUDA error(malloc d_A) = %s\n",cudaGetErrorString(Error));

	size = N*sizeof(float);
	d_B = (float*)malloc(size);
        Error = cudaMalloc((void**)&d_B, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_B) = %s\n",cudaGetErrorString(Error));
	d_x = (float*)malloc(size);
        Error = cudaMalloc((void**)&d_x, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_x) = %s\n",cudaGetErrorString(Error));
	d_R = (float*)malloc(size);
        Error = cudaMalloc((void**)&d_R, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_R) = %s\n",cudaGetErrorString(Error));
	d_P =  (float*)malloc(size);
        Error = cudaMalloc((void**)&d_P, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_P) = %s\n",cudaGetErrorString(Error));
	d_AP =  (float*)malloc(size);
        Error = cudaMalloc((void**)&d_AP, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_AP) = %s\n",cudaGetErrorString(Error));

	size = 5*sizeof(float);
	d_scalars = (float*)malloc(size);
	Error = cudaMalloc((void**)&d_scalars, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_scalars) = %s\n",cudaGetErrorString(Error));

	size = (N+256-1)/256*sizeof(float);
        d_temp = (float*)malloc(size);
        Error = cudaMalloc((void**)&d_temp, size);
        if (Error != cudaSuccess)
	printf("CUDA error(malloc d_temp) = %s\n", cudaGetErrorString(Error));
}


void Free_Memory() {

        // Now we better free the memory on the GPU
        free(h_A);
        free(h_B);
        free(h_x);
        free(h_R);
        free(h_P);
        free(h_AP);
        free(h_scalars);

	// Free GPU memory
	cudaFree(d_A);cudaFree(d_B);cudaFree(d_x);
	cudaFree(d_R);cudaFree(d_P);cudaFree(d_AP);
	cudaFree(d_scalars);cudaFree(d_temp);
}
/*
void Init() {

        int i;
        int index;
        int x_cell, y_cell;
        // Set up our A and B matrix

        // Full 2D Heat Transfer test
        for (i = 0; i < N; i++) {

                y_cell = (int)(i/NX);
                x_cell = i - y_cell*NX;

                // Find the diagonal
                index = i*N + i;
                h_A[index] = 2.0*PHI_X + 2.0*PHI_Y;

                // Modify A for left and right
                if (x_cell > 0) {
                        h_A[index-1] = -PHI_X;
                }

                if (x_cell < (NX-1)) {
                        h_A[index+1] = -PHI_X;
                }

                // Modifiy for up and down
                if (y_cell > 0) {
                        h_A[index-NX] = -PHI_Y;
                }
                if (y_cell < (NY-1)) {
                        h_A[index+NX] = -PHI_Y;
                }


                // Set B now
                if (y_cell == 0) {
                        h_B[i] = PHI_Y;
                } else {
                        h_B[i] = 0.0;
                }

                // And our initial guess x
                h_x[i] = 0.0;
        }


}
*/
void Read_File(){
	start_cu = clock();

	FILE *pFile;
        int i;
	// Read the matrix B
        pFile = fopen("B.txt","r");
        for (i = 0; i < N; i++) {
		h_B[i] = 0.0;
		fscanf(pFile, "%g", &h_B[i]);
	}
	fclose(pFile);

	// Rean the matrix A
	pFile = fopen("A.txt","r");
	for (i = 0; i < N*N; i++) {
                h_A[i] = 0.0;
                fscanf(pFile, "%g,", &h_A[i]);
        }
	fclose(pFile);

	// And our initial guess x
        for (i = 0; i < N; i++) {
		h_x[i] = 0.0;
	}

}


void Save_Result() {

        FILE *pFile;
        int i;//,j;
        //int index;
	/*
        pFile = fopen("A_ans.txt","w");
        // Save the matrix A
        for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                        index = i*N + j;
                        fprintf(pFile, "%g", h_A[index]);
                        if (j == (N-1)) {
                                fprintf(pFile, "\n");
                        } else {
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);

        pFile = fopen("B_ans.txt","w");
        // Save the vector B
        for (i = 0; i < N; i++) {
                fprintf(pFile, "%g\n", h_B[i]);
        }
        fclose(pFile);
	*/

        pFile = fopen("X_CG.txt","w");
        // Save the vector X
        for (i = 0; i < N; i++) {
                fprintf(pFile, "%g\n", h_x[i]);
        }
        fclose(pFile);

        pFile = fopen("R_CG.txt","w");
        // Save the vector R
        for (i = 0; i < N; i++) {
                fprintf(pFile, "%g\n", h_R[i]);
        }
        fclose(pFile);


}

void Send_To_Device(){
	//cutCreateTimer(&gpu_timer);
	//cudaThreadSynchronize();
	//cutStartTimer(gpu_timer);

	cudaError_t Error;
	size_t size = N*N*sizeof(float);
	Error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (Error != cudaSuccess)
	printf("CUDA error(copy h_A->d_A) = %s\n",cudaGetErrorString(Error));
	size = N*sizeof(float);
	Error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if (Error != cudaSuccess)
	printf("CUDA error(copy h_B->d_B) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
	printf("CUDA error(copy h_x->d_x) = %s\n",cudaGetErrorString(Error));
}

__global__ void GPU_Setup(float *x, float *y, float *z, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<n){
 		y[i] = x[i];
		z[i] = x[i];
	}
}

void SetUp_CG_GPU(){
	int tpb = 256;
	int bpg = (N+tpb-1)/tpb;

	GPU_Setup<<<bpg, tpb>>>(d_B, d_R, d_P, N);
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
/*
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
*/
void DotFunction1(){
	int tpb = 256;
	int bpg = (N+tpb-1)/tpb;

	GPU_DotProduct<<<bpg, tpb>>>(d_P, d_AP, d_temp, N);
	GPU_DotProduct_Sum<<<1, 1>>>(d_temp, d_scalars, 2, bpg);
	//GPU_DotProduct_Sum_reduction<<<1, 4>>>(d_temp, d_scalars, 2);
	GPU_DotProduct<<<bpg, tpb>>>(d_R, d_R, d_temp, N);
	GPU_DotProduct_Sum<<<1, 1>>>(d_temp, d_scalars, 3, bpg);
	//GPU_DotProduct_Sum_reduction<<<1, 4>>>(d_temp, d_scalars, 3);
}

void DotFunction2(){
        int tpb = 256;
        int bpg = (N+tpb-1)/tpb;

        GPU_DotProduct<<<bpg, tpb>>>(d_R, d_R, d_temp, N);
	GPU_DotProduct_Sum<<<1, 1>>>(d_temp, d_scalars, 4, bpg);
	//GPU_DotProduct_Sum_reduction<<<1, 4>>>(d_temp, d_scalars, 4);
}


__global__ void GPU_ScalarsCal_alpha(float *x){
	x[0] = x[3]/x[2];
}

void Cal_alpha(){
	int tpb = 1;
        int bpg = 1;

	GPU_ScalarsCal_alpha<<<bpg, tpb>>>(d_scalars);
}

__global__ void GPU_ScalarsCal_beta(float *x){
        x[1] = x[4]/x[3];
}

void Cal_beta(){
        int tpb = 1;
        int bpg = 1;

        GPU_ScalarsCal_beta<<<bpg, tpb>>>(d_scalars);
}

__global__ void GPU_MatrixVectorProduct(float *x, float *y, float *z, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;

	if(i<n){
		z[i] = 0.0;
		for(j=0;j<n;j++){
			z[i] = z[i] + x[i*n+j]*y[j];
		}
		
	}
} 

void MatrixVectorFunction(){
	int tpb = 256;
	int bpg = (N+tpb-1)/tpb;

	GPU_MatrixVectorProduct<<<bpg, tpb>>>(d_A, d_P, d_AP, N);
}

__global__ void GPU_Update_x(float *x, float *y, float *z, int n){
	int j = threadIdx.x;
        int i = blockDim.x * blockIdx.x + j;

        __shared__ float sx[256];
        __shared__ float sy[256];

        sx[j] = x[i];
        sy[j] = y[i];

        if(i<n) sx[j] = sx[j] + z[0] * sy[j];

        x[i] = sx[j];

	//if(i<n) x[i] = x[i] + z[0] * y[i];
}

__global__ void GPU_Update_R(float *x, float *y, float *z, int n){
        int j = threadIdx.x;
        int i = blockDim.x * blockIdx.x + j;

        __shared__ float sx[256];
        __shared__ float sy[256];

        sx[j] = x[i];
        sy[j] = y[i];

        if(i<n) sx[j] = sx[j] - z[0] * sy[j];

        x[i] = sx[j];

	//if(i<n) x[i] = x[i] - z[0] * y[i];
}

void Update_xandR(){
	int tpb = 256;
        int bpg = (N+tpb-1)/tpb;

	GPU_Update_x<<<bpg, tpb>>>(d_x, d_P, d_scalars, N);
	GPU_Update_R<<<bpg, tpb>>>(d_R, d_AP, d_scalars, N);
}

__global__ void GPU_Update_P(float *x, float *y, float *z, int n){
	int j = threadIdx.x;
        int i = blockDim.x * blockIdx.x + j;

        __shared__ float sx[256];
        __shared__ float sy[256];

        sx[j] = x[i];
        sy[j] = y[i];

        if(i<n) sx[j] = sy[j] + z[1] * sx[j];

        x[i] = sx[j];

	//if(i<n) x[i] = y[i] + z[1] * x[i];
}

void Update_P(){
	int tpb = 256;
        int bpg = (N+tpb-1)/tpb;

        GPU_Update_P<<<bpg, tpb>>>(d_P, d_R, d_scalars, N);
}

void Send_For_Print(){
	cudaError_t Error;
        size_t size = 5*sizeof(float);
        Error = cudaMemcpy(h_scalars, d_scalars, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_scalars->h_scalars) = %s\n",cudaGetErrorString(Error));
}

void Send_To_Host(){
	cudaError_t Error;
        size_t size = N*sizeof(float);
	/*
        Error = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_B->h_B) = %s\n",cudaGetErrorString(Error));
	*/
	Error = cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_x->h_x) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_R->h_R) = %s\n",cudaGetErrorString(Error));
	/*
	size = N*N*sizeof(float);
	Error = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_A->h_A) = %s\n",cudaGetErrorString(Error));
	*/

	end_cu = clock();
        time_used_cu = (float)(end_cu - start_cu)/ CLOCKS_PER_SEC;
        printf("\ntime in cu = %f s\n",time_used_cu);
	//cudaDeviceSynchronize();
}
