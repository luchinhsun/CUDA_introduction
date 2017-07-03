#include <stdio.h>
#include <malloc.h>

#define N 10000

float *A;
float *B;
float *temp;
float *x_cu;
float *x_CG;

int main(){

	int i,j;
	size_t size = N*N*sizeof(float);

	A = (float *)malloc(size);
	size = N*sizeof(float);
	temp = (float *)malloc(size);
	B = (float *)malloc(size);
	x_cu = (float *)malloc(size);
	x_CG = (float *)malloc(size);

	for(i=0;i<N;i++){
		A[i*N+i] = -2;
		B[i] = 0;
	}
	B[0] = -1;
	for(i=0;i<N-1;i++){
                A[i*N+i+1] = 1;
		A[(i+1)*N+i] = 1;
        }

	FILE *pFile;
	pFile = fopen("cusparse/x_cu.txt","r");
        for (i = 0; i < N; i++) {
                x_cu[i] = 0.0;
                fscanf(pFile, "%g", &x_cu[i]);
        }
        fclose(pFile);
	pFile = fopen("CG/x_CG.txt","r");
        for (i = 0; i < N; i++) {
                x_CG[i] = 0.0;
                fscanf(pFile, "%g", &x_CG[i]);
        }
        fclose(pFile);

	float err;
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			temp[i] = 0.0;
			temp[i] += A[i*N+j]*B[j];
		}
	}

	for(i=0;i<N;i++){
		if(temp[i]>x_cu[i])	err = temp[i];
		else	err = x_cu[i];
	}

	printf("cu err = %f\n",err);

	for(i=0;i<N;i++){
                if(temp[i]>x_CG[i])     err = temp[i];
                else    err = x_CG[i];
        }

	printf("CG err = %f\n",err);

	free(A);
	free(B);
	free(x_cu);
	free(x_CG);
	return 0;
}
