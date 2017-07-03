#include <stdio.h>
#include <malloc.h>

#define N 10000

float *A;
float *B;

int main(){

	int i,j;
	size_t size = N*N*sizeof(float);

	A = (float *)malloc(size);
	B = (float *)malloc(N*sizeof(float));

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
        int index;
	pFile = fopen("A.txt","w");
        // Save the matrix A
        for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                        index = i*N + j;
                        fprintf(pFile, "%g", A[index]);
                        if (j == (N-1)) {
                                fprintf(pFile, "\n");
                        } else {
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);

	pFile = fopen("B.txt","w");
        // Save the vector B
        for (i = 0; i < N; i++) {
        	fprintf(pFile, "%g\n", B[i]);
        }
        fclose(pFile);

	free(A);
	free(B);
	return 0;
}
