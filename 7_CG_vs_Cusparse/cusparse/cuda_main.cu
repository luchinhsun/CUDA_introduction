#include "head.h"

//variable for cusparse
cusparseStatus_t status;
cusparseHandle_t handle=0;
cusparseMatDescr_t descr=0;
cusparseMatDescr_t descrL=0;
cusparseMatDescr_t descrU=0;
cusparseSolveAnalysisInfo_t infoA=0;
cusparseSolveAnalysisInfo_t info_u=0;
int *cooRowIndexHostPtr;
int * cooColIndexHostPtr;
float * cooValHostPtr;
int *cooRowIndex;
int * cooColIndex;
float * cooVal;
float * cooValLU;
float * yHostPtr;
float * y;
float * xHostPtr;
float * x;
float * temp;
int * csrRowPtr;

float dzero =0.0;
float done =1.0;
float dtwo =2.0;
float dthree=3.0;
float dfive =5.0;

clock_t start_cu;
clock_t end_cu;
float time_used_cu;


void Allocate_Memory(){
	//cusparse
	size_t size = nnz*sizeof(int);
	cooRowIndexHostPtr = (int *) malloc(size);
	cooColIndexHostPtr = (int *) malloc(size);
	cooValHostPtr = (float *)malloc(nnz*sizeof(float));

	cooRowIndexHostPtr[0] = 0;cooColIndexHostPtr[0]=0;cooValHostPtr[0]=-2.0;
	cooRowIndexHostPtr[1] = 0;cooColIndexHostPtr[1]=1;cooValHostPtr[1]=1.0;

	cooRowIndexHostPtr[2] = 1;cooColIndexHostPtr[2]=0;cooValHostPtr[2]=1.0;
	cooRowIndexHostPtr[3] = 1;cooColIndexHostPtr[3]=1;cooValHostPtr[3]=-2.0;
	cooRowIndexHostPtr[4] = 1;cooColIndexHostPtr[4]=2;cooValHostPtr[4]=1.0;
	int i;
	for(i=5;i<(nnz-3);i=i+3){
		cooRowIndexHostPtr[i] = cooRowIndexHostPtr[i-3]+1;	
		cooColIndexHostPtr[i] = cooColIndexHostPtr[i-3]+1;
		cooRowIndexHostPtr[i+1] = cooRowIndexHostPtr[i];	
		cooColIndexHostPtr[i+1] = cooColIndexHostPtr[i]+1;
		cooRowIndexHostPtr[i+2] = cooRowIndexHostPtr[i+1];	
		cooColIndexHostPtr[i+2] = cooColIndexHostPtr[i+1]+1;
		cooValHostPtr[i]=1.0;
		cooValHostPtr[i+1]=-2.0;
		cooValHostPtr[i+2]=1.0;
	}
	cooRowIndexHostPtr[nnz-2] = n-1;cooColIndexHostPtr[nnz-2]=n-2;cooValHostPtr[nnz-2]=1.0;
        cooRowIndexHostPtr[nnz-1] = n-1;cooColIndexHostPtr[nnz-1]=n-1;cooValHostPtr[nnz-1]=-2.0;

	yHostPtr    = (float *)malloc(n*sizeof(float));
	for (i=1;i<n;i++){
		yHostPtr[i] = 0.0;
	}
	yHostPtr[0] = -1.0;

	xHostPtr    = (float *)malloc(n*sizeof(float));

	cudaError_t Error;

	Error = cudaMalloc((void**)&cooRowIndex, size);
	printf("CUDA error(malloc RowIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooColIndex, size);
	printf("CUDA error(malloc ColIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooVal, nnz*sizeof(float));
	printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&cooValLU, nnz*sizeof(float));
        printf("CUDA error(malloc Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&y, n*sizeof(float));
	printf("CUDA error(malloc y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&x, n*sizeof(float));
        printf("CUDA error(malloc x) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&temp, n*sizeof(float));
        printf("CUDA error(malloc temp) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(int));
        printf("CUDA error(malloc csrRowPtr) = %s\n",cudaGetErrorString(Error));

	status= cusparseCreate(&handle);
	status= cusparseCreateMatDescr(&descr);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

        status = cusparseCreateSolveAnalysisInfo(&infoA);
	status = cusparseCreateSolveAnalysisInfo(&info_u);

	status = cusparseCreateMatDescr(&descrL);
	cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

    	status = cusparseCreateMatDescr(&descrU);
    	cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    	cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
}

void Send_To_Device(){
	start_cu = clock();
	cudaError_t Error;
	size_t size = nnz*sizeof(int);
	Error = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, size, cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy RowIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(cooColIndex, cooColIndexHostPtr, size, cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy ColIndex) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(cooVal, cooValHostPtr, (size_t)(nnz*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy Val) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(y, yHostPtr, (size_t)(n*sizeof(float)), cudaMemcpyHostToDevice);
	printf("CUDA error(memcpy y) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(x, xHostPtr, (size_t)(n*sizeof(float)), cudaMemcpyHostToDevice);
        printf("CUDA error(memcpy x) = %s\n",cudaGetErrorString(Error));
}

void Call_GPUFunction(){
	status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n, csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("shit");
	}

	status= cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descr,
						cooVal, csrRowPtr, cooColIndex, infoA);
	cudaMemcpy(cooValLU, cooVal, nnz*sizeof(float), cudaMemcpyDeviceToDevice);
	status = cusparseScsrilu0(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, descr,
						cooValLU, csrRowPtr, cooColIndex, infoA);
	status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descrU,
						cooVal, csrRowPtr, cooColIndex, info_u);
	status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, descrL,
                                              cooValLU, csrRowPtr, cooColIndex, infoA, y, temp);
	status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, descrU,
                                              cooValLU, csrRowPtr, cooColIndex, info_u, temp , x);

	if (status != CUSPARSE_STATUS_SUCCESS) {
                printf("shit");
        }

}


void Send_To_Host(){
	cudaError_t Error;
	Error = cudaMemcpy(yHostPtr, y, (size_t)(n*sizeof(float)), cudaMemcpyDeviceToHost);
	printf("CUDA error(memcpy y->yHostPtr) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(xHostPtr, x, (size_t)(n*sizeof(float)), cudaMemcpyDeviceToHost);
        printf("CUDA error(memcpy x->xHostPtr) = %s\n",cudaGetErrorString(Error));

	end_cu = clock();
        time_used_cu = (float)(end_cu - start_cu)/ CLOCKS_PER_SEC;
        printf("\ntime in cu = %f s\n",time_used_cu);
}

void Free_Memory(){

	status = cusparseDestroyMatDescr(descr); descr = 0;
	status = cusparseDestroy(handle); handle = 0;
	status = cusparseDestroyMatDescr(descrL); descrL = 0;
	status = cusparseDestroyMatDescr(descrU); descrU = 0;


	if (yHostPtr) free(yHostPtr);
	if (xHostPtr) free(xHostPtr);
        if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);
        if (cooColIndexHostPtr) free(cooColIndexHostPtr);
        if (cooValHostPtr) free(cooValHostPtr);
        if (y) cudaFree(y);
	if (x) cudaFree(x);
	if (temp) cudaFree(temp);
        if (csrRowPtr) cudaFree(csrRowPtr);
        if (cooRowIndex) cudaFree(cooRowIndex);
        if (cooColIndex) cudaFree(cooColIndex);
        if (cooVal) cudaFree(cooVal);
	if (cooValLU) cudaFree(cooValLU);
        if (descr) cusparseDestroyMatDescr(descr);
        if (handle) cusparseDestroy(handle);

	cusparseDestroySolveAnalysisInfo(infoA);
        cusparseDestroySolveAnalysisInfo(info_u);

}

void Save_Result() {

        FILE *pFile;
        int i;

	/*
        // Save the matrix
	pFile = fopen("B_ans.txt","w");
        // Save the vector B
        for (i = 0; i < n; i++) {
                fprintf(pFile, "%g\n", yHostPtr[i]);
        }
        fclose(pFile);
	*/

        pFile = fopen("X_ans.txt","w");
        // Save the vector X
        for (i = 0; i < n; i++) {
                fprintf(pFile, "%g\n", xHostPtr[i]);
        }
        fclose(pFile);
}
