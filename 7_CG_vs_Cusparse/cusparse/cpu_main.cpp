#include "head.h"

extern float * yHostPtr;
extern float * xHostPtr;

int main()
{
	int i;
        struct timeb start, end;
        int diff;

	Allocate_Memory();

	ftime(&start);
	Send_To_Device();

	Call_GPUFunction();

	Send_To_Host();
	//cudaDeviceSynchronize();
	ftime(&end);

        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);

	Save_Result();

	/*
	printf("Y = ");
	for(i=0;i<n;i++){
		printf("%g, ",yHostPtr[i]);
	}

	printf("X = ");
        for(i=0;i<n;i++){
                printf("%g, ",xHostPtr[i]);
        }
	*/

	printf("\n");
	Free_Memory();

}
