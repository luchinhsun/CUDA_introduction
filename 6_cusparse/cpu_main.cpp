#include "head.h"
/*
extern int *cooRowIndexHostPtr;
extern int * cooRowIndexHostPtr;
extern int * cooColIndexHostPtr;
extern float * cooValHostPtr;
extern float * yHostPtr;
*/
extern float * yHostPtr;
extern float * xHostPtr;
//extern int * csrRowHostPtr;

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
	ftime(&end);
        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);

	printf("Y = ");
	for(i=0;i<n;i++){
		printf("%g, ",yHostPtr[i]);
	}

	printf("X = ");
        for(i=0;i<n;i++){
                printf("%g, ",xHostPtr[i]);
        }

	/*
	printf("csrRowHostPtr = ");
        for(i=0;i<n+1;i++){
                printf("%d, ",csrRowHostPtr[i]);
        }
	*/

	printf("\n");
	Free_Memory();

	return 0;
}
