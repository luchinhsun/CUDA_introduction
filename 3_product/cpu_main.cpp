#include "head.h"

extern float *h_a;
extern float *h_b;
extern float *h_c;

int main(){
	struct timeb start1,end1,start2,end2,start3,end3;
        int diff;

	CPU_malloc();
	GPU_malloc();

	ftime(&start1);
	Init();

	/*
	printf("a = \n");
	print(h_a);
	printf("b = \n");
        print(h_b);
	*/

        product1();
        ftime(&end1);

	ftime(&start2);
	Sent_to_device();
	product2();
	Sent_to_host();
	ftime(&end2);

	ftime(&start3);
	Sent_to_device();
        product3();
        Sent_to_host();
	ftime(&end3);

	//printf("c = \n");
	//print(h_c);

	diff = (int)(1000.0*(end1.time-start1.time)+(end1.millitm-start1.millitm));
        printf("CPU Time = %d ms\n",diff);
	diff = (int)(1000.0*(end2.time-start2.time)+(end2.millitm-start2.millitm));
        printf("GPU serial Time = %d ms\n",diff);
	diff = (int)(1000.0*(end3.time-start3.time)+(end3.millitm-start3.millitm));
        printf("GPU Time = %d ms\n",diff);

	Free();


	printf("complete\n");
	return 0;
}
