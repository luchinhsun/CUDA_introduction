#include "head.h"

extern float *h_a;
extern float *h_b;

int main(){

	CPU_malloc();
	GPU_malloc();

	Init();
	printf("a = \n");
	print(h_a);
	Sent_to_device();
	Sent_to_host();
	printf("b = \n");
	print(h_b);
	Free();


	printf("complete\n");
	return 0;
}
