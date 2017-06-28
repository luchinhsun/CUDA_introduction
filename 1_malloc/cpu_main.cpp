#include "head.h"

extern float *h_a;
extern float *h_b;

int main(){

	CPU_malloc();
	GPU_malloc();
	Free();

	printf("complete\n");
	return 0;
}
