#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>

#define N 1000000

void CPU_malloc();
void GPU_malloc();
void Free();
void Init();
void Sent_to_device();
void Sent_to_host();
void print(float *a);

void product1();
void product2();
void product3();
