#include <stdio.h>
#include <malloc.h>

#define N 10

void CPU_malloc();
void GPU_malloc();
void Free();

void Init();
void Sent_to_device();
void Sent_to_host();
void print(float *a);
