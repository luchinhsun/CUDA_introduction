#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>

#define n 3
#define N (n*n)

void CPU_malloc();
void GPU_malloc();
void Free();
void Init();
void Sent_to_device();
void Sent_to_host();
void print(float *a);

void CPU_add();
void GPU_add();
