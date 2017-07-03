// CG (Conjugate Gradient) Demonstration code
// Code rewrite from Prof. Matthew Smith, NCKU, 2016 in Introduction to Multi-Core (CPU) and GPU Computing class
// All together in one file to make
// it easier to build

// This code will solve a steady 2D heat transfer problem.
// Here we use 2D central differences to create an Ax = B
// matrix. The init() function creates A and B - A is a
// NxN matrix while B is a 1xN vector. Our solution is x.

#include <stdio.h>
#include <malloc.h>
//#include <sys/time.h>
#include <sys/timeb.h>
#include <time.h>
#include <cuda_runtime.h>

//#define NX 25   // Solve a 25x25 matrix
//#define NY 25
#define N 10000//(NX*NY)                       // Number of cells - our matrix is N*N
#define L 1.0
#define H 1.0
#define DX (L/NX)
#define DY (H/NY)
#define K 0.1                           // Conductivity
#define PHI_X (K/(DX*DX))
#define PHI_Y (K/(DY*DY))
//#define float double

void Allocate_Memory();
void Free_Memory();
void Read_File();
void Init();
void Save_Result();

//GPU function
void Send_To_Device();
void SetUp_CG_GPU();
void MatrixVectorFunction();
void DotFunction1();
void Cal_alpha();
void Update_xandR();
void DotFunction2();
void Cal_beta();
void Update_P();
void Send_For_Print();
void Send_To_Host();

