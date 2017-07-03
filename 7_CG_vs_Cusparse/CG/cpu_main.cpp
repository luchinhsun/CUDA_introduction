#include "head.h"
#include <math.h>

extern float *h_A;     // A matrix
extern float *h_B;     // B vector
extern float *h_x;     // x (solution) vector
extern float *h_R;     // Residual vector
extern float *h_P;
extern float *h_AP;
extern float *h_scalars;  // Scalars

int main() {

	struct timeb start,end;
	int diff;

        int i;
        float alpha;
        float beta;
        float PTAP, RTR, RTR_NEW;
        // Allocate the memory
        Allocate_Memory();

    	//Init();
	Read_File();
	ftime(&start);

	// Send Init to Device
	Send_To_Device();

	//SetUp_CG(h_B, h_R, h_P, N);
	// SetUpCG in Device
	SetUp_CG_GPU();

    // Here I will fix the number of iterations to 33.
    // Theoretically we need 625 (worst case) but practically speaking
    // we know CG will converge with sqrt(N) ~ 25 iterations.
        for (i = 0; i < N; i++) {

                // Compute AP
                //MatrixVectorProduct(h_A, h_P, h_AP, N); // Send it to R so we can get it back and have a look
		MatrixVectorFunction();

                // Now to compute PTAP
                //PTAP = DotProduct(h_P, h_AP, N);
                // Compute RTR
                //RTR = DotProduct(h_R, h_R, N);
		DotFunction1();

		// Update X (and compute alpha)
                //alpha = RTR/PTAP;
		Cal_alpha();
                //Update_x(h_x, h_P, alpha, N);
                //Update_R(h_R, h_AP, alpha, N);
		Update_xandR();

                // Compute the new residual
                //RTR_NEW = DotProduct(h_R, h_R, N);
		DotFunction2();

                // Update P (and compute beta)
                //beta = RTR_NEW/RTR;
		Cal_beta();
                //Update_P(h_P, h_R, beta, N);
		Update_P();

		Send_For_Print();
		RTR_NEW = h_scalars[4];
		if(RTR_NEW<1e-6){
			Send_For_Print();
                	alpha = h_scalars[0];   beta = h_scalars[1];    PTAP = h_scalars[2];
                	RTR = h_scalars[3];     //RTR_NEW = h_scalars[4];
                	printf("Iteration %i = RTR = %g, RTR_new = %g, PTAP = %g, ALPHA = %g, BETA = %g\n", i, RTR, RTR_NEW, PTAP, alpha, beta);
			break;
		}

        }

	Send_To_Host();

        ftime(&end);

        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("Time = %d ms\n",diff);

        // Save result
        Save_Result();

        // Free memory
        Free_Memory();

        printf("Complete\n");

}

