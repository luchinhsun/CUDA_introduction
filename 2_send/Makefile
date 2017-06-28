CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart

all: CPU GPU
	g++ cpu_main.o cuda_main.o -o test.run -L $(CUDA_LIB)
CPU:
	g++ cpu_main.cpp -c
GPU:
	nvcc -arch=sm_61 cuda_main.cu -c
clean:
	rm *.o *.run
