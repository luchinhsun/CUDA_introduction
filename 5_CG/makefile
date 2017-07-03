CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart

all: CPU GPU 
	nvcc cuda_main.o cpu_main.o -o test.run -L $(CUDA_LIB)
CPU:
	nvcc cpu_main.cpp -c
GPU:
	nvcc -arch=sm_61 cuda_main.cu -c
	#nvcc -arch=sm_61 --ptxas-options=-v cuda_main.cu -c
clean:
	rm *.o *.run *.txt
