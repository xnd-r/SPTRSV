CC=icpc

INCLUDES=-I/common/cuda-11.2/include

OPENCL_LIBS=-L/common/cuda-11.2/lib64 -lOpenCL

LIBS = $(OPENCL_LIBS)
make:
	$(CC) -O3 *.cpp -qopenmp -mkl:parallel -std=gnu++11 -o SPTRSV $(INCLUDES) $(LIBS)
