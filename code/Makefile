CC=icpc

INCLUDES=-I/common/cuda-11.2/include -I"${MKLROOT}/include"

USING_LIBS=-L/common/cuda-11.2/lib64 -lOpenCL -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

LIBS = $(USING_LIBS)

$(info    MKLROOT is $(MKLROOT))

make:
	$(CC) -O3 *.cpp -qopenmp -DMKL_LP64 -Wextra -Wall -pedantic-errors -qopt-report=5 -xAVX -mkl=parallel -std=c++11 -o SPTRSV $(INCLUDES) $(LIBS)

