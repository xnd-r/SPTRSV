CC=icpc

INCLUDES=-I/common/cuda-11.2/include -I"${MKLROOT}/include"

LIBS=-L/common/cuda-11.2/lib64 -lOpenCL -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

$(info    MKLROOT is $(MKLROOT))

make:
	$(CC) -O3 *.cpp -qopenmp -Wextra -Wall -pedantic-errors -xAVX -mkl=parallel -o SPTRSV -std=c++11 $(INCLUDES) $(LIBS)

