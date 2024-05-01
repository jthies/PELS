
# GCC
CC=gcc
CFLAGS=-fPIC -O3 -fopenmp -march=native

# Intel LLVM-based compiler
#CC=icx
#CFLAGS=-fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather

OBJ=vector_kernels.o spmv_kernels.o

libkernels.so: ${OBJ} Makefile
	${CC} ${CFLAGS} -shared -o libkernels.so ${OBJ}

%.o: %.c Makefile
	${CC} ${CFLAGS} -c $<
