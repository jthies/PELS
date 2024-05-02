
include make.inc

OBJ=vector_kernels.o spmv_kernels.o

libkernels.so: ${OBJ} Makefile
	${CC} ${CFLAGS} -shared -o libkernels.so ${OBJ}

%.o: %.c Makefile
	${CC} ${CFLAGS} -c $<
