
CC=icx
CFLAGS=-fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather

OBJ=vector_kernels.o spmv_kernels.o

libkernels.so: ${OBJ}
	${CC} ${CFLAGS} -shared -o libkernels.so ${OBJ}

%.o: %.c
	${CC} ${CFLAGS} -c $<
