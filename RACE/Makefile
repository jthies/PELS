CXX=icpx
#CXXFLAGS=-Iinclude -fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather
CXXFLAGS=-Iinclude -fPIC -O3 -qopenmp -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather

OBJ=sparsemat.o densemat.o mpk.o

libkernels.so: ${OBJ}
	${CXX} ${CXXFLAGS} -shared -o lib/libmpk.so ${OBJ} -L./lib/ -lhwloc -lRACE -Wl,-rpath=${PWD}/lib/

%.o: %.c
	${CXX} ${CXXFLAGS} -c $<
