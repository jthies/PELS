include ../make.inc

OBJ=sparsemat.o densemat.o mpk.o

libmpk.so: ${OBJ} Makefile ../make.inc
	${CXX} ${CXXFLAGS} -I${RACE_INCDIR} -I./include/ -shared -o lib/libmpk.so ${OBJ} -L${RACE_LIBDIR} -lhwloc -lRACE -Wl,-rpath=${PWD}/lib/

%.o: %.c Makefile ../make.inc
	${CXX} ${CXXFLAGS} -I${RACE_INCDIR} -I./include/ -c $<
