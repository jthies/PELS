include ../make.inc

OBJ=sparsemat.o densemat.o mpk.o

libmpk.so: ${OBJ}
	${CXX} ${CXXFLAGS} -I${RACE_INCDIR} -I./include/ -shared -o lib/libmpk.so ${OBJ} -L${RACE_LIBDIR} -lhwloc -lRACE -Wl,-rpath=${PWD}/lib/

%.o: %.c
	${CXX} ${CXXFLAGS} -I${RACE_INCDIR} -I./include/ -c $<
