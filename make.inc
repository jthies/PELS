# GCC
CC=gcc
CFLAGS=-fPIC -O3 -fopenmp -march=native

CXX=g++
CXXFLAGS=${CFLAGS}

# Intel LLVM-based compiler
#CC=icx
#CFLAGS=-fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather

#CXX=icpx
#CXXFLAGS=${CFLAGS}

# In order to enable RACE cache-blocking, you also have to set these variables:
RACE_INCDIR=$HOME}/local/include
RACE_LIBDIR=${HOME}/local/lib/RACE/
