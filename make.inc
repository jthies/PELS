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
