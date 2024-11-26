clang++ -fPIC -O3 -march=native -fopenmp -lblas -shared -o libgemm.so ./gemm_c.cc
