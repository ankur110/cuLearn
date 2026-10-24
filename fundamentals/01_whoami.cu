#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

#define CUDA_CHECK(call)                                                \
    do{                                                                 \
        cudaError_t err=(call);                                         \
        if(err!=cudaSuccess){                                           \
            cerr<<"CUDA error: "<<cudaGetErrorString(err)               \
                <<" at "<<__FILE__<<":"<<__LINE__<<'\n';                \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }while(0)

__global__ void whoami(){
    const int col=blockDim.x*blockIdx.x+threadIdx.x;
    const int row=blockDim.y*blockIdx.y+threadIdx.y;
    printf("block(%u,%u) thread(%u,%u) -> global(row=%d, col=%d)\n",
           blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,row,col);
}

int main(){
    const dim3 grid(2,2);
    const dim3 block(3,2);
    whoami<<<grid,block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
