#include <iostream>
#include <vector>
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

__global__ void vecAdd(const float* A,const float* B,float* C,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)C[i]=A[i]+B[i];
}

int main(){
    constexpr int n=1<<20;
    const size_t bytes=n*sizeof(float);

    vector<float> h_A(n),h_B(n),h_C(n);
    for(int i=0;i<n;++i){
        h_A[i]=static_cast<float>(i);
        h_B[i]=2.0f*i;
    }

    float *d_A=nullptr,*d_B=nullptr,*d_C=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,bytes));
    CUDA_CHECK(cudaMalloc(&d_B,bytes));
    CUDA_CHECK(cudaMalloc(&d_C,bytes));

    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B.data(),bytes,cudaMemcpyHostToDevice));

    constexpr int threads=256;
    const int blocks=(n+threads-1)/threads;
    vecAdd<<<blocks,threads>>>(d_A,d_B,d_C,n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C.data(),d_C,bytes,cudaMemcpyDeviceToHost));

    cout<<"blocks = "<<blocks<<", threads/block = "<<threads<<'\n';
    for(int i=0;i<5;++i)
        cout<<"C["<<i<<"] = "<<h_C[i]<<'\n';

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
