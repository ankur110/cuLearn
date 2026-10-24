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

__global__ void scaleMatrix(const float* A,float* C,float k,int rows,int cols){
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    const int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(row<rows&&col<cols){
        const int idx=row*cols+col;
        C[idx]=k*A[idx];
    }
}

int main(){
    constexpr int rows=1000,cols=1500;
    constexpr float k=3.0f;
    constexpr int count=rows*cols;
    const size_t bytes=count*sizeof(float);

    vector<float> h_A(count),h_C(count);
    for(int i=0;i<count;++i)
        h_A[i]=static_cast<float>(i%100)-50.0f;

    float *d_A=nullptr,*d_C=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,bytes));
    CUDA_CHECK(cudaMalloc(&d_C,bytes));
    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),bytes,cudaMemcpyHostToDevice));

    const dim3 block(32,8);
    const dim3 grid((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);
    cout<<"grid  = ("<<grid.x<<", "<<grid.y<<")\n"
        <<"block = ("<<block.x<<", "<<block.y<<")\n";

    scaleMatrix<<<grid,block>>>(d_A,d_C,k,rows,cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C.data(),d_C,bytes,cudaMemcpyDeviceToHost));

    for(int i=0;i<5;++i)
        cout<<"C["<<i<<"] = "<<h_C[i]<<'\n';

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
