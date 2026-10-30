#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

#define CUDA_CHECK(call)                                                    \
    do{                                                                    \
        cudaError_t err=(call);                                            \
        if(err!=cudaSuccess){                                              \
            cerr<<"CUDA error: "<<cudaGetErrorString(err)                   \
                <<" at "<<__FILE__<<":"<<__LINE__<<'\n';                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }while(0)

__global__ void copyKernel(const float* in,float* out,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=in[i];
}

int main(){
    constexpr int n=1<<24;
    constexpr size_t bytes=static_cast<size_t>(n)*sizeof(float);
    constexpr int reps=20;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    const int numSMs=prop.multiProcessorCount;
    const int maxWarpsPerSM=prop.maxThreadsPerMultiProcessor/prop.warpSize;

    int memClockKHz=0,busWidthBits=0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz,cudaDevAttrMemoryClockRate,0));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits,cudaDevAttrGlobalMemoryBusWidth,0));
    const double peakGBs=2.0*memClockKHz*1e3*(busWidthBits/8.0)/1e9;

    float *d_in=nullptr,*d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,bytes));
    CUDA_CHECK(cudaMalloc(&d_out,bytes));
    CUDA_CHECK(cudaMemset(d_in,0,bytes));
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cout<<fixed<<setprecision(1);
    cout<<prop.name<<": "<<numSMs<<" SMs, peak "<<peakGBs<<" GB/s\n\n";
    cout<<right
        <<setw(8)<<"threads"<<setw(9)<<"blocks"
        <<setw(8)<<"blk/SM"<<setw(7)<<"warps"
        <<setw(7)<<"occ%"<<setw(8)<<"waves"
        <<setw(8)<<"us"<<setw(8)<<"GB/s"
        <<setw(7)<<"%peak"<<'\n';

    for(int threads:{32,64,128,256,512,1024}){
        const int blocks=(n+threads-1)/threads;

        int blocksPerSM=0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &blocksPerSM,copyKernel,threads,0));
        const int warpsPerSM=blocksPerSM*threads/prop.warpSize;
        const double occupancy=100.0*warpsPerSM/maxWarpsPerSM;
        const double waves=static_cast<double>(blocks)/(blocksPerSM*numSMs);

        copyKernel<<<blocks,threads>>>(d_in,d_out,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for(int r=0;r<reps;++r)
            copyKernel<<<blocks,threads>>>(d_in,d_out,n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms=0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        ms/=reps;
        const double gbs=2.0*bytes/(ms*1e6);

        cout<<setw(8)<<threads<<setw(9)<<blocks
            <<setw(8)<<blocksPerSM<<setw(7)<<warpsPerSM
            <<setw(7)<<occupancy<<setw(8)<<waves
            <<setw(8)<<ms*1000.0<<setw(8)<<gbs
            <<setw(7)<<100.0*gbs/peakGBs<<'\n';
    }

    cout<<"\nWhat-if: 256 threads/block + shared memory per block\n";
    for(int kb:{0,8,16,24,32,48}){
        int blocksPerSM=0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &blocksPerSM,copyKernel,256,kb*1024));
        cout<<setw(4)<<kb<<" KB -> "<<blocksPerSM<<" blocks/SM, occupancy "
            <<100.0*blocksPerSM*256/prop.maxThreadsPerMultiProcessor<<"%\n";
    }

    int minGridSize=0,suggested=0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize,&suggested,copyKernel,0,0));
    cout<<"\nSuggested block size: "<<suggested
        <<"  (min grid for full GPU: "<<minGridSize<<" blocks)\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
