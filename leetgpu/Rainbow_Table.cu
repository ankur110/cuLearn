#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

// __global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
//     int i= blockDim.x*blockIdx.x+threadIdx.x;
//     if(i<N){
//         unsigned int hash=input[i];
//         for(int j=0;j<R;j++){
//             hash=fnv1a_hash(hash);
//         }
//         output[i]=hash;
//     }
    
// }

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int i= blockDim.x*blockIdx.x+threadIdx.x;
    int stride=gridDim.x*blockDim.x;
    for(int start=i;start<N;start+=stride){
        unsigned int hash=input[start];
        for(int j=0;j<R;j++){
            hash=fnv1a_hash(hash);
        }
        output[start]=hash;
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
