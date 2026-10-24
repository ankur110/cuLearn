#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < width*height){
        int base=i*4;
        image[base+0]=255-image[base+0];
        image[base+1]=255-image[base+1];
        image[base+2]=255-image[base+2];
    }
}
// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
