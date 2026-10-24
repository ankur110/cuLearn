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

constexpr int BLUR_SIZE=3;

__global__ void invertKernel(uchar4* img,int numPixels){
    const int start=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=gridDim.x*blockDim.x;
    for(int p=start;p<numPixels;p+=stride){
        uchar4 px=img[p];
        px.x=255-px.x;
        px.y=255-px.y;
        px.z=255-px.z;
        img[p]=px;
    }
}

__global__ void grayKernel(const uchar4* rgba,unsigned char* gray,int width,int height){
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<width&&y<height){
        const int p=y*width+x;
        const uchar4 px=rgba[p];
        const float l=0.299f*px.x+0.587f*px.y+0.114f*px.z;
        gray[p]=static_cast<unsigned char>(l+0.5f);
    }
}

__global__ void blurKernel(const unsigned char* in,unsigned char* out,int width,int height){
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width||y>=height)return;

    int sum=0,count=0;
    for(int dy=-BLUR_SIZE;dy<=BLUR_SIZE;++dy)
        for(int dx=-BLUR_SIZE;dx<=BLUR_SIZE;++dx){
            const int ny=y+dy,nx=x+dx;
            if(ny>=0&&ny<height&&nx>=0&&nx<width){
                sum+=in[ny*width+nx];
                ++count;
            }
        }
    out[y*width+x]=static_cast<unsigned char>((sum+count/2)/count);
}

int main(){
    const int W=500,H=375;
    const int numPixels=W*H;

    vector<uchar4> h_rgba(numPixels);
    for(int y=0;y<H;++y)
        for(int x=0;x<W;++x){
            unsigned char r=255*x/(W-1);
            unsigned char g=255*y/(H-1);
            h_rgba[y*W+x]=make_uchar4(r,g,100,255);
        }

    uchar4* d_rgba=nullptr;
    unsigned char *d_gray=nullptr,*d_blur=nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgba,numPixels*sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_gray,numPixels));
    CUDA_CHECK(cudaMalloc(&d_blur,numPixels));

    CUDA_CHECK(cudaMemcpy(d_rgba,h_rgba.data(),numPixels*sizeof(uchar4),
                          cudaMemcpyHostToDevice));

    const dim3 block(32,8);
    const dim3 grid((W+block.x-1)/block.x,(H+block.y-1)/block.y);

    grayKernel<<<grid,block>>>(d_rgba,d_gray,W,H);
    CUDA_CHECK(cudaGetLastError());

    blurKernel<<<grid,block>>>(d_gray,d_blur,W,H);
    CUDA_CHECK(cudaGetLastError());

    const int threads=256;
    const int blocks=(numPixels+threads-1)/threads;
    invertKernel<<<blocks,threads>>>(d_rgba,numPixels);
    CUDA_CHECK(cudaGetLastError());

    vector<uchar4> h_inv(numPixels);
    vector<unsigned char> h_gray(numPixels),h_blur(numPixels);
    CUDA_CHECK(cudaMemcpy(h_inv.data(),d_rgba,numPixels*sizeof(uchar4),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gray.data(),d_gray,numPixels,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blur.data(),d_blur,numPixels,cudaMemcpyDeviceToHost));

    const int p=(H/2)*W+(W/2);
    cout<<"original RGB : "<<+h_rgba[p].x<<' '<<+h_rgba[p].y<<' '<<+h_rgba[p].z<<'\n';
    cout<<"inverted RGB : "<<+h_inv[p].x<<' '<<+h_inv[p].y<<' '<<+h_inv[p].z<<'\n';
    cout<<"gray         : "<<+h_gray[p]<<'\n';
    cout<<"blurred gray : "<<+h_blur[p]<<'\n';

    CUDA_CHECK(cudaFree(d_rgba));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blur));
    return 0;
}
