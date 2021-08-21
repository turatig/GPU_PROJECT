#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "imgStuff.h"

#define H_PIXEL 32
#define V_PIXEL 32
#define H_BYTES ((3*H_PIXEL+3)&~3)
#define MAX_DWELL 512
#define THREADS_PER_BLOCK 4

/*Bullshit for treating complex numbers*/
typedef struct comp{
    float re;
    float img;
}complex;

__host__ __device__ complex *complex_alloc(float re,float img){
    complex *c=(complex*)malloc(2*sizeof(float));
    c->re=re;
    c->img=img;
    return c;
}

__device__ complex *gpu_complex_alloc(float re,float img){
    complex *c;
    cudaMalloc((void**)&c,2*sizeof(float));
    c->re=re;
    c->img=img;
    return c;
}

__host__ __device__ void add(complex *z1,complex *z2,complex *res){
    res->re=z1->re+z2->re;
    res->img=z1->img+z2->img;
}

__host__ __device__ void sub(complex *z1,complex *z2,complex *res){
    res->re=z1->re-z2->re;
    res->img=z1->img-z2->img;
}

__host__ __device__ void mul(complex *z1,complex *z2,complex *res){
    float re=(z1->re*z2->re-z1->img*z2->img);
    float img=(z1->re*z2->img+z1->img*z2->re);
    res->re=re;
    res->img=img;
}

__host__ __device__ float abs2(complex *z){
    return sqrt(z->re*z->re+z->img*z->img);
}

/* Args: 
w=width_of_img 
h=height_of_img
cmin=coordinates_bottom_left
cmax=coordinates_top_right
x,y=pixel_coordinates
*/
__host__ int timeEscapeAlgorithm(int w,int h,complex *cmin,complex *cmax,int x,int y){
    complex *dc=complex_alloc(0.0,0.0);
    sub(cmax,cmin,dc);
    
    float fx=(float)x/w;
    float fy=(float)y/h;
    complex *z=complex_alloc(fx*dc->re,fy*dc->img);
    complex *c=complex_alloc(0.0,0.0);
    add(cmin,z,c);
    z->re=c->re;
    z->img=c->img;
    int dwell=0;
    while(dwell<MAX_DWELL && abs2(z)<4){
        mul(z,z,z);
        add(z,c,z);
        dwell++;
    }
    free(dc);
    free(z);
    free(c);
    return dwell;
}

//__constant__ complex *cmin;
//__constant__ complex *cmax;

/*Time escape algorithm gpu version*/
__global__ void tea(unsigned char *img,complex *cmin,complex *cmax){
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;

    if(x<H_PIXEL && y<V_PIXEL){
        
        complex *dc=complex_alloc(0.0,0.0);
        
        /*
        Converting pixel coordinates (x,y) into complex coordinates knowing the top-right and
        bottom-left complex coordinates
        */
        sub(cmax,cmin,dc);
        float fx=(float)x/H_PIXEL;
        float fy=(float)y/V_PIXEL;
        
        complex *z=complex_alloc(fx*dc->re,fy*dc->img);
        complex *c=complex_alloc(0.0,0.0);
        
        /*
        z_0=c
        */
        add(cmin,z,c);
        z->re=c->re;
        z->img=c->img;
        
        int dwell=0;
        
        while(dwell<MAX_DWELL && abs2(z)<4){
            mul(z,z,z);
            add(z,c,z);
            dwell++;
        }
        
        /*if(dwell==MAX_DWELL){
            printf("If\n");
            img[y*H_BYTES+x*3]=0;
            img[y*H_BYTES+x*3+1]=100;
            img[y*H_BYTES+x*3+2]=200;
        }
        else{
            printf("Else\n");
            img[y*H_BYTES+x*3]=200;
            img[y*H_BYTES+x*3+1]=100;
            img[y*H_BYTES+x*3+2]=0;
        }
        */
        img[y*H_BYTES+x*3]=200;
        img[y*H_BYTES+x*3+1]=100;
        img[y*H_BYTES+x*3+2]=0;
        free(dc);
        free(z);
        free(c);

    }
}

__global__ void useless(unsigned char *img){
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    
    if(x<H_PIXEL && y<V_PIXEL){
        img[y*H_BYTES+x*3]=(unsigned char)200;
        img[y*H_BYTES+x*3+1]=(unsigned char)100;
        img[y*H_BYTES+x*3+2]=(unsigned char)0;
    }
}
int main(){

    /*Allocate space for pixel matrix*/
    unsigned char *img=(unsigned char*)malloc(H_BYTES*V_PIXEL);
    ImgProp *ip=trueColorHeader(V_PIXEL,H_PIXEL);
    int i,j,dwell;

    complex *cmin=complex_alloc(-1.5,-1.0);
    complex *cmax=complex_alloc(0.5,1);

    /*Fill image data*/
    /*
    for(i=0;i<V_PIXEL;i++){
        //printf("Row %d\n",i);
        for(j=0;j<H_PIXEL*3;j+=3){
            dwell=timeEscapeAlgorithm(H_PIXEL,V_PIXEL,cmin,cmax,j/3,i);
            //printf("Dwell %d\n",dwell);
            if(dwell==MAX_DWELL){
                img[i*H_BYTES+j]=0;
                img[i*H_BYTES+j+1]=0;
                img[i*H_BYTES+j+2]=0;
            }
            else{
                img[i*H_BYTES+j]=200;
                img[i*H_BYTES+j+1]=100;
                img[i*H_BYTES+j+2]=0;
            }
        }
        /*Add pad for BMP format*/
        /*for(j=3*H_PIXEL;j<H_BYTES;j++)
            img[i*V_PIXEL+j]=0;
    }
    printf("CPU finished\n");
    WriteBMP(img,"test.bmp",ip);
    */
    for(i=0;i<V_PIXEL;i++){
        for(j=0;j<H_BYTES;j++)
            img[i*H_BYTES+j]=200;
    }
    /*Allocate GPU memory*/
    unsigned char *gpu_img;
    complex *gpu_c_min,*gpu_c_max;
    
    cudaMalloc((void**)&gpu_img,H_BYTES*V_PIXEL);
    cudaMalloc((void**)&gpu_c_min,2*sizeof(float));
    cudaMalloc((void**)&gpu_c_max,2*sizeof(float));

    cudaMemcpy(gpu_img,img,H_BYTES*V_PIXEL*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_min,cmin,2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_max,cmax,2*sizeof(float),cudaMemcpyHostToDevice);

    /*Setting parameters of the kernel*/
    int blockSide=(int)sqrt(THREADS_PER_BLOCK);
    
    dim3 blockSize=dim3(blockSide,blockSide);
    dim3 gridSize=dim3((H_PIXEL+blockSide-1)/blockSide,(V_PIXEL+blockSide-1)/blockSide);

    /*
    useless<<<gridSize,blockSize>>>(gpu_img);
    cudaMemcpy(img,gpu_img,H_BYTES*V_PIXEL*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    /*for(i=0;i<V_PIXEL;i++){
        for(j=0;j<H_PIXEL;j++){
            printf("[%d %d %d] ",img[i*H_BYTES+j],img[i*H_BYTES+j+1],img[i*H_BYTES+j+2]);
        }
        printf()
    }
    WriteBMP(img,"test2.bmp",ip);
    exit(1);*/
    tea<<<gridSize,blockSize>>>(gpu_img,gpu_c_min,gpu_c_max);

    cudaMemcpy(img,gpu_img,H_BYTES*V_PIXEL*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    /*for(i=0;i<V_PIXEL;i++){
        for(j=0;j<H_PIXEL*3;j+=3)
            printf("[%d %d %d] ",img[i*H_BYTES+j],img[i*H_BYTES+j+1],img[i*H_BYTES+j+2]);

        printf("\n");
    }*/

    WriteBMP(img,"toast.bmp",ip);

    cudaFree(gpu_c_min);
    cudaFree(gpu_c_max);
    cudaFree(gpu_img);

    free(img);
    free(ip->HeaderInfo);
    free(ip);

}