#include <stdio.h>
#include <stdlib.h>

#define IN_SIZE 32
#define TH_X_BLK 32


// It's not work efficient and results must be adjusted by adjusted
__global__ void prefixSumNaive(int *in,int *out){
    __shared__ int smem[TH_X_BLK];

    int x=threadIdx.x+blockIdx.x*blockDim.x;

    if(x<IN_SIZE)
        smem[threadIdx.x]=in[x];

    for(int i=1;i<TH_X_BLK;i*=2){
        __syncthreads();
        if(threadIdx.x>=i)
            smem[threadIdx.x]+=smem[threadIdx.x-i];

    }
    if(x<IN_SIZE)
        out[x]=smem[threadIdx.x];
}

//work-efficient implementation of the algorithm
__global__ void prefixSumWE(int *in,int *out){
    __shared__ int smem[TH_X_BLK];

    int x=threadIdx.x+blockIdx.x*blockDim.x;
    unsigned int i;
    int temp;

    if(x<IN_SIZE)
        smem[threadIdx.x]=in[x];

    //reduction phase
    for(i=1;i<TH_X_BLK;i*=2){
        __syncthreads();
        if(threadIdx.x<TH_X_BLK/(i*2))
            smem[(threadIdx.x+1)*i*2-1]+=smem[(threadIdx.x+1)*i*2-1-i];
    }
    //sweep-down phase
    if(threadIdx.x==0)
        smem[blockDim.x-1]=0;

    for(i=TH_X_BLK/2;i>0;i/=2){
        __syncthreads();
        if(threadIdx.x<TH_X_BLK/(i*2)){
            //taking value of the current root
            temp=smem[(threadIdx.x+1)*i*2-1];
            //the current root value is root_value+left_child value
            smem[(threadIdx.x+1)*i*2-1]+=smem[(threadIdx.x+1)*i*2-1-i];
            //left_child value is the old value of the root
            smem[(threadIdx.x+1)*i*2-1-i]=temp;
        }
    }
    if(x<IN_SIZE)
        out[x]=smem[threadIdx.x];
}
int main(){
    bool log=true;

    int *in=(int*)malloc(IN_SIZE*sizeof(int)),
        *out=(int*)malloc(IN_SIZE*sizeof(int)),
        *gpu_in,*gpu_out;

    cudaMalloc((void**)&gpu_in,IN_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_out,IN_SIZE*sizeof(int));

    for(int i=0;i<IN_SIZE;i++)
        in[i]=1;

    if(log){
        printf("Input array\n");
        for(int i=0;i<IN_SIZE;i++)
            printf("%d ",in[i]);
        printf("\n");
    }
    
    cudaMemcpy(gpu_in,in,IN_SIZE*sizeof(int),cudaMemcpyHostToDevice);

    prefixSumNaive<<<(IN_SIZE+TH_X_BLK-1)/TH_X_BLK,TH_X_BLK>>>(gpu_in,gpu_out);
    cudaMemcpy(out,gpu_out,IN_SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    if(log){
        printf("Output array(naive version)\n");
        for(int i=0;i<IN_SIZE;i++)
            printf("%d ",out[i]);
        printf("\n");
    }

    prefixSumWE<<<(IN_SIZE+TH_X_BLK-1)/TH_X_BLK,TH_X_BLK>>>(gpu_in,gpu_out);
    cudaMemcpy(out,gpu_out,IN_SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    if(log){
        printf("Output array\n");
        for(int i=0;i<IN_SIZE;i++)
            printf("%d ",out[i]);
        printf("\n");
    }

    cudaFree(gpu_in);
    cudaFree(gpu_out);
    free(in);
    free(out);

}