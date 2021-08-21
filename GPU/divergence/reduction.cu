#include<stdio.h>
#include<stdlib.h>
#include<math.h>

/*
Strategy: every block reduce a part of the input of size: THREADS_PER_BLOCK. Partial values are then 
reduced on the CPU
*/
#define INPUT_SIZE 65536
#define THREADS_PER_BLOCK 128
#define N_BLOCKS (INPUT_SIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK

/*
Compute reduction with divergence
*/
__global__ void reductionWDiv(int *in,int *res){
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    int i;

    for(i=2;i<=THREADS_PER_BLOCK;i*=2){
        /*
        Divergence problem: iteration 0: thread 0,2,4...
                            iteration 1: thread 0,4,8...
                            ...and so on...
        Threads in the same warp will follow different paths
        */
        if(threadIdx.x%i==0)
            in[n]+=in[n+i/2];
        /*
        Synchronize threads in the block after every iteration
        */
        __syncthreads();
    }
    
    if(threadIdx.x==0)
        res[blockIdx.x]=in[n];
}
/*
Compute reduction without divergence
*/
__global__ void reductionWODivergence(int *in,int *res){
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    int i;

    for(i=2;i<=THREADS_PER_BLOCK;i*=2){
        /*
        This condition allows subsequent threads in the warp to be active together till the computation
        is done
        */
        if(threadIdx.x<THREADS_PER_BLOCK/i)
            in[blockIdx.x*blockDim.x+threadIdx.x*i]+=in[blockIdx.x*blockDim.x+
                                                        threadIdx.x*i+
                                                        i/2];
        __syncthreads();
    }
    if(threadIdx.x==0){
        res[blockIdx.x]=in[n];
        /*printf("Computed %d\n");*/
    }

}
/*
Compute reduction without divergence and accessing subsequent memory cells.
This is the most efficient approach to the problem of reduction
*/
__global__ void reductionSubsMemory(int *in,int *res){
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    int i;

    for(i=2;i<=THREADS_PER_BLOCK;i*=2){
        /*
        This condition allows subsequent threads in the warp to be active together till the computation
        is done
        */
        if(threadIdx.x<THREADS_PER_BLOCK/i)
            in[n]+=in[n+THREADS_PER_BLOCK/i];
        __syncthreads();
    }
    if(threadIdx.x==0){
        res[blockIdx.x]=in[n];
        /*printf("Computed %d\n");*/
    }

}
int main(){
    bool log=false;
    /*
    Allocate cpu memory for input and array of partial results
    */
    int *in=(int*)malloc(INPUT_SIZE*sizeof(int)),
        *part=(int*)malloc(N_BLOCKS*sizeof(int));
    /*
    Declare pointers for GPU memory
    */
    int *gpu_in,*gpu_part;
    /*
    Allocate global memory for GPU
    */
    cudaMalloc((void**)&gpu_in,INPUT_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_part,N_BLOCKS*sizeof(int));
    /*
    Fill input data
    */
    int i;
    for(i=0;i<INPUT_SIZE;i++){in[i]=2;}
    
    if(log){
        printf("Input array is:\n");
        for(i=0;i<INPUT_SIZE;i++){printf("%d ",in[i]);}
        printf("\n");
    }
    /*
    Copy memory to device pointers
    */
    cudaMemcpy(gpu_in,in,INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Invoke kernel
    */
    reductionWDiv<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gpu_in,gpu_part);
    /*
    Copy memory from device to host
    */
    cudaMemcpy(part,gpu_part,N_BLOCKS*sizeof(int),cudaMemcpyDeviceToHost);
    /*
    Reduce array of partials to obtain the result of the reduction operation
    */
    int res=0;
    for(i=0;i<N_BLOCKS;i++){res+=part[i];}
    if(log)
        printf("The array is reduced with sum into %d\n",res);

    if(log){
        printf("Input array is:\n");
        for(i=0;i<INPUT_SIZE;i++){printf("%d ",in[i]);}
        printf("\n");
    }

    /*
    Copy input from CPU to global device memory to avoid problems with values kept from the last call
    */
    cudaMemcpy(gpu_in,in,INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Zero partial values array and copy to device
    */
    for(i=0;i<N_BLOCKS;i++){part[i]=0;}
    cudaMemcpy(gpu_part,part,N_BLOCKS*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Invoke kernel
    */
    reductionWODivergence<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gpu_in,gpu_part);
    /*
    Copy memory from device to host
    */
    cudaMemcpy(part,gpu_part,N_BLOCKS*sizeof(int),cudaMemcpyDeviceToHost);
    res=0;
    for(i=0;i<N_BLOCKS;i++){res+=part[i];}
    if(log)
        printf("The array is reduced with sum into %d\n",res);

    if(log){
        printf("Input array is:\n");
        for(i=0;i<INPUT_SIZE;i++){printf("%d ",in[i]);}
        printf("\n");
    }
    /*
    Copy input from CPU to global device memory to avoid problems with values kept from the last call
    */
    cudaMemcpy(gpu_in,in,INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Zero partial values array and copy to device
    */
    for(i=0;i<N_BLOCKS;i++){part[i]=0;}
    cudaMemcpy(gpu_part,part,N_BLOCKS*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Invoke kernel
    */
    reductionSubsMemory<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gpu_in,gpu_part);
    /*
    Copy memory from device to host
    */
    cudaMemcpy(part,gpu_part,N_BLOCKS*sizeof(int),cudaMemcpyDeviceToHost);
    res=0;
    for(i=0;i<N_BLOCKS;i++){res+=part[i];}
    if(log)
        printf("The array is reduced with sum into %d\n",res);
    /*
    Free GPU memory
    */
    cudaFree(gpu_in);
    cudaFree(gpu_part);
    /*
    Free CPU memory
    */
    free(in);
    free(part);    

}
