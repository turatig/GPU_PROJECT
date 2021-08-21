#include <stdio.h>
#include <stdlib.h>

#define IN_SIZE 16
#define TH_X_BLK 8

//Since every thread is a comparator of two elements, every block sorts a list of TH_X_BLK*2 elements
//Refers to Mullapudi paper figure to understand the offsets
__global__ void bitonicMergeSort(int *in,int *out){
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ int tile[TH_X_BLK*2];
    int tmp;
    unsigned int dir,stage_oset,step_oset;

    if(x<IN_SIZE/2){
        tile[threadIdx.x]=in[x];
        tile[threadIdx.x+TH_X_BLK]=in[x+TH_X_BLK];
    }

    for(unsigned int stage=1;stage<TH_X_BLK*2;stage*=2){
        //direction determines if the thread swap up or down
        dir=threadIdx.x%(stage*2);
        //stage_oset determines the starting point from which the threads do comparisons
        stage_oset=stage*(threadIdx.x/stage);
        //step=...4,2,1. Powers of 2
        for(unsigned int step=stage;step>0;step/=2){
            //step_oset determines the starting point from which a thread do a comparison inside a stage
            step_oset=step*((threadIdx.x%stage)/step);
            __syncthreads();
            //sort in ascending order
            if(dir<stage){
                if(tile[threadIdx.x+stage_oset+step_oset]>tile[threadIdx.x+stage_oset+step_oset+step]){
                    tmp=tile[threadIdx.x+stage_oset+step_oset];
                    tile[threadIdx.x+stage_oset+step_oset]=tile[threadIdx.x+stage_oset+step_oset+step];
                    tile[threadIdx.x+stage_oset+step_oset+step]=tmp;
                }
            }
            //sort in descending order
            else{
                if(tile[threadIdx.x+stage_oset+step_oset]<tile[threadIdx.x+stage_oset+step_oset+step]){
                    tmp=tile[threadIdx.x+stage_oset+step_oset];
                    tile[threadIdx.x+stage_oset+step_oset]=tile[threadIdx.x+stage_oset+step_oset+step];
                    tile[threadIdx.x+stage_oset+step_oset+step]=tmp;
                }
            }
        }
    }
    if(x<IN_SIZE/2){
        out[x]=tile[threadIdx.x];
        out[x+TH_X_BLK]=tile[threadIdx.x+TH_X_BLK];
    }
}

int main(){
    bool log=true;

    int *in=(int*)malloc(IN_SIZE*sizeof(int)),
        *out=(int*)malloc(IN_SIZE*sizeof(int)),
        *gpu_in,*gpu_out;

    for(int i=0;i<IN_SIZE;i++)
        in[i]=rand()%5000;

    if(log){
        printf("Input array\n");
        for(int i=0;i<IN_SIZE;i++)
            printf("%d ",in[i]);
        printf("\n");
    }

    cudaMalloc((void**)&gpu_in,IN_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_out,IN_SIZE*sizeof(int));
    cudaMemcpy(gpu_in,in,IN_SIZE*sizeof(int),cudaMemcpyHostToDevice);

    bitonicMergeSort<<<(IN_SIZE/2+TH_X_BLK-1)/TH_X_BLK,TH_X_BLK>>>(gpu_in,gpu_out);

    cudaMemcpy(out,gpu_out,IN_SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    if(log){
        printf("Output array\n");
        for(int i=0;i<IN_SIZE;i++)
            printf("%d ",out[i]);
        printf("\n");
    }
    
    cudaFree(gpu_in);
    free(in);

}