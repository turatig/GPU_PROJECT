#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 8
#define MASK_SIZE 4
#define RES_SIZE (MASK_SIZE+INPUT_SIZE-1)
#define THREAD_PER_BLOCK 2
#define N_BLOCKS (RES_SIZE+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK

/*
Debug runtime API function
*/
#define CHECK(call){\
    const cudaError_t error=call;\
    if( error!= cudaSuccess ){\
        printf("Error %s %d, ",__FILE__,__LINE__);\
        printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));\
        exit(1);\
    }\
}\

/*
Compute the convolution between in and mask in the straightforward way
*/
__global__ void convolution(int *in,int *mask,int *res){
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int acc=0,j;
    
    if(n<RES_SIZE){
        for(j=0;j<MASK_SIZE;j++){
            if(n-j>=0 && n-j<INPUT_SIZE)
                acc+=in[n-j]*mask[j];
        }
        res[n]=acc;
    }
}
int main(){
    /*
    Allocate memory for input and convolution mask
    */
    int *input=(int*)malloc(INPUT_SIZE*sizeof(int)), 
        *mask=(int*)malloc(MASK_SIZE*sizeof(int)),
        *res=(int*)malloc(RES_SIZE*sizeof(int));
    int i;
    /*
    Declare pointer for device memory
    */
    int *gpu_in,*gpu_mask,*gpu_res;
    /*
    Allocate memory on the device
    */
    cudaMalloc((void**)&gpu_in,INPUT_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_mask,MASK_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_res,RES_SIZE*sizeof(int));
    /*
    Fill data arrays with integer values
    */
    for(i=0;i<INPUT_SIZE;i++){input[i]=1;}
    for(i=0;i<MASK_SIZE;i++){mask[i]=1;}

    /*
    Print data arrays
    */
    printf("Input array\n");
    for(i=0;i<INPUT_SIZE;i++){printf("%d ",input[i]);}
    printf("\n");

    printf("Convolution mask\n");
    for(i=0;i<MASK_SIZE;i++){printf("%d ",mask[i]);}
    printf("\n");
    /*
    Copy data to device memory
    */
    cudaMemcpy(gpu_in,input,INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mask,mask,MASK_SIZE*sizeof(int),cudaMemcpyHostToDevice);

    /*
    Call CUDA kernel with params
    */
    convolution<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_in,gpu_mask,gpu_res);
    cudaDeviceSynchronize();
    /*
    Copy the result from device to host memory
    */
    printf("Convolution mask: %d\n",MASK_SIZE);
    printf("Input array size: %d\n",INPUT_SIZE);
    printf("Result array size: %d\n",RES_SIZE);
    printf("Number of blocks: %d\n",N_BLOCKS);
    
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    /*
    Display result
    */
    printf("Result\n\n");
    for(i=0;i<RES_SIZE;i++){printf("%d ",res[i]);}
    printf("\n");

    /*
    Free devcie memory
    */
    cudaFree(gpu_in);
    cudaFree(gpu_mask);
    cudaFree(gpu_res);
    /*
    Free host memory
    */
    free(input);
    free(mask);

}