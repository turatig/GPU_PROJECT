#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 65536
/*
N.B.: a proper mask size can show the efficiency of tiledConvolution respect to the others.
If mask size is too small divergence of threads in the could result in a cumbersome computation
*/
#define MASK_SIZE 64
#define RADIUS MASK_SIZE/2
#define RES_SIZE (MASK_SIZE+INPUT_SIZE-1)
#define THREAD_PER_BLOCK 64
#define TILE_SIZE THREAD_PER_BLOCK
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

void logResult(int *res);
/*
Compute the convolution between in and mask in the straightforward way
*/
__global__ void convolution_easy(int *in,int *mask,int *res){
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

/*
Compute the convolution between in and mask in the straightforward way (alternative version)
*/
__global__ void convolution_alternate(int *in,int *mask,int *res){
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int acc=0,j;

    int start_point=n-(MASK_SIZE/2);

    for(j=0;j<MASK_SIZE;j++){
        if(start_point+j>=0 && start_point+j<RES_SIZE){
            acc+=in[start_point+j]*mask[j];
        }
    }
    res[n]=acc;
}
/*
Declaring constant memory mask
*/
__constant__ int MASK[MASK_SIZE];

/*
Tiled convolution
N.B: input array must be padded of MASK_SIZE/2(RADIUS) to work
*/
__global__ void tiledConvolution(int *in,int *res){

    int n=blockDim.x*blockIdx.x+threadIdx.x;
    /*
    The effective size of the tile is the tile itself+the halo (2*RADIUS)
    */
    __shared__ int tile[TILE_SIZE+MASK_SIZE];

    if(n<RES_SIZE){
        res[n]=0;
        /*
        Loading into shared memory left halo
        */
        if(threadIdx.x<RADIUS)
            /*If at the start of the input, left halo is zero-padding*/
            tile[threadIdx.x]= blockIdx.x==0 ? 0 : in[n-RADIUS];
        /*
        Loading into shared memory right halo
        */
        if(threadIdx.x+RADIUS>=TILE_SIZE)
            /*If at the end of the input, right halo is zero-padding*/
            tile[threadIdx.x+RADIUS*2]= n>=INPUT_SIZE ? 0 : in[n+RADIUS];

        /*
        Every thread loads at least this single data
        */
        tile[threadIdx.x+RADIUS]=in[n];
        /*
        Synchronize all the threads in the block
        */
        __syncthreads();
        /*
        Compute the convolution output element of which the thread is responsible
        */
        for(int i=0; i<MASK_SIZE; i++)
            res[n]+=MASK[i]*tile[threadIdx.x+i];
    }
}
/*
Tiled convolution alternate version
*/
__global__ void convolution1D(int *result, int *data, int n) {

	// shared memory size = TILE + MASK
	__shared__ float tile[TILE_SIZE+MASK_SIZE];

	// edges
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int left = blockIdx.x * blockDim.x - RADIUS;
	int right = (blockIdx.x + 1) * blockDim.x;

	if (threadIdx.x < RADIUS)                      // left
		tile[threadIdx.x] = left < 0 ? 0 : data[left + threadIdx.x];
	else if (threadIdx.x >= blockDim.x - RADIUS)  // right
		tile[threadIdx.x + MASK_SIZE - 1] = right >= n ? 0 : data[right + threadIdx.x - blockDim.x + RADIUS];

	// center
	tile[threadIdx.x + RADIUS] = data[idx];

	__syncthreads();

	// convoluzione: tile * mask
	int sum = 0;
	for (int i = -RADIUS; i <= RADIUS; i++)
		sum += tile[threadIdx.x + RADIUS + i] * MASK[i + RADIUS];

	// store conv result
	result[idx] = sum;
}
int main(){
    /*
    Allocate memory for input and convolution mask
    */
    int *input=(int*)malloc(INPUT_SIZE*sizeof(int)),
        /*
        This is to have correct output from tiled convolution
        */
        *padded_in=(int*)malloc( (INPUT_SIZE+RADIUS)*sizeof(int) ), 
        *mask=(int*)malloc(MASK_SIZE*sizeof(int)),
        *res=(int*)malloc(RES_SIZE*sizeof(int));
    int i;
    bool log=false;
    /*
    Declare pointer for device memory
    */
    int *gpu_in,*gpu_padded,*gpu_mask,*gpu_res;
    /*
    Allocate memory on the device
    */
    cudaMalloc((void**)&gpu_in,INPUT_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_padded,(INPUT_SIZE+RADIUS)*sizeof(int));
    cudaMalloc((void**)&gpu_mask,MASK_SIZE*sizeof(int));
    cudaMalloc((void**)&gpu_res,RES_SIZE*sizeof(int));
    /*
    Fill data arrays with integer values
    */
    for(i=0;i<INPUT_SIZE;i++){input[i]=1;}
    for(i=0;i<INPUT_SIZE+RADIUS;i++){padded_in[i]= i<RADIUS ? 0:1;}
    for(i=0;i<MASK_SIZE;i++){mask[i]=1;}

    /*
    Print data arrays
    */
    if(log){
        printf("Input array\n");
        for(i=0;i<INPUT_SIZE;i++){printf("%d ",input[i]);}
        printf("\n");

        printf("Convolution mask\n");
        for(i=0;i<MASK_SIZE;i++){printf("%d ",mask[i]);}
        printf("\n");
    }
    /*
    Copy data to device memory
    */
    cudaMemcpy(gpu_in,input,INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mask,mask,MASK_SIZE*sizeof(int),cudaMemcpyHostToDevice);

    /*
    Call CUDA kernel to perform the convolution product in the straightforward way
    */
    convolution_easy<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_in,gpu_mask,gpu_res);
    /*
    Copy the result from device to host memory
    */
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    if(log)
        logResult(res);

    /*
    Copy padded input
    */
    cudaMemcpy(gpu_padded,padded_in,(INPUT_SIZE+RADIUS)*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Test convolution results calling the alternative kernel
    */
    convolution_alternate<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_padded,gpu_mask,gpu_res);
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    if(log)
        logResult(res);
    /*
    Test convolution results calling the alternative kernel
    */
    convolution_alternate<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_padded,gpu_mask,gpu_res);
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    /*
    Copy the mask from host array to the constant memory
    */
    cudaMemcpyToSymbol(MASK,mask,MASK_SIZE*sizeof(int));

    /*
    Call the CUDA kernel to compute tiled version of convolution
    */
    tiledConvolution<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_padded,gpu_res);
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    if(log)
        logResult(res);
    /*
    Call the CUDA kernel to compute tiled version of convolution
    */
    convolution1D<<<N_BLOCKS,THREAD_PER_BLOCK>>>(gpu_res,gpu_padded,INPUT_SIZE+RADIUS);
    cudaMemcpy(res,gpu_res,RES_SIZE*sizeof(int),cudaMemcpyDeviceToHost);
    if(log)
        logResult(res);    

    /*
    Free devcie memory
    */
    cudaFree(gpu_in);
    cudaFree(gpu_padded);
    cudaFree(gpu_mask);
    cudaFree(gpu_res);
    /*
    Free host memory
    */
    free(input);
    free(padded_in);
    free(mask);

}

void logResult(int *res){

    int i;
    printf("Convolution mask: %d\n",MASK_SIZE);
    printf("Input array size: %d\n",INPUT_SIZE);
    printf("Result array size: %d\n",RES_SIZE);
    printf("Number of blocks: %d\n",N_BLOCKS);

    /*
    Display result
    */
    printf("Result\n\n");
    for(i=0;i<RES_SIZE;i++){printf("%d ",res[i]);}
    printf("\n");

}