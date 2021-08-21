#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define A_COL 7
#define A_ROW 2
#define B_COL 12
#define B_ROW 7
/*
Must be a power of 2
*/
#define THREADS_PER_BLOCK 4


__global__ void matProd(int *a,int *b,int *res){
    int colIdx=threadIdx.x+blockDim.x*blockIdx.x;
    int rowIdx=threadIdx.y+blockDim.y*blockIdx.y;
    int i;
    /*
    If the index are in the range of the indexes of the output matrix
    */
    if(colIdx<B_COL && rowIdx<A_ROW){
        /*
        Compute one element of the output matrix
        */
        res[rowIdx*B_COL+colIdx]=0;
        for(i=0;i<A_COL;i++){
            res[rowIdx*B_COL+colIdx]+=a[rowIdx*A_COL+i]*b[i*B_COL+colIdx];
        }

    }
}
int main(){
    bool log=true;
    /*
    Allocate CPU memory
    */
    int *a=(int*)malloc(A_COL*A_ROW*sizeof(int)),
        *b=(int*)malloc(B_COL*B_ROW*sizeof(int)),
        *res=(int*)malloc(A_ROW*B_COL*sizeof(int));
    /*
    Allocate GPU memory
    */
    int *gpu_a,*gpu_b,*gpu_res;
    cudaMalloc((void**)&gpu_a,A_COL*A_ROW*sizeof(int));
    cudaMalloc((void**)&gpu_b,B_COL*B_ROW*sizeof(int));
    cudaMalloc((void**)&gpu_res,A_ROW*B_COL*sizeof(int));
    /*
    Fill with random data
    */
    int i,j;
    for(i=0;i<A_ROW*A_COL;i++){a[i]=1/*rand()%5*/;}
    for(i=0;i<B_ROW*B_COL;i++){b[i]=1/*rand()%5*/;}
    
    if(log){
        printf("A:\n");
        for(i=0;i<A_ROW;i++){
            for(j=0;j<A_COL;j++)
                printf("%d ",a[i*A_COL+j]);
            printf("\n");
        }
        printf("\nB:\n");
        for(i=0;i<B_ROW;i++){
            for(j=0;j<B_COL;j++)
                printf("%d ",b[i*B_COL+j]);
            printf("\n");
        }
    }
    /*
    Copy data to GPU memory
    */
    cudaMemcpy(gpu_a,a,A_ROW*A_COL*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b,b,B_ROW*B_COL*sizeof(int),cudaMemcpyHostToDevice);
    /*
    Setting block dim: square block of size THREADS_PER_BLOCK^(1/2) x THREADS_PER_BLOCK^(1/2)
    */
    int blockSide=(int)sqrt(THREADS_PER_BLOCK);
    dim3 blockSize=dim3(blockSide,blockSide);
    /*
    The grid must cover the size of the output matrix grid (x_ax_dimension,y_ax_dimension)
    */
    dim3 gridSize=dim3((B_COL+blockSide-1)/blockSide,(A_ROW+blockSide-1)/blockSide);
    /*
    Invoke kernel
    */
    matProd<<<gridSize,blockSize>>>(gpu_a,gpu_b,gpu_res);
    /*
    Copy data from device to host
    */
    cudaMemcpy(res,gpu_res,A_ROW*B_COL*sizeof(int),cudaMemcpyDeviceToHost);
    if(log){
        printf("\n\n\nA*B:\n");
        for(i=0;i<A_ROW;i++){
            for(j=0;j<B_COL;j++)
                printf("%d ",res[i*B_COL+j]);
            printf("\n");
        }
    }
    /*
    Free device memory
    */
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_res);
    /*
    Free host memory
    */
    free(a);
    free(b);
    free(res);


}