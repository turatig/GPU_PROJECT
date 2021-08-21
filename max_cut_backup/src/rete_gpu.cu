/*
Cuda kernel and functions used to optimize a Hopefield network while solving the max_cut problem
*/

/*
Update the status of the i-th node in the graph by summing
*/
#include <stdio.h>
#include <iostream>
#include "../inc/Graph.cuh"


#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void statusUpdate(int *adjlist,int *status,int size,int *res){
    /*declare smem to have a size equal to the double of the max number of threads per block*/
    __shared__ int smem[2048];
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    
    if(n<size)  smem[threadIdx.x]=-(adjlist[n]*status[n]);
    else smem[threadIdx.x]=0;

    __syncthreads();
    
    for(int i=2;i<=blockDim.x;i*=2){
        if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
        __syncthreads();
    }

    if(threadIdx.x==0) res[blockIdx.x]=smem[0];
}

/*  -txb: threads_per_block
*/
int *stabilizeHopfieldNet(Graph g,int txb){
    int *status;
    int *res;
    int n,prev;
    
    std::cout<<"Graph obj is\n";
    g.print();

    gpuErrCheck(cudaMallocManaged((void**)&status,g.getSize()*sizeof(int)));
    for(int i=0;i<g.getSize();i++) status[i]=0;
    bool end=false;
    
    int n_blocks=(int)(g.getSize()+txb-1)/txb;
    gpuErrCheck(cudaMallocManaged((void**)&res,n_blocks*sizeof(int)));
    int count=0;

    while(!end/*&&count<2*/){
        end=true;
        for(int i=0;i<g.getSize();i++){
            
           /* std::cout<<"Adjacency list of the node "<<i<<"\n";
            for(int k=0;k<g.getSize();k++) std::cout<<g.getAdjmat()[i][k]<<" ";
            std::cout<<"\n";
            std::cout<<"Status of the network\n";
            for(int k=0;k<g.getSize();k++)std::cout<<status[k]<<" ";
            std::cout<<"\n";
            for(int k=0;k<n_blocks;k++) res[k]=0;
            for(int k=0;k<n_blocks;k++) std::cout<<res[k]<<" ";
            std::cout<<"\n";*/

            /*Compute sum reduction on device*/
            //std::cout<<"Calling kernel\n";
            statusUpdate<<<n_blocks,txb>>>(g.getAdjmat()[i],status,g.getSize(),res);
            gpuErrCheck(cudaDeviceSynchronize());
            //std::cout<<"Kernel terminated successfully\n";
            
            /*Sum partial results together*/
            n=0;
            /*std::cout<<"n_blocks "<<n_blocks<<"\n";
            std::cout<<"Sum "<<res[0]<<"\n";*/
            for(int j=0;j<n_blocks;j++) n+=res[j];
            //std::cout<<"Sum="<<n<<"\n";
            prev=status[i];

            if(n<0) status[i]=-1;
            else status[i]=1;

            if(status[i]!=prev) end=false;
        }
        /*std::cout<<"-------ITERATION "<<count<<"-------\n";
        std::cout<<"End of the cycle: "<<end<<"\n";
        count+=1;*/
    }
    
    return status;
}
