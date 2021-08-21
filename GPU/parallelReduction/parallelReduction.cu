#include<stdio.h>
#include<stdlib.h>

#define N 1024
#define threadPerBlock 256

__global__ void summation(int *in,int *out,int length){
	//One block reduce in from blockidx.x*blockdim.x*2 to blockidx.x*blockdim.x*2+blockdim.x*2
	int offset=blockIdx.x*blockDim.x*2;
	//if(>length/2) return;

	//<<2==*2
	for(int stride=1;stride<=blockDim.x;stride*=2){
		if( threadIdx.x < blockDim.x/stride)
			in[offset+threadIdx.x]+=in[offset+(blockDim.x/stride)+threadIdx.x ];
		__syncthreads();
	}
	out[blockIdx.x]=in[offset];:

//	printf("%d + %d\n",in[tid],in[tid+blockDim.x]);
}



void sumCPU(int *v,int length){
	int res=0;
	int i;
	
	for(i=0;i<length;i++){
		res+=v[i];
	}
	printf("Sum on CPU  %d\n",res);
}

int main(){
	int *vect=(int *)malloc(sizeof(int)*N),*gpuVect,*gpuRes,*res;
	int i;
	int numBlock=(N/2+threadPerBlock-1) / threadPerBlock;
	
	printf("%d\n",numBlock);
	for(i=0;i<N;i++){
		vect[i]=1;
		if(i==72)
			vect[i]+=2;
	}

	sumCPU(vect,N);

	cudaMalloc((void **)&gpuVect,sizeof(int)*N);
	cudaMalloc((void **)&gpuRes,sizeof(int)*numBlock );
	res=(int *)malloc(sizeof(int)*numBlock);

	cudaMemcpy(gpuVect,vect,sizeof(int)*N,cudaMemcpyHostToDevice);
	
	summation<<<numBlock,threadPerBlock>>>(gpuVect,gpuRes,N);
	cudaMemcpy(res,gpuRes,sizeof(int)*numBlock,cudaMemcpyDeviceToHost);
	cudaFree(gpuRes);
	cudaFree(gpuVect);
	
	int r=0;

	for(i=0;i<numBlock;i++)
		r+=res[i];

	printf("Sum on GPU %d\n",r);
	free(vect);
	free(res);
}
	

