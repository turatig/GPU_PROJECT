/*
 * imFlip.cu
 *
 *  Created on: 29/mar/2020
 *      Author: jack
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imgStuff.h"

//Function performing vertical flip.
//Every thread switch one pixel. Matrix m and res have linearized indexes
__global__ void VFlip(unsigned char *m,unsigned char  *res,int VPixel,int HPixel){

	//Linearearized index in matrix of pixels (3 bytes) depending on linear thread indexes
	int t=(blockDim.x*blockIdx.x+threadIdx.x);
	
	//Computing the number of padding bytes per row according to  bmp format
	int pad=((HPixel*3+3)&~3)-HPixel*3;
	int rowIdx=t/HPixel;
	int colIdx=t%HPixel;

	//First index of symmetric row in linearized mat
	int symmetricRow=((VPixel-1) * HPixel*3) - (rowIdx * HPixel*3);

	if(t*3<VPixel*HPixel*3){
		res[ symmetricRow + colIdx*3 + (VPixel-rowIdx-1)*pad ]=m[ t*3 + rowIdx*pad ];
		res[ symmetricRow + colIdx*3 + (VPixel-rowIdx-1)*pad + 1 ]=m[ t*3 + rowIdx*pad + 1 ];
		res[ symmetricRow + colIdx*3 + (VPixel-rowIdx-1)*pad + 2 ]=m[ t*3 + rowIdx*pad + 2 ];
	}

}

//Function performing horizontal flip.
__global__ void HFlip(unsigned char *m,unsigned char  *res,int VPixel,int HPixel){

	int t=(blockDim.x*blockIdx.x+threadIdx.x);
	
	//Computing the number of padding bytes per row according to  bmp format
	int pad=((HPixel*3+3)&~3)-HPixel*3;

	int rowIdx=t/HPixel;
	int colIdx=t%HPixel;

	int symmetricCol=(HPixel*3)-colIdx*3;

	if(t*3<VPixel*HPixel*3){
		res[ rowIdx*HPixel*3 + rowIdx*pad + symmetricCol - 3 ]=m[ t*3 + rowIdx*pad ];
		res[ rowIdx*HPixel*3 + rowIdx*pad + symmetricCol - 2 ]=m[ t*3 + rowIdx*pad + 1 ];
		res[ rowIdx*HPixel*3 + rowIdx*pad + symmetricCol - 1 ]=m[ t*3 + rowIdx*pad + 2 ];
	}

}

int main(int argc,char **argv){
	int threadPerBlock=32,dimGrid;
	unsigned char *GPUImgRes,*GPUImg;
	
	//Creates test bmp img
	//randFourSquares("test1.bmp",480,640);

	//Img properties obj
	ImgProp *ip=(ImgProp *)malloc(sizeof(ImgProp));
	unsigned char *mat=ReadBMP("dog.bmp",ip);
	
	/*printf("Ya\n");
	int i,j
	for(i=0;i<ip->VPixel;i++){
		for(j=0;j<ip->HPixel*3;j+=3){
			printf("* %d %d %d *",mat[i*ip->HBytes+j],mat[i*ip->HBytes+j+1],mat[i*ip->HBytes+j+2]);
		}
		for(j=ip->HPixel*3;j<ip->HBytes;j++){
			printf(" %d ",mat[i* ip->HBytes+j]);
		}
		printf("\n");
	}*/

	//Number of blocks. Every thread switch one pixel
	dimGrid=(ip->HPixel*ip->VPixel + threadPerBlock - 1)/threadPerBlock;

	//Arguments in cuda are passed by reference
	cudaMalloc((void **)&GPUImgRes,sizeof(unsigned char)*ip->HBytes*ip->VPixel);
	cudaMalloc((void **)&GPUImg,sizeof(unsigned char)*ip->HBytes*ip->VPixel);
	
	cudaMemcpy(GPUImg,mat,ip->HBytes*ip->VPixel,cudaMemcpyHostToDevice);
	cudaMemcpy(GPUImgRes,mat,ip->HBytes*ip->VPixel,cudaMemcpyHostToDevice);

	VFlip<<<dimGrid,threadPerBlock>>>(GPUImg,GPUImgRes,ip->VPixel,ip->HPixel);
	

	cudaMemcpy(mat,GPUImgRes,ip->HBytes*ip->VPixel,cudaMemcpyDeviceToHost);
	cudaFree(GPUImgRes);
	cudaFree(GPUImg);

	
	/*printf("Yo\n");
	for(i=0;i<ip->VPixel;i++){
		for(j=0;j<ip->HPixel*3;j+=3){
			printf("* %d %d %d *",mat[i*ip->HBytes+j],mat[i*ip->HBytes+j+1],mat[i*ip->HBytes+j+2]);
		}
		for(j=ip->HPixel*3;j<ip->HBytes;j++){
			printf(" %d ",mat[i* ip->HBytes+j]);
		}
		printf("\n");
	}*/


	WriteBMP(mat,"dogVFlip.bmp",ip);

	free(mat);
	free(ip->HeaderInfo);
	free(ip);

}


