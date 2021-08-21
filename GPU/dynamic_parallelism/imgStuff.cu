/*
 * Same version of image flip CPU but allocates linearized mat
 * imageStuff.c
 *
 *  Created on: 31/mar/2020
 *      Author: jack
 */


/*
 * imageStuff.c
 *
 *  Created on: 26/mar/2020
 *      Author: jack
 */
#include "imgStuff.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<dirent.h>

/*Start byte of header field. Type of header field specified in comment*/
/*CHAR[2]*/
#define SIGNATURE (0)
/*INT*/
#define FILE_SIZE (2)
/*INT*/
#define RESERVED (6)
/*INT*/
#define DATA_OFFSET (10)
/*INT*/
#define DIB_SIZE (14)
/*INT*/
#define HPIXEL (18)
/*INT*/
#define VPIXEL (22)
/*SHORT: always 1*/
#define PLANES (26)
/*SHORT*/
#define BIT_PER_PIXEL (28)
/*INT*/
#define COMP_TYPE (30)
/*INT*/
#define IMG_SIZE (34)
/*INT*/
#define HPM (38)
/*INT*/
#define VPM (42)
/*INT*/
#define PALETTE (46)
/*INT*/
#define IMPORTANT (50)

void test(){printf("Ciao\n");}

/*Debug function*/
void printProperties(ImgProp *ip){
	printf("Signature : %c%c\n",ip->HeaderInfo[0],ip->HeaderInfo[1]);

	printf("File size : %d\n",*(int *)&ip->HeaderInfo[FILE_SIZE]);
	printf("Reserved : %d\n",*(int *)&ip->HeaderInfo[RESERVED]);
	printf("Data offset : %d\n",*(int *)&ip->HeaderInfo[DATA_OFFSET]);
	printf("Information header size : %d\n",*(int *)&ip->HeaderInfo[DIB_SIZE]);
	printf("HPixel : %d\n",*(int *)&ip->HeaderInfo[HPIXEL]);
	printf("VPixel : %d\n",*(int *)&ip->HeaderInfo[VPIXEL]);

	printf("Planes : %hi\n",*(short *)&ip->HeaderInfo[PLANES]);
	printf("Bit per pixel : %hi\n",*(short *)&ip->HeaderInfo[BIT_PER_PIXEL]);

	printf("Compression : %d\n",*(int *)&ip->HeaderInfo[COMP_TYPE]);
	printf("Raw data size : %d\n",*(int *)&ip->HeaderInfo[IMG_SIZE]);
	printf("HPM : %d\n",*(int *)&ip->HeaderInfo[HPM]);
	printf("VPM : %d\n",*(int *)&ip->HeaderInfo[VPM]);
	printf("Palette : %d\n",*(int *)&ip->HeaderInfo[PALETTE]);
	printf("Important : %d\n",*(int *)&ip->HeaderInfo[IMPORTANT]);

}
void WriteBMP(unsigned char * mat,char *filename,ImgProp *ip){
	FILE *fp;
	int i,j,headSz;
	/*b for writing in binary mode*/
	fp=fopen(filename,"wb");
	if(fp==NULL){
		printf("Error while trying to open output file %s\n",filename);
		exit(1);
	}

	headSz=*(int *)&ip->HeaderInfo[10];

	printProperties(ip);

	/*Writing header info*/
	for(i=0;i<headSz;i++) {fputc(ip->HeaderInfo[i],fp);}

	for(i=0;i<ip->VPixel;i++){
		/*3 bytes per pixel (R,G.B) rounded up to the nearest multiple of 4. HBytes=(3*HPixel+3)&(~3)*/
		for(j=0;j<ip->HBytes;j++){fputc(mat[i*ip->HBytes+j],fp);}
	}
	printf("Output file %s successfully created\n",filename);
	fclose(fp);

}

ImgProp *trueColorHeader(int VPixel,int HPixel){

	ImgProp *ip=(ImgProp *)malloc(sizeof(ImgProp));
	/*38 bytes header*/
	ip->HeaderInfo=(unsigned char*)malloc(sizeof(unsigned char)*54);
	ip->VPixel=VPixel;
	ip->HPixel=HPixel;
	/*N.B: the number of bytes per row must be multiple of 4*/
	ip->HBytes=(3*HPixel+3)&~3;

	/*FIELDS OF THE BMP FORMAT*/
	/*File header*/
	unsigned char sig[]={'B','M'};
	int totalSize,reserved=0,dataOffset=54;
	totalSize=dataOffset+VPixel*ip->HBytes;

	/*Information header*/
	int DIBSize=40;
	short planes=1,bpp=24;
	int compType=0;
	int imgSize=ip->VPixel*ip->HBytes,hpm=0,vpm=0,palette=0,important=0;

	memcpy((void *)&ip->HeaderInfo[SIGNATURE],(void *)sig,2);

	memcpy((void *)&ip->HeaderInfo[FILE_SIZE],(void *)&totalSize,4);
	memcpy((void *)&ip->HeaderInfo[RESERVED],(void *)&reserved,4);
	memcpy((void *)&ip->HeaderInfo[DATA_OFFSET],(void *)&dataOffset,4);
	memcpy((void *)&ip->HeaderInfo[DIB_SIZE],(void *)&DIBSize,4);
	memcpy((void *)&ip->HeaderInfo[HPIXEL],(void *)&ip->HPixel,4);
	memcpy((void *)&ip->HeaderInfo[VPIXEL],(void *)&ip->VPixel,4);

	memcpy((void *)&ip->HeaderInfo[PLANES],(void *)&planes,2);
	memcpy((void *)&ip->HeaderInfo[BIT_PER_PIXEL],(void *)&bpp,2);

	memcpy((void *)&ip->HeaderInfo[COMP_TYPE],(void *)&compType,4);
	memcpy((void *)&ip->HeaderInfo[IMG_SIZE],(void *)&imgSize,4);

	memcpy((void *)&ip->HeaderInfo[HPM],(void *)&hpm,4);
	memcpy((void *)&ip->HeaderInfo[VPM],(void *)&vpm,4);
	memcpy((void *)&ip->HeaderInfo[PALETTE],(void *)&palette,4);
	memcpy((void *)&ip->HeaderInfo[IMPORTANT],(void *)&important,4);


	return ip;
}

unsigned char *ReadBMP(char *filename,ImgProp *ip){
	FILE *fp;
	int i,headSz;
	unsigned char *mat;

	fp=fopen(filename,"rb");

	if(fp==NULL){
		printf("Error while trying to open input file %s\n",filename);
		exit(1);
	}
	/*Reading header size in int. Header bytes [10-13]*/
	fseek(fp,10,SEEK_SET);
	fread(&headSz,sizeof(int),1,fp);
	rewind(fp);
	/*Allocating space for header in image properties structure*/
	ip->HeaderInfo=(unsigned char *)malloc(sizeof(unsigned char)*headSz);
	/*Filling imageProperties data structure*/
	fread(ip->HeaderInfo,sizeof(unsigned char),headSz,fp);


	ip->HPixel=*(int *)&ip->HeaderInfo[HPIXEL];
	ip->VPixel=*(int *)&ip->HeaderInfo[VPIXEL];

	/*Number of bytes per each row: 3 bytes per pixel rounded up to the nearest multiple of 4*/
	/*N.B: valid only for true color images*/
	ip->HBytes=(3*ip->HPixel+3)&(~3);

	printProperties(ip);

	/*Allocating matrix to be returned*/
	mat=(unsigned char*)malloc(sizeof(unsigned char )*ip->HBytes*ip->VPixel);

	for(i=0;i<ip->VPixel;i++){
		fread(&mat[i*ip->HBytes],sizeof(unsigned char),ip->HBytes,fp);
	}
	printf("Input file %s successfully read\n",filename);
	fclose(fp);
	return mat;

}

/*Get a BMP image with four squares of random color*/
/*Images are true color uncompressed*/
void randFourSquares(char *filename,int VPixel,int HPixel){

	ImgProp *ip=trueColorHeader(VPixel,HPixel);
	Pixel *pixels[4];
	int i,j;
/*	for(i=0;i<4;i++){
		pixels[i]=(Pixel *)malloc(sizeof(Pixel));
		pixels[i]->R=(unsigned char)(rand()%256);
		pixels[i]->G=(unsigned char)(rand()%256);
		pixels[i]->B=(unsigned char)(rand()%256);
		printf("Pixel %d : %d %d %d\n",i,pixels[i]->R,pixels[i]->G,pixels[i]->B);
	}
*/

	
	pixels[0]=(Pixel *)malloc(sizeof(Pixel));
	pixels[0]->R=(unsigned char)0;
	pixels[0]->G=(unsigned char)0;
	pixels[0]->B=(unsigned char)255;
	pixels[1]=(Pixel *)malloc(sizeof(Pixel));
	pixels[1]->R=(unsigned char)0;
	pixels[1]->G=(unsigned char)255;
	pixels[1]->B=(unsigned char)0;
	pixels[2]=(Pixel *)malloc(sizeof(Pixel));
	pixels[2]->R=(unsigned char)255;
	pixels[2]->G=(unsigned char)0;
	pixels[2]->B=(unsigned char)0;
	pixels[3]=(Pixel *)malloc(sizeof(Pixel));
	pixels[3]->R=(unsigned char)0;
	pixels[3]->G=(unsigned char)0;
	pixels[3]->B=(unsigned char)0;

	unsigned char *mat=(unsigned char*)malloc(sizeof(unsigned char )*ip->HBytes*ip->VPixel);

	for(i=0;i<ip->VPixel;i++){
		/*Creating four blocks of random colors*/

		for(j=0;j<ip->HPixel*3;j+=3){
			if(i<ip->VPixel/2 && j/3<ip->HPixel/2){
				mat[i*ip->HBytes+j]=pixels[0]->R;
				mat[i*ip->HBytes+j+1]=pixels[0]->G;
				mat[i*ip->HBytes+j+2]=pixels[0]->B;
			}
			else if(i<ip->VPixel/2 && j/3>=ip->HPixel/2){
				mat[i*ip->HBytes+j]=pixels[1]->R;
				mat[i*ip->HBytes+j+1]=pixels[1]->G;
				mat[i*ip->HBytes+j+2]=pixels[1]->B;
			}
			else if(i>=ip->VPixel/2 && j/3<ip->HPixel/2){
				mat[i*ip->HBytes+j]=pixels[2]->R;
				mat[i*ip->HBytes+j+1]=pixels[2]->G;
				mat[i*ip->HBytes+j+2]=pixels[2]->B;
			}
			else if(i>=ip->VPixel/2 && j/3>=ip->HPixel/2){
				mat[i*ip->HBytes+j]=pixels[3]->R;
				mat[i*ip->HBytes+j+1]=pixels[3]->G;
				mat[i*ip->HBytes+j+2]=pixels[3]->B;
			}
		}
		/*Filling padding to have rows of numBytes multiple of 4*/
		for(j=ip->HPixel*3;j<ip->HBytes;j++)
			mat[i*ip->HBytes+j]=(unsigned char)0;
	}

	WriteBMP(mat,filename,ip);

	/*Deallocate memory*/

	for(i=0;i<4;i++)
		free(pixels[i]);

	free(mat);
	free(ip->HeaderInfo);
	free(ip);

}



