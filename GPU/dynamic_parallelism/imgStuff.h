/*
 * Same version of image flip CPU but allocates linearized mat
 * imageStuff.h
 *
 *  Created on: 31/mar/2020
 *      Author: jack
 */

#ifndef IMAGESTUFF_H_
#define IMAGESTUFF_H_
#define YA 1

typedef struct{
	int HPixel; /*Number of horizontal pixels*/
	int VPixel;	/*Number of vertical pixels*/
	unsigned char *HeaderInfo;/*Header bytes array*/
	unsigned long HBytes;/*Number of bytes per row rounded up to the nearest (higher) multiple of 4*/
}ImgProp;

typedef struct{
	unsigned char R;
	unsigned char G;
	unsigned char B;
}Pixel;

void test();
unsigned char *ReadBMP(char *,ImgProp *);/*Read a bmp image in arg into a matrix of unsigned char. arg[2] struct filled with image properties*/
void WriteBMP(unsigned char *,char *,ImgProp *);/*Ouput bmp file arg[2] from matrix arg[1]. arg[3] is for header info*/
void randFourSquares(char *,int,int);/*Create a bmp image of four random colored blocks*/
ImgProp *trueColorHeader(int,int);/*Return a header of a true color V_Pixel*H_pixel image*/




#endif /* IMAGESTUFF_H_ */
