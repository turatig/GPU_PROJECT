#include<stdlib.h>
#include<stdio.h>

//Efficient implementation for matrix multiplication
//Constraint:
//-Matrices must be block diagonal
//Efficency target:
//-Allocated memory
//-Computational time

typedef struct{
	int *values;
	int order;
}SqMat;

SqMat *newSqMat(int order){
	SqMat *m=malloc(sizeof(SqMat));
	m->order=order;
	m->values=malloc(order*order*sizeof(int));
	return m;
}

int getMemIdx(SqMat *m,int i,int j){
	//Index out of bound
	if(i<0||j<0||i>=m->order||j>=m->order) return -1;
	return m->order*i+j;
}

SqMat *zeroSqMat(int order){
	int i;
	SqMat *m=newSqMat(order);
	for(i=0;i<m->order*m->order;i++)m->values[i]=0;
	return m;
}
	
SqMat *randomSqMat(int order){
	int i;
	SqMat *m=newSqMat(order);
	for(i=0;i<m->order*m->order;i++)m->values[i]=rand()%10;
	return m;
}

//Product between square matrices
SqMat *SqProduct(SqMat *m1,SqMat *m2){
	int i,j,k;
	SqMat *m;
	if(m1->order!=m2->order) return NULL;
	m=zeroSqMat(m1->order);
	for(i=0;i<m1->order;i++){
		for(j=0;j<m1->order;j++){
			for(k=0;k<m1->order;k++)
				m->values[getMemIdx(m,i,j)]+=m1->values[getMemIdx(m1,i,k)]*m2->values[getMemIdx(m2,k,j)];
		}
	}
	return m;
}

void printSqMat(SqMat *m){
	int i,j;
	for(i=0;i<m->order;i++){
		for(j=0;j<m->order;j++)
			printf("%d",m->values[getMemIdx(m,i,j)]);
		printf("\n");
	}
}

//Block diagonal matrix
//Only stores diagonal square matrices
//Matrices are indexed by blocks for space efficency
typedef struct{
	SqMat **blocks;
	int nblocks;
	int order;
}BdMat;

BdMat *newBdMat(int nblocks){
	BdMat *m=malloc(sizeof(BdMat));
	int i;
	m->nblocks=nblocks;
	m->blocks=malloc(nblocks*sizeof(SqMat *));
	m->order=0;
	return m;
}

BdMat *randomBdMat(int nblocks,int *sizeList){
	BdMat *m=newBdMat(nblocks);
	int i;
	for(i=0;i<nblocks;i++){
		m->blocks[i]=randomSqMat(sizeList[i]);
		m->order+=m->blocks[i]->order;
	}
	return m;
}

BdMat *emptyBdMat(int nblocks,int *sizeList){
	BdMat *m=newBdMat(nblocks);
	int i;
	for(i=0;i<nblocks;i++){
		m->blocks[i]=newSqMat(sizeList[i]);
		m->order+=m->blocks[i]->order;
	}
	return m;
}
	
//Optimized version for block diagonal matrix
BdMat *BdProduct(BdMat *m1,BdMat *m2){
	int i;
	int *sizeList;
	BdMat *m;
	//Checking the two matrices have the same number of block
	if(m1->nblocks!=m2->nblocks) return NULL;
	sizeList=malloc(m1->nblocks*sizeof(int));
	//Checking the two matrices have blocks of the same size
	for(i=0;i<m1->nblocks;i++){
		if(m1->blocks[i]->order!=m2->blocks[i]->order) return NULL;
	}
	m=newBdMat(m1->nblocks);
	for(i=0;i<m1->nblocks;i++){
		m->order+=m1->blocks[i]->order;
		m->blocks[i]=SqProduct(m1->blocks[i],m2->blocks[i]);
	}
	return m;
}

void printBdMat(BdMat *m){
	int i,j,curblock=0,offset=0;
	printf("Block diagonal matrix of order %d\n",m->order);
	for(i=0;i<m->order;i++){
		for(j=0;j<m->order;j++){
			if(j<offset||j>=offset+m->blocks[curblock]->order){printf(" ");}
			else{printf("%d",m->blocks[curblock]->values[getMemIdx(m->blocks[curblock],i-offset,j-offset)]);}
		}
		printf("\n");
		if(i+1==offset+m->blocks[curblock]->order){
			//printf("Block number %d of order %d\n",curblock,m->blocks[curblock]->order);
			offset+=m->blocks[curblock]->order;
			curblock++;
		}
	}
}
	
int main(){
	int sizeList[]={4,3,2,1};
	BdMat *m1=randomBdMat(4,sizeList);
	BdMat *m2=randomBdMat(4,sizeList);
	printBdMat(m1);
	printBdMat(m2);
	printBdMat(BdProduct(m1,m2));
}
