#include<stdio.h>
#include<stdlib.h>
#include<string.h>

typedef struct{
	int num_blocks;
	int order;
	int *blk_size_list;
	int **blocks;
}BdMat;

BdMat *randBdMat(int num_blocks,int *blk_size_list);
void freeBdMat(BdMat *m);
void printBdMat(BdMat *m);

int main(){
	int d[]={4,3,1,2};
	BdMat *A=randBdMat(4,d);
	BdMat *B=randBdMat(4,d);
	printBdMat(A);
	freeBdMat(A);
}

int *randSqMat(int order){
	int *m=malloc(order*order*sizeof(int));
	int i;
	for(i=0;i<order*order;i++) m[i]=rand();
	return m;
}

BdMat *randBdMat(int num_blocks,int *blk_size_list){
	BdMat *m=malloc(sizeof(BdMat));
	m->num_blocks=num_blocks;
	memcpy(m->blk_size_list,blk_size_list,num_blocks*sizeof(int));
	m->blocks=malloc(num_blocks*sizeof(int *));
	int i;
	for(i=0;i<num_blocks;i++){
		m->blocks[i]=randSqMat(blk_size_list[i]);
		m->order+=m->blk_size_list[i];
	}
	return m;
}

void freeBdMat(BdMat *m){
	int i;
	for(i=0;i<m->num_blocks;i++)free(m->blocks[i]);
	free(m->blk_size_list);
	free(m);
}

void printBdMat(BdMat *m){
	int i,j,idx=0,offset=0;
	for(i=0;i<m->order;i++){
		if(i-offset==m->blk_size_list[idx]) offset+=m->blk_size_list[idx++];	
		for(j=0;j<m->order;j++){
			if(j<offset||j>=offset+m->blk_size_list[idx]) printf(" ");
			else{
				printf("%d",m->blocks[idx][m->blk_size_list[idx]*(i-offset)+j]);
			}
		}
	}
}
			
