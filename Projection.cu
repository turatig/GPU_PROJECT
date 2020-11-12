#include<stdio.h>
#include<math.h>
#define N 32

typedef struct{
    uint id;
    // Dimensionality of the point
    uint dim;
    float *point;
}Node;

Node nodeAlloc(int dim){
    static uint nextId=0; 
}

float getDistance(Node n,Node m){
    if((n.dim!=m.dim){return -1;}

}
int main(){
    int *AdjMat=(int*)malloc(size);
}