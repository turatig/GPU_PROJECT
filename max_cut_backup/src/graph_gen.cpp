#include<iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

/*
Params: -size: number of nodes in the graph
        -p: fraction (p%) of nodes connected
*/
float **rndWeightGraph(int size,int p){
    float **adjmat=(float **)malloc(size*sizeof(float*));
    
    srand(time(NULL));
    for(int i=0;i<size;i++){
        adjmat[i]=(float *)malloc(size*sizeof(float));
        for(int j=0;j<size;j++){

            //Graph is supposed to be undirected
            if(j<i){ adjmat[i][j]=adjmat[j][i]; }
            else{
                if(rand()%100<p){  adjmat[i][j]=(float)rand()/(float)RAND_MAX; } 
            }
        }
    }
    return adjmat;
}

void printGraph(float **adjmat,int size){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++)
            cout<<adjmat[i][j]<<" ";
        cout<<"\n";
    }
}
            
