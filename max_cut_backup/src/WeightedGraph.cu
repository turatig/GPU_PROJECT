#include<iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../inc/WeightedGraph.cuh"

WeightedGraph::WeightedGraph(int s,int p){
    size=s;   
    cudaMallocManaged(&adjmat,size*sizeof(float*));
    
    srand(time(NULL));
    for(int i=0;i<size;i++){

        cudaMallocManaged(&adjmat[i],size*sizeof(float));

        for(int j=0;j<size;j++){

            //Graph is supposed to be undirected
            if(j<i){ adjmat[i][j]=adjmat[j][i]; }

            else{

                if(rand()%100<p){  adjmat[i][j]=(float)rand()/(float)RAND_MAX; }
                else{ adjmat[i][j]=0; }

            }
        }
    }
}

float **WeightedGraph::getAdjmat(){ return adjmat;}
int WeightedGraph::getSize(){ return size;}

void WeightedGraph::print(){

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++)
            std::cout<<adjmat[i][j]<<" ";
        std::cout<<"\n";
    }
}
