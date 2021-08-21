#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../inc/graph_gen.h"
#include "../inc/rete.h"
#include "../inc/rete_gpu.cuh"
#include "../inc/Graph.cuh"

/*
Function used to check the correctness of the parallel algorithm with respect to its sequential implementation
*/
bool check_output(int *seq_partitions,int *par_partitions,int size){
    for(int i=0;i<size;i++){
        if(par_partitions[i]!=seq_partitions[i]) return false;
    }
    return true;
}

int main(){
    /*Logging gpu infos*/
    /*int dev;
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props,dev);
    std::cout<<"Compute capability "<<props.major<<"."<<props.minor<<"\n";*/

    Graph g=Graph("graph.txt");
    inizializza_strutture(g.getAdjmat(),g.getSize());
    stampa_Adjmat();
    stabilizza_rete_Hopfield();
    stampa_stato_rete();
    
    std::cout<<"GPU is starting to compute\n";
    int *status=stabilizeHopfieldNet(g);
    std::cout<<"GPU finished\n";
    for(int i=0;i<g.getSize();i++)
        std::cout<<(int)((status[i]+1)/2)<<" ";

    std::cout<<"\n";
    if(check_output(get_stato_rete(),status,g.getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";
}
