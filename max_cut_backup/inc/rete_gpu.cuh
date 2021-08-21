/*  -adjlist: adjacency list of one node
    -status: vector representing the status of the network: status[i]=0/1 -> node[i] belongs to partition 0/1
    -size: size of the graph
    -res: vector in which partial results of the sum reduction are stored
*/
#include "Graph.cuh"

void statusUpdate(int *adjlist,int *status,int size,int *res);
/*  -g: Graph object representing the graph
    -txb: threads_per_block
*/
int *stabilizeHopfieldNet(Graph g,int txb=256);
