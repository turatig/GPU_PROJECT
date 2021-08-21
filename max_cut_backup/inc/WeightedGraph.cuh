/*
Class to manage a random generated undirected weighted graph.
*/
#ifndef W_GRAPH
#define W_GRAPH
class WeightedGraph{
    private:
        /*  -adjmat: adjacency matrix. Allocated both on host and device.
            -size: number of nodes in the graph.
        */
        float **adjmat;
        int size;
    public:
        /*  -s: size.
            -p: fraction of connected nodes.
        */
        WeightedGraph(int s,int p=75);
        float **getAdjmat();
        int getSize();
        void print();
};
#endif
