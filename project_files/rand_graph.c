/* 
Generatore di grafi p-random

- compilare con:   cc -o rndgraph rndgraph.c -lm
- eseguire con:    rndgraph n m

  dove:
       - n         indica il size del grafo
       - m=p*100   (0 < m < 100) indica la % di nodi

- produce il file (ASCII) "graph" con il seg. formato:
   
  I riga:     un intero che indica la dimensione del grafo
  righe seg.: la matrice di adiacenza del grafo

- Scrive su stdout il risltato di Matula 

ESEMPIO:
6
011001	 
101110
110100
011000
010001
100010
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

FILE *out, *csv;

int N,P;
typedef int *Row;
static Row *adjmat;

double F(double);

double F(double x) {
  return x*log(x)-x+0.5*log(x)+log(sqrt(6.28));
}



int main(int argc, char *argv[]) {
  int i,j;
  double p,n,k;
  
  srand(time(NULL)&32767);
  N = atoi(argv[1]);
  P = atoi(argv[2]);

  adjmat = (Row *) malloc(N*sizeof(Row));

  for (i=0;i<N;i++) 
    adjmat[i] = (int *) calloc(N,sizeof(int));
  
  for (i=0; i<N-1; i++)
    for (j=i+1; j<N; j++)
      if ( (rand() % 100) > P )
        adjmat[i][j] = adjmat[j][i] = 0;
      else 
        adjmat[i][j] = adjmat[j][i] = 1;

  out = fopen("graph.txt","w");
  fputs(argv[1], out);
  fputc('\n', out);
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++)
      fputc(adjmat[i][j]+48,out);
    fputc('\n',out);
   }
   fclose(out);

  csv = fopen("graph.csv","w");
  fputs(argv[1], csv);
  fputc('\n', csv);
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      fputc(adjmat[i][j]+48,csv);
      if (j<N-1)
        fputc(',',csv);
    }
    fputc('\n',csv);
   }
   fclose(csv);



  /*calcolo del valore atteso della massima clique utilizzando la
    formula di Matula */

   n = (double) N;
   p = (double) P/100;     

   k = 1;
   while (F(n)-F(k)-F(n-k) > (k*(k-1)/2)*log(1/p))
     k = k+0.001; 
   printf("MATULA -> %.2f\n",k);

   return(0);
 }
