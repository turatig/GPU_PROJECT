/**************************************************************

Algoritmo approssimato per la ricerca della cricca max di un
grafo basato su di una successione di reti di Hopfield discrete.
  
  Esecuzione:
        
	 % run <graph_name> 
 dove
		- graph_name = file contenente la matrice di incidenza.

 **************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


FILE *in;
typedef int *Row;
static int N, *stato_rete;                 
static Row *Adjmat;


/**************************************************************/

void stampa_stato_rete() {
  int i;

  printf("\n\n Stato della rete:\n");
  for (i=0; i<N; i++)
    printf("%3d", (stato_rete[i]+1)/2 );
  printf("\n");
}

/****************************************************************/

void energia()
{
  register int i,j;
  int s=0;
  static int T=0;

  for (i=0; i<N-1; i++)
    for (j=i+1; j<N; j++)
      s += Adjmat[i][j]*(1 - stato_rete[i]*stato_rete[j]); 
  printf("\n Energia -> %d\n",s/2);
  }

/****************************************************************/

void stabilizza_rete_Hopfield() {
  register int i,j;
  int pred, FINE=1, somme;

  while (FINE) {
    FINE = 0;
    for (i=0; i<N; i++) {
      somme = 0;

      // somma pesata dei vicini
      for (j=0; j<N; j++)
        somme -= Adjmat[i][j]*stato_rete[j];

      // analisi dello stato
      pred = stato_rete[i];
      if (somme<0)    
        stato_rete[i] = -1;
      else            
        stato_rete[i] = 1;

      // test cambio stato
      if (pred != stato_rete[i]) 
        FINE = 1;   
    }
  }   
}

/****************************************************************/

  void inizializza_strutture(char *argv[]) {
    register int i,j;
    char s[11];

    if( (in = fopen(argv[1],"r")) == NULL ) {
      printf("File non trovato!\n");
      exit(1);
    }

    N = atoi( fgets(s,10,in));
    stato_rete = (int *) malloc(N*sizeof(int));
    Adjmat = (Row *) malloc(N*sizeof(Row));
   
    for (i=0; i<N; i++) {
      Adjmat[i] = (int *) malloc(N*sizeof(int));
      stato_rete[i] = 0;
      for(j=0; j<N; j++)
        Adjmat[i][j] = fgetc(in)-48;
      fgetc(in);
    }

   fclose(in);
 }

/**************************************************************/

 void stampa_Adjmat() {
  int i,j;

  printf("\n Matrice di incidenza del Grafo:\n");
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++)
      printf("%3d", Adjmat[i][j] );
    printf("\n");
 }
 printf("\n");
}


/******************************    Main     ******************************/

int main(int argc, char *argv[]) {
  register int k;

  if( argc < 2 ) {
    printf("Manca un parametro!\n");
    exit(1);
  }
  inizializza_strutture(argv);
  stabilizza_rete_Hopfield();
  energia();
  // stampa_stato_rete();  
  return(0);
}