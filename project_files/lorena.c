#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define Min(x,y) (fabs(x) < fabs(y) ? fabs(x) : fabs(y))

#define EPSILON 0.1
#define PI 3.14159265358979323846


FILE *in, *out;
typedef int *Row;

static int N;                 
static Row *Adjmat;
double *A,*B,*teta,alfa;


/****************************************************************/

void modifica_A_B(int k, double a, double b) {
	register int i;

	for (i=0; i<N; i++) {
		A[i] += (double) a*Adjmat[k][i];
		B[i] += (double) b*Adjmat[k][i];
	}
}

/****************************************************************/

double taglio(int *r) {
	register int i,j;
	double s=0;

	for (i=0; i<N-1; i++)
		for (j=i+1; j<N; j++)
			s += Adjmat[i][j]*(1-r[i]*r[j]);
		return s/2;
	}

/****************************************************************/

void taglio_massimo() {
	register int i,j,k;
	register double t;
	int *R,T=0;
	double alfa;

	R = (int *) calloc(N,sizeof(int));
	for (i=0; i<N; i++) {
		alfa = (teta[i] > PI) ? teta[i]-PI : teta[i];
		for (j=0; j<N; j++)
			R[j] = ((teta[j] >= alfa) && (teta[j] < alfa+PI)) ? 1 : -1;
		t = taglio(R);
		if ( t > T )
			T = t;
	}
	printf("Lorena -> %d\n",T);
}


/****************************************************************/

void inizializza_strutture(int argc, char *argv[]){
	register int i,j;
	char s[11];

	if ( argc < 3 ) {
		printf("Manca un parametro!\n");
		exit(1);
	}
	if ((in = fopen(argv[1],"r")) == NULL) {
		printf("File non trovato!\n");
		exit(1);
	}
	srand(atoi(argv[2]));  // seme fissato x ripetere stesso exp
	
	/* srand(time(NULL)&RAND_MAX); */
	N = atoi(fgets(s,10,in));
	Adjmat = (Row *) malloc(N*sizeof(Row));
	teta = (double *) calloc(N,sizeof(double));
	A = (double *) calloc(N,sizeof(double));
	B = (double *) calloc(N,sizeof(double));
	for (i=0; i<N; i++)
		teta[i] = (double) 2*PI*rand()/RAND_MAX;
	for (i=0; i<N; i++) {
		Adjmat[i] = (int *) malloc(N*sizeof(int));
		A[i] = B[i] = 0;
		for(j=0; j<N; j++) {
			Adjmat[i][j] = fgetc(in)-48;
			A[i] += Adjmat[i][j]*cos(teta[j]);
			B[i] += Adjmat[i][j]*sin(teta[j]);
		}
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

/**************************************************************/

void stampa_teta() {
	int k;
	static int t=0; 

	printf("Teta[%d]:\n",t);
	for (k=0; k<N; k++)
		printf("teta[%2d] -> %f\n",k+1,teta[k]);
	t++;
}

/******************************    Main     ******************************/

int main(int argc, char *argv[]) {
	register int k;
	int OK = 1, nround = 0;
	double p;

	inizializza_strutture(argc,argv);
	while ( OK ) {
		nround++;
		OK = 0;
		for (k=0; k<N; k++) {
			alfa = teta[k];
			teta[k] = atan(B[k]/A[k]); 
			if (A[k] >= 0) 
				teta[k] += PI; 
			else if (B[k] > 0) 
				teta[k] += 2*PI;
			modifica_A_B(k, cos(teta[k])-cos(alfa), sin(teta[k])-sin(alfa));
			if ( Min(alfa-teta[k],2*PI-alfa+teta[k]) > EPSILON )
				OK = 1;
		}
	}
	taglio_massimo();
	stampa_teta();
	printf("Num rounds = %d\n",nround);
	return(0);
}




