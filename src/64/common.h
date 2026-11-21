#ifndef SPECTRAL_COMMON
#define SPECTRAL_COMMON

#define	type	double
#define	align	32

#define	MATRIX		type*
#define	VECTOR		type*

// Numero di vicini del grafo K-NN: costante del problema
#define	K	8


typedef struct{
	// ---------------- Ingresso ----------------
	MATRIX DS;					// dataset (N x D)
	// ---------------- Uscita ----------------
	MATRIX Y;					// i c autovettori dominanti (N x c, row-major);
	// al termine le righe sono normalizzate (matrice Y)
	// ---------------- Parametri e dimensioni ----------------
	int N;						// numero di righe del dataset
	int D;						// numero di colonne/feature del dataset
	int c;						// numero di cluster (= numero di autovettori)
	int eig_iters;				// numero di iterazioni T per il calcolo degli autovettori
	int silent;					// modalita' silenziosa
} params;

#endif
