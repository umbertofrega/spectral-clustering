#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <math.h>
#include <omp.h>
#include "common.h"

// Funzione di esempio scritta in assembly (spectral32.nasm)
extern void prova(params* input);

/*
*	knngraph
*	========
*
*	Per ogni punto del dataset calcola i K vicini e le relative
*	distanze al quadrato, memorizzandoli in input->nn_id e input->nn_dist.
*	Le liste sono ordinate per distanza; a parita' vince l'indice minore.
*/
void knngraph(params* input){
	// TODO
}

/*
*	affinity
*	========
*
*	A partire dalle liste dei K vicini costruisce il grafo simmetrico,
*	le scale locali, la matrice di affinita' W, i gradi e la matrice
*	normalizzata M, memorizzata in forma sparsa (input->row_ptr,
*	input->col_idx, input->val, input->nnz).
*/
void affinity(params* input){
	// TODO
}

/*
*	eigenvectors
*	============
*
*	Calcola i c autovettori dominanti di M mediante iterazione
*	ortogonale (input->eig_iters iterazioni) e li memorizza in input->U
*	(N x c, row-major, gia' allocata dal chiamante).
*/
void eigenvectors(params* input){
	// TODO
}

/*
*	normalize
*	=========
*
*	Normalizza ogni riga di input->U in modo che abbia norma unitaria,
*	ottenendo (in place) la matrice Y. Una riga nulla resta nulla.
*/
void normalize(params* input){
	// TODO
}

/*
*	spectral
*	========
*
*	Calcola la matrice Y, ossia la matrice U dei c autovettori dominanti
*	della matrice di affinita' normalizzata del grafo dei K vicini,
*	con le righe normalizzate a norma unitaria.
*	Il clustering delle righe di Y (k-means) e' eseguito in Python.
*
*	input->U deve essere allocata dal chiamante (N x c elementi);
*	al termine contiene Y.
*	Le strutture intermedie allocate qui vanno liberate con spectral_free.
*/
void spectral(params* input){
	knngraph(input);
	affinity(input);
	eigenvectors(input);
	normalize(input);
	// Esempio di chiamata a una funzione assembly: da rimuovere
	prova(input);
}

/*
*	spectral_free
*	=============
*
*	Libera eventuali strutture intermedie
*/
void spectral_free(params* input){
}
