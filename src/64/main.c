#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include <immintrin.h>

#include <omp.h>

#include "common.h"

#include "spectral64.c"

/*
*
* 	load_data
* 	=========
*
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
*
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*sizeof(type) byte: matrix data in row-major order --> numeri floating-point
*/
MATRIX load_data(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status;

	fp = fopen(filename, "rb");

	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(1);
	}

	status = fread(&rows, sizeof(int), 1, fp);
	status = fread(&cols, sizeof(int), 1, fp);

	MATRIX data = _mm_malloc(rows*cols*sizeof(type), align);
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);

	*n = rows;
	*k = cols;

	return data;
}

/*
* 	save_data
* 	=========
*
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
*
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*size byte: matrix data in row-major order
*	(size e' la dimensione in byte di un elemento: sizeof(int) o sizeof(type))
*/
void save_data(char* filename, void* X, int n, int k, int size) {
	FILE* fp;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);
		fwrite(X, size, n*k, fp);
	}
	else{
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

/*
*	now
*	===
*
*	Restituisce il tempo reale (wall-clock) in secondi
*/
double now(){
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec*1e-9;
}

/*
*	check_params
*	============
*
*	Verifica che i parametri siano applicabili; in caso contrario
*	stampa un messaggio e termina
*/
void check_params(params* input){
	if(input->N <= K){
		printf("Errore: il dataset deve contenere almeno K+1 = %d punti (N = %d)\n", K+1, input->N);
		exit(1);
	}
	if(input->D <= 0){
		printf("Errore: il dataset deve avere almeno una dimensione (D = %d)\n", input->D);
		exit(1);
	}
	if(input->c < 2 || input->c > input->N){
		printf("Errore: il numero di cluster c deve essere compreso tra 2 e N = %d (c = %d)\n", input->N, input->c);
		exit(1);
	}
	if(input->eig_iters <= 0){
		printf("Errore: T deve essere un intero positivo (T = %d)\n", input->eig_iters);
		exit(1);
	}
}

int main(int argc, char** argv) {

	// ================= Parametri di ingresso =================
	char* dsfilename = "ds.ds2";
	int c = 2;
	int T = 1000;
	int silent = 0;
	// =========================================================

	params* input = malloc(sizeof(params));

	input->DS = load_data(dsfilename, &input->N, &input->D);
	input->c = c;
	input->eig_iters = T;
	input->silent = silent;

	check_params(input);

	input->Y = _mm_malloc(input->N*input->c*sizeof(type), align);

	double t = now();
	// =========================================================
	spectral(input);
	// =========================================================
	t = now() - t;

	if(!input->silent)
		printf("EIG time = %.5f secs\n", t);
	else
		printf("%.5f\n", t);

	// Salva il risultato
	char* outname = "out_Y.ds2";
	save_data(outname, input->Y, input->N, input->c, sizeof(type));

	if(!input->silent){
		int rows = input->N < 10 ? input->N : 10;
		printf("Y (prime %d righe):\n", rows);
		for(int i=0; i<rows; i++){
			printf("( ");
			for(int j=0; j<input->c; j++)
				printf("%f ", input->Y[i*input->c + j]);
			printf(")\n");
		}
	}

	spectral_free(input);
	_mm_free(input->DS);
	_mm_free(input->Y);
	free(input);

	return 0;
}
