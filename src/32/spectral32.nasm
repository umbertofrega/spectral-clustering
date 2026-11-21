%include "sseutils32.nasm"

default rel

section .data			; Sezione contenente dati inizializzati

msgN		db	'N:',32,0
msgC		db	', c:',32,0
nl			db	10,0
	align 16
uno			dd	1.0

section .bss			; Sezione contenente dati non inizializzati
	alignb 16
	n		resd		1
	c		resd		1

section .text			; Sezione contenente il codice macchina

; ------------------------------------------------------------
; Funzioni
; ------------------------------------------------------------

global prova


prova:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		rbp		; salva il Base Pointer
		mov		rbp, rsp	; il Base Pointer punta al Record di Attivazione corrente
		push		rbx		; salva i registri da preservare
		push		rsi
		push		rdi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------

		; elaborazione
		; [RDI]    input->DS;		// dataset
		; [RDI+8]  input->Y;		// autovettori (N x c)
		; [RDI+16] input->N;		// numero di righe del dataset
		; [RDI+20] input->D;		// numero di colonne/feature del dataset
		; [RDI+24] input->c;		// numero di cluster
		; [RDI+28] input->eig_iters;	// iterazioni per gli autovettori
		; [RDI+32] input->silent;	// modalita' silenziosa

		; STAMPA I PARAMETRI N e c
		MOV EAX, [RDI+16]
		MOV [n], EAX
		MOV EAX, [RDI+24]
		MOV [c], EAX
		prints msgN
		printsi n
		prints msgC
		printsi c
		prints nl

		; SALVA 1.0 IN Y[0][0]
		MOV RAX, [RDI+8]	; indirizzo di Y
		MOVSS XMM0, [uno]
		MOVSS [RAX], XMM0

		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------

		pop	rdi		; ripristina i registri da preservare
		pop	rsi
		pop	rbx
		mov	rsp, rbp	; ripristina lo Stack Pointer
		pop	rbp		; ripristina il Base Pointer
		ret			; torna alla funzione C chiamante
