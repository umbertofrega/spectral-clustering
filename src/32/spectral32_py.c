#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <xmmintrin.h>

#include "common.h"

#include "spectral32.c"

static void mm_free_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, NULL);
    if (ptr != NULL) {
        _mm_free(ptr);
    }
}

// Funzione embedding(dataset, n_clusters, eig_iters, silent=False)
static PyObject* py_embedding(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *ds_array;

	int c, T, silent = 0;

	static char *kwlist[] = {"dataset", "n_clusters", "eig_iters", "silent", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii|p", kwlist,
									&PyArray_Type, &ds_array,
									&c, &T, &silent)) {
		return NULL;
	}

	// Verifica che sia un array NumPy valido
	if (PyArray_NDIM(ds_array) != 2) {
		PyErr_SetString(PyExc_ValueError, "Data must be a 2D array");
		return NULL;
	}

	// Verifica che sia float32
	if (PyArray_TYPE(ds_array) != NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "Data must be float32");
		return NULL;
	}

	// Verifica che sia contiguo
	if (!PyArray_IS_C_CONTIGUOUS(ds_array)) {
		PyErr_SetString(PyExc_ValueError, "Data must be C-contiguous");
		return NULL;
	}

	type* dataset = (type*)PyArray_DATA(ds_array);

	// Verifica che sia allineato
	uintptr_t addr = (uintptr_t)dataset;
	int is_aligned = (addr % align == 0);

	if(!is_aligned){
		PyErr_SetString(PyExc_ValueError, "Input array (DS) not aligned");
		return NULL;
	}

	int N = (int)PyArray_DIM(ds_array, 0);
	int D = (int)PyArray_DIM(ds_array, 1);

	// Verifica che i parametri siano applicabili
	if (N <= K) {
		PyErr_Format(PyExc_ValueError, "The dataset must contain at least K+1 = %d points", K+1);
		return NULL;
	}
	if (D <= 0) {
		PyErr_SetString(PyExc_ValueError, "The dataset must have at least one feature");
		return NULL;
	}
	if (c < 2 || c > N) {
		PyErr_Format(PyExc_ValueError, "n_clusters must be between 2 and N = %d", N);
		return NULL;
	}
	if (T <= 0) {
		PyErr_SetString(PyExc_ValueError, "eig_iters must be a positive integer");
		return NULL;
	}

	// Inizializzazione parametri
	params input;
	input.DS = dataset;
	input.N = N;
	input.D = D;
	input.c = c;
	input.eig_iters = T;
	input.silent = silent;

	input.Y = (type*) _mm_malloc(N * c * sizeof(type), align);
	if (input.Y == NULL)
		return PyErr_NoMemory();

	// ========================================= //
	spectral(&input);
	// ========================================= //

	spectral_free(&input);

	npy_intp dims[2] = {N, c};

	PyArrayObject* Y_array = (PyArrayObject*)PyArray_SimpleNewFromData(
		2,				// ndim
		dims,			// shape
		NPY_FLOAT32,	// dtype
		input.Y			// data pointer (usa la memoria allineata)
	);
	// Crea un capsule per gestire la deallocazione
	PyObject* capsule = PyCapsule_New(input.Y, NULL, mm_free_destructor);

	// Associa il capsule all'array così quando l'array viene distrutto,
	// la memoria allineata viene liberata
	PyArray_SetBaseObject(Y_array, capsule);

	return (PyObject*)Y_array;
}

// Tabella delle funzioni
static PyMethodDef spectral32_methods[] = {
	{
		"embedding",
		(PyCFunction)py_embedding,
		METH_VARARGS | METH_KEYWORDS,
		"Compute the spectral embedding Y: the c dominant eigenvectors of\n"
		"the normalized affinity matrix of the K-NN graph, with rows\n"
		"normalized to unit norm\n\n"
		"Parameters:\n"
		"  dataset: numpy array of shape (N, D)\n"
		"  n_clusters: number of clusters c (= number of eigenvectors)\n"
		"  eig_iters: number of orthogonal iterations T\n"
		"  silent: silent mode (default=False)\n"
		"\n"
		"Returns:\n"
		"  numpy array Y of shape (N, c)"
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef spectral32_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_spectral32",          // Nome del modulo C
    .m_doc = "Spectral embedding on the K-NN graph (32bit)",  // Docstring
    .m_size = -1,                     // -1 significa che il modulo non mantiene stato
    .m_methods = spectral32_methods,
};

// Inizializzazione del modulo
PyMODINIT_FUNC PyInit__spectral32(void) {
	// Inizializza NumPy
	import_array();

	return PyModule_Create(&spectral32_module);
}
