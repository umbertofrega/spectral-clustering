"""
Spectral clustering on the K-NN graph

La matrice Y (i c autovettori dominanti, con righe a norma unitaria)
e' calcolata dal modulo C/assembly; il clustering delle righe di Y
e' eseguito con il k-means di scikit-learn.
"""

import time
import numpy as np
from sklearn.cluster import KMeans

from . import spectral32, spectral64, spectral64omp

# Funzione C/assembly che calcola Y, per ciascuna versione
_EMBEDDING = {
    '32':    spectral32.embedding,      # float,  SSE
    '64':    spectral64.embedding,      # double, AVX
    '64omp': spectral64omp.embedding,   # double, AVX, OpenMP
}


class SpectralClustering:
    """
    Spectral clustering on the K-NN graph

    Parameters:
      n_clusters: numero di cluster c
      eig_iters: numero di iterazioni T per il calcolo degli autovettori
      km_iters: numero massimo di iterazioni I del k-means
      version: '32' (float, SSE), '64' (double, AVX)
               o '64omp' (double, AVX, OpenMP)
      silent: modalita' silenziosa
    """

    def __init__(self, n_clusters, eig_iters=1000, km_iters=100,
                 version='32', silent=False):
        if version not in _EMBEDDING:
            raise ValueError(f"version must be one of {list(_EMBEDDING)}")
        if n_clusters < 2:
            raise ValueError("n_clusters must be an integer >= 2")
        if eig_iters <= 0:
            raise ValueError("eig_iters must be a positive integer")
        if km_iters <= 0:
            raise ValueError("km_iters must be a positive integer")
        self.n_clusters = n_clusters    # c
        self.eig_iters = eig_iters      # T
        self.km_iters = km_iters        # I
        self.version = version
        self.silent = silent

    def init_centroids(self, Y, c):
        n = Y.shape[0]
        idx = [0]
        dmin = ((Y - Y[0]) ** 2).sum(1)
        for _ in range(1, c):
            i = int(np.argmax(dmin))  # argmax -> primo indice in caso di parita'
            idx.append(i)
            dmin = np.minimum(dmin, ((Y - Y[i]) ** 2).sum(1))
        return Y[idx].copy()

    def fit(self, dataset):
        """
        Calcola le etichette dei punti del dataset.

        Parameters:
          dataset: numpy array di forma (N, D), allineato, di tipo
                   float32 (versione '32') o float64 ('64', '64omp')

        Returns:
          self; le etichette sono in self.labels_, la matrice Y
          in self.embedding_
        """
        embedding = _EMBEDDING[self.version]

        # ========== C/assembly: matrice Y ==========
        start = time.time()
        Y = embedding(dataset, self.n_clusters, self.eig_iters, self.silent)
        self.eig_time_ = time.time() - start
        self.embedding_ = Y

        # ========== Python: k-means sulle righe di Y ==========
        start = time.time()
        # Centroidi iniziali deterministici: righe in posizione floor(N/c)*j
        n, c = Y.shape
        init = self.init_centroids(Y, c)

        kmeans = KMeans(n_clusters=c, init=init, n_init=1,
                        max_iter=self.km_iters, tol=0, algorithm='lloyd')
        self.labels_ = kmeans.fit_predict(Y).astype(np.int32)
        self.km_time_ = time.time() - start

        return self

    def fit_predict(self, dataset):
        """
        Esegue fit e restituisce le etichette (numpy array di N interi)
        """
        return self.fit(dataset).labels_
