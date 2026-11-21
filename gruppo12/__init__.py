"""
Spectral Clustering on the K-NN graph Package
"""

from . import spectral32
from . import spectral64
from . import spectral64omp
from ._spectral import SpectralClustering

__version__ = '1.0'
__all__ = ['SpectralClustering', 'spectral32', 'spectral64', 'spectral64omp']
