"""Diffusion maps.

Author: Stefan van der Walt <stefan@sun.ac.za>
"""

import sys
import numpy as np
from scipy.sparse.linalg import eigs

def gauss_kernel(x,y,**kernel_params):
    eps = kernel_params.get('eps',1.0)
    return np.exp(-((x - y)**2).sum()/eps)

class DiffusionMap(object):
    def __init__(self, data, kernel=gauss_kernel, kernel_params={},
                 cache_filename=None):
        """
        Parameters
        ----------
        data : ndarray or filename
            Data features as row-vectors or filename (.npy) where these
            are stored.
        cache_filename : string
            Filename to/from which calculated normalised diffusion matrix, P,
            is stored.  If None, then no caching is done.

        """
        if isinstance(data,str):
            data = np.load(data)

        self.data = data
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.cache_filename = cache_filename
        self.H = None

    def _compute_diffusion_matrix(self):
        # This can be done more efficiently in Cython
        if self.H is not None:
            return

        if self.cache_filename is not None:
            try:
                self.H = np.load(self.cache_filename)
                print("Using cached diffusion matrix.")
                return
            except IOError:
                pass

        N = len(self.data)
        H = np.empty((N,N),float)
        data = self.data
        irange = range(N)
        kernel = self.kernel
        kp = self.kernel_params

        for i in irange:
            di = data[i]
            for j in range(i+1):
                H[i,j] = kernel(di,data[j],**kp)

        for i in irange:
            for j in range(i+1,N):
                    H[i,j] = H[j,i]

        self.H = ((H.T / H.sum(axis=1)).T).copy() # Markov normalisation, along row

    def map(self, ndim=2, time=1):
        self._compute_diffusion_matrix()
        if self.H is None:
            raise ValueError("Diffusion matrix was not computer correctly")

        w,v = eigs(self.H,k=ndim+1)

        w = w[1:ndim+1]
        v = (w.real**time) * np.array(v[:,1:ndim+1].real.astype(float))

        return w, v
