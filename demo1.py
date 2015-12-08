#!/usr/bin/env python
"""Diffusion maps -- simple demo.

Along the lines of Fig 1 in:

  'Diffusion Maps', Coifman & Lafon, doi:10.1016/j.acha.2006.04.006.
"""

import os
import random

import numpy as np
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

from nutils import rescale_arr,mpower,make_colors

FIGDIR = '/home/fperez/prof/grants/0804_nsf_cdi/fig/'
SAVEFIGS = 1

def savefigs(figs,dpi=400,figdir=FIGDIR):
    if not SAVEFIGS:
        return

    for name,fig in figs.iteritems():
        fig.savefig(os.path.join(figdir,name+'.png'),dpi=dpi)

# Main code
if __name__ == '__main__':

    # Construct a dataset made of 'clouds' of points (each is just a Gaussian
    # distribution of points)

    # variance of the distribution (same in x-y for simplicity)
    N = 100  # number of points per cloud
    centers = ([0,0.5],[1.5,0.5],[1.5,-1.5],[0,-1.5]) # Centers of clouds
    variances = ([0.1,0.01],[0.02,0.2],0.03,0.03)
    # times at which to plot the operator (keep them powers of 2)
    times = 1,8,32,256

    # Build the dataset
    normal = np.random.multivariate_normal
    id2 = np.eye(2)
    data = tuple(normal(c,v*id2,N) for c,v in zip(centers,variances))
    points = np.vstack(data)
    
    # Ensure we don't feed the data into the algorithm with any preconcieved
    # notion of the actual cluster structure, to avoid any bias
    np.random.shuffle(points)

    # Compute the Diffusion operator
    eps = 0.7  # sharpness parameter for gaussian kernel
    # First, compute the kernel
    k = np.exp(-((points - points[:,np.newaxis])**2).sum(axis=2)/eps)
    # Then, define the actual diffusion matrix
    P = (k / k.sum(axis=1)) # markov normalisation in x-dir

    # raw data
    x = points[:,0]
    y = points[:,1]

    # Compute diffusion map for P
    ndim = 2
    w0,v0 = eigs(P,k=ndim+1)
    w = (w0[1:ndim+1]).real
    v = np.array(v0[:,1:ndim+1].real.astype(float))
    vx = v[:,0]
    vy = v[:,1]

    xdbounds = vx.min(),vx.max()
    ydbounds = vy.min(),vy.max()

    col = make_colors(vx,vy,xbounds=xdbounds,ybounds=ydbounds)

    # Choose whether you want edges around the points or not
    #edgecolors = None  # draw edges
    edgecolors = 'None'  # no edges
    
    # figs dictionary for writing later figures to disk with a simple call
    figs = {}

    fig = plt.figure(figsize=(8.7,4))

    plt.subplot(131)
    plt.scatter(x,y,edgecolors=edgecolors)
    plt.title('Original data')
    plt.xticks([]); plt.yticks([])

    plt.subplot(132)
    plt.scatter(vx,vy,edgecolors=edgecolors,c=col)
    plt.title('Diffusion coordinates')
    plt.xticks([]); plt.yticks([])

    # We need the x/y range of this plot so we can later plot the entire
    # diffusion map, rescaled in each direction by the proper power of the
    # corresponding eigenvalue, to show how the clustering changes in the
    # diffusion coordinates
    ax = plt.gca()
    xlim_dif = ax.get_xlim()
    ylim_dif = ax.get_ylim()
    
    plt.subplot(133)
    plt.scatter(x,y,edgecolors=edgecolors,c=col)
    plt.title('Pseudo-colored data')
    plt.xticks([]); plt.yticks([])

    fig.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.85)

    figs['clust_ori'] = fig

    for t in times:
        # Display the raw data
        wt  = w**t
        psix = wt[0]*vx
        psiy = wt[1]*vy

        # Define colormap based on diffusion map
        col = make_colors(psix,psiy,xbounds=xdbounds,ybounds=ydbounds)

        # Left panel, show the data with diffusion colors
        fig = plt.figure()
        fig.text(0.5,0.92,'t=%s' % t,ha='center',va='center')

        plt.subplot(122)
        plt.scatter(x,y,edgecolors=edgecolors,c=col)
        plt.xticks([]); plt.yticks([])

        # Right panel
        if 0:
            # And the corresponding power of P
            plt.subplot(121)
            Pt = mpower(P,t)
            plt.imshow(Pt)
            plt.xticks([]); plt.yticks([])

        if 1:
            # The diffusion coordinates scaled by the eigenvalue, but using the
            # original x/y range so we see the anisotropic shrinking
            plt.subplot(121)
            plt.scatter(psix,psiy,edgecolors=edgecolors,c=col)
            plt.xlim(xlim_dif)
            plt.ylim(ylim_dif)
            plt.xticks([]); plt.yticks([])

        fig.subplots_adjust(left=0.05,bottom=0.07,right=0.95,top=0.87)

        figs['clust_t%s' % t] = fig
        
    plt.show()
