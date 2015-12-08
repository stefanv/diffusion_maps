#!/usr/bin/env python
"""Diffusion maps -- organising images"""

# Standard library imports
import math
import random
import sys

# Third party imports
import numpy as np
from scipy import ndimage as ndi
from scipy.misc.pilutil import imsave
from matplotlib import pyplot as plt

# Imports from this project
from difmap import DiffusionMap, gauss_kernel

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import rotate

from nutils import rescale_arr

def stretch(a,b,s):
    c = (a+b)/2.0
    d = s * (b-a)/2.0
    return (c-d,c+d)
    
def plot_images(x,y,images,shape):
    """Plot image sequence at coordinates x,y.
    """
    s = 1.4 # scale axes out so images don't fall off edge of figure
    xlim = stretch(x.min(),x.max(),s)
    ylim = stretch(y.min(),y.max(),s)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1],xlim=xlim,ylim=ylim,xticks=[],yticks=[])

    forward = ax.transData.transform
    reverse = ax.transAxes.inverted().transform

    for i,img in enumerate(images):
        xi,yi = reverse(forward([x[i],y[i]]))
        a_sub = plt.axes([xi,yi,0.1,0.1],axisbg='w',aspect=1.0,
                         xticks=[],yticks=[])
        a_sub.imshow(images[i].reshape(shape),cmap=plt.cm.gray)


def makemat(nx,ny,images,shape):
    nr,nc = shape
    mat = np.empty((nx*nr,ny*nc))

    ir = range(nx)
    jr = range(ny)
    ni = 0
    for i in ir:
        for j in jr:
            mat[i*nr:(i+1)*nr,j*nc:(j+1)*nc] = images[ni].reshape(shape)
            ni += 1
    return mat

if __name__ == '__main__':
    # Number of images to generate
    nx,ny = (8,8)
    N = nx*ny
    max_angle = 360

    #angles = np.linspace(0,max_angle,N,endpoint=False)
    angles = np.random.uniform(0,max_angle,N)

    angles_ord = angles.copy() # keep copy to plot unsorted images
    random.shuffle(angles)   # truly randomized inputs

    # Generate randomized dataset
    template = img_as_ubyte(imread('template.png', as_grey=True))
    shape = np.array(template.shape)
    x_off,y_off = (shape-1)/2

    data = np.empty((len(angles),np.prod(template.shape)),float)
    for i,a in enumerate(angles):
        print("Rotating %i" % i)
        data[i] = rotate(template, angle=a, cval=1).flat

    # Compute diffusion map on dataset
    dm = DiffusionMap(data, kernel=gauss_kernel, kernel_params={'eps':1e9})
    w,v = dm.map()

    # # Plot the raw data as seen by the diffusion map
    # plt.figure()
    # dx,dy = data.shape
    # plt.imshow(data),aspect=float(dy)/dx,cmap=plt.cm.gray)
    # plt.title('raw')
    # plt.xticks([])

    # Plot the matrices as a little table
    m = makemat(nx,ny,data,template.shape)
    plt.matshow(m,cmap=plt.cm.gray)
    plt.xticks([]); plt.yticks([])
    #plt.savefig('images_table.png',dpi=200)

    # Plot the unsorted images on a circle on the x-y plane 
    plot_images(np.cos(angles_ord),np.sin(angles_ord),data,template.shape)
    #plt.savefig('images_unordered.png')

    # Plot the images on the diffusion coordinates, which should be sorted
    plot_images(v[:,0],v[:,1],data,template.shape)
    plt.savefig('images_ordered.png',dpi=200)

    plt.show()
