#!/usr/bin/env python
"""Diffusion maps -- organising images"""

# Standard library imports
import math
import random
import sys

# Third party imports
import numpy as np
from matplotlib import pyplot as plt

# Imports from this project
from difmap import DiffusionMap, gauss_kernel
from skimage import io
from transform import homography

from nutils import rescale_arr

# Utility functions

def rotate_around_centre(img,angle=0,*args,**kwargs):
    sa = math.sin(angle)
    ca = math.cos(angle)

    T = np.array([[1,0,-x_off],
                  [0,1,-y_off],
                  [0,0,1]])
    R = np.array([[ca,-sa,0],
                  [sa,ca,0],
                  [0,0,1]])

    T_back = T.copy()
    T_back[:2,2] *= -1

    M = np.dot(T_back,np.dot(R,T))

    return homography(img,M,*args,**kwargs)

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


if __name__ == '__main__':
    # Number of images to generate
    N = 25
    max_angle = np.pi*2

    #angles = np.linspace(0,max_angle,N,endpoint=False)
    angles = np.random.uniform(0,max_angle,N)

    angles_ord = angles.copy() # keep copy to plot unsorted images
    #random.shuffle(angles)   # truly randomized inputs

    # Generate randomized dataset
    template = io.imread('template.png', as_grey=True)
    shape = np.array(template.shape)
    x_off,y_off = (shape-1)/2

    data = np.empty((len(angles),np.prod(template.shape)),float)
    dx,dy = data.shape
    for i,a in enumerate(angles):
        print("Rotating %i" % i)
        data[i] = rotate_around_centre(template,angle=a,cval=255).flat

    # EXPERIMENT: Simulate colour image
    old_data = data.copy()
    data = data.repeat(3, axis=1)

    # Compute diffusion map on dataset
    dm = DiffusionMap(data, kernel=gauss_kernel, kernel_params={'eps':1e9})
    w,v = dm.map()

    # Plot the raw data as seen by the diffusion map
    plt.figure()
    plt.imshow(data,aspect=float(dy)/dx,cmap=plt.cm.gray)
    plt.xticks([])

    # Plot the unsorted images on a circle on the x-y plane
    plot_images(np.cos(angles_ord),np.sin(angles_ord),old_data,template.shape)
    #plt.savefig('images_unordered.png')

    # Plot the images on the diffusion coordinates, which should be sorted
    plot_images(v[:,0],v[:,1],old_data,template.shape)
    #plt.savefig('images_ordered.png')

    plt.show()
