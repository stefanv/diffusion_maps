"""Diffusion maps -- clustering on a spiral dataset"""

import numpy as np
from numpy import sin,cos,pi
from matplotlib import pyplot as plt

from difmap import DiffusionMap, gauss_kernel

eps = 100
N = 100
T = 1

theta = np.linspace(0,4*pi,N)
t = np.linspace(0,1,N)
R = 10
noise = 1/1.25

# Generate dataset

x = R*sin(theta)*np.exp(-t) + noise*np.random.random(len(t))
y = R*cos(theta)*np.exp(-t) + noise*np.random.random(len(t))
shuffle = np.random.permutation(np.arange(len(x)))
data = np.vstack((x[shuffle],y[shuffle])).T

x = R*sin(-theta)*np.exp(-t) + noise*np.random.random(len(t))
y = R*cos(-theta)*np.exp(-t) + noise*np.random.random(len(t))
data = np.vstack((data,np.vstack((x,y)).T))

dm = DiffusionMap(data, kernel=gauss_kernel, kernel_params={'eps': eps},
                  cache_filename=None)

w,v = dm.map(ndim=2)

plt.title('Time step %s' % T)

plt.subplot(311)
plt.plot(data[:,0],data[:,1],'x')
plt.axis('equal')

plt.subplot(312)
plt.imshow(dm.H, cmap='viridis')

plt.subplot(313)
for i,(x,y) in enumerate(zip(v[:,0].flat,v[:,1].flat)):
    plt.plot([x],[y],'.b')
    plt.xlim([-0.2,0.2])
    plt.ylim([-0.2,0.2])
    #plt.axis('equal')

plt.show()



