"""Various numerical utilities.
"""

import numpy as np

def mpower(A,n):
    """Simple matrix power"""

    if n == 0:
        # This only makes sense for square matrices, we return 
        return np.identity(A.shape[0],A.dtype)
    
    B = A
    p = int(np.floor(np.log(n)/np.log(2)))
    for i in range(p):
        B = np.dot(B,B)

    for i in range(n-2**p):
        B = np.dot(B,A)

    return B

def rescale_arr(arr,amin,amax,omin=None,omax=None):
    """Rescale an array to a new range.

    Return a new array whose range of values is (amin,amax).

    :Parameters:
      arr : array-like
      
      amin : float
        new minimum value
        
      amax : float
        new maximum value

    :Examples:
    >>> a = np.arange(5)

    >>> rescale_arr(a,3,6)
    array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])

    >>> rescale_arr(a,3,6,-1,5)
    array([ 3.5,  4. ,  4.5,  5. ,  5.5])

    >>> rescale_arr(a,3,6,-1)
    array([ 3.6,  4.2,  4.8,  5.4,  6. ])
    """
    
    # old bounds
    m = arr.min() if omin is None else omin
    M = arr.max() if omax is None else omax

    # scale/offset
    s = float(amax-amin)/(M-m)
    d = amin - s*m
    
    # Apply clip before returning to cut off possible overflows outside the
    # intended range due to roundoff error, so that we can absolutely guarantee
    # that on output, there are no values > amax or < amin.
    return np.clip(s*arr+d,amin,amax)

def make_colors(x,y,xc='r',yc='g',xbounds=(0,1),ybounds=(0,1)):
    """Return a color array for a pair of x/y arrays, suitable for scatter()
    """
    # Sanity checks
    npts = len(x)
    assert npts==len(y),"Incompatible lengths for x and y"

    # map primary colors to columns
    colors = dict(r=0,g=1,b=2)
    xcidx = colors[xc]
    ycidx = colors[yc]

    # Convert values of input arrays to 0-1 range, so we can interpret
    # them as color components.

    xc = rescale_arr(x,0,1,*xbounds)
    yc = rescale_arr(y,0,1,*ybounds)

    # Assemble into an actual color spec that MPL understands
    colspec = np.zeros((npts,3))
    colspec[:,xcidx] = xc
    colspec[:,ycidx] = yc
    # Unfortunately mpl doesn't understand an Nx3 array as a color spec, it
    # wants an actual list of tuples... Otherwise we could just return colspec.
    return colspec
