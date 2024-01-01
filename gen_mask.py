"""How to edit a great discribtion.
Cheak the web of python.

Hello.
"""

import numpy as np

def generate_mask(img_height,img_width,radius,center_x,center_y):
    """generate a circle mask.

    Parameters
    ----------
    img_height : int
        height of img.
    img_width : int
        width of img.

    Returns
    -------
    mask : ndarray of shape (img_height,img_width)
        a circle mask.

    See Also
    --------

    Examples
    --------
    >>> generate_mask(2, 2, 10, 1, 1)
    array([[true, true],
           [true, true]])
    """

 
    y,x=np.ogrid[0:img_height,0:img_width]
 
    # circle mask
 
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
 
    return mask
