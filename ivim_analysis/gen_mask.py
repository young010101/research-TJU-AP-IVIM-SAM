"""How to edit a great discribtion.
Cheak the web of python.

Hello.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi

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


def plot_map(raw_data, variable, limits, title):
    fig, ax = plt.subplots(1)
    lower, upper = limits
    ax.set_title('Map for {}'.format(variable))
    im = ax.imshow(raw_data.T, origin='lower', clim=(lower, upper),
                   cmap="gray", interpolation='nearest')
    fig.colorbar(im)
    return fig


def plot_map_roi(raw_data, variable, limits, filename, x_roi=100, y_roi=100, radius=5):
    """plot map with roi
    
    Parametes
    ---------
    filename: str
    filename without roi and format"""
    plot_map(raw_data, variable, limits, filename+'_roi.png')
    plt.scatter(y_roi, x_roi, marker='o', s=radius**2*pi, edgecolors='r', facecolors='None')


# Function to save a figure
def save_figure(fig, filename, output_dir='output'):
    # Save the figure to the output directory with the specified filename
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory