import matplotlib.pyplot as plt

def create_intensity_map(data, limits, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='gray', clim=limits)
    fig.colorbar(im)
    fig.savefig(filename)