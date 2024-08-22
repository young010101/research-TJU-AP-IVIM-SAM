import matplotlib.pyplot as plt

def visualize_data(data, title):
    plt.imshow(data, cmap='gray')
    plt.title(title)

def save_figure(fig, filename):
    fig.savefig(filename)