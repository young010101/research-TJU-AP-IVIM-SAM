# image_plotter.py

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@DeprecationWarning
class ImagePlotter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.fig, self.ax = None, None

    def display(self, subplot_spec, title=None):
        if self.fig and self.ax:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(2, 2)
        self.ax[0, 0].imshow(self.image, interpolation="none")
        self.ax[0, 1].imshow(self.image, interpolation="none")
        self.ax[1, 0].imshow(self.image, interpolation="none")
        self.ax[1, 1].imshow(self.image, interpolation="none")
        if title:
            self.ax[0, 0].set_title(title)
        self.ax.axis("off")

        # plt.subplot(subplot_spec)
        # plt.imshow(self.image, interpolation="none")
        # if title:
        #     plt.title(title)
        # plt.axis("off")


if __name__ == "__main__":
    # 图像路径列表
    image_paths = [
        "perfusion_coeff.png",
        "perfusion_fraction.png",
        "perfusion_fraction.png",
    ]

    # 创建ImagePlotter对象列表
    image_plotters = [
        ImagePlotter(os.path.join("ivim_analysis", path)) for path in image_paths
    ]

    # 创建一个2x2的图表布局
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 将每个图像添加到subplot中
    for idx, (ax, image_plotter) in enumerate(zip(axes.flatten(), image_plotters)):
        image_plotter.display(ax, title=f"Image {idx + 1}")

    # 调整布局
    plt.tight_layout()
    plt.show()
