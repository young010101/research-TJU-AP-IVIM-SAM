import matplotlib.pyplot as plt
from data.data_loader import extract_layer

def plot_slice(data_slice, title="Image Slice"):
    plt.imshow(data_slice, 'gray')


def interactive_plot_slice(i_slice, i_bval, x_roi, y_roi, rad):
    # Code to create an interactive plot
    pass

def plot_specific_layer(image_path, layer_index, b_value_index, title="Image Slice"):
    """
    提取特定层并绘制图像。
    
    参数:
    - image_path (str): 图像文件的路径。
    - layer_index (int): 要提取的层的索引。
    - title (str): 可选的，绘图窗口的标题。
    
    返回:
    - 无。
    """
    # 提取特定层
    layer = extract_layer(image_path, layer_index, b_value_index)
    
    # 绘制图像切片
    plot_slice(layer, title=title)