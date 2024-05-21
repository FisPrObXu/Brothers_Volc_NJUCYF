import numpy as np
from mayavi import mlab

def visualize_3d_mayavi(file_path):
    # 加载 .npz 文件
    data = np.load(file_path)
    stacked_image = data['stacked_image']
    
    # 使用 mayavi 可视化 3D 图像数据
    mlab.figure(size=(800, 800))
    mlab.volume_slice(stacked_image, plane_orientation='x_axes')
    mlab.volume_slice(stacked_image, plane_orientation='y_axes')
    mlab.volume_slice(stacked_image, plane_orientation='z_axes')
    mlab.colorbar(title='Gray Value', orientation='vertical')
    mlab.show()

def main():
    file_path = 'C:/Users/XuuX/Desktop/Ocean Drilling Data/mineCV/1530A/stacked_image.npz'  # 替换为你的 .npz 文件路径
    visualize_3d_mayavi(file_path)

if __name__ == '__main__':
    main()
