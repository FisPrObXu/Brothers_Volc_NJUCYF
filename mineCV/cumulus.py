import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_images(root_dir):
    img_names = os.listdir(root_dir)
    
    def parse_filename(name):
        try:
            return int(name.split('A')[1].split('_')[0])
        except (IndexError, ValueError):
            return None
    
    img_names = [name for name in img_names if parse_filename(name) is not None]
    img_names.sort(key=parse_filename)
    
    images = []
    for name in img_names:
        img_path = os.path.join(root_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        img = crop_image(img)
        images.append(img)
    return images

def crop_image(img):
    """Crop the black borders around the image."""
    gray = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def stack_images(images):
    return np.stack(images, axis=0)

def visualize_3d(stacked_image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    depth, height, width = stacked_image.shape
    x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
    ax.scatter(x, y, z, c=stacked_image.flatten(), cmap='gray', marker='.')
    plt.show()

def main():
    root_dir = 'C:/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephototrimnormalize'  # 替换为实际的图像目录路径
    images = load_images(root_dir)
    
    if not images:
        print("No images loaded. Exiting.")
        return
    
    # 确保所有图像的尺寸一致
    target_size = (images[0].shape[1], images[0].shape[0])  # (width, height)
    images = [cv2.resize(img, target_size) for img in images]
    
    # 打印一些调试信息
    print(f"Loaded {len(images)} images with size {target_size}")
    
    # 堆叠图像形成3D图像
    stacked_image = stack_images(images)
    
    # 可视化3D图像
    visualize_3d(stacked_image)
    
    # 保存3D图像为npz文件
    output_path = 'C:/Users/XuuX/Desktop/Ocean Drilling Data/mineCV/1530A/stacked_image.npz'
    np.savez(output_path, stacked_image=stacked_image)
    print(f"3D image saved as {output_path}")

if __name__ == '__main__':
    main()
