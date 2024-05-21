import os
import cv2
import numpy as np

def center_image_in_square(img):
    h, w = img.shape[:2]
    side_length = max(h, w)  # 获取最大的边长
    square = np.zeros((side_length, side_length), dtype=np.uint8)  # 创建一个新的正方形图像
    top, left = (side_length - h) // 2, (side_length - w) // 2  # 计算顶部和左侧的边距以居中图像
    square[top:top + h, left:left + w] = img  # 将图像置于正方形的中心
    return square

def auto_crop_and_save_images(root_dir, output_dir, threshold=200, min_area=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录

    img_names = os.listdir(root_dir)
    img_names = [n for n in img_names if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_names.sort()

    for name in img_names:
        img_path = os.path.join(root_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > min_area:
                cropped_img = img[y:y+h, x:x+w]
                centered_img = center_image_in_square(cropped_img)  # 居中裁切图像
                output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_crop_centered_{i}.png")
                cv2.imwrite(output_path, centered_img)  # 保存居中后的正方形图像

if __name__ == '__main__':
    root_dir = './ex-corephoto2024-1-21-20-1705838854133'
    output_dir = './ex-corephototrim'
    threshold = 100  # 提高阈值减少裁切区域
    min_area = 11000  # 增大最小区域阈值以减少小区域的裁切
    auto_crop_and_save_images(root_dir, output_dir, threshold, min_area)
