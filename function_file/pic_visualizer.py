import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 加载数据集
data = pd.read_csv('/Users/XuuX/Desktop/selected_points_near_planes.csv')

# 定义点击事件处理函数
def on_click(event):
    ix, iy = event.xdata, event.ydata
    # 检查点击位置是否接近任一特定点
    for (x, y, z), image_path in zip(special_coords, images):
        if abs(ix - x) < 0.01 and abs(iy - y) < 0.01:  # 可以根据需要调整阈值
            img = plt.imread(image_path)
            fig, ax = plt.subplots()
            ax.imshow(img)
            plt.axis('off')  # 不显示坐标轴
            plt.show()

# 特定点的坐标和对应的Z值
special_coords = [
    (179.077083, -34.885417, -3005),
    (179.077083, -34.885417, -2705),
    (179.077083, -34.885417, -2405),
    (179.077083, -34.885417, -2105),
    (179.077083, -34.885417, -1805),
    (179.077083, -34.885417, -1505)
]

# 对应的图片路径
images = [
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A93.png',  # 请将'/path/to/'替换为实际的图片路径
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A74.png',
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A56.png',
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A37.png',
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A19.png',
    '/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephoto2024-1-21-20-1705838854133/376U1530A1.png'
]

# 绘制所有点
plt.figure(figsize=(10, 8))
plt.scatter(data['X'], data['Y'], color='blue', label='All Points')

# 标记特定点
for x, y, _ in special_coords:
    plt.scatter(x, y, color='red', s=100)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Click on Red Points to Display Images')
plt.legend()

# 连接点击事件
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
