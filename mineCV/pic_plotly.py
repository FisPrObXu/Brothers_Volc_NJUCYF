import pandas as pd
import plotly.graph_objects as go

# GitHub上的CSV文件URL
csv_url = 'https://raw.githubusercontent.com/FisPrObXu/Brothers_Volc_NJUCYF/main/data_file/brothers_cross-section_points_rotated.csv'

# 从GitHub直接加载数据
data = pd.read_csv(csv_url)

# 过滤出X坐标为179.0729的点，并准备图片URLs
# 假定这些点的图片名称按照特定的规则命名，并存储在GitHub的特定目录下
image_numbers = [1, 19, 37, 56, 74, 93]
image_base_url = 'https://raw.githubusercontent.com/FisPrObXu/Brothers_Volc_NJUCYF/main/corephoto/1530A/376U1530A'
images_urls = {num: f"{image_base_url}{num}.png" for num in image_numbers}

# 计算Z值的范围用于颜色映射
z_min, z_max = data['Z'].min(), data['Z'].max()

# 创建图表
fig = go.Figure()

for index, row in data.iterrows():
    if row['X'] == 179.0729 and index + 1 in image_numbers:
        color = 'red'
        hover_text = f"<img src='{images_urls.pop(0)}' width='200px'>"
    else:
        # 根据Z值映射蓝色的不同深浅
        z_normalized = (row['Z'] - z_min) / (z_max - z_min)
        color = f"rgba(0, 0, 255, {0.3 + 0.7 * z_normalized})"
        hover_text = "Z value: {:.2f}".format(row['Z'])
    
    fig.add_trace(go.Scatter(
        x=[row['X']],  
        y=[row['Z']],
        mode='markers',
        marker=dict(size=10, color=color),
        hovertext=hover_text,
        hoverinfo='text'
    ))

# 更新图表布局
fig.update_layout(
    title="鼠标悬停在特定点显示图片",
    xaxis_title="X坐标",
    yaxis_title="Z值",
    hovermode="closest"
)

# 显示图表
fig.show()
