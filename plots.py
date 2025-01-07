import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# 设置渲染器
pio.renderers.default = 'notebook'

# 读取CSV文件
df_final_state = pd.read_csv("3D-EP.csv")

# 示例数据（需要替换为你的实际数据）
theta_bins_rad = np.linspace(0, np.pi / 2, 10)  # theta 角分区（弧度制，0到90度）
phi_bins_rad = np.linspace(0, 2 * np.pi, 20)    # phi 角分区（弧度制，0到360度）

# 将角度从度转换为弧度
df_final_state['theta_f_rad'] = df_final_state['theta_f'] * np.pi / 180
df_final_state['phi_rad'] = df_final_state['phi'] * np.pi / 180

# 计算每个区域中的散点数并归一化为概率
H, theta_edges, phi_edges = np.histogram2d(df_final_state['theta_f_rad'], df_final_state['phi_rad'], bins=[theta_bins_rad, phi_bins_rad])
probability = H / H.sum()  # 归一化得到概率分布

ekt_avg = np.zeros(H.shape)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        region_mask = (df_final_state['theta_f_rad'] >= theta_edges[i]) & (df_final_state['theta_f_rad'] < theta_edges[i+1]) & \
                      (df_final_state['phi_rad'] >= phi_edges[j]) & (df_final_state['phi_rad'] < phi_edges[j+1])
        ekt_avg[i, j] = df_final_state.loc[region_mask, 'ekt'].mean()

# 创建极坐标柱状图数据
x_faces = []
y_faces = []
z_faces = []
colors = []

# 遍历每个 bin，构造对应的极坐标柱状图
for i in range(len(theta_bins_rad) - 1):
    for j in range(len(phi_bins_rad) - 1):
        # 获取 bin 的边界
        theta_min, theta_max = theta_bins_rad[i], theta_bins_rad[i + 1]
        phi_min, phi_max = phi_bins_rad[j], phi_bins_rad[j + 1]

        # 获取概率分布高度
        dz = probability[i, j]  # 概率分布作为柱子高度
        color = ekt_avg[i, j]   # 颜色映射到能量分布

        # 构造该区域的四个角点
        # 极坐标转换为笛卡尔坐标
        x_corners = [
            theta_min * np.cos(phi_min),
            theta_min * np.cos(phi_max),
            theta_max * np.cos(phi_max),
            theta_max * np.cos(phi_min),
        ]
        y_corners = [
            theta_min * np.sin(phi_min),
            theta_min * np.sin(phi_max),
            theta_max * np.sin(phi_max),
            theta_max * np.sin(phi_min),
        ]
        z_corners = [0, 0, 0, 0]  # 柱子底部坐标

        # 将顶面坐标抬高到 dz
        x_top = x_corners
        y_top = y_corners
        z_top = [dz, dz, dz, dz]

        # 添加底面
        x_faces.append(x_corners)
        y_faces.append(y_corners)
        z_faces.append(z_corners)
        colors.append(color)

        # 添加顶面
        x_faces.append(x_top)
        y_faces.append(y_top)
        z_faces.append(z_top)
        colors.append(color)

        # 添加侧面（连接底面和顶面）
        for k in range(4):
            x_faces.append([x_corners[k], x_corners[(k + 1) % 4], x_top[(k + 1) % 4], x_top[k]])
            y_faces.append([y_corners[k], y_corners[(k + 1) % 4], y_top[(k + 1) % 4], y_top[k]])
            z_faces.append([z_corners[k], z_corners[(k + 1) % 4], z_top[(k + 1) % 4], z_top[k]])
            colors.append(color)

# 创建 3D 表面
fig = go.Figure()

# 遍历每个面添加到图形中
for i in range(len(x_faces)):
    fig.add_trace(go.Mesh3d(
        x=x_faces[i],
        y=y_faces[i],
        z=z_faces[i],
        colorbar_title='E<sub>f</sub> (kcal/mol)',
        colorscale=[[0, 'black'], [0.14, 'blue'], [0.28, 'lightblue'], [0.42, 'lightgreen'], [0.57, 'yellow'], [0.71, 'orange'], [0.85, 'red'], [1.0, 'brown']],
        intensity=[colors[i]] * 4,  # 使用同一颜色填充整个面
        showscale=True if i == 0 else False,  # 仅显示一次颜色条
        cmin=np.min(colors),
        cmax=np.max(colors),
        opacity=1,
        hoverinfo='skip'  # 跳过 hover 信息
    ))

# 设置布局
fig.update_layout(
    scene=dict(
        xaxis=dict(title=r'θ (rad)', range=[-2, 2]),
        yaxis=dict(title=r'φ (rad)', range=[-2, 2]),
        zaxis=dict(title='Probability', range=[0, np.max(probability)]),
    ),
    title="3D Energy-Probability Distribution",
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="black",
    )
)

# 显示图形
fig.show()
