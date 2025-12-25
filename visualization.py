# =============================================================================
# 可视化模块
# 统一管理裂纹扩展预测结果的可视化功能
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


def plot_crack_length_prediction(data, filename, step_count):
    """
    绘制裂纹长度预测结果

    参数:
        data: 裂纹长度数据 [循环数, 2.5%分位数, 均值, 97.5%分位数]
        filename: 保存的文件名
        step_count: 当前步数，用于标题显示
    """
    # 提取数据
    cycles = data[:, 0]           # 循环数
    percentile_2_5 = data[:, 1]   # 2.5%分位数
    mean = data[:, 2]             # 均值
    percentile_97_5 = data[:, 3]  # 97.5%分位数

    # 创建图形
    plt.figure(figsize=(12, 7))

    # 绘制置信区间（阴影）
    plt.fill_between(cycles, percentile_2_5, percentile_97_5,
                     alpha=0.3, color='skyblue', label='95% Confidence Interval')

    # 绘制分位数线
    plt.plot(cycles, percentile_2_5, '--', color='blue', linewidth=1.5, alpha=0.7, label='2.5% Percentile')
    plt.plot(cycles, percentile_97_5, '--', color='blue', linewidth=1.5, alpha=0.7, label='97.5% Percentile')

    # 绘制均值线（主曲线）
    plt.plot(cycles, mean, '-', color='red', linewidth=2.5, label='Mean')

    # 设置标签
    plt.xlabel('Loading Cycles', fontsize=14, fontweight='bold')
    plt.ylabel('Crack Length (mm)', fontsize=14, fontweight='bold')
    plt.title(f'{filename.split("_")[0]} Model - Crack Length Prediction (Step {step_count})',
              fontsize=16, fontweight='bold')

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=12)

    # 优化显示
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.tight_layout()

    # 保存图片
    output_file = filename.replace('.csv', f'_step_{step_count}_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Step {step_count}: 图片已保存: {output_file}")

    # 显示图形（可选，如果不需要显示可以注释掉）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


# =============================================================================
# 扩展的可视化功能
# =============================================================================

def plot_particle_parameters_evolution(csv_filename, n_particles, save_dir='.', param_names=None):
    """
    通过读取CSV文件绘制粒子参数随时间演化的边缘分布可视化

    CSV文件格式：每一行是一个粒子，包含四个参数[D, p, dKthr, A]
    数据按时间顺序叠加：前N行是初始分布，N+1到2N行是第一次更新，依此类推

    参数:
        csv_filename: CSV文件名
        n_particles: 粒子总数N
        save_dir: 保存目录
        param_names: 参数名称列表，默认['D', 'p', 'dKthr', 'A']
    """
    import pandas as pd
    import os

    if param_names is None:
        param_names = ['D', 'p', 'dKthr', 'A']

    # 检查CSV文件是否存在
    if not os.path.exists(csv_filename):
        print(f"警告：CSV文件 {csv_filename} 不存在，跳过可视化")
        return

    # 读取CSV文件
    try:
        data = pd.read_csv(csv_filename)
        total_rows = len(data)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 计算更新步骤数
    n_steps = total_rows // n_particles
    if n_steps == 0:
        print(f"警告：CSV文件行数({total_rows})少于粒子数({n_particles})，跳过可视化")
        return

    # 创建子图 - 四个参数分别展示
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 为每个参数创建时间演化图
    for param_idx in range(min(len(param_names), 4)):
        ax = axes[param_idx]
        param_name = param_names[param_idx]

        # 为每个时间步骤绘制数据
        for step in range(n_steps):
            start_row = step * n_particles
            end_row = min((step + 1) * n_particles, total_rows)

            # 获取当前步骤的参数值
            param_values = data[param_name].iloc[start_row:end_row].values

            # 在x=step的位置绘制所有粒子的参数值
            x_positions = [step] * len(param_values)
            y_positions = param_values

            # 绘制散点
            ax.scatter(x_positions, y_positions, alpha=0.6, s=15,
                      label=f'Step {step}' if step < 5 else "")

        # 设置标签和标题
        ax.set_xlabel('Update Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{param_name} Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{param_name} Parameter Evolution', fontsize=14, fontweight='bold')

        # 设置x轴为整数
        ax.set_xticks(range(n_steps))

        # 添加网格
        ax.grid(True, alpha=0.3)

    # 添加图例（只在第一个子图显示，避免重复）
    if n_steps <= 5:
        axes[0].legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_file = f'{save_dir}/particle_params_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"粒子参数演化图已保存: {output_file}")
    plt.close()


def plot_particle_parameters_evolution_3d(csv_filename, n_particles, save_dir='.', param_names=None):
    import pandas as pd
    import os
    from matplotlib import cm

    if param_names is None:
        param_names = ['D', 'p', 'dKthr', 'A']

    if not os.path.exists(csv_filename):
        print(f"警告：CSV文件 {csv_filename} 不存在，跳过可视化")
        return

    try:
        data = pd.read_csv(csv_filename)
        total_rows = len(data)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    n_steps = total_rows // n_particles
    if n_steps == 0:
        print(f"警告：CSV文件行数({total_rows})少于粒子数({n_particles})，跳过可视化")
        return

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('white')
    
    colors = cm.viridis(np.linspace(0, 1, n_steps))

    for param_idx in range(min(len(param_names), 4)):
        ax = fig.add_subplot(2, 2, param_idx + 1, projection='3d')
        param_name = param_names[param_idx]

        for step in range(n_steps):
            start_row = step * n_particles
            end_row = min((step + 1) * n_particles, total_rows)
            param_values = data[param_name].iloc[start_row:end_row].values

            if len(param_values) > 5:
                try:
                    kde = gaussian_kde(param_values)
                    y_grid = np.linspace(np.min(param_values), np.max(param_values), 100)
                    z_values = kde(y_grid)

                    x_vals = np.full_like(y_grid, step)
                    
                    ax.plot(x_vals, y_grid, z_values, color=colors[step], 
                           alpha=0.85, linewidth=2.5, zorder=step)
                    
                    ax.plot(x_vals, y_grid, np.zeros_like(z_values), 
                           color=colors[step], alpha=0.15, linewidth=1, zorder=0)
                    
                    vertices = [(x_vals[0], y_grid[0], 0)] + \
                              list(zip(x_vals, y_grid, z_values)) + \
                              [(x_vals[-1], y_grid[-1], 0)]
                    from matplotlib.patches import Polygon
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    
                    verts = [list(zip(x_vals, y_grid, z_values))]
                    poly = Poly3DCollection(verts, alpha=0.25, facecolor=colors[step], 
                                           edgecolor='none', zorder=step)
                    ax.add_collection3d(poly)
                    
                except:
                    pass

        ax.set_xlabel('Update Step', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel(f'{param_name}', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel('Probability Density', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(f'{param_name} Parameter Evolution', fontsize=14, fontweight='bold', pad=15)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=10)
        ax.view_init(elev=20, azim=45)
        
        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['inward_factor'] = 0

    plt.tight_layout(pad=3.0)
    output_file = f'{save_dir}/particle_params_evolution_3d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"粒子参数3D演化图已保存: {output_file}")
    plt.close()


