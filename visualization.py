# =============================================================================
# 可视化模块
# 统一管理裂纹扩展预测结果的可视化功能
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np


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


