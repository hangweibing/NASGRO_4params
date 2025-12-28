import csv
import os

# 简化的数值计算函数
def calc_K_for_tip(a, b, dSigma):
    """计算应力强度因子范围ΔK"""
    return max(dSigma * 139.94684 * (a / 1000) ** 0.31759, 0)

def predict_NASGRO_tf(a, b, D, p, A, dKthr, dSigma):
    """基于NASGRO方程计算裂纹扩展速率"""
    deltaK = calc_K_for_tip(a, b, dSigma)
    # 添加数值保护防止NaN
    ratio = deltaK / 0.9 / A
    safe_ratio = min(ratio, 0.999)
    if safe_ratio >= 1.0:
        dka = 0
    else:
        dka = max((deltaK - dKthr) / (1 - safe_ratio) ** 0.5, 0)
    da = max(1000 * D * dka ** p, 0)
    return da

def generate_ground_truth_data(params, output_file='ground_truth.csv'):
    """
    生成真实裂纹扩展数据的函数

    参数:
        params: 参数字典，包含所有必要参数
        output_file: 输出文件路径

    返回:
        生成的数据点数量
    """
    # 提取参数
    D = params['D']
    p = params['p']
    dKthr = params['dKthr']
    A = params['A']
    dSigma = params['dSigma']
    b = params['b']
    a0 = params['a0']
    target_length = params['target_length']
    record_interval = params.get('record_interval', 10000)

    print(f"生成真实数据：D={D}, p={p}, dKthr={dKthr}, A={A}")
    print(f"初始裂纹长度: {a0} mm, 目标长度: {target_length} mm")

    # 初始化裂纹长度
    crack_length = a0

    # 记录数据
    step_data = []
    length_data = []

    # 循环扩展
    step = 0
    max_steps = 1000000  # 防止无限循环
    while crack_length < target_length and step < max_steps:
        # 计算裂纹扩展增量
        da = predict_NASGRO_tf(crack_length, b, D, p, A, dKthr, dSigma)
        # 更新裂纹长度
        crack_length += da

        # 按指定间隔记录
        if step % record_interval == 0:
            step_data.append(step)
            length_data.append(crack_length)

        step += 1

        # 如果增量太小，停止循环（防止无限循环）
        if da < 1e-10:
            break

    # 保存为CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for s, l in zip(step_data, length_data):
            writer.writerow([s, l])

    print(f"真实数据生成完成，共 {len(step_data)} 个数据点")
    print(f"文件保存至: {output_file}")

    return len(step_data)


# 如果直接运行此文件，则使用默认参数生成数据
if __name__ == '__main__':
    # 默认参数（向后兼容）
    default_params = {
        'D': 2e-9,        # NASGRO常数D
        'p': 1.2,         # NASGRO指数p
        'dKthr': 7.23,    # 阈值应力强度因子范围
        'A': 74.1,        # 断裂韧度参数
        'dSigma': 1.0,    # 应力范围
        'b': 50,          # 板宽
        'a0': 1.5,        # 初始裂纹长度
        'target_length': 50.0,  # 目标裂纹长度
        'record_interval': 10000  # 记录间隔步数
    }

    generate_ground_truth_data(default_params)
