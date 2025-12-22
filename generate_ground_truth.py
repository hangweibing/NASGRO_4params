import csv

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

# 参数设置
D = 2e-9        # NASGRO常数D
p = 1.2         # NASGRO指数p
dKthr = 7.23    # 阈值应力强度因子范围
A = 74.1        # 断裂韧度参数
dSigma = 1.0    # 应力范围
b = 50          # 板宽
a0 = 1.5        # 初始裂纹长度
target_length = 50.0  # 目标裂纹长度

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

    # 每10000步记录一次
    if step % 10000 == 0:
        step_data.append(step)
        length_data.append(crack_length)

    step += 1

    # 如果增量太小，停止循环（防止无限循环）
    if da < 1e-10:
        break

# 保存为CSV文件
with open('ground_truth.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for s, l in zip(step_data, length_data):
        writer.writerow([s, l])
