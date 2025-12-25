"""
参数配置文件
包含所有可配置的参数设置
"""

import os

# ------------------------------ 全局参数设置 ------------------------------
# 模型选择：'PARIS' 或 'NASGRO'
MODE = 'NASGRO'

# 粒子初始化方法：'tf' 使用TensorFlow内置随机采样, 'lhs' 使用拉丁超立方采样
INIT_METHOD = 'lhs'

# 结果输出目录
RESULTS_DIR = 'results'

# 粒子数量与可靠性指标设置
# NPARTICLE: 粒子总数
# reliability_rate: 可靠性指标阈值
# MAXCOUNT: 最大允许超限粒子数
NPARTICLE = int(1E4)
reliability_rate = 0.025
MAXCOUNT = int(NPARTICLE * reliability_rate)

# ------------------------------ 计算资源配置 ------------------------------
# CPU计算开关设置
# 注意：粒子数量超过1e7后请勿在一般电脑上使用CPU计算
# 如需强制使用CPU，请取消下面的注释
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------------------------------ 材料与载荷参数定义 ------------------------------
# Kc: 断裂韧度 (MPa*m^0.5)
# safe_factor: 安全系数
# critical_crack_length: 临界裂纹尺寸 (mm)
# crack_limit: 裂纹临界长度限制 (mm) = critical_crack_length / safe_factor
# width: 板材宽度 (mm)
# load: 载荷参数 [均值, 标准差]
# stress: 应力参数（确定值）
# length: 初始裂纹长度参数 [均值, 标准差, 偏移]
Kc = 50
safe_factor = 2
critical_crack_length = 25  # 临界裂纹尺寸
crack_limit = critical_crack_length / safe_factor  # 裂纹临界长度限制
width = 50
load = [1, 0.1]
stress = load[0]  # 确定性应力值
length = [0.01, 0.01, 1.27]

# 材料参数字典，包含PARIS和NASGRO两种模型的参数
# 'N' 表示正态分布参数格式: [均值, 标准差] 或 [均值, 标准差, 偏移]
paramDict = {
    'PARIS': {
        'C': [3.2e-10, 3e-11],      # Paris方程常数C [均值, 标准差]
        'm': [1.6, 0.08],             # Paris方程指数m [均值, 标准差]
        '_uncertainParam': {'C': 'N', 'm': 'N', 'crack_length': 'N'}  # 不确定参数分布类型
    },
    'NASGRO': {
        'D': [2e-9, 8e-10],         # NASGRO方程常数D [均值, 标准差]
        'p': [1.2, 0.05],            # NASGRO方程指数p [均值, 标准差]
        'dKthr': [7.23, 0.5],        # 阈值应力强度因子范围 [均值, 标准差]
        'A': [74.1, 2.0],            # 断裂韧度参数 [均值, 标准差]
        '_uncertainParam': {'D': 'N', 'p': 'N', 'dKthr': 'N', 'A': 'N', 'crack_length': 'N'}  # 不确定参数分布类型
    }
}
