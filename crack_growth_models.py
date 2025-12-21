# =============================================================================
# 裂纹扩展模型模块
# 包含多种疲劳裂纹扩展模型的实现，方便后续扩展和替换
# =============================================================================

import tensorflow as tf
from stress_intensity_factor import calc_K_for_tip


def predict_NASGRO_tf(a, b, D, p, A, dKthr, dSigma):
    """
    基于Hartman-Schieve NASGRO方程计算裂纹扩展速率

    参数:
        a: 裂纹长度 (mm)
        b: 板宽 (mm)
        D, p: NASGRO方程材料常数
        A: 断裂韧度参数
        dKthr: 阈值应力强度因子范围
        dSigma: 应力范围 (MPa)

    返回:
        da: 逐循环的裂纹长度增量 (mm/cycle)
    """
    deltaK = calc_K_for_tip(a, b, dSigma)
    dka = tf.maximum((deltaK - dKthr) / (1 - (deltaK / 0.9 / A)) ** 0.5, 0)
    da = tf.maximum(1000 * D * dka ** p, 0)
    return da


def predict_PARIS_tf(a, b, C, m, dSigma):
    """
    基于Paris方程计算裂纹扩展速率

    Paris方程是最经典的疲劳裂纹扩展模型，da/dN = C*(ΔK)^m

    参数:
        a: 裂纹长度 (mm)
        b: 板宽 (mm)
        C, m: Paris方程材料常数
        dSigma: 应力范围 (MPa)

    返回:
        da: 逐循环的裂纹长度增量 (mm/cycle)
    """
    deltaK = calc_K_for_tip(a, b, dSigma)
    da = tf.maximum(1000 * C * deltaK ** m, 0)
    return da


# =============================================================================
# 未来扩展接口
# =============================================================================


