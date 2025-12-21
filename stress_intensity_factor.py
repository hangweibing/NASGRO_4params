# =============================================================================
# 应力强度因子计算模块
# 用于计算裂纹尖端应力强度因子，方便后续替换为神经网络等复杂模型
# =============================================================================

import tensorflow as tf


def calc_K_for_tip(a, b, dSigma):
    """
    计算方形板长边中点裂纹的应力强度因子范围ΔK

    参数:
        a: 裂纹长度 (mm)
        b: 板宽 (mm)
        dSigma: 应力范围 (MPa)

    返回:
        deltaK: 应力强度因子范围 (MPa*m^0.5)
    """
    deltaK = tf.maximum(dSigma * 139.94684 * (a / 1000) ** 0.31759, 0)
    return deltaK


# =============================================================================
# 未来扩展接口
# =============================================================================

def calc_K_for_tip_nn(a, b, dSigma, model=None):
    """
    使用神经网络计算应力强度因子的接口函数
    用于未来替换传统公式计算

    参数:
        a: 裂纹长度 (mm)
        b: 板宽 (mm)
        dSigma: 应力范围 (MPa)
        model: 神经网络模型（预训练模型）

    返回:
        deltaK: 应力强度因子范围 (MPa*m^0.5)
    """
    if model is None:
        # 如果没有提供模型，使用传统公式
        return calc_K_for_tip(a, b, dSigma)
    else:
        # 使用神经网络模型进行预测
        # 这里需要根据实际模型接口进行调整
        # 示例代码：
        # inputs = tf.stack([a, b, dSigma], axis=-1)
        # deltaK = model.predict(inputs)
        # return deltaK
        raise NotImplementedError("神经网络模型接口尚未实现")

    return calc_K_for_tip(a, b, dSigma)
