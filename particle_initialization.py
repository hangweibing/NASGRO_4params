import numpy as np
import tensorflow as tf
from scipy.stats import norm


def lhs_normal_sampling(n_samples, mean, std):
    """拉丁超立方采样 - 正态分布"""
    lhs_samples = np.zeros(n_samples)
    intervals = np.arange(n_samples, dtype=float) / n_samples
    
    for i in range(n_samples):
        lower = intervals[i]
        upper = intervals[i] + 1.0 / n_samples
        lhs_samples[i] = np.random.uniform(lower, upper)
    
    np.random.shuffle(lhs_samples)
    
    # 防止极端值：将范围限制在[1e-10, 1-1e-10]之间
    # 避免norm.ppf(0)=-inf和norm.ppf(1)=+inf的情况
    lhs_samples = np.clip(lhs_samples, 1e-8, 1-1e-8)
    normal_samples = norm.ppf(lhs_samples, loc=mean, scale=std)
    
    return normal_samples


def lhs_uniform_sampling(n_samples, min_val, max_val):
    """拉丁超立方采样 - 均匀分布"""
    lhs_samples = np.zeros(n_samples)
    intervals = np.arange(n_samples, dtype=float) / n_samples
    
    for i in range(n_samples):
        lower = intervals[i]
        upper = intervals[i] + 1.0 / n_samples
        lhs_samples[i] = np.random.uniform(lower, upper)
    
    np.random.shuffle(lhs_samples)
    uniform_samples = min_val + lhs_samples * (max_val - min_val)
    
    return uniform_samples


def init_uncertain_param_lhs(shape, dtype, dist, value, param_name=None):
    """
    使用LHS采样初始化不确定参数

    参数:
        shape: TensorFlow张量形状
        dtype: 数据类型
        dist: 分布类型 ('N'或'U')
        value: 分布参数列表
        param_name: 参数名称，用于特殊处理（如D参数的对数变换）

    返回:
        TensorFlow Variable
    """
    n_particles = shape[-1]

    if dist == 'N':
        if len(value) == 3:
            mean, std, offset = value[0], value[1], value[2]
            # 特殊处理D参数：对log10(D)进行采样，确保D始终为正数
            if param_name == 'D':
                # 计算log10空间的参数
                log10_mean = np.log10(mean + offset) if (mean + offset) > 0 else np.log10(abs(mean + offset))
                log10_std = std / (mean + offset) / np.log(10) if (mean + offset) != 0 else 0.1
                log10_samples = lhs_normal_sampling(n_particles, log10_mean, log10_std)
                samples = 10 ** log10_samples
            else:
                samples = lhs_normal_sampling(n_particles, mean, std) + offset
        elif len(value) == 2:
            mean, std = value[0], value[1]
            # 特殊处理D参数：对log10(D)进行采样
            if param_name == 'D':
                log10_mean = np.log10(mean) if mean > 0 else np.log10(abs(mean))
                log10_std = std / mean / np.log(10) if mean != 0 else 0.1
                log10_samples = lhs_normal_sampling(n_particles, log10_mean, log10_std)
                samples = 10 ** log10_samples
            else:
                samples = lhs_normal_sampling(n_particles, mean, std)
        else:
            raise ValueError("正态分布参数数量错误，应为2或3个参数")
    elif dist == 'U':
        if len(value) == 2:
            min_val, max_val = value[0], value[1]
            # 对于D参数的均匀分布，也使用对数变换
            if param_name == 'D':
                if min_val > 0 and max_val > 0:
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    log_samples = lhs_uniform_sampling(n_particles, log_min, log_max)
                    samples = 10 ** log_samples
                else:
                    samples = lhs_uniform_sampling(n_particles, min_val, max_val)
            else:
                samples = lhs_uniform_sampling(n_particles, min_val, max_val)
        else:
            raise ValueError("均匀分布参数数量错误，应为2个参数")
    else:
        raise ValueError("不支持的分布类型")

    return tf.Variable(samples.reshape(shape), dtype=dtype)


def init_uncertain_param_tf(shape, dtype, dist, value, param_name=None):
    """
    使用TensorFlow内置随机采样初始化不确定参数

    参数:
        shape: TensorFlow张量形状
        dtype: 数据类型
        dist: 分布类型 ('N'或'U')
        value: 分布参数列表
        param_name: 参数名称，用于特殊处理（如D参数的对数变换）

    返回:
        TensorFlow Variable
    """
    if dist == 'N':
        if len(value) == 3:
            mean, std, offset = value[0], value[1], value[2]
            # 特殊处理D参数：对log10(D)进行采样，确保D始终为正数
            if param_name == 'D':
                # 计算log10空间的参数
                log10_mean = np.log10(mean + offset) if (mean + offset) > 0 else np.log10(abs(mean + offset))
                log10_std = std / (mean + offset) / np.log(10) if (mean + offset) != 0 else 0.1
                log10_samples = tf.random.normal(shape=shape, mean=log10_mean, stddev=log10_std, dtype=dtype)
                samples = 10 ** log10_samples
                return tf.Variable(samples)
            else:
                return tf.Variable(tf.random.normal(shape=shape, mean=value[0], stddev=value[1], dtype=dtype) + value[2])
        elif len(value) == 2:
            mean, std = value[0], value[1]
            # 特殊处理D参数：对log10(D)进行采样
            if param_name == 'D':
                log10_mean = np.log10(mean) if mean > 0 else np.log10(abs(mean))
                log10_std = std / mean / np.log(10) if mean != 0 else 0.1
                log10_samples = tf.random.normal(shape=shape, mean=log10_mean, stddev=log10_std, dtype=dtype)
                samples = 10 ** log10_samples
                return tf.Variable(samples)
            else:
                return tf.Variable(tf.random.normal(shape=shape, mean=value[0], stddev=value[1], dtype=dtype))
        else:
            raise ValueError("正态分布参数数量错误，应为2或3个参数")
    elif dist == 'U':
        if len(value) == 2:
            min_val, max_val = value[0], value[1]
            # 对于D参数的均匀分布，也使用对数变换
            if param_name == 'D':
                if min_val > 0 and max_val > 0:
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    log_samples = tf.random.uniform(shape=shape, minval=log_min, maxval=log_max, dtype=dtype)
                    samples = 10 ** log_samples
                    return tf.Variable(samples)
                else:
                    return tf.Variable(tf.random.uniform(shape=shape, minval=value[0], maxval=value[1], dtype=dtype))
            else:
                return tf.Variable(tf.random.uniform(shape=shape, minval=value[0], maxval=value[1], dtype=dtype))
        else:
            raise ValueError("均匀分布参数数量错误，应为2个参数")
    else:
        raise ValueError("不支持的分布类型")

