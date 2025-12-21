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


def init_uncertain_param_lhs(shape, dtype, dist, value):
    """
    使用LHS采样初始化不确定参数
    
    参数:
        shape: TensorFlow张量形状
        dtype: 数据类型
        dist: 分布类型 ('N'或'U')
        value: 分布参数列表
    
    返回:
        TensorFlow Variable
    """
    n_particles = shape[-1]
    
    if dist == 'N':
        if len(value) == 3:
            mean, std, offset = value[0], value[1], value[2]
            samples = lhs_normal_sampling(n_particles, mean, std) + offset
        elif len(value) == 2:
            mean, std = value[0], value[1]
            samples = lhs_normal_sampling(n_particles, mean, std)
        else:
            raise ValueError("正态分布参数数量错误，应为2或3个参数")
    elif dist == 'U':
        if len(value) == 2:
            min_val, max_val = value[0], value[1]
            samples = lhs_uniform_sampling(n_particles, min_val, max_val)
        else:
            raise ValueError("均匀分布参数数量错误，应为2个参数")
    else:
        raise ValueError("不支持的分布类型")
    
    return tf.Variable(samples.reshape(shape), dtype=dtype)


def init_uncertain_param_tf(shape, dtype, dist, value):
    """
    使用TensorFlow内置随机采样初始化不确定参数
    
    参数:
        shape: TensorFlow张量形状
        dtype: 数据类型
        dist: 分布类型 ('N'或'U')
        value: 分布参数列表
    
    返回:
        TensorFlow Variable
    """
    if dist == 'N':
        if len(value) == 3:
            return tf.Variable(tf.random.normal(shape=shape, mean=value[0], stddev=value[1], dtype=dtype) + value[2])
        elif len(value) == 2:
            return tf.Variable(tf.random.normal(shape=shape, mean=value[0], stddev=value[1], dtype=dtype))
        else:
            raise ValueError("正态分布参数数量错误，应为2或3个参数")
    elif dist == 'U':
        if len(value) == 2:
            return tf.Variable(tf.random.uniform(shape=shape, minval=value[0], maxval=value[1], dtype=dtype))
        else:
            raise ValueError("均匀分布参数数量错误，应为2个参数")
    else:
        raise ValueError("不支持的分布类型")

