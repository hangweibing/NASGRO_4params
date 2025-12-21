
# ------------------------------ 导入模块 ------------------------------
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 导入应力强度因子计算模块
from stress_intensity_factor import calc_K_for_tip
# 导入裂纹扩展模型模块
from crack_growth_models import predict_NASGRO_tf, predict_PARIS_tf
# 导入可视化模块
from visualization import plot_crack_length_prediction
# 导入粒子初始化模块
from particle_initialization import init_uncertain_param_lhs, init_uncertain_param_tf




# ------------------------------ 全局参数设置 ------------------------------
# 模型选择：'PARIS' 或 'NASGRO'
MODE = 'NASGRO'

# 粒子初始化方法：'tf' 使用TensorFlow内置随机采样, 'lhs' 使用拉丁超立方采样
INIT_METHOD = 'tf'

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
# width: 板材宽度 (mm)
# load: 载荷参数 [均值, 标准差]
# stress: 应力参数 [均值, 标准差]
# length: 初始裂纹长度参数 [均值, 标准差, 偏移]
Kc = 50
safe_factor = 2
width = 50
load = [1, 0.1]
stress = [load[0], 0]
length = [0.01, 0.001, 1.27]

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
        'A': [74.1, 5.0],            # 断裂韧度参数 [均值, 标准差]
        '_uncertainParam': {'D': 'N', 'p': 'N', 'dKthr': 'N', 'A': 'N', 'crack_length': 'N'}  # 不确定参数分布类型
    }
}

# ------------------------------ 裂纹扩展模型方程 ------------------------------




# 裂纹扩展预测函数已移至 crack_growth_models.py 模块
# ------------------------------ 裂纹扩展主类 ------------------------------

class CrackGrowth():
    """
    裂纹扩展模拟主类

    基于粒子滤波的方法，模拟裂纹在循环载荷下的扩展过程
    支持PARIS和NASGRO两种疲劳裂纹扩展模型
    """

    # 各模型所需参数列表
    paramDict = {
        'NASGRO': ['D', 'p', 'dKthr', 'A'],  # NASGRO模型参数
        'PARIS': ['C', 'm']                   # Paris模型参数
    }

    # 常量参数列表
    constList = ['crack_length', 'b', 'crack_limit', 'dSigma']

    def __init__(self, mode='NASGRO', crack_limit=0, dtype=tf.dtypes.float32, shape=[]):
        """
        初始化裂纹扩展模拟器

        参数:
            mode: 模型类型 ('NASGRO' 或 'PARIS')
            crack_limit: 裂纹临界长度限制 (mm)
            dtype: TensorFlow数据类型
            shape: 张量形状，用于粒子滤波
        """
        from partialfilter import ParticleFilter

        # 初始化计算步数计数器
        self.step = tf.Variable(0)

        # 模型配置
        self.mode = mode
        self.crack_limit = crack_limit

        # 统计计数器
        self.crack_count = tf.Variable(0, dtype='int64')

        # 数据类型和形状配置
        self.dtype = dtype
        self.shape = shape

        # 初始化粒子滤波器
        self.check = ParticleFilter(self)

        # 95%置信度裂纹长度初始化
        self.crack_length_95 = 0

        # 时间戳记录，用于性能监控
        self.start = tf.timestamp()

    def log_particle_params(self, filename='particle_params.csv'):
        """
        记录所有粒子的参数状态到CSV文件

        将每个粒子的四个NASGRO参数（D, p, dKthr, A）保存到CSV文件中。
        每一列代表一个参数，每一行代表一个粒子。
        每次调用时在文件末尾追加新的数据。

        参数:
            filename: CSV文件名，默认为'particle_params.csv'
        """
        import csv
        import os

        # 获取当前所有粒子的参数值
        D_vals = self.D.numpy().flatten()
        p_vals = self.p.numpy().flatten()
        dKthr_vals = self.dKthr.numpy().flatten()
        A_vals = self.A.numpy().flatten()

        # 创建数据行：每一行是一个粒子，包含四个参数
        data_rows = []
        for i in range(len(D_vals)):
            data_rows.append([D_vals[i], p_vals[i], dKthr_vals[i], A_vals[i]])

        # 检查文件是否存在以确定是否需要写入表头
        file_exists = os.path.isfile(filename)

        # 追加模式写入CSV文件
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 如果是新文件，写入表头
            if not file_exists:
                writer.writerow(['D', 'p', 'dKthr', 'A'])

            # 写入所有粒子的参数数据
            writer.writerows(data_rows)
    def log(self):
        """
        记录当前裂纹长度到GPU历史记录

        将当前时刻的所有粒子裂纹长度添加到历史记录中
        """
        self.crack_length_history_GPU = tf.concat([self.crack_length_history_GPU, self.crack_length], axis=0)

    def log_to_cpu(self):
        """
        将GPU上的历史记录转移到CPU内存

        在GPU计算时使用，避免显存溢出
        定期将历史数据转移到CPU，释放GPU内存
        """
        if 'crack_length_history_CPU' in self.__dict__.keys():
            # 追加新数据到现有CPU历史记录
            self.crack_length_history_CPU = np.vstack([self.crack_length_history_CPU, self.crack_length_history_GPU[1:]])
        else:
            # 首次转移，创建CPU历史记录
            self.crack_length_history_CPU = self.crack_length_history_GPU.numpy()

        # 清理GPU历史记录，保留当前状态作为新起点
        del self.crack_length_history_GPU
        self.crack_length_history_GPU = self.crack_length
        return
    def setParamDict(self, paramDict='', **kwargs):
        """
        批量设置模型参数，自动分离确定参数和不确定参数

        根据参数字典中的_uncertainParam配置，自动判断参数类型
        确定参数使用固定值，不确定参数使用随机分布采样

        参数:
            paramDict: 参数字典，包含参数值和不确定性配置
        """
        for key in paramDict:
            if not '_' in key:  # 跳过配置参数（如_uncertainParam）
                if key in paramDict['_uncertainParam']:
                    # 不确定参数：使用指定分布类型
                    self.setParam(param=key, dist=paramDict['_uncertainParam'][key], value=paramDict[key])
                else:
                    # 确定参数：使用固定值
                    self.setParam(param=key, dist='single', value=paramDict[key])

    def setParam(self, param='', dist='N', **kwargs):
        """
        设置单个模型参数

        支持确定参数（固定值）和不确定参数（随机分布）

        参数:
            param: 参数名称
            dist: 分布类型 ('single'为确定参数，'N'为正态分布，'U'为均匀分布)
            **kwargs: 必须包含'value'键，参数值或分布参数
        """
        if param in self.paramDict[self.mode] or param in self.constList:
            if dist == 'single':
                # 创建确定参数常量
                exec('self.%s=tf.constant(kwargs["value"], dtype=self.dtype)' % (param))
            else:
                # 创建不确定参数变量（随机采样）
                exec('self.%s=self.setUncertainParam(dist, kwargs["value"])' % (param))
        else:
            raise KeyError("参数 '%s' 不在允许的参数列表中" % param)
    def getParam(self, param=''):
        """
        获取参数的当前数值

        参数:
            param: 参数名称

        返回:
            numpy数组格式的参数值
        """
        if param in self.paramDict[self.mode] or param in self.constList:
            return eval('self.%s.numpy()' % param)
        else:
            raise KeyError("参数 '%s' 不在允许的参数列表中" % param)

    def setUncertainParam(self, dist, value):
        """
        对不确定参数进行随机采样初始化

        支持正态分布(N)和均匀分布(U)两种类型

        正态分布参数格式:
            2参数: [均值, 标准差]
            3参数: [均值, 标准差, 均值偏移]

        均匀分布参数格式:
            2参数: [最小值, 最大值]

        参数:
            dist: 分布类型 ('N'正态分布 或 'U'均匀分布)
            value: 分布参数列表

        返回:
            TensorFlow Variable，包含随机采样的参数值
        """
        if INIT_METHOD == 'lhs':
            return init_uncertain_param_lhs(self.shape, self.dtype, dist, value)
        else:
            return init_uncertain_param_tf(self.shape, self.dtype, dist, value)
    def summary(self):
        """
        执行单步统计和日志记录

        功能：
        1. 统计超过临界裂纹长度的粒子数量
        2. 检测数值计算错误（NaN值）
        3. 记录当前裂纹长度历史
        4. 每10000步输出统计信息并转移数据到CPU
        """
        # 统计超过临界值的粒子数量
        self.crack_count = tf.math.count_nonzero(tf.maximum(self.crack_length - self.crack_limit, 0))

        # 统计数值错误（NaN值）
        self.error_count = tf.math.count_nonzero(tf.math.is_nan(self.crack_length))

        # 记录当前状态到历史
        self.log()

        # 每10000步进行统计输出和数据转移
        if self.step % 10000 == 0:
            if self.step > 0:
                # 计算时间间隔
                stop = tf.timestamp()
                # 输出详细统计信息
                tf.print(
                    self.step, 'Step\ttime:', stop - self.start,
                    '\tmax:', tf.reduce_max(self.crack_length_history_GPU[-1]),
                    '\tmin:', tf.reduce_min(self.crack_length_history_GPU[-1]),
                    '\tavg:', tf.reduce_mean(self.crack_length_history_GPU[-1]),
                    '\tcount:', self.crack_count
                )

            # 数据转移到CPU（释放GPU内存）
            if True:  # 总是执行数据转移
                self.log_to_cpu()

            # 重置时间戳
            self.start = tf.timestamp()

        # 步数递增
        self.step = self.step + 1
        return
    def growth(self):
        """
        执行单步裂纹扩展计算

        根据选择的模型（PARIS或NASGRO）计算裂纹长度增量，
        更新所有粒子的裂纹长度，然后执行统计和记录
        """
        if self.mode == 'NASGRO':
            # 使用NASGRO方程计算裂纹扩展
            self.crack_length = self.crack_length.assign_add(
                predict_NASGRO_tf(self.crack_length, self.b, self.D, self.p, self.A, self.dKthr, self.dSigma)
            )
        elif self.mode == 'PARIS':
            # 使用Paris方程计算裂纹扩展
            self.crack_length = self.crack_length.assign_add(
                predict_PARIS_tf(self.crack_length, self.b, self.C, self.m, self.dSigma)
            )

        # 执行统计和日志记录
        self.summary()

    # @tf.function  # 可选：启用TensorFlow图模式优化
    def growth_loop(self):
        """
        主模拟循环控制

        控制整个疲劳裂纹扩展过程：
        1. 内循环：当超限粒子数未达到阈值时，继续模拟
        2. 外循环：当95%置信度裂纹长度未达到临界值时，执行粒子滤波更新
        3. 定期保存中间结果到文件
        """
        # 检查所有必需参数是否已设置
        if all(param in self.__dict__.keys() for param in self.paramDict[self.mode]) and \
           all(const in self.__dict__.keys() for const in self.constList):

            # 初始化GPU历史记录
            self.crack_length_history_GPU = self.crack_length
            maxcount = tf.constant(MAXCOUNT, dtype='int64')

            # 初始化参数更新计数器
            update_count = 0

            # 外循环：继续直到97.5%置信度裂纹长度达到临界值 12.5mm
            while self.crack_length_95 < self.crack_limit:

                # 内循环：继续直到超限粒子数达到阈值
                while self.crack_count < maxcount:
                    self.growth() # 执行单步裂纹扩展计算，记录粒子裂纹长度到GPU，每10000步输出统计信息并转移数据到CPU，统计超过临界值的粒子数量

                # 数据转移到CPU
                if True:  # 总是执行
                    self.log_to_cpu()

                # 准备输出数据：计算统计量
                t_ = self.crack_length_history_CPU.copy()
                t_.sort(axis=-1)  # 按裂纹长度排序
                
                # 创建输出数组：
                t = np.vstack((
                    np.arange(1, t_.shape[0]),           # 循环数
                    t_[1:, MAXCOUNT],
                    np.average(t_[1:], axis=-1),
                    t_[1:, t_.shape[-1] - MAXCOUNT]
                )).T

                # 保存结果到CSV文件
                np.savetxt('%s_%d.csv' % (MODE, load[0]), t, delimiter=',')

                # 执行粒子滤波更新参数
                self.crack_length_95 = self.check.check()

                # 记录更新后的粒子参数状态
                self.log_particle_params()

                # 更新参数分布可视化
                from visualization import plot_particle_parameters_evolution
                plot_particle_parameters_evolution('particle_params.csv', NPARTICLE)

                # 增加更新计数器
                update_count += 1

                # 绘制当前预测效果
                filename = '%s_%d.csv' % (MODE, load[0])
                plot_crack_length_prediction(t, filename, update_count)

                # 重置超限计数器
                self.crack_count = 0
        else:
            raise KeyError("缺少必需的模型参数或常量参数")


# ------------------------------ 主函数 ------------------------------


def main():
    """
    主程序入口

    执行完整的疲劳裂纹扩展模拟：
    1. 初始化CrackGrowth模拟器
    2. 设置模型参数（确定参数和不确定参数）
    3. 运行模拟循环
    4. 自动保存结果到CSV文件
    """
    # 初始化裂纹扩展模拟器
    # mode: 模型类型 (PARIS/NASGRO)
    # crack_limit: 裂纹临界长度 (25/2 = 12.5mm)
    # shape: 张量形状 [1, NPARTICLE] 表示单批次，NPARTICLE个粒子
    cg = CrackGrowth(mode=MODE, crack_limit=25/2, shape=tf.TensorShape([1, NPARTICLE]))

    # 设置模型参数字典（自动区分确定参数和不确定参数）
    cg.setParamDict(paramDict=paramDict[MODE])

    # 记录初始粒子参数状态
    cg.log_particle_params()

    # 生成初始参数分布可视化
    from visualization import plot_particle_parameters_evolution
    plot_particle_parameters_evolution('particle_params.csv', NPARTICLE)

    # 单独设置其他参数
    # 初始裂纹长度：正态分布 [均值, 标准差, 偏移]
    cg.setParam(param='crack_length', dist='N', value=length)
    # 应力范围：正态分布
    cg.setParam(param='dSigma', dist='N', value=stress)
    # 板宽：确定参数
    cg.setParam(param='b', dist='single', value=50)

    # 执行主模拟循环
    cg.growth_loop()

    print()  # 输出空行


    return


# ------------------------------ 程序入口 ------------------------------
if __name__ == '__main__':
    main()

