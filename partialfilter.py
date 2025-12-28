# =============================================================================
# 粒子滤波模块 - 正则化粒子滤波实现
# 用于裂纹扩展参数的实时更新和不确定性量化
# =============================================================================
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d

# ================================= 导主程序 =================================
import config
# =============================================================================


# 空间维度参数
pi = 3.1416
n = 5  # 空间维度

# 球体积参数（用于核密度估计）
v_sphere = 1

# 测量噪声方差（观测误差）
R = 0.1

# 正则化参数A（基于空间维度和球体积计算）
A = (8 / v_sphere * (n + 4) * (2 * tf.sqrt(pi)) ** n) ** (1 / (n + 4))

# 平滑带宽h（自适应带宽，基于粒子数量）
h = A * config.NPARTICLE ** (-1 / (n + 4))
# ------------------------------ 粒子滤波器类 ------------------------------

class ParticleFilter():
    """
    正则化粒子滤波器

    实现基于测量数据的参数更新，用于实时校正裂纹扩展模型参数
    支持多维参数空间的粒子重采样和核密度 smoothing
    """

    def __init__(self, growth):
        """
        初始化粒子滤波器

        参数:
            growth: CrackGrowth实例，用于访问裂纹扩展模型
        """
        # 保存裂纹扩展模型引用
        self.growth = growth

        # 获取不确定参数列表（需要通过粒子滤波更新的参数）
        self.checkParamList = config.paramDict[config.MODE]['_uncertainParam'].keys()

        # 检查计数器初始化
        self.Ncheck = 0

        # 加载观测数据文件（用于参数更新的基准数据）
        # 文件格式：第一列为循环数，第二列为观测的裂纹长度
        # check = np.loadtxt('%d.csv' % config.load[0], dtype=float, delimiter=',', encoding='UTF-8')  # 注释：原来加载1.csv等文件
        check = np.loadtxt('ground_truth.csv', dtype=float, delimiter=',', encoding='UTF-8')  # 修改：加载ground_truth.csv

        # 创建观测数据插值函数（3次样条插值）
        # 用于根据循环数获取对应的观测裂纹长度
        self.check_interp_func = interp1d(check.T[0], check.T[1], 3)

        return
    def setParamforPF(self):
        """
        从裂纹扩展模型获取当前参数状态，用于粒子滤波

        将所有不确定参数收集到xp数组中，用于后续的滤波计算
        xp的每一行对应一个参数，每一列对应一个粒子
        """
        # 重置xp数组，避免重复堆叠
        first_param = True
        for key in self.checkParamList:
            if first_param:
                # 初始化xp数组（第一个参数）
                self.xp = self.growth.getParam(key)
                first_param = False
            else:
                # 添加其他参数到xp数组
                self.xp = np.vstack((self.xp, self.growth.getParam(key)))

        # self.xp = self.xp.T  # 可选：转置为[粒子数, 参数数]格式

        # 初始化参数历史记录
        if not 'xphistory' in self.__dict__.keys():
            self.xphistory = [self.xp]

    def setParamforGrowth(self):
        """
        将更新后的参数设置回裂纹扩展模型

        使用粒子滤波更新后的参数值替换模型中的旧参数值
        """
        for i, key in enumerate(self.checkParamList):
            # 获取对应的模型参数变量
            growthParam = eval('self.growth.%s' % key)
            # 更新参数值为滤波后的值（取第一个粒子作为代表值）
            growthParam.assign([self.xp[i]])
    def writeOutput(self):
        """
        输出参数历史记录（待完善的方法）

        计划功能：将参数演化历史保存到文件
        当前实现不完整，仅作为占位符
        """
        # 转置参数历史数组
        np.transpose(np.array(self.xphistory))

        # 遍历历史记录中的每个时间步
        for j in len(self.xphistory):
            # 遍历每个参数
            for i, key in enumerate(self.checkParamList):
                # 访问历史参数值（代码不完整）
                self.xphistory[j][i]
            return
    def checkFN(self):
        """
        执行粒子滤波的主要算法

        包含三个子函数：
        1. randomr: 基于权重的随机重采样
        2. kernelsampling: Epanechnikov核密度估计采样
        3. rearrange: 参数重排序保持相关性

        返回:
            length_95: 95%置信度的裂纹长度预测值
        """


        def randomr(w):
            """
            随机重采样函数

            基于粒子权重进行系统性重采样，选择高权重粒子进行复制

            参数:
                w: 粒子权重数组

            返回:
                outindex: 重采样后的粒子索引
            """
            c = np.cumsum(w)  # 计算累积权重分布
            outindex = np.zeros(config.NPARTICLE, dtype=int)
            for i in range(config.NPARTICLE):
                # 随机选择一个权重值，找到对应的粒子索引
                outindex[i] = np.where(np.random.rand() <= c)[0][0]
            return outindex

        def kernelsampling(N, e):
            """
            Epanechnikov核密度估计采样

            使用Epanechnikov核函数生成光滑的随机扰动，
            避免粒子退化问题

            参数:
                N: 采样数量
                e: 输出数组，用于存储采样结果
            """
            n = 0
            while n < N:
                # 在[-1, 1]区间均匀采样
                t = np.random.rand() * 2 - 1
                # Epanechnikov核函数
                f = 0.75 * (1 - t**2)
                # 接受-拒绝采样
                r = np.random.rand()
                if r <= f:
                    e[n] = t
                    n = n + 1

        def rearrange(b, a):
            """
            参数重排序函数

            保持参数间的相关性，通过排序重新排列参数值

            参数:
                b: 待重排列的数组
                a: 参考排序数组

            返回:
                c: 重排序后的数组
            """
            index = np.argsort(a)  # 获取排序索引
            b1 = np.sort(b)        # 对b进行排序
            c = np.zeros(len(a))
            for i in range(len(a)):
                c[index[i]] = b1[i]
            return c
        # 获取裂纹长度参数在参数列表中的索引
        index = list(self.checkParamList).index('crack_length')

        # 获取当前检查步数（基于历史记录长度）
        checkstep = len(self.growth.crack_length_history_CPU) - 1

        # 从观测数据插值函数获取当前步的观测裂纹长度
        check_result = float(self.check_interp_func(checkstep))

        # 确保观测值不小于6mm
        # if check_result < 6:
        #     check_result = 6

        # 计算观测误差（实际观测值与粒子预测值的差）
        z = check_result - self.xp[index]

        # 计算似然概率
        # 更新权重
        weight = 1 / np.sqrt(2 * pi * R) * np.exp(-0.5 * z / R * z) + 1e-99

        # 计算权重和
        weight_sum = np.sum(weight)

        # 归一化权重
        weight = weight / weight_sum

        # 计算参数的加权均值（滤波后的参数估计）
        Xpf = np.matmul(self.xp, weight)

        # 计算参数协方差矩阵（用于核密度估计的带宽计算）
        Xp_cov = np.zeros((self.xp.shape[0], self.xp.shape[0]))
        # self.xp = self.xp.T  # 可选：调整数组方向

        # 遍历所有粒子，计算协方差矩阵
        for i in range(config.NPARTICLE):
            Xp_cov = Xp_cov + weight[i] * np.matmul(
                np.array([self.xp[:, i] - Xpf]).T,
                np.array([self.xp[:, i] - Xpf])
            )

        # 执行随机重采样
        outindex = randomr(weight)

        # 初始化重采样后的参数数组
        Xp_resample = np.zeros(self.xp.shape)

        # 计算各参数的标准差（用于核密度带宽）
        D = np.zeros(self.xp.shape[0])

        # 初始化核采样扰动数组
        e = np.zeros(self.xp.shape)

        # 执行重采样
        for i in range(self.xp.shape[-1]):
            Xp_resample[:, i] = self.xp[:, outindex[i]]

        # 对每个参数维度应用正则化（核密度 smoothing）
        for i in range(self.xp.shape[0]):
            # 计算参数的标准差
            D[i] = np.sqrt(Xp_cov[i][i])

            # 生成核密度扰动
            kernelsampling(config.NPARTICLE, e[i])

            # 应用正则化：重采样参数 + 核密度扰动
            self.xp[i] = Xp_resample[i] + h * D[i] * e[i]

            # 重排序以保持参数间的相关性
            self.xp[i] = rearrange(self.xp[i], Xp_resample[i])

        # 保存参数历史
        self.xphistory.append(self.xp)

        # 对所有粒子的裂纹长度参数进行排序，取97.5%分位数
        length_95 = np.sort(self.xp.T[:, index].copy())[int(0.975 * config.NPARTICLE)]

        return length_95
    def check(self):
        """
        执行完整的粒子滤波更新步骤

        这是外部调用的主要接口方法，整合了粒子滤波的完整流程：
        1. 获取当前模型参数状态
        2. 执行粒子滤波算法（重采样、正则化、参数更新）
        3. 将更新后的参数设置回模型

        返回:
            length_95
        """
        # 从模型获取当前参数状态
        self.setParamforPF()

        # 执行粒子滤波算法
        length_95 = self.checkFN()

        # 将更新后的参数设置回模型
        self.setParamforGrowth()

        return length_95

