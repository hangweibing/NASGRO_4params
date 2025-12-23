# 裂纹扩展预测与参数更新系统
## 代码模块与流程详细说明文档

---

## 目录
1. [应力强度因子计算模块](#1-应力强度因子计算模块)
2. [裂纹扩展模型模块](#2-裂纹扩展模型模块)
3. [粒子滤波模块](#3-粒子滤波模块)
4. [主程序模块](#4-主程序模块)
5. [总体规划](#5-当前阶段总体规划)

---

## 1. 应力强度因子计算模块

### 1.1 模块概述

**文件名**: `stress_intensity_factor.py`

该模块通过调用不同的模型计算应力强度因子 K（Stress Intensity Factor）。目前使用的是一个简单的经验公式进行计算，后续可能考虑采用神经网络等更为复杂的计算模型。

### 1.2 核心函数

#### 1.2.1 简单模型 - `calc_K_for_tip(a, b, dSigma)`

**功能说明**：计算方形板长边中点裂纹的应力强度因子范围 ΔK

**运行逻辑**：输入当前的裂纹长度 `a`、板宽 `b` 和应力变化量 `dSigma`，计算出 K 值，作为裂纹扩展模型的输入。

**代码实现**：

```python
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
```

**参数说明**：
- `a`：裂纹长度（单位：mm）
- `b`：板宽（单位：mm）
- `dSigma`：应力范围（单位：MPa）
- **返回值**：`deltaK` - 应力强度因子范围（单位：MPa·m^0.5）

#### 1.2.2 扩展接口 - `calc_K_for_tip_nn()`

该模块预留了神经网络模型接口，用于未来替换传统公式计算，目前尚未实现。

### 1.3 模块开发需求

- **前端展示**：该模块仅作为计算函数，不需要在前端界面中进行操作和展示
- **调用方式**：需要能够在裂纹扩展模型中被调用，并提取各个粒子的状态参数进行 K 的计算
- **扩展性**：保留神经网络接口，为后续升级做准备

---

## 2. 裂纹扩展模型模块

### 2.1 模块概述

**文件名**: `crack_growth_models.py`

该模块调用不同的裂纹扩展模型，计算出裂纹增长量 da（单位：mm/cycle）。目前模块中包含 **Paris** 和 **NASGRO H-S** 两种模型，运行程序中实际调用的是 **NASGRO H-S** 模型。

### 2.2 核心模型

#### 2.2.1 NASGRO H-S 模型 - `predict_NASGRO_tf()`

**运行逻辑**：
1. 输入粒子的不确定参数（D, p, A, dKthr）
2. 输入当前裂纹长度 `a` 和应力变化量 `dSigma`
3. 调用 `calc_K_for_tip()` 计算应力强度因子 K
4. 根据 NASGRO Hartman-Schieve 方程计算裂纹增长量 da

**代码实现**：

```python
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
```

**参数说明**：
- **不确定参数**（需要通过粒子滤波更新）：
  - `D`：NASGRO 方程材料常数
  - `p`：NASGRO 方程指数
  - `A`：断裂韧度参数
  - `dKthr`：阈值应力强度因子范围
- **状态参数**：
  - `a`：当前裂纹长度
  - `dSigma`：应力变化量
- **返回值**：`da` - 裂纹长度增量（mm/cycle）

#### 2.2.2 Paris 模型 - `predict_PARIS_tf()`

Paris 方程是最经典的疲劳裂纹扩展模型：da/dN = C·(ΔK)^m

```python
def predict_PARIS_tf(a, b, C, m, dSigma):
    """
    基于Paris方程计算裂纹扩展速率

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
```

### 2.3 模块开发需求

- **定位**：该模块类似模块1，仅在主程序中作为计算函数
- **前端交互**：不需要在前端中进行操作和展示
- **数据流**：仅需要提取粒子的状态参数，进行各个粒子裂纹长度的独立计算

---

## 3. 粒子滤波模块

### 3.1 模块概述

**文件名**: `partialfilter.py`

当 97.5% 分位数的粒子裂纹长度超过主程序中定义的 `crack_limit` 时，就会进入该模块，进行粒子的权重更新和粒子重采样。

### 3.2 核心类 - `ParticleFilter`

#### 3.2.1 初始化

```python
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
        self.checkParamList = paramDict[MODE]['_uncertainParam'].keys()

        # 检查计数器初始化
        self.Ncheck = 0

        # 加载观测数据文件（用于参数更新的基准数据）
        check = np.loadtxt('ground_truth.csv', dtype=float, delimiter=',', encoding='UTF-8')

        # 创建观测数据插值函数（3次样条插值）
        self.check_interp_func = interp1d(check.T[0], check.T[1], 3)
```

#### 3.2.2 核心算法 - `checkFN()`

该函数包含三个关键子函数：

**1. `randomr(w)` - 随机重采样**

基于粒子权重进行系统性重采样，选择高权重粒子进行复制：

```python
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
    outindex = np.zeros(NPARTICLE, dtype=int)
    for i in range(NPARTICLE):
        # 随机选择一个权重值，找到对应的粒子索引
        outindex[i] = np.where(np.random.rand() <= c)[0][0]
    return outindex
```

**2. `kernelsampling(N, e)` - Epanechnikov 核密度估计采样**

使用 Epanechnikov 核函数生成光滑的随机扰动，避免粒子退化问题：

```python
def kernelsampling(N, e):
    """
    Epanechnikov核密度估计采样

    使用Epanechnikov核函数生成光滑的随机扰动，避免粒子退化问题

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
```

**3. `rearrange(b, a)` - 参数重排序**

保持参数间的相关性，通过排序重新排列参数值。

#### 3.2.3 主接口函数 - `check()`

```python
def check(self):
    """
    执行完整的粒子滤波更新步骤

    这是外部调用的主要接口方法，整合了粒子滤波的完整流程：
    1. 获取当前模型参数状态
    2. 执行粒子滤波算法（重采样、正则化、参数更新）
    3. 将更新后的参数设置回模型

    返回:
        length_95: 97.5%置信度的裂纹长度预测值
    """
    # 从模型获取当前参数状态
    self.setParamforPF()

    # 执行粒子滤波算法
    length_95 = self.checkFN()

    # 将更新后的参数设置回模型
    self.setParamforGrowth()

    return length_95
```

### 3.3 粒子滤波算法流程

1. **获取观测数据**：从 `ground_truth.csv` 加载真实观测裂纹长度
2. **计算权重**：根据观测值与粒子预测值的差异，计算似然概率和权重
3. **权重归一化**：确保所有粒子权重之和为1
4. **重采样**：基于权重选择高质量粒子
5. **核密度正则化**：添加扰动避免粒子退化
6. **参数更新**：将更新后的参数返回给主程序

### 3.4 模块开发需求

- **黑盒处理**：可以把这一部分看成一个黑盒函数
- **数据流向**：从主程序读取粒子状态信息 → 参数更新 → 返回新粒子给主程序
- **前端开发**：在前端开发时应该不涉及这一部分的修改和功能开发

---

## 4. 主程序模块

### 4.1 模块概述

**文件名**: `DT_2D_tf_fn_4param.py`

主程序是整个系统的核心控制模块，负责参数定义、粒子初始化、循环计算和结果输出。

### 4.2 全局参数设置

```python
# ------------------------------ 全局参数设置 ------------------------------
# 模型选择：'PARIS' 或 'NASGRO'
MODE = 'NASGRO'

# 粒子初始化方法：'tf' 使用TensorFlow内置随机采样, 'lhs' 使用拉丁超立方采样
INIT_METHOD = 'lhs'

# 结果输出目录
RESULTS_DIR = 'results'

# 粒子数量与可靠性指标设置
NPARTICLE = int(1E4)
reliability_rate = 0.025
MAXCOUNT = int(NPARTICLE * reliability_rate)
```

### 4.3 参数定义与初始化

#### 4.3.1 材料与载荷参数

```python
# ------------------------------ 材料与载荷参数定义 ------------------------------
# Kc: 断裂韧度 (MPa*m^0.5)
# safe_factor: 安全系数
# width: 板材宽度 (mm)
# load: 载荷参数 [均值, 标准差]
# stress: 应力参数（确定值）
# length: 初始裂纹长度参数 [均值, 标准差, 偏移]
Kc = 50
safe_factor = 2
width = 50
load = [1, 0.1]
stress = load[0]  # 确定性应力值
length = [0.01, 0.01, 1.27]
```

#### 4.3.2 模型参数字典

定义了 PARIS 和 NASGRO 两种模型的不确定参数均值和方差：

```python
paramDict = {
    'PARIS': {
        'C': [3.2e-10, 3e-11],      # Paris方程常数C [均值, 标准差]
        'm': [1.6, 0.08],            # Paris方程指数m [均值, 标准差]
        '_uncertainParam': {'C': 'N', 'm': 'N', 'crack_length': 'N'}
    },
    'NASGRO': {
        'D': [2e-9, 8e-10],         # NASGRO方程常数D [均值, 标准差]
        'p': [1.2, 0.05],            # NASGRO方程指数p [均值, 标准差]
        'dKthr': [7.23, 0.5],        # 阈值应力强度因子范围 [均值, 标准差]
        'A': [74.1, 2.0],            # 断裂韧度参数 [均值, 标准差]
        '_uncertainParam': {'D': 'N', 'p': 'N', 'dKthr': 'N', 'A': 'N', 'crack_length': 'N'}
    }
}
```

**参数说明**：
- **'N'** 表示正态分布
- **参数格式**：`[均值, 标准差]` 或 `[均值, 标准差, 偏移]`

#### 4.3.3 粒子不确定参数的采样（粒子状态初始化）

通过 `setParamDict()` 方法，根据上述均值和方差，调用 `particle_initialization.py` 中的抽样函数，对粒子的模型参数进行初始化（对于 NASGRO H-S 模型，需要抽样的参数就是 D, p, A, dKthr）：

```python
# 设置模型参数字典（自动区分确定参数和不确定参数）
cg.setParamDict(paramDict=paramDict[MODE])
```

内部实现：

```python
def setParamDict(self, paramDict='', **kwargs):
    """
    批量设置模型参数，自动分离确定参数和不确定参数

    根据参数字典中的_uncertainParam配置，自动判断参数类型
    确定参数使用固定值，不确定参数使用随机分布采样
    """
    for key in paramDict:
        if not '_' in key:  # 跳过配置参数（如_uncertainParam）
            if key in paramDict['_uncertainParam']:
                # 不确定参数：使用指定分布类型
                self.setParam(param=key, dist=paramDict['_uncertainParam'][key], value=paramDict[key])
            else:
                # 确定参数：使用固定值
                self.setParam(param=key, dist='single', value=paramDict[key])
```

#### 4.3.4 其它计算所需参数的初始化

调用 `setParam()` 函数，可以选择不同的分布方式（正态分布 'N'，固定值 'single'）对所需要的参数进行初始化：

```python
# 初始裂纹长度：正态分布 [均值, 标准差, 偏移]
cg.setParam(param='crack_length', dist='N', value=length)

# 应力范围：确定参数
cg.setParam(param='dSigma', dist='single', value=stress)

# 板宽：确定参数
cg.setParam(param='b', dist='single', value=50)
```

#### 4.3.5 裂纹尺寸限制 `crack_limit`

根据安全系数 `safe_factor`（默认为2）和临界裂纹尺寸计算 `crack_limit`：

```python
# 初始化裂纹扩展模拟器
# mode: 模型类型 (PARIS/NASGRO)
# crack_limit: 裂纹临界长度 (25/2 = 12.5mm)
# shape: 张量形状 [1, NPARTICLE] 表示单批次，NPARTICLE个粒子
cg = CrackGrowth(mode=MODE, crack_limit=25/2, shape=tf.TensorShape([1, NPARTICLE]))
```

**说明**：
- 临界裂纹尺寸设置为 25mm
- 除以安全系数2，得到裂纹最大尺寸限制值 `crack_limit = 12.5mm`
- 当 97.5% 分位数的粒子裂纹长度超过此值时，触发粒子滤波更新

### 4.4 核心类 - `CrackGrowth`

#### 4.4.1 单步裂纹扩展 - `growth()`

```python
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
```

#### 4.4.2 主模拟循环 - `growth_loop()`

这是整个系统的核心控制逻辑：

```python
def growth_loop(self):
    """
    主模拟循环控制

    控制整个疲劳裂纹扩展过程：
    1. 内循环：当超限粒子数未达到阈值时，继续模拟
    2. 外循环：当97.5%置信度裂纹长度未达到临界值时，执行粒子滤波更新
    3. 定期保存中间结果到文件
    """
    # 初始化GPU历史记录
    self.crack_length_history_GPU = self.crack_length
    maxcount = tf.constant(MAXCOUNT, dtype='int64')

    # 初始化参数更新计数器
    update_count = 0

    # 外循环：继续直到97.5%置信度裂纹长度达到临界值 12.5mm
    while self.crack_length_95 < self.crack_limit:

        # 内循环：继续直到超限粒子数达到阈值
        while self.crack_count < maxcount:
            self.growth()  # 执行单步裂纹扩展计算

        # 数据转移到CPU
        self.log_to_cpu()

        # 准备输出数据：计算统计量
        t_ = self.crack_length_history_CPU.copy()
        t_.sort(axis=-1)  # 按裂纹长度排序

        # 创建输出数组
        t = np.vstack((
            np.arange(1, t_.shape[0]),           # 循环数
            t_[1:, MAXCOUNT],                    # 2.5%分位数
            np.average(t_[1:], axis=-1),         # 加权平均
            t_[1:, t_.shape[-1] - MAXCOUNT]      # 97.5%分位数
        )).T

        # 保存结果到CSV文件
        result_filename = os.path.join(RESULTS_DIR, '%s_%d.csv' % (MODE, load[0]))
        np.savetxt(result_filename, t, delimiter=',')

        # 执行粒子滤波更新参数
        self.crack_length_95 = self.check.check()

        # 记录更新后的粒子参数状态
        self.log_particle_params()

        # 更新参数分布可视化
        from visualization import plot_particle_parameters_evolution
        plot_particle_parameters_evolution(os.path.join(RESULTS_DIR, 'particle_params.csv'), 
                                          NPARTICLE, save_dir=RESULTS_DIR)

        # 增加更新计数器
        update_count += 1

        # 绘制当前预测效果
        filename = os.path.join(RESULTS_DIR, '%s_%d.csv' % (MODE, load[0]))
        plot_crack_length_prediction(t, filename, update_count)

        # 重置超限计数器
        self.crack_count = 0
```

**循环逻辑详解**：

1. **外循环条件**：`while self.crack_length_95 < self.crack_limit`
   - 当 97.5% 分位数的裂纹长度小于临界值（12.5mm）时继续
   
2. **内循环条件**：`while self.crack_count < maxcount`
   - 当超过临界值的粒子数量小于阈值（2.5%×粒子总数）时继续
   - 每次调用 `self.growth()` 执行一个载荷循环的裂纹扩展计算
   
3. **触发粒子滤波**：内循环结束后（超限粒子数达到阈值）
   - 保存当前结果到 CSV 文件
   - 调用 `self.check.check()` 执行粒子滤波参数更新
   - 绘制可视化图像
   - 重置计数器，继续外循环

### 4.5 统计与日志功能

#### 4.5.1 `summary()` - 单步统计

```python
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
        self.log_to_cpu()

        # 重置时间戳
        self.start = tf.timestamp()

    # 步数递增
    self.step = self.step + 1
```

#### 4.5.2 输出数据格式

程序会生成以下输出：

1. **裂纹长度历史 CSV**（`results/NASGRO_1.csv`）：
   - 列1：循环数
   - 列2：2.5% 分位数裂纹长度
   - 列3：加权平均裂纹长度
   - 列4：97.5% 分位数裂纹长度

2. **粒子参数历史 CSV**（`results/particle_params.csv`）：
   - 列：D, p, dKthr, A
   - 行：每个粒子的参数值（每次更新后追加）

3. **可视化图像**：
   - 裂纹长度预测图
   - 参数分布演化图

### 4.6 主函数

```python
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
    cg = CrackGrowth(mode=MODE, crack_limit=25/2, shape=tf.TensorShape([1, NPARTICLE]))

    # 设置模型参数字典（自动区分确定参数和不确定参数）
    cg.setParamDict(paramDict=paramDict[MODE])

    # 确保results目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 删除之前的CSV文件，避免数据叠加
    csv_filename = os.path.join(RESULTS_DIR, 'particle_params.csv')
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    # 记录初始粒子参数状态
    cg.log_particle_params()

    # 生成初始参数分布可视化
    from visualization import plot_particle_parameters_evolution
    plot_particle_parameters_evolution(os.path.join(RESULTS_DIR, 'particle_params.csv'), 
                                      NPARTICLE, save_dir=RESULTS_DIR)

    # 单独设置其他参数
    cg.setParam(param='crack_length', dist='N', value=length)
    cg.setParam(param='dSigma', dist='single', value=stress)
    cg.setParam(param='b', dist='single', value=50)

    # 执行主模拟循环
    cg.growth_loop()

    return
```

### 4.7 模块开发需求

#### 4.7.1 参数定义与初始化界面

**前端功能需求**：

1. **参数输入窗口**：设置一个参数定义的前端界面
   - NASGRO 模型四个参数（D, p, A, dKthr）的均值和方差输入框
   - 初始裂纹长度的均值、方差、偏移输入框
   - 应力参数、板宽等确定参数输入
   - 安全系数设置

2. **示例输入格式**：
   ```
   D均值: 2e-9          D标准差: 8e-10
   p均值: 1.2           p标准差: 0.05
   dKthr均值: 7.23      dKthr标准差: 0.5
   A均值: 74.1          A标准差: 2.0
   
   初始裂纹长度均值: 0.01    标准差: 0.01    偏移: 1.27
   应力: 1.0
   板宽: 50
   安全系数: 2
   粒子数: 10000
   ```

#### 4.7.2 粒子裂纹长度循环更新界面

**（1）粒子状态的提取与裂纹长度可视化更新**

在之前二级界面的基础上，实现实时可视化：

- **上方蓝色虚线**：97.5% 分位数的裂纹长度
- **下方蓝色虚线**：2.5% 分位数的裂纹长度
- **中间白色线**：所有粒子裂纹长度的加权平均数

**当前代码逻辑**：
- 只有 `while self.crack_count < maxcount:` 循环结束时，才会一次性输出长度值
- 可以进行调整：每运行 `self.growth()` 1000次，就输出一次裂纹长度，同时更新图像
- 实现裂纹长度的可视化实时追踪

**（2）输入检查值，进入粒子滤波模块进行参数更新**

- **前端输入**：提供一个检查值输入框
- **触发机制**：输入检查值后，执行粒子滤波模块的权重计算和重采样
- **参数更新**：对参数分布进行更新
- **当前逻辑**：`self.crack_count < maxcount` 自动触发（97.5% 分位数的裂纹长度大于 crack_limit）


## 5. 当前阶段总体规划

### 5.1 首要目标

**实现各模块在系统中的集成**，根据设置的参数初始分布，能够成功在平台中实时计算并更新裂纹长度的图像。

#### 开发步骤：

1. **理解代码运行逻辑和基本流程**
   - 熟悉四个核心模块的功能和调用关系
   - 理解粒子滤波的触发机制
   - 掌握数据流向和参数传递方式

2. **测试裂纹扩展预测功能**
   - 在平台中成功执行主程序的 `growth_loop()`
   - 验证裂纹扩展预测能否正常工作

3. **开发参数更新功能**
   - 确认基本功能正常后实现输入检查数据并执行 `partialfilter` 参数更新的功能


### 5.2 界面调整建议

**失效概率预测界面**：
- 先前演示视频中的失效概率预测界面，在后续开发中应该不会用到
- 后续可能需要取消这一部分的界面

### 5.3 数据管理功能（后续扩展）

后续在平台中预计需要有**储存并管理应变、加速度等数据的功能**：

1. **数据管理储存功能**

2. **数据可视化**
   - 在可视化界面中，需要有一部分用于展示收集到数据的时间历程
   - 开发过程可以先用随机数进行测试




## 6. 系统架构总结

### 6.1 数据流向图

```
用户输入参数（前端）
    ↓
参数初始化（DT_2D_tf_fn_4param.py）
    ↓
粒子初始化（particle_initialization.py）
    ↓
┌─────────────────────────────────┐
│  主循环 (growth_loop)            │
│  ┌─────────────────────────┐    │
│  │ 内循环 (growth)          │    │
│  │   ↓                      │    │
│  │ 计算应力强度因子 K       │    │
│  │ (stress_intensity_factor)│    │
│  │   ↓                      │    │
│  │ 计算裂纹增长量 da        │    │
│  │ (crack_growth_models)    │    │
│  │   ↓                      │    │
│  │ 更新裂纹长度            │    │
│  │   ↓                      │    │
│  │ 统计分析 (summary)       │    │
│  └─────────────────────────┘    │
│         ↓                        │
│  超限粒子数达到阈值？             │
│         ↓ Yes                    │
│  执行粒子滤波 (partialfilter)    │
│         ↓                        │
│  参数更新                        │
│         ↓                        │
│  保存结果与可视化                │
│         ↓                        │
│  97.5%分位数达到临界值？          │
│         ↓ No (返回主循环)         │
└─────────────────────────────────┘
         ↓ Yes
    结束程序
```

### 6.2 模块调用关系

```
DT_2D_tf_fn_4param.py (主程序)
├── stress_intensity_factor.py (应力强度因子)
│   └── calc_K_for_tip()
├── crack_growth_models.py (裂纹扩展模型)
│   ├── predict_NASGRO_tf()
│   └── predict_PARIS_tf()
├── partialfilter.py (粒子滤波)
│   ├── ParticleFilter.check()
│   ├── ParticleFilter.checkFN()
│   └── ParticleFilter.setParamforGrowth()
├── particle_initialization.py (粒子初始化)
│   ├── init_uncertain_param_lhs()
│   └── init_uncertain_param_tf()
└── visualization.py (可视化)
    ├── plot_crack_length_prediction()
    └── plot_particle_parameters_evolution()
```

### 6.3 关键参数列表

| 参数名称 | 说明 | 默认值 | 单位 |
|---------|------|--------|------|
| `NPARTICLE` | 粒子数量 | 10000 | - |
| `MODE` | 模型类型 | 'NASGRO' | - |
| `safe_factor` | 安全系数 | 2 | - |
| `crack_limit` | 裂纹临界长度 | 12.5 | mm |
| `reliability_rate` | 可靠性指标阈值 | 0.025 | - |
| `MAXCOUNT` | 最大允许超限粒子数 | 250 | - |
| `D` | NASGRO 常数 | [2e-9, 8e-10] | - |
| `p` | NASGRO 指数 | [1.2, 0.05] | - |
| `A` | 断裂韧度参数 | [74.1, 2.0] | MPa·m^0.5 |
| `dKthr` | 阈值应力强度因子 | [7.23, 0.5] | MPa·m^0.5 |
| `length` | 初始裂纹长度 | [0.01, 0.01, 1.27] | mm |
| `stress` | 应力范围 | 1.0 | MPa |
| `width` | 板宽 | 50 | mm |

---

## 附录 A：文件清单

| 文件名 | 功能 | 关键函数/类 |
|--------|------|------------|
| `stress_intensity_factor.py` | 应力强度因子计算 | `calc_K_for_tip()` |
| `crack_growth_models.py` | 裂纹扩展模型 | `predict_NASGRO_tf()`, `predict_PARIS_tf()` |
| `partialfilter.py` | 粒子滤波 | `ParticleFilter` |
| `DT_2D_tf_fn_4param.py` | 主程序 | `CrackGrowth`, `main()` |
| `particle_initialization.py` | 粒子初始化 | `init_uncertain_param_lhs()` |
| `visualization.py` | 可视化 | `plot_crack_length_prediction()` |
| `ground_truth.csv` | 观测数据 | - |

---

## 附录 B：快速开始指南

### 运行程序

```bash
python DT_2D_tf_fn_4param.py
```

### 修改参数

编辑 `DT_2D_tf_fn_4param.py` 中的全局参数：

```python
# 修改粒子数量
NPARTICLE = int(1E4)

# 修改模型类型
MODE = 'NASGRO'  # 或 'PARIS'

# 修改初始裂纹长度
length = [0.01, 0.01, 1.27]  # [均值, 标准差, 偏移]

# 修改应力
stress = 1.0
```

### 查看结果

1. **裂纹长度历史**：`results/NASGRO_1.csv`
2. **参数演化历史**：`results/particle_params.csv`
3. **可视化图像**：`results/` 目录下的 PNG 文件

---


