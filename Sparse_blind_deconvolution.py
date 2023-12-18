# 导入所需的库
import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy import signal

from scipy import fft, optimize
import matplotlib.pyplot as plt

# Parameters 设置参数
T = 400        # 时间序列的长度
k = 20         # 滤波器的长度
N = T + k      # FFT计算中使用的长度
p = 0.1        # 稀疏信号的稀疏程度
sigma = 0.0001 # 噪声水平

# Random Model with fixed seed 使用固定的随机数种子生成模型数据
rn = RandomState(364)

# 生成真实滤波器，并对其进行处理以满足特定的形式
w_true = rn.rand(k)               # 生成随机滤波器
index = np.argmax(np.abs(w_true)) # 找到滤波器中最大幅度的元素索引
w_true = np.roll(w_true, -index)  # 将滤波器滚动，使最大元素位于起始位置
w_true = w_true / w_true[0]       # 规范化滤波器

# 生成稀疏信号和观测值
x_true = rn.randn(T)                              # 生成随机信号
x_true = rn.binomial(1, p, np.shape(x_true)) * x_true # 使信号稀疏
y_true = np.real(np.fft.ifft(np.fft.fft(x_true, N) / np.fft.fft(w_true, N), N))
y = y_true[k:T+k] + sigma * rn.randn(T)           # 添加噪声生成最终的观测值

# 定义逆核函数
def inverse_ker(w, len=N):
    w_inv = np.real(np.fft.ifft(1/np.fft.fft(w, len), len))
    return w_inv

# 定义用于优化的 l1 范数函数
def l1_norm(w, y, T, k):
    x = signal.convolve(w, y, mode='full')    # 计算 w 和 y 的卷积
    x_truncated = x[k-1:T]                    # 只考虑 x[k:T] 部分
    return np.sum(np.abs(x_truncated))        # 计算 l1 范数

# 设置优化问题的约束条件：滤波器 w 的元素之和为 1
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# 随机初始化 w
w_initial = np.random.rand(k)

# 使用 optimize.minimize 函数解优化问题
result = optimize.minimize(l1_norm, w_initial, args=(y, T, k), constraints=constraint, method='SLSQP')

# 检查优化结果
if result.success:
    w_optimal = result.x  # 获取最优滤波器
    x_optimal = signal.convolve(w_optimal, y, mode='full')[k-1:T]  # 通过卷积得到稀疏信号
    print("Success", '\n', result.message, '\n', "w_optimal", '\n', w_optimal[:], '\n', "x_optimal", '\n', x_optimal[:])  # 显示 w_optimal 和 x_optimal
else:
    w_optimal = x_optimal = None
    print("False", result.message)



# 计算逆核函数
def compute_inverse_kernel(w, N):
    Fw = fft.fft(w, N)
    inverse_Fw = 1 / np.where(Fw != 0, Fw, 1)
    return np.real(fft.ifft(inverse_Fw))

# 计算逆核 w^-1
w_inverse = compute_inverse_kernel(w_optimal, N)

# 绘制结果图
plt.figure(figsize=(24, 20))

# 绘制最优滤波器 w
plt.subplot(4, 1, 1)
plt.stem(w_optimal)
plt.title('Optimal Filter w')
plt.xlabel('Index')
plt.ylabel('Amplitude')

# 绘制稀疏信号 x
plt.subplot(4, 1, 2)
plt.plot(x_optimal)
plt.title('Sparse Signal x')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 绘制观测 y
plt.subplot(4, 1, 3)
plt.plot(y)
plt.title('Observation y')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 绘制逆核 w^-1
plt.subplot(4, 1, 4)
plt.stem(w_inverse)
plt.title('Inverse Kernel w^-1')
plt.xlabel('Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
plt.savefig('results.jpg')
