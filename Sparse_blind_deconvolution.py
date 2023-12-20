"""
15.15 Sparse blind deconvolution. We are given a time series observation y ∈ R^T, and seek a filter (convolution kernel) w ∈ R^k, so that the convolution x = w * y ∈ R^(T + k - 1) is sparse after truncating the first and last k - 1 entries, i.e., x_k:T = (x_k, x_k+1, ..., x_T) is sparse. Here * denotes convolution,
x_i = Σ w_j y_i-j, i = 1,...,T + k - 1, j = 1
where we assume that y_t = 0 for t ≤ 0. Typically we have k ≪ T.
As a convex surrogate for sparsity of x, we minimize its l1-norm, ||x||_1. To preclude the trivial solution w = 0, we normalize w by imposing the constraint w_1 = 1.
Interpretations. (These are not needed to solve the problem.) In signal processing dialect, we can say that w is a filter which, when applied to the signal y, results in x, a simpler, sparse signal. As a second interpretation, we can say that y = w^(-1) * x, where w^(-1) is the convolution inverse of w, defined as
w^(-1) = F^(-1)(1/F(w)),
where F is discrete Fourier transform at length N = T + k and F^(-1) is its inverse transform. In this interpretation, we can say that we have decomposed the signal into the convolution of a sparse signal x and a signal with short (k-long) inverse, w^(-1).
Carry out blind deconvolution on the signal given in blind_deconv_data.*. This file also defines the kernel length k. Plot optimal w and x, and also the given observation y. Also plot the inverse kernel w^(-1), use the function inverse_ker that we provided in blind_deconv_data.*.
Hint. The function conv(w, y) is overloaded to work with CVX*.
"""

# 导入所需的库
import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy import signal
from scipy import optimize
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
# blind_deconv_data.py 信号生成函数此


# 定义用于优化的 l1 范数函数
def l1_norm(w, y, T, k):
    x = signal.convolve(w, y, mode='full')    # 计算 w 和 y 的卷积
    x_truncated = x[k-1:T]                    # 只考虑 x[k:T] 部分
    return np.sum(np.abs(x_truncated))        # 计算 l1 范数

# 设置优化问题的约束条件：滤波器 w 的元素之和为 1
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# 随机初始化 w
w_initial = np.random.RandomState(1).rand(k)

# 使用回调函数来计算并存储每一步的 MSE/NMSE
mse_values = []
nmse_values = []

def callback(w):
    mse = np.mean((w - w_true) ** 2)
    nmse = np.linalg.norm(w - w_true) ** 2 / np.linalg.norm(w_true) ** 2
    mse_values.append(mse)
    nmse_values.append(nmse)

# 使用 optimize.minimize 函数解优化问题
result = optimize.minimize(l1_norm, w_initial, args=(y, T, k), constraints=constraint, method='SLSQP', callback=callback)

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
    Fw = np.fft.fft(w, N)
    inverse_Fw = 1 / np.where(Fw != 0, Fw, 1)
    return np.real(np.fft.ifft(inverse_Fw))

# 计算逆核 w^-1
w_inverse = compute_inverse_kernel(w_optimal, N)

# 绘制结果图
# 绘制最优滤波器 w
plt.figure(figsize=(24, 5))
plt.stem(w_optimal)
plt.title('Fig1.Optimal Filter w')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
plt.savefig('Fig1.Optimal Filter w.jpg')

# 绘制稀疏信号 x
plt.figure(figsize=(24, 5))
plt.plot(x_optimal)
plt.title('Fig2.Sparse Signal x')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
plt.savefig('Fig2.Sparse Signal x.jpg')

# 绘制观测 y
plt.figure(figsize=(24, 5))
plt.plot(y)
plt.title('Fig3.Observation y')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
plt.savefig('Fig3.Observation y.jpg')

# 绘制逆核 w^-1
plt.figure(figsize=(24, 5))
plt.stem(w_inverse)
plt.title('Fig4.Inverse Kernel w^-1')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
plt.savefig('Fig4.Inverse Kernel w^-1.jpg')

# 绘制 NMSE 图像
plt.figure(figsize=(24, 5))
# 创建一个坐标轴 MSE
ax1 = plt.gca()
ax1.plot(mse_values, label='MSE')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('MSE')
ax1.tick_params('y')
# 创建另一个坐标轴 NMSE，共享同一个 X 轴
ax2 = ax1.twinx()
ax2.semilogy(nmse_values, 'r--', label='NMSE (log scale)')  # 'r--' 表示红色虚线
ax2.set_ylabel('NMSE (log scale)')
ax2.tick_params('y')
# 添加图例和标题
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Fig5.MSE and NMSE Over Iterations')
plt.tight_layout()
plt.show()
plt.savefig('Fig5.MSE_and_NMSE_Over_Iterations.jpg')


# 计算信号 x 的均方误差 (MSE)
# 注意：x_optimal 可能需要被截断或填充以与 x_true 的长度匹配
length_difference = len(x_true) - len(x_optimal)
if length_difference > 0:
    x_optimal_padded = np.append(x_optimal, np.zeros(length_difference))
elif length_difference < 0:
    x_optimal_padded = x_optimal[:length_difference]
else:
    x_optimal_padded = x_optimal

mse_x = np.mean((x_optimal_padded - x_true) ** 2)

print(mse_x)