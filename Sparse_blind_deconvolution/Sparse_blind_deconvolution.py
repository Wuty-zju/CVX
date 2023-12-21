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

15.15 稀疏盲反卷积。我们得到了一个时间序列观察 y ∈ R^T,并寻求一个滤波器(卷积核)w ∈ R^k,使得卷积 x = w * y ∈ R^(T + k - 1) 在截断前后 k - 1 项后变得稀疏，即 x_k:T = (x_k, x_k+1, ..., x_T) 是稀疏的。这里 * 表示卷积，
x_i = Σ w_j y_i-j, i = 1,...,T + k - 1, j = 1,
其中我们假设 y_t = 0 对于 t ≤ 0。通常我们有 k ≪ T。
作为 x 稀疏性的凸替代，我们最小化其 l1-范数，||x||_1。为了排除平凡解 w = 0,我们通过施加约束 w_1 = 1 来规范化 w。
解释。（这些不是解决问题所必需的。）在信号处理方言中，我们可以说 w 是一个滤波器，当应用于信号 y 时，会得到 x,一个更简单、稀疏的信号。作为第二种解释，我们可以说 y = w^(-1) * x,其中 w^(-1) 是 w 的卷积逆，定义为
w^(-1) = F^(-1)(1/F(w)),
其中 F 是长度为 N = T + k 的离散傅立叶变换,F^(-1) 是其逆变换。在这种解释中，我们可以说我们已经将信号分解成稀疏信号 x 和具有短(k 长）逆的信号 w^(-1) 的卷积。
对 blind_deconv_data.* 中给出的信号执行盲反卷积。这个文件还定义了核长度 k。绘制最佳 w 和 x,以及给定的观测 y。还要绘制逆核 w^(-1)，使用我们在 blind_deconv_data.* 中提供的 inverse_ker 函数。
提示。函数 conv(w, y) 已重载以适用于 CVX*。
"""

# 导入所需的库
import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import os
import pandas as pd

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
# blind_deconv_data.py 信号生成函数文件部分到此结束


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
    print("Success", '\n', result.message)
    print("w_optimal", '\n', w_optimal[:], '\n', "x_optimal", '\n', x_optimal[:])  # 显示 w_optimal 和 x_optimal
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

# 绘制结果图并输出 CSV
def plot_and_save_with_csv(data, title, xlabel, ylabel, output_folder_path, fig_size=(24, 5), plot_type='plot'):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 绘制图像
    plt.figure(figsize=fig_size)
    if plot_type == 'plot':
        plt.plot(data)
    elif plot_type == 'stem':
        plt.stem(data)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    # 保存图像
    safe_title = title.replace(' ', '_').replace('.', '').replace('-', '_')  # 替换标题中的非法文件名字符
    image_file_name = f"{safe_title}.jpg"
    plt.savefig(os.path.join(output_folder_path, image_file_name))
    plt.show()

    # 保存数据到 CSV
    csv_file_name = f"{safe_title}.csv"
    pd.DataFrame(data).to_csv(os.path.join(output_folder_path, csv_file_name), index=False, header=True)

# 设置输出路径
results_output_folder_path = 'Sparse_blind_deconvolution/results'   # 输出题目要求的 results 路径
error_output_folder_path = 'Sparse_blind_deconvolution/errors'   # 输出误差随迭代次数 errors 路径

# 设置输出参数
plot_and_save_with_csv(w_optimal, 'Optimal Filter w', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem')    # 绘制最优滤波器 w
plot_and_save_with_csv(x_optimal, 'Sparse Signal x', 'Time', 'Amplitude', results_output_folder_path)    # 绘制稀疏信号 x
plot_and_save_with_csv(y, 'Observation y', 'Time', 'Amplitude', results_output_folder_path)      # 绘制观测 y
plot_and_save_with_csv(w_inverse, 'Inverse Kernel w^-1', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem')     # 绘制逆核 w^-1
plot_and_save_with_csv(mse_values, 'MSE Over Iterations', 'Iteration', 'MSE', error_output_folder_path)  # 绘制 MSE
plot_and_save_with_csv(nmse_values, 'NMSE Over Iterations', 'Iteration', 'NMSE', error_output_folder_path)   # 绘制 NMSE