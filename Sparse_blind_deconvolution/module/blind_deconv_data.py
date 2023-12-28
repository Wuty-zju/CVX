import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy import signal


# 波形生成
def generate_waveform():
    # 参数设置
    T = 400        # 时间序列的长度
    k = 20         # 滤波器的长度
    N = T + k      # FFT计算中使用的长度
    p = 0.1        # 稀疏信号的稀疏程度
    sigma = 0.0001 # 噪声水平

    # 使用固定的随机数种子生成模型数据
    rn = RandomState(364)

    # 生成真实滤波器，并对其进行处理以满足特定的形式
    w_true = rn.rand(k)
    index = np.argmax(np.abs(w_true))
    w_true = np.roll(w_true, -index)
    w_true = w_true / w_true[0]

    # 生成稀疏信号和观测值
    x_true = rn.randn(T)
    x_true = rn.binomial(1, p, np.shape(x_true)) * x_true
    y_true = np.real(np.fft.ifft(np.fft.fft(x_true, N) / np.fft.fft(w_true, N), N))
    y = y_true[k:T+k] + sigma * rn.randn(T)

    return w_true, y, T, k, N

# 定义逆核函数
def inverse_ker(w, len):
    w_inv = np.real(np.fft.ifft(1/np.fft.fft(w, len), len))
    return w_inv
