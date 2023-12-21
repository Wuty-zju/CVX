import numpy as np
from scipy import optimize
from scipy import signal

# 优化算法的代码
def optimize_filter(w_true, y, T, k):
    # 定义用于优化的 l1 范数函数
    def l1_norm(w, y, T, k):
        x = signal.convolve(w, y, mode='full')
        x_truncated = x[k-1:T]
        return np.sum(np.abs(x_truncated))

    # 设置优化问题的约束条件
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
        w_optimal = result.x
        x_optimal = signal.convolve(w_optimal, y, mode='full')[k-1:T]
        print("w_optimal", '\n', w_optimal[:], '\n', "x_optimal", '\n', x_optimal[:])
        print("Success", '\n', result.message)
    else:
        w_optimal = x_optimal = None
        print("False", result.message)

    return w_optimal, x_optimal, mse_values, nmse_values
