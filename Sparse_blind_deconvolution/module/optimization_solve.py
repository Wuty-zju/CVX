import numpy as np
import cvxpy as cp
from scipy import signal

# 优化算法
def optimization_solve(w_true, y, T, k):
    # 定义优化变量
    w = cp.Variable(k)

    # 构造卷积矩阵
    Y = np.zeros((T, k))
    for i in range(T):
        Y[i, :min(i+1, k)] = y[max(0, i-k+1):i+1][::-1]

    # 定义卷积操作
    x = Y @ w

    # 目标函数：最小化 x 的 l1 范数
    objective = cp.Minimize(cp.norm(x, 1))

    # 约束条件：w 的第一个元素等于 1
    constraints = [w[0] == 1]

    # 定义并求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    # 提取优化后的 w 和 x
    w_optimal = w.value
    x_optimal = Y @ w_optimal

    # 误差计算
    rmms = np.linalg.norm(w_optimal - w_true) / np.linalg.norm(w_true)      # RRMS（Relative Root Mean Square Error）误差计算
    mse = np.mean((w_optimal - w_true) ** 2)        # 计算 MSE
    nmse = np.linalg.norm(w_optimal - w_true) ** 2 / np.linalg.norm(w_true) ** 2        # 计算 NMSE
    print('Optimize Success')

    return w_optimal, x_optimal, rmms, mse, nmse
