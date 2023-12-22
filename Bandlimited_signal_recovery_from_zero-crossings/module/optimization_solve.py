import numpy as np
import cvxpy as cvx

def optimization_solve(n, f_min, B, s, y):
    """
    使用带限正弦波构造矩阵 A，并最小化目标函数。
    
    参数:
    n (int): 信号长度。
    B (int): 带宽。
    f_min (float): 最小频率。
    s (np.array): 符号约束数组。

    返回:
    np.array: 恢复后的信号 y_recovered。
    """
    # 构造矩阵 A，其列为带限正弦波
    C = np.zeros((n, B))
    S = np.zeros((n, B))
    for j in range(B):
        C[:, j] = np.cos(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
        S[:, j] = np.sin(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
    A = np.hstack((C, S))

    # 最小化目标函数，同时考虑 L1 规范化和符号约束
    x = cvx.Variable(2 * B)
    obj = cvx.norm(A @ x)
    constraints = [cvx.multiply(s, A @ x) >= 0, s.T @ (A @ x) == n]
    problem = cvx.Problem(cvx.Minimize(obj), constraints)
    problem.solve(solver=cvx.ECOS)

    y_recovered = A @ x.value

    error = np.linalg.norm(y - y_recovered) / np.linalg.norm(y)
    print('Recovery error', error)

    return y_recovered
