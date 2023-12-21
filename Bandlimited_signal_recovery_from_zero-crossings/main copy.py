'''
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from module.zero_crossings_data import *

# 构造矩阵A，其列为带限正弦波
C = np.zeros((n, B))
S = np.zeros((n, B))
for j in range(B):
    C[:, j] = np.cos(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
    S[:, j] = np.sin(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
A = np.hstack((C, S))

# 最小化目标函数，同时考虑L1规范化和符号约束
x = cvx.Variable(2 * B)
obj = cvx.norm(A @ x)
constraints = [cvx.multiply(s, A @ x) >= 0, s.T @ (A @ x) == n]
problem = cvx.Problem(cvx.Minimize(obj), constraints)
problem.solve(solver=cvx.ECOS)

y_hat = A @ x.value
print('Recovery error: {}'.format(np.linalg.norm(y - y_hat) / np.linalg.norm(y)))

plt.figure(figsize=(24, 5))
plt.plot(np.arange(0, n), y, label='original');
plt.plot(np.arange(0, n), y_hat, label='recovered');
plt.xlim([0, n])
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('./Bandlimited_signal_recovery_from_zero-crossings/outputs/original_and_recovered_y_signal.jpg')
plt.show()
# 绘制原始信号y和估计信号y_hat的图像，以比较二者。使用matplotlib.pyplot绘制折线图，设置图例和x轴范围。
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# 导入用户上传的数据文件
from module.zero_crossings_data import *

# 构造矩阵A，其列为带限的正弦和余弦波
C = np.zeros((n, B))
S = np.zeros((n, B))
for j in range(B):
    C[:, j] = np.cos(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
    S[:, j] = np.sin(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
A = np.hstack((C, S))

# 定义优化问题的目标函数：最小化A*x的二范数
def objective(x):
    return np.linalg.norm(np.dot(A, x), 2)

# 定义符号约束函数
def sign_constraints(x):
    return np.multiply(s, np.dot(A, x))

# 定义L1范数规范化条件
def l1_norm_constraint(x):
    return np.sum(np.abs(np.dot(A, x))) - n

# 初始化变量
x0 = np.random.randn(2 * B)

# 设置约束条件
cons = ({'type': 'ineq', 'fun': sign_constraints},  # 符号约束
        {'type': 'eq', 'fun': l1_norm_constraint})  # L1范数规范化

# 使用scipy.optimize的minimize函数求解
result = minimize(objective, x0, constraints=cons)

# 恢复信号
y_hat = np.dot(A, result.x)

# 计算恢复误差
recovery_error = np.linalg.norm(y - y_hat) / np.linalg.norm(y)
print(f'恢复误差: {recovery_error}')

# 绘制原始信号和恢复信号
plt.figure()
plt.plot(np.arange(n), y, label='original')
plt.plot(np.arange(n), y_hat, label='recovered')
plt.xlim([0, n])
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('./Bandlimited_signal_recovery_from_zero-crossings/outputs/original_and_recovered_y_signal2.jpg')
plt.show()
