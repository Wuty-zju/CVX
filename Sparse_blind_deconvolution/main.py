import os
from module.blind_deconv_data import generate_waveform, inverse_ker
from module.optimization_solve import optimization_solve
from module.results_output import plot_and_save_with_csv

# 生成波形
w_true, y, T, k, N = generate_waveform()

# 优化求解
w_optimal, x_optimal, rrms_values, mse_values, nmse_values = optimization_solve(w_true, y, T, k)

# 计算逆核
w_inverse = inverse_ker(w_optimal, N)

# 输出路径
current_dir = os.path.dirname(os.path.realpath(__file__))
results_output_folder_path = os.path.join(current_dir, 'outputs/results')
error_output_folder_path = os.path.join(current_dir, 'outputs/errors')

# 输出结果
plot_and_save_with_csv(y, 'Observation y', 'Time', 'Amplitude', results_output_folder_path, legend1='y')
plot_and_save_with_csv(w_optimal, 'Optimal Filter w', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem', legend1='w_optimal')
plot_and_save_with_csv(x_optimal, 'Sparse Signal x', 'Time', 'Amplitude', results_output_folder_path, legend1='x_optimal')
plot_and_save_with_csv(w_inverse, 'Inverse Kernel w^-1', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem', legend1='w_inverse')
plot_and_save_with_csv(rrms_values, 'RRMS Error', 'Iteration', 'Error', error_output_folder_path, legend1='RRMS')
plot_and_save_with_csv(mse_values, 'MSE Error', 'Iteration', 'Error', error_output_folder_path, legend1='MSE')
plot_and_save_with_csv(nmse_values, 'NMSE Error', 'Iteration', 'Error', error_output_folder_path, legend1='NMSE')
