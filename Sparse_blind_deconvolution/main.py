import os
from module.blind_deconv_data import generate_waveform, inverse_ker
from module.optimization_algorithm import optimize_filter
from module.results_output import plot_and_save_with_csv

# 生成波形
w_true, y, T, k, N = generate_waveform()

# 优化求解
w_optimal, x_optimal, mse_values, nmse_values = optimize_filter(w_true, y, T, k)

# 计算逆核
w_inverse = inverse_ker(w_optimal, N)

# 输出路径
current_dir = os.path.dirname(os.path.realpath(__file__))
results_output_folder_path = os.path.join(current_dir, 'outputs/results')
error_output_folder_path = os.path.join(current_dir, 'outputs/errors')

# 输出结果
plot_and_save_with_csv(w_optimal, 'Optimal Filter w', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem')
plot_and_save_with_csv(x_optimal, 'Sparse Signal x', 'Time', 'Amplitude', results_output_folder_path)
plot_and_save_with_csv(y, 'Observation y', 'Time', 'Amplitude', results_output_folder_path)
plot_and_save_with_csv(w_inverse, 'Inverse Kernel w^-1', 'Index', 'Amplitude', results_output_folder_path, plot_type='stem')
plot_and_save_with_csv(mse_values, 'MSE Over Iterations', 'Iteration', 'MSE', error_output_folder_path)
plot_and_save_with_csv(nmse_values, 'NMSE Over Iterations', 'Iteration', 'NMSE', error_output_folder_path)