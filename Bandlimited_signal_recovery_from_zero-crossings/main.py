import os
from module.zero_crossings_data import zero_crossings_data
from module.optimization_solve import optimization_solve
from module.results_output import plot_and_save_with_csv

n, f_min, B, s, y = zero_crossings_data()

y_recovered, RRMS = optimization_solve(n, f_min, B, s, y)

# 输出路径
current_dir = os.path.dirname(os.path.realpath(__file__))
results_output_folder_path = os.path.join(current_dir, 'outputs/results')
error_output_folder_path = os.path.join(current_dir, 'outputs/errors')

# 输出结果
# 绘制原始信号y和估计信号y_recovered的图像
plot_and_save_with_csv(y, 'original and recovered bandlimited signals', 'Sample Index', 'bandlimited signals', results_output_folder_path, y2_data=y_recovered, y2_label='bandlimited signals', legend1='original', legend2='recovered')
plot_and_save_with_csv(RRMS, 'RRMS Error', 'Iteration', 'Error', error_output_folder_path, legend1='RRMS')
