import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_and_save_with_csv(y1_data, title, xlabel, ylabel, output_folder_path, fig_size=(24, 5), plot_type='plot', y2_data=None, y2_label=None, legend1=None, legend2=None):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 绘制图像
    fig, ax1 = plt.subplots(figsize=fig_size)

    if plot_type == 'plot':
        ax1.plot(y1_data, color='tab:blue', label=legend1)
    elif plot_type == 'stem':
        ax1.stem(y1_data, linefmt='tab:blue', markerfmt='bo', label=legend1)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.tick_params(axis='y')

    # 第二个 Y 轴的处理
    if y2_data is not None and y2_label is not None:
        ax2 = ax1.twinx()
        ax2.plot(y2_data, '--', color='tab:red', label=legend2)
        ax2.set_ylabel(y2_label)
        ax2.tick_params(axis='y')

    plt.title(title)

    # 显示图例
    if legend1 is not None:
        ax1.legend(loc='upper left')
    if y2_data is not None and legend2 is not None:
        ax2.legend(loc='upper right')

    plt.tight_layout()

    # 保存图像
    safe_title = title.replace(' ', '_').replace('.', '').replace('-', '_')
    image_file_name = f"{safe_title}.jpg"
    plt.savefig(os.path.join(output_folder_path, image_file_name))
    plt.show()

    # 合并数据并保存到 CSV
    csv_file_name = f"{safe_title}.csv"
    if y2_data is not None:
        combined_data = pd.DataFrame({legend1: y1_data, legend2: y2_data})
    else:
        combined_data = pd.DataFrame({legend1: y1_data})
    combined_data.to_csv(os.path.join(output_folder_path, csv_file_name), index=False, header=True)
