import matplotlib.pyplot as plt
import pandas as pd
import os

# 输出结果的代码
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
    safe_title = title.replace(' ', '_').replace('.', '').replace('-', '_')
    image_file_name = f"{safe_title}.jpg"
    plt.savefig(os.path.join(output_folder_path, image_file_name))
    plt.show()

    # 保存数据到 CSV
    csv_file_name = f"{safe_title}.csv"
    pd.DataFrame(data).to_csv(os.path.join(output_folder_path, csv_file_name), index=False, header=True)
    