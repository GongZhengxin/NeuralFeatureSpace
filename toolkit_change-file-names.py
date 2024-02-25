import os

# 指定根目录路径
wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
root_dir = os.path.join(wk_dir, 'build/featurewise-corr/filtered')

# 遍历根目录下的子目录
for i in range(1, 10):
    sub_dir = f"sub-{i:02}"  # 生成子目录名称，如sub-01, sub-02, ..., sub-09
    full_sub_dir_path = os.path.join(root_dir, sub_dir)

    # 检查子目录是否存在
    if os.path.isdir(full_sub_dir_path):
        # 遍历子目录下的所有文件
        for filename in os.listdir(full_sub_dir_path):
            if "pvalus" in filename:  # 检查文件名中是否含有错误单词"pvalus"
                # 生成新的文件名，将"pvalus"替换为"pvalue"
                new_filename = filename.replace("pvalus", "pvalue")
                # 生成旧文件和新文件的完整路径
                old_file_path = os.path.join(full_sub_dir_path, filename)
                new_file_path = os.path.join(full_sub_dir_path, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")
