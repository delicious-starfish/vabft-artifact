import torch
import torch_npu
import os

_FAULTY_ROOT = os.environ.get('LLM_DATA_ROOT_FAULTY', './checksum_llm_data_faulty')
import glob
import utils
from datetime import datetime

def check_folder(folder_path, log_file_path):
    """
    检查指定文件夹中的a/b/c文件
    :param folder_path: 待检查的目标文件夹绝对路径
    :param log_file_path: 日志文件的绝对路径
    """
    print(f"开始检查文件夹: {folder_path}")
    
    try:
        device = torch.device(f"npu:0")
        # 1. 直接使用 os.path.join 获取目标路径下的文件，不切换工作目录
        # glob 支持直接传入带路径的 pattern
        pattern_a = os.path.join(folder_path, "*_a_*.pth")
        files_a = sorted(glob.glob(pattern_a))
        
        # 获取文件名（不带路径），方便后续处理前缀
        files_a_names = [os.path.basename(f) for f in files_a]
        
        # 统计数量 (glob 可能会扫到 b 和 c，这里仅做数量参考，实际逻辑依赖 prefixes)
        count_a = len(files_a)
        count_b = len(glob.glob(os.path.join(folder_path, "*_b_*.pth")))
        count_c = len(glob.glob(os.path.join(folder_path, "*_c_*.pth")))
        
        print(f"在 {folder_path} 中找到: A={count_a}, B={count_b}, C={count_c}")
        
        # 2. 提取前缀
        prefixes = []
        for f_name in files_a_names:
            try:
                # 假设文件名格式为 prefix_a_suffix.pth
                parts = f_name.split("_a_")
                if len(parts) == 2:
                    prefixes.append((parts[0], parts[1]))
            except:
                continue
        
        print(f"提取到 {len(prefixes)} 个文件前缀")
        
        if len(prefixes) == 0:
            with open(log_file_path, "a") as f:
                f.write(f"{datetime.now()} - 错误: 在 {folder_path} 中未找到有效的文件对\n")
            return
        
        failed_files = []
        
        # 3. 遍历检查
        for i, prefix in enumerate(prefixes):
            try:
                # 构造完整路径
                base_name_a = f"{prefix[0]}_a_{prefix[1]}"
                base_name_b = f"{prefix[0]}_b_{prefix[1]}"
                base_name_c = f"{prefix[0]}_c_{prefix[1]}"
                
                path_a = os.path.join(folder_path, base_name_a)
                path_b = os.path.join(folder_path, base_name_b)
                path_c = os.path.join(folder_path, base_name_c)
                
                # 检查文件是否存在
                if not (os.path.exists(path_a) and os.path.exists(path_b) and os.path.exists(path_c)):
                    failed_files.append(f"文件缺失: {base_name_a} (或对应的b/c)")
                    continue
                
                # 加载 PyTorch 张量
                a = torch.load(path_a).to(device)
                b = torch.load(path_b).to(device)
                # c = torch.load(path_c) # 如果 utils.FT_matmul 不需要 c，可以不加载以节省IO
                # print(f"[{i+1}/{len(prefixes)}] Checking {base_name_a}...") 
                
                # 使用FT_matmul计算
                success, _ = utils.FT_matmul(a, b)
                
                if not success:
                    msg = f"FT_matmul失败: {base_name_a}"
                    failed_files.append(msg)
                    print(msg)
                    continue
                
                # 每100个文件输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已完成 {i+1}/{len(prefixes)} 个文件的检查")
                    
            except Exception as e:
                failed_files.append(f"处理异常: {prefix[0]}_*_{prefix[1]}, 错误: {str(e)}")
                continue
        
        # 4. 记录结果到日志 (使用绝对路径)
        with open(log_file_path, "a") as f:
            if failed_files:
                f.write(f"\n{datetime.now()} - 文件夹 {folder_path} 检查结果:\n")
                f.write(f"总共检查 {len(prefixes)} 个文件对，失败 {len(failed_files)} 个:\n")
                for failed in failed_files:
                    f.write(f"  {failed}\n")
                f.write("-" * 80 + "\n")
                print(f"文件夹 {folder_path} 检查完成，存在失败文件: {len(failed_files)}")
            else:
                f.write(f"{datetime.now()} - 文件夹 {folder_path} 所有文件检查通过!\n")
                print(f"文件夹 {folder_path} 所有文件检查通过!")
        
    except Exception as e:
        # 顶层异常捕获
        with open(log_file_path, "a") as f:
            f.write(f"{datetime.now()} - 检查文件夹 {folder_path} 时发生致命错误: {str(e)}\n")
        print(f"错误: {e}")

def main():
    # 1. 获取日志文件的【绝对路径】
    # os.getcwd() 获取当前脚本运行目录
    log_file_name = "check_log4.txt"
    log_file_path = os.path.join(os.getcwd(), log_file_name)
    
    print(f"日志文件将保存在: {log_file_path}")
    
    # 清空或创建日志文件
    with open(log_file_path, "w") as f:
        f.write(f"检查开始时间: {datetime.now()}\n")
        f.write("=" * 80 + "\n")
    
    # 要检查的三个文件夹
    folders_to_check = [
        os.path.join(_FAULTY_ROOT, "mlp_r"),
        os.path.join(_FAULTY_ROOT, "atte_c"),
        os.path.join(_FAULTY_ROOT, "atte_r")
    ]
    
    # 检查每个文件夹
    for folder in folders_to_check:
        if os.path.exists(folder):
            check_folder(folder, log_file_path)
        else:
            with open(log_file_path, "a") as f:
                f.write(f"{datetime.now()} - 错误: 文件夹不存在 {folder}\n")
            print(f"跳过不存在的文件夹: {folder}")
    
    # 检查完成
    with open(log_file_path, "a") as f:
        f.write(f"\n检查结束时间: {datetime.now()}\n")
        f.write("=" * 80 + "\n")
    
    print(f"所有检查完成! 详情请查看 {log_file_path}")

if __name__ == "__main__":
    main()
