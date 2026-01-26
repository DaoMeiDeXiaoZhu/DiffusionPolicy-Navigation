import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def check_zarr_file(file_path):
    print(f"\n{'='*20} 正在检查: {os.path.basename(file_path)} {'='*20}")
    
    try:
        # 1. 打开 Zarr 文件
        # mode='r' 只读模式
        root = zarr.open(file_path, mode='r')
        
        # 2. 检查基本组结构
        print(f"[结构检查]")
        if 'data' not in root or 'meta' not in root:
            print(f"❌ 失败: 根目录必须包含 'data' 和 'meta' 组。当前包含: {list(root.keys())}")
            return False
            
        data_group = root['data']
        meta_group = root['meta']
        
        keys = list(data_group.keys())
        print(f"✅ data 组包含键: {keys}")
        
        # 3. 获取各数组形状并检查对齐
        # 假设你保存的键名是 'img', 'state', 'action'
        # 如果你之前用了 'image' 或 'scan'，这里会报错，正好帮你发现命名不一致
        try:
            img_arr = data_group['img']
            state_arr = data_group['state']
            action_arr = data_group['action']
            episode_ends = meta_group['episode_ends']
        except KeyError as e:
            print(f"❌ 失败: 缺少关键数据键 {e}。请检查你的保存代码命名。")
            return False

        T_img = img_arr.shape[0]
        T_state = state_arr.shape[0]
        T_action = action_arr.shape[0]
        
        print(f"\n[维度检查]")
        print(f"  img   : {img_arr.shape} | 类型: {img_arr.dtype} | Chunks: {img_arr.chunks}")
        print(f"  state : {state_arr.shape}       | 类型: {state_arr.dtype}")
        print(f"  action: {action_arr.shape}         | 类型: {action_arr.dtype}")
        print(f"  ends  : {episode_ends[:]}           | 类型: {episode_ends.dtype}")

        # 校验时间维度是否一致
        if not (T_img == T_state == T_action):
            print(f"❌ 失败: 时间维度不一致! img={T_img}, state={T_state}, action={T_action}")
            return False
        
        # 校验 episode_ends
        last_end = episode_ends[-1]
        if last_end != T_img:
            print(f"❌ 失败: meta/episode_ends ({last_end}) 与实际数据长度 ({T_img}) 不匹配")
            return False
        
        print(f"✅ 所有数组时间维度对齐，episode_ends 匹配。")

        # 4. 数据内容抽检 (可视化)
        print(f"\n[内容抽检]")
        
        # 随机取一帧
        idx = np.random.randint(0, T_img)
        sample_img = img_arr[idx]
        sample_state = state_arr[idx]
        sample_action = action_arr[idx]
        
        print(f"  随机采样第 {idx} 帧:")
        print(f"  -> Action (v, w): {sample_action}")
        print(f"  -> State (Scan) mean: {np.mean(sample_state):.4f}")
        print(f"  -> Img range: [{sample_img.min()}, {sample_img.max()}]")

        # 简单绘图
        plt.figure(figsize=(10, 4))
        
        # 显示图片
        plt.subplot(1, 2, 1)
        plt.title(f"Frame {idx} Image")
        plt.imshow(sample_img) # 如果颜色不对，可能是 RGB/BGR 问题
        plt.axis('off')
        
        # 显示雷达数据
        plt.subplot(1, 2, 2)
        plt.title(f"Frame {idx} Scan (State)")
        plt.plot(sample_state)
        plt.xlabel("Angle Index")
        plt.ylabel("Distance")
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 检查完成: 文件结构格式正确 (Diffusion Policy 兼容)")
        return True

    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 自动查找 expert_data_collection 下所有的 .zarr
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'expert_data_collection')
    
    zarr_files = glob.glob(os.path.join(data_dir, "*.zarr"))
    
    if not zarr_files:
        print(f"未在 {data_dir} 找到任何 .zarr 文件。请先运行采集节点。")
    else:
        # 默认检查最新的一个文件
        latest_file = max(zarr_files, key=os.path.getctime)
        check_zarr_file(latest_file)