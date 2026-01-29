import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def check_zarr_file(file_path):
    print(f"\n{'='*20} 正在全量检查: {os.path.basename(file_path)} {'='*20}")
    
    try:
        root = zarr.open(file_path, mode='r')
        data = root['data']
        
        has_img = 'img' in data
        has_state = 'state' in data
        has_action = 'action' in data
        
        # --- 命令行数据概览提示 ---
        print(f"[数据详情]")
        if has_img:
            print(f"  📷 图像: {data['img'].shape} | dtype: {data['img'].dtype}")
        
        if has_state:
            lidar_sample = data['state'][:]
            # 在这里增加雷达数据的数值提示
            print(f"  📡 雷达: {data['state'].shape} | dtype: {data['state'].dtype}")
            print(f"      -> 范围: [{lidar_sample.min():.2f}m, {lidar_sample.max():.2f}m] | 均值: {lidar_sample.mean():.2f}m")
            
            # 自动预警：如果雷达数据全是 0 或全是最大值
            if lidar_sample.max() == 0:
                print("      ⚠️  警告: 雷达数据全为 0，请检查激光雷达是否启动或话题是否正确！")
            if lidar_sample.min() == lidar_sample.max():
                 print("      ⚠️  警告: 雷达数据无变化（死数），请检查传感器状态！")
        
        if has_action:
            act_sample = data['action'][:]
            print(f"  🎮 动作: {data['action'].shape} | v_range: [{act_sample[:,0].min():.2f}, {act_sample[:,0].max():.2f}]")

        T = data['action'].shape[0]
        
        # --- 1. 动态布局 ---
        plot_count = sum([has_img, has_state, has_action])
        fig = plt.figure(figsize=(5 * plot_count, 5))
        current_plot = 1

        # --- 2. 图像子图 ---
        if has_img:
            ax_img = fig.add_subplot(1, plot_count, current_plot)
            im_display = ax_img.imshow(data['img'][0])
            ax_img.set_title("Camera Feed")
            ax_img.axis('off')
            current_plot += 1

        # --- 3. 雷达子图 (极坐标) ---
        if has_state:
            ax_lidar = fig.add_subplot(1, plot_count, current_plot, projection='polar')
            lidar_init = data['state'][0]
            angles = np.linspace(0, 2*np.pi, len(lidar_init))
            
            # 极坐标散点图
            lidar_plot, = ax_lidar.plot(angles, lidar_init, '.', markersize=3, color='#00FF00')
            ax_lidar.fill(angles, lidar_init, color='g', alpha=0.1) # 增加阴影更易观察
            
            # 严格使用 config 中的归一化范围
            ax_lidar.set_ylim(config['lidar_min'], config['lidar_max']) 
            ax_lidar.set_title(f"Lidar Scan\nRange: {config['lidar_min']} - {config['lidar_max']}m")
            current_plot += 1

        # --- 4. 动作子图 ---
        ax_vel = fig.add_subplot(1, plot_count, current_plot)
        actions_np = data['action'][:]
        line_v, = ax_vel.plot([], [], label='Linear (v)', color='r', lw=1.5)
        line_w, = ax_vel.plot([], [], label='Angular (w)', color='b', lw=1.5)
        
        ax_vel.set_xlim(0, T)
        ax_vel.set_ylim(config['action_stats']['v_min'] - 0.2, config['action_stats']['v_max'] + 0.2)
        ax_vel.set_title("Expert Action Commands")
        ax_vel.legend(loc='upper right')
        ax_vel.grid(True, alpha=0.3)

        # --- 5. 动画播放 ---
        print(f"\n[播放] 正在预览数据步 (共 {T} 帧)...")
        plt.ion()
        step = max(1, T // 150) # 动态步进避免太慢
        
        for i in range(0, T, step):
            if not plt.fignum_exists(fig.number): break
            
            if has_img: im_display.set_data(data['img'][i])
            if has_state: lidar_plot.set_ydata(data['state'][i])
            
            line_v.set_data(np.arange(i), actions_np[:i, 0])
            line_w.set_data(np.arange(i), actions_np[:i, 1])
            
            fig.suptitle(f"Zarr Check: {os.path.basename(file_path)} | Step: {i}/{T}")
            plt.pause(0.005)

        plt.ioff()
        print("✅ 预览结束。")
        plt.show()
        return True

    except Exception as e:
        print(f"❌ 检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    data_dir = config['dataset_path']
    zarr_files = glob.glob(os.path.join(data_dir, "*.zarr"))
    
    if not zarr_files:
        print(f"❌ 错误: 目录 {data_dir} 下未找到 .zarr 文件")
    else:
        latest_file = max(zarr_files, key=os.path.getctime)
        check_zarr_file(latest_file)