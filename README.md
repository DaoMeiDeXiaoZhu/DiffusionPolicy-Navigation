**项目说明：**

该项目是我自己搭建的一个导航，该模型以幻尔的 ROS2 小车为真实载体，从固定起点到固定终点的整个过程中，每 0.1s 收集一次专家经验（图像+雷达+速度话题消息），每隔一定的回合，环境会有小范围变化并不是一成不变的。该项目我们从 专家经验收集 -> 网络训练 -> 真机部署 介绍完整的流程，整个项目的文件结果如下所示：

<img width="2026" height="430" alt="yuque_diagram (1)" src="https://github.com/user-attachments/assets/a0bd08bf-2de2-4122-9c4c-a9b05319b2f3" />

**文件说明：**

<img width="1754" height="487" alt="image" src="https://github.com/user-attachments/assets/b6cf4313-7ad6-4ddb-8734-b9330ea3b4f7" />

**算法流程：**

![DiffusionPolicy原理图](https://github.com/user-attachments/assets/38d2b14e-a747-472a-b798-6dda75559c6f)

**如何运行：**

1. 连接ROS2小车，通过ros2 topic list 查看话题是否存在，如果存在则运行程序collect_expert_experience/collect_data.py
2. 收集完数据后运行程序train_diffusion_policy/train.py
3. 训练完成后连接小车运行程序reality_run.py


1. collect_data.py：收集专家经验，将不稳定的雷达话题/scan_raw的雷达消息处理为固定的 360 条光束并每次采样都取出最新的 360 维数据，图像信息每次采样都从/ascamera/camera_publisher/rgb0/image话题中取出最新的数据，动作的收集依赖/cmd_vel话题每次采样都从该话题中取出最新的 linear_x, angular_z。
2. check_data.py：随机抽取专家经验进行检查，看一下雷达、图像、动作格式是否正确，图像是否为彩色正常显示。
3. reality_run.py：使用训练好的 DiffusionPolicy，通过话题接收/发送的形式控制 ROS2 小车进行导航。由于采集专家经验的时候，我是通过 rviz2 实时查看图像和雷达并通过向/cmd_vel话题发送速度指令来控制小车。消息的接收和发送本身就存在延迟，这种延迟也是 DiffusionPolicy 学习的一部分，直接部署到真机上会破坏学习到的策略，所以该模型不推荐部署到真机上，而是通过服务器转发的形式进行控制。
4. dataset.py作用（切分轨迹）：假设我一共收集了$n$条完整的轨迹，每个轨迹$\tau_i$包含的 $T_i$帧，那么该函数的作用是根据时间窗口horizon来切分，每个轨迹$\tau_i$就会被切分为$T_i - horizon$个时间上连续的序列$[(o_0,a_0),(o_1,a_1), ..., (o_{horizon-1}, a_{horizon-1})]$，每个序列包含完整的观测和动作，因此$n$条轨迹一共会被切分为$\sum_{i=1}^{i=n} (T_i-horizon)$个时间上连续的序列。
5. diffusion.py作用：计算累计噪声系数$ᾱₖ$，并根据随机噪声$\epsilon$、扩散步$k$、原始专家动作$x_0$，生成任意$k$的带噪动作$x_k$。
6. inference.py作用：根据最大扩散步数$K=1000$，该文件随机生成一个初始噪声作为当前动作$x_{1000}$，根据一步步根据当前观测$o$、$\bar{\alpha}_t$、$\alpha_t$、$\beta_t$、$\epsilon_{\theta}$、$t = 1000,999,...,1$从后向前计算出$x_{999},...,x_0$，这个$x_0$就是专家动作。
7. model.py作用：根据 transformer（ 图像序列 + 雷达序列 + 带噪动作序列 + 扩散步 t ） 得到动作噪声$\epsilon_{\theta}$
8. trainer.py作用：通过$transformer$提供的预测噪声$\epsilon_{\theta}$和真实噪声$\epsilon$做均方误差从而更新该网络。
9. train.py作用：集中调用前面提到的所有模块，按照顺序：选择训练设备 -> 找到数据集路径 -> 创建 Dataset（切片专家轨迹）-> 创建 DataLoader（批量提供训练数据）-> 创建 Diffusion Policy 模型 -> 创建 Diffusion 对象 -> 创建 Trainer（训练循环） -> 开始训练
