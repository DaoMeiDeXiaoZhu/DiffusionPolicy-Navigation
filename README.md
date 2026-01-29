**项目说明：**

该项目是我自己搭建的一个导航，该模型以幻尔的 ROS2 小车为真实载体，从固定起点到固定终点的整个过程中，每 0.1s 收集一次专家经验（图像+雷达+速度话题消息），每隔一定的回合，环境会有小范围变化并不是一成不变的。该项目我们从 专家经验收集 -> 网络训练 -> 真机部署 介绍完整的流程，整个项目的文件结果如下所示：

<img width="2192" height="430" alt="yuque_diagram (2)" src="https://github.com/user-attachments/assets/eff235a8-b827-4385-92fc-5aa6579628bf" />


**文件说明：**

<img width="1520" height="1086" alt="image" src="https://github.com/user-attachments/assets/92dc6d3e-b585-47ed-bdf7-f0a528e35688" />



**算法流程：**

![DiffusionPolicy原理图](https://github.com/user-attachments/assets/38d2b14e-a747-472a-b798-6dda75559c6f)

**如何运行：**

1. 连接ROS2小车，通过ros2 topic list 查看话题是否存在，如果存在则运行程序collect_expert_experience/collect_data.py
2. 收集完数据后运行程序train_diffusion_policy/train.py
3. 训练完成后连接小车运行程序reality_run.py

**专家经验收集：**

