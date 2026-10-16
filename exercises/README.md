# 精选练习代码

这些文件对应正文中的核心算例与练习，保留公式和算法步骤的注释。包括小型合成数据训练；不包含完整VLA训练、数据集或机械臂项目。

## 环境与运行

建议使用 Python 3.10 或更新版本，在虚拟环境中安装依赖。CPU 即可运行下列验证：

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
# macOS / Linux 可改用：source .venv/bin/activate
python -m pip install -r exercises/requirements.txt
python exercises/test_mechanisms.py
python exercises/basics/verify_tutorials.py
```

前两条创建和激活虚拟环境的命令可按个人环境调整。已有 NumPy 和 PyTorch 时，可直接执行最后两条 Python 命令。验证记录分别写入脚本所在目录的 verification_results.json。

## 代码与正文

| 文件 | 主要内容 | 对应章节 |
| --- | --- | --- |
| basics/autograd_freezing.py | 参数冻结与 no_grad 对梯度路径的影响 | 1.2 |
| basics/attention_mha.py、basics/transformer_block.py | 多头注意力与完整 Transformer 块 | 1.4—1.5 |
| basics/dino_head_signal.py | DINO 预测头的目标、居中与梯度算例 | 1.3 |
| attention_numpy.py、attention_torch.py | 遮蔽注意力、手写反向传播、多头、缓存、LoRA | 1.4—1.6 |
| generative_actions.py | 动作掩码损失、扩散加噪、流匹配与欧拉采样 | 1.7—1.10 |
| rl_estimators.py | GAE、PPO 裁剪目标、期望分位损失与动作概率修正 | 第 3 章 |
| robotics_algorithms.py | 刚体变换、平面机械臂雅可比、阻尼逆解、RANSAC 与 RRT | 第 4、8、9 章 |
| systems_debug.py | 因果时间对齐、分布式损失缩放与经验回放 | 第 7、10 章 |
| efficient_attention.py | 在线 softmax、量化点积与 KV 缓存容量 | 第 11 章 |
| test_mechanisms.py、basics/verify_tutorials.py | 数值算例、梯度和关键失败边界检查 | 对应上述实现 |

运行 `python exercises/generative_actions.py` 还会训练一维条件流模型，并将采样统计写入 flow_training_results.json。它使用人工分布说明训练与采样的联系，不是完整 VLA 模型。

DINO 文件只展开预测头上的一个教学批次；RRT 使用二维点机器人及矩形障碍。相应简化条件见源码和正文。

## 按机制继续学习

| 目录 | 可运行入口 | 对应内容 |
| --- | --- | --- |
| transformer/ | gradient_case.py、cache_execution.py | 1.5：残差梯度、冻结路径与缓存 |
| conditioning/ | conditioning_paths.py、geometry_inputs.py | 1.6：条件调制、零初始化与几何输入 |
| latent/ | cvae_step.py | 1.8：CVAE重建、KL与一次更新 |
| diffusion/ | diffusion_core.py、train_denoiser.py | 1.9：解析去噪、双峰去噪器训练与采样 |
| flow/ | flow_core.py、train_bimodal_flow.py | 1.10：速度场、积分与双峰流训练 |
| action_tokens/ | fast_steps.py、consistency_gaussian.py | 1.11：动作分词与一致性端点学习 |
| vla/ | hierarchy_trace.py、memory_routes.py | 1.12—1.13：分层预测与记忆读取 |
| adaptation/ | lora_paths.py、attention_routes.py、joint_paths.py | 1.14：LoRA与联合训练梯度路径 |
| imitation/ | trace_dagger_rounds.py、temporal_ensemble.py、action_interface_trace.py | 第2章：纠错、时间集成与接口 |
| planning/ | rrt_trace.py、verify_rrt.py | 8.3：完整搜索、碰撞拒绝、回溯与重连 |
| simulation/ | single_joint/simulate_joint.py、vector_boundary.py | 6.3—6.4：单关节和回合边界 |

在仓库根目录运行，例如：

```bash
python exercises/latent/cvae_step.py
python exercises/conditioning/conditioning_paths.py
python exercises/diffusion/train_denoiser.py --seeds 7 --updates 8000
python exercises/flow/train_bimodal_flow.py --seeds 13 --updates 8000
python exercises/planning/rrt_trace.py --out exercises/planning/results
python exercises/planning/verify_rrt.py
```

生成模型训练脚本默认可比较多个种子；上面只运行正文的代表种子。训练在CPU上进行，结果与临时权重写入各目录的results/，由命令生成，不作为预先提供的实验数据。其他脚本同样用 `python exercises/目录/文件.py` 执行。

仿真练习需另外安装依赖：

```bash
python -m pip install -r exercises/requirements-simulation.txt
python exercises/simulation/single_joint/simulate_joint.py
python exercises/simulation/vector_boundary.py
```

单关节只推进CPU物理，不需要图像窗口。第6.8节机械臂抓放以书内图文案例说明，资产入口见正文。FAST的整数教学实现可直接运行；真实发布词表和分词器应使用正文所链的官方实现。


本轮在Python 3.12、NumPy 2.3.2、PyTorch 2.8.0+cpu环境运行了原有52项机制检查、16项基础算例检查，以及新增目录的核心脚本。另在独立虚拟环境运行了单关节和Gymnasium回合边界示例。书中的GPU、多机和真机应用需要相应设备。
