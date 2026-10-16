# 精选练习代码

这些文件来自书中已有的练习实现，保留对应算法步骤的注释。本目录不包含完整训练、数据或仿真项目。

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

书内的 unified_practice/mechanisms/ 与 unified_practice/tutorials/，在本仓库中分别对应 exercises/ 与 exercises/basics/。未列在表中的其他配套项目不在此次发布范围内。

此次发布前已在 Python 3.12.8、NumPy 2.3.2、PyTorch 2.8.0+cpu 环境中运行：52 项机制检查与 16 项基础算例检查均通过。
