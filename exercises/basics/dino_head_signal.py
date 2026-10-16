import torch
from torch.nn import functional as F

# 步骤1：用前面的三张图片分数，学生暂时都预测均匀
teacher_logits = torch.tensor([[6., 2., 0.],
                               [5., 3., 0.],
                               [5., 0., 3.]])
student_logits = torch.zeros(3, 3, requires_grad=True)
center = teacher_logits.mean(dim=0, keepdim=True)
# 步骤2：这个例子用当前批次均值展示居中；实际用移动中心
with torch.no_grad():
    target = F.softmax((teacher_logits - center) / 0.1, dim=-1)
# 步骤3：固定教师目标，只训练学生去匹配
log_prediction = F.log_softmax(student_logits, dim=-1)
loss = -(target * log_prediction).sum(dim=-1).mean()
loss.backward()
# 步骤4：三张图的主要目标维度分别为 0、1、2
print(target.argmax(dim=-1))
print(student_logits.grad)
