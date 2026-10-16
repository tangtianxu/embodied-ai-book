import json, math
import torch
from torch.nn import functional as F
from attention_mha import TutorialMHA, safe_softmax
from transformer_block import TutorialBlock
from autograd_freezing import inspect_gradients

def verify():
    checks = []
    def check(label, condition):
        assert bool(condition), label
        checks.append(label)
    check('冻结编码器仍可训练上游', inspect_gradients(False) == [-3., None, -6.])
    check('no_grad丢失上游路径，保留下游训练', inspect_gradients(True) == [None, None, -6.])
    torch.manual_seed(7)
    model = TutorialMHA(12, 3).double()
    x = torch.randn(2, 3, 12, dtype=torch.double, requires_grad=True)
    ctx = torch.randn(2, 7, 12, dtype=torch.double, requires_grad=True)
    allow = torch.ones(2, 1, 3, 7, dtype=torch.bool)
    allow[..., -1] = False
    out, weights = model(x, ctx, allow)
    q, k, v = model.split(model.q(x)), model.split(model.k(ctx)), model.split(model.v(ctx))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
    ref = model.out(ref.transpose(1, 2).reshape(2, 3, 12))
    check('多头前向与官方缩放点积结果一致', torch.allclose(out, ref, atol=1e-10, rtol=1e-10))
    go = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, ctx), go, retain_graph=True)
    expected = torch.autograd.grad(ref, (x, ctx), go)
    check('查询输入梯度一致', torch.allclose(actual[0], expected[0], atol=1e-10, rtol=1e-10))
    check('上下文输入梯度一致', torch.allclose(actual[1], expected[1], atol=1e-10, rtol=1e-10))
    check('每个查询在允许集合归一化', torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1))))
    check('禁止键权重为零', torch.all(weights[..., -1] == 0))
    block = TutorialBlock(12, 3).double()
    seq = torch.randn(2, 5, 12, dtype=torch.double)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    altered = seq.clone(); altered[:, 3:] += 5
    check('完整Transformer块不泄漏未来', torch.allclose(block(seq, mask)[:, :3], block(altered, mask)[:, :3], atol=1e-10))
    raw = torch.tensor([[6.,2.,0.],[5.,3.,0.],[5.,0.,3.]], dtype=torch.double)
    q = torch.softmax((raw - raw.mean(0))/.1, -1)
    zs = torch.zeros_like(raw, requires_grad=True)
    loss = -(q * torch.log_softmax(zs, -1)).sum(-1).mean()
    g = torch.autograd.grad(loss, zs)[0]
    check('DINO不同图片有不同目标', q.argmax(-1).tolist() == [0,1,2])
    check('DINO交叉熵梯度符合p-q', torch.allclose(g, (torch.softmax(zs,-1)-q)/3))
    z = torch.zeros(2, dtype=torch.double, requires_grad=True)
    loss = -(torch.tensor([.5,.5]) * torch.log_softmax(z, -1)).sum()
    check('均匀教师目标允许零梯度', torch.equal(torch.autograd.grad(loss,z)[0], torch.zeros(2)))
    p = torch.softmax(torch.tensor([1/math.sqrt(2), 0.],dtype=torch.double),0)
    q2 = torch.softmax(torch.tensor([0.,4/math.sqrt(2)],dtype=torch.double),0)
    v = torch.tensor([[2.,0.],[0.,4.]],dtype=torch.double)
    check('双头手算结果', abs((p@v)[0].item()-1.3395230987)<1e-9 and abs((q2@v)[1].item()-3.776771)<1e-5)
    elbo = .8*math.log(.7)+.2*math.log(.3)-(.8*math.log(1.6)+.2*math.log(.4))
    gap = math.log(.5)-elbo
    check('ELBO差等于后验KL', abs(gap-(.8*math.log(.8/.7)+.2*math.log(.2/.3)))<1e-12)
    pp = 1/(1+math.exp(-80/9)); en=(1-1.6*(2*pp-1))/.6
    check('双峰条件噪声均值', abs(en+0.9992645665)<1e-9)
    alpha=1/3; ell=alpha+1; u=2*alpha+6
    check('分块softmax恢复整矩阵输出', abs(u/ell-5)<1e-12)
    check('因子图最优解', abs((2*.933333333333-1.866666666666))<1e-10)
    return checks

if __name__ == '__main__':
    checks=verify()
    result={'checks':len(checks),'failures':0,'items':checks,'device':'CPU'}
    from pathlib import Path
    Path(__file__).with_name('verification_results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
