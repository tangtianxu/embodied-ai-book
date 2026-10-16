"""CPU toy learning: matched label budgets, narrow BC vs visited-state DAgger.
Fixed 1-D goal; this is neither a VLA nor a real robot experiment.
"""
import json,copy
from pathlib import Path
import numpy as np

GOAL=.8
def teacher(x): return np.clip(.35*(GOAL-x),-.12,.12)

class Policy:
    def __init__(self, seed):
        rng=np.random.default_rng(seed)
        self.p=[rng.normal(0,.5,(1,32)),np.zeros((1,32)),rng.normal(0,.05,(32,1)),np.zeros((1,1))]
        self.m=[np.zeros_like(x) for x in self.p];self.v=[np.zeros_like(x) for x in self.p];self.step=0
    def forward(self,x):
        h=np.tanh(x@self.p[0]+self.p[1]);z=np.tanh(h@self.p[2]+self.p[3]);return .15*z,h,z
    def act(self,x): return float(self.forward(np.array([[x]]))[0][0,0])
    def fit(self,x,y,rng,steps=350):
        x=np.asarray(x).reshape(-1,1);y=np.asarray(y).reshape(-1,1)
        for _ in range(steps):
            # 步骤1：从累计监督集采样批次
            ix=rng.integers(0,len(x),128);xb=x[ix];yb=y[ix]
            # 步骤2：计算动作与中间特征
            out,h,z=self.forward(xb)
            # 步骤3：由动作误差反传到两层参数
            gz=2*(out-yb)/len(ix)*.15*(1-z*z)
            gh=(gz@self.p[2].T)*(1-h*h)
            grads=[xb.T@gh,gh.sum(0,keepdims=True),h.T@gz,gz.sum(0,keepdims=True)]
            # 步骤4：更新 Adam 状态和策略参数
            self.step+=1
            for j,g in enumerate(grads):
                self.m[j]=.9*self.m[j]+.1*g;self.v[j]=.999*self.v[j]+.001*g*g
                self.p[j]-=.007*(self.m[j]/(1-.9**self.step))/(np.sqrt(self.v[j]/(1-.999**self.step))+1e-8)
        return float(np.mean((self.forward(x)[0]-y)**2))

def rollout(model,start):
    x=float(start);states=[];dwell=0;ok=False
    for _ in range(40):
        states.append(x);x+=model.act(x)
        dwell=dwell+1 if abs(x-GOAL)<.025 else 0
        if dwell>=3:ok=True;break
    return states,ok,abs(x-GOAL)

def run(seed):
    rng=np.random.default_rng(seed)
    # 步骤1：建立初始专家数据与相同模型
    xs=rng.uniform(-.9,-.5,160);ys=teacher(xs)
    bc=Policy(seed);bc.fit(xs,ys,rng);dag=copy.deepcopy(bc);broad=copy.deepcopy(bc)
    bx=list(xs);dx=list(xs);wx=list(xs);rows=[];test=np.linspace(-1,.75,100)
    def evaluate(model):
        rows=[rollout(model,s) for s in test]
        return {'success':float(np.mean([r[1] for r in rows])),'final_error':float(np.mean([r[2] for r in rows]))}
    # 步骤6：在固定起点记录闭环结果
    rows.append({'round':0,'expert_labels':160,'bc':evaluate(bc),'dagger':evaluate(dag),'broad_bc':evaluate(broad)})
    for step in range(1,7):
        # 步骤2：让当前 DAgger 策略访问状态
        visited=[x for start in rng.uniform(-1,.75,24) for x in rollout(dag,start)[0]]
        # 步骤3：在已访问状态取得相同预算的专家标签
        chosen=rng.choice(np.asarray(visited),160,replace=len(visited)<160)
        # 步骤4：汇总数据；对照分别补窄覆盖和宽覆盖
        dx.extend(chosen);bx.extend(rng.uniform(-.9,-.5,160));wx.extend(rng.uniform(-1,1.3,160))
        # 步骤5：用对应监督数据重新训练三种策略
        b_loss=bc.fit(bx,teacher(np.array(bx)),rng);d_loss=dag.fit(dx,teacher(np.array(dx)),rng)
        w_loss=broad.fit(wx,teacher(np.array(wx)),rng)
        # 步骤6：在固定起点记录闭环结果
        rows.append({'round':step,'expert_labels':160*(step+1),'bc':evaluate(bc),'dagger':evaluate(dag),'broad_bc':evaluate(broad),'bc_training_mse':b_loss,'dagger_training_mse':d_loss,'broad_bc_training_mse':w_loss})
    return {'seed':seed,'history':rows}

if __name__=='__main__':
    result={'numpy':np.__version__,'goal':GOAL,'horizon':40,'dwell_steps':3,'seeds':[run(s) for s in [3,7,11,19,23]]}
    Path(__file__).with_name('bc_dagger_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'initial':result['seeds'][0]['history'][0],'final':[r['history'][-1] for r in result['seeds']]}))
