"""Train a real small denoiser; the analytic reference is used only for evaluation."""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch
from torch import nn
from diffusion_core import (ROOT,TRAIN_LEVELS,THETA_MAX,DATA_STD,alpha_bar,analytic_epsilon,
                            estimate_clean,ddim,equal_probability_noise,target_quantiles,
                            distribution_metrics)


class NoiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,64),nn.SiLU(),
                               nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,candidate,time_level,condition):
        angles=math.pi*time_level
        time_features=torch.cat([angles.sin(),angles.cos(),(2*angles).sin(),(2*angles).cos()],1)
        return self.net(torch.cat([candidate,time_features,condition],1))


def batch(generator,count):
    # P1: a fixed visible condition leaves two possible action modes.
    route=torch.randint(0,2,(count,1),generator=generator)
    clean=(2*route-1)*2.+DATA_STD*torch.randn((count,1),generator=generator)
    condition=torch.zeros((count,1))
    # P2: choose a level and manufacture a noisy input with its known noise label.
    index=torch.randint(1,TRAIN_LEVELS+1,(count,1),generator=generator)
    level=index.float()/TRAIN_LEVELS
    keep=torch.cos(THETA_MAX*level).square()
    noise=torch.randn((count,1),generator=generator)
    candidate=keep.sqrt()*clean+(1-keep).sqrt()*noise
    return clean,condition,index,level,candidate,noise


@torch.no_grad()
def evaluate(model,validation):
    _,condition,index,level,candidate,noise=validation
    prediction=model(candidate,level,condition)
    oracle=np.array([analytic_epsilon(np.array([float(x)]),int(k))[0]
                     for x,k in zip(candidate[:,0],index[:,0])],dtype=np.float32)
    oracle=torch.from_numpy(oracle[:,None])
    return {"noise_label_mse":float((prediction-noise).square().mean()),
            "oracle_noise_label_mse":float((oracle-noise).square().mean()),
            "mean_square_error_to_analytic_conditional_mean":float((prediction-oracle).square().mean())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[7,11,23])
    parser.add_argument("--updates",type=int,default=8000)
    args=parser.parse_args()
    if args.updates<1:
        raise ValueError("Need at least one update.")
    torch.set_num_threads(1)
    initial=equal_probability_noise()
    reference=target_quantiles()
    validation=batch(torch.Generator().manual_seed(90901),2048)
    all_results=[]
    for seed in args.seeds:
        started=time.perf_counter()
        torch.manual_seed(seed)
        generator=torch.Generator().manual_seed(seed+100000)
        model=NoiseMLP()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        logs=[{"update":0,**evaluate(model,validation)}]
        first_update=None
        model.train()
        for update in range(1,args.updates+1):
            clean,condition,index,level,candidate,noise=batch(generator,256)
            # P3: the network receives the current candidate, level and available condition.
            prediction=model(candidate,level,condition)
            # P4: epsilon is known because P2 sampled it; no analytic teacher is used here.
            loss=(prediction-noise).square().mean()
            optimizer.zero_grad()
            loss.backward()
            if update==1:
                old=torch.cat([p.detach().flatten().clone() for p in model.parameters()])
                grad_norm=math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
                first_update={"clean":float(clean[0]),"noise_level_index":int(index[0]),
                              "candidate":float(candidate[0]),"sampled_noise_label":float(noise[0]),
                              "predicted_noise_before":float(prediction[0].detach()),
                              "output_gradient_for_this_row":float(2*(prediction[0]-noise[0]).detach()/256),
                              "batch_loss_before":float(loss.detach()),"parameter_gradient_l2":grad_norm}
            # P5: change network parameters, then train on a fresh sample/noise pair.
            optimizer.step()
            if update==1:
                new=torch.cat([p.detach().flatten() for p in model.parameters()])
                first_update["parameter_change_l2"]=float((new-old).norm())
            if update%500==0 or update==args.updates:
                logs.append({"update":update,"last_batch_mse":float(loss.detach()),
                             **evaluate(model,validation)})
        model.eval()

        @torch.no_grad()
        def predict(x,index):
            candidate=torch.from_numpy(np.asarray(x,dtype=np.float32).reshape(-1,1))
            level=torch.full_like(candidate,index/TRAIN_LEVELS)
            return model(candidate,level,torch.zeros_like(candidate)).numpy().reshape(np.asarray(x).shape)

        metrics={}
        arrays={}
        for steps in [16,64,256]:
            final,snapshots,_=ddim(initial,steps,predict)
            metrics[str(steps)]=distribution_metrics(final,reference)
            arrays[f"generated_{steps}"]=final
            if steps==64:
                arrays.update({f"snapshot_{k}":v for k,v in snapshots.items()})
        one_call=estimate_clean(initial,predict(initial,1000),1000)
        arrays["one_call_clean_estimate"]=one_call
        probe=np.array([-1.5,-.5,-.1,0.,.1,.5,1.5])
        probe_final,_,trace=ddim(probe,64,predict,record=True)
        result={"seed":seed,"updates":args.updates,"batch_size":256,
                "scope":"one fixed condition, synthetic scalar two-mode actions; no robot execution",
                "torch":torch.__version__,"data_generator_seed":seed+100000,"validation_seed":90901,
                "validation_used_for_gradient_or_selection":False,
                "analytic_reference_used_for_training":False,"parameter_count":sum(p.numel() for p in model.parameters()),
                "first_update":first_update,"learning_log":logs,
                "ddim":metrics,"one_call":distribution_metrics(one_call,reference),
                "probe_initial":probe.tolist(),"probe_final":probe_final.tolist(),"probe_trace":trace,
                "seconds":time.perf_counter()-started}
        target=ROOT/"results"/f"learned_seed_{seed}"
        target.mkdir(parents=True,exist_ok=True)
        (target/"training.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        np.savez_compressed(target/"arrays.npz",**arrays)
        torch.save(model.state_dict(),target/"model.pt")
        all_results.append({"seed":seed,"validation_before":logs[0],"validation_after":logs[-1],
                            "ddim":metrics,"one_call":result["one_call"],"seconds":result["seconds"]})
        print(json.dumps(all_results[-1],ensure_ascii=False),flush=True)
    (ROOT/"results/learning_summary.json").write_text(json.dumps(all_results,ensure_ascii=False,indent=2),
                                                     encoding="utf-8")


if __name__=="__main__":
    main()
