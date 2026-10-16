"""Extend the legacy 3->48->48->1 learner to two modes under the SAME condition."""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch import nn
from flow_core import (ROOT, DATA_STD, analytic_field, integrate, equal_probability_noise,
                       target_quantiles, metrics)


class VelocityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3,48), nn.SiLU(), nn.Linear(48,48),
                                 nn.SiLU(), nn.Linear(48,1))

    def forward(self,candidate,tau,condition):
        return self.net(torch.cat([candidate,tau,condition],dim=1))


def batch(generator,count):
    # P1: the same visible condition admits two clean action modes.
    route = (2*torch.randint(0,2,(count,1),generator=generator)-1).float()
    clean = 2*route+DATA_STD*torch.randn((count,1),generator=generator)
    condition = torch.zeros((count,1))
    # P2: sample source noise and generation time; construct a known path label.
    noise = torch.randn((count,1),generator=generator)
    tau = torch.rand((count,1),generator=generator)
    candidate = (1-tau)*noise+tau*clean
    velocity_label = clean-noise
    return clean,noise,tau,condition,candidate,velocity_label


@torch.no_grad()
def evaluate(model,validation):
    _,_,tau,condition,candidate,label = validation
    predicted = model(candidate,tau,condition)
    reference = torch.from_numpy(analytic_field(candidate.numpy(),tau.numpy()).astype(np.float32))
    return {"velocity_label_mse":float((predicted-label).square().mean()),
            "analytic_label_mse":float((reference-label).square().mean()),
            "mse_to_analytic_velocity":float((predicted-reference).square().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[13,17,29])
    parser.add_argument("--updates",type=int,default=8000)
    args = parser.parse_args()
    assert args.updates > 0
    torch.set_num_threads(1)
    initial,target = equal_probability_noise(),target_quantiles()
    validation = batch(torch.Generator().manual_seed(90901),2048)
    summary = []
    for seed in args.seeds:
        started = time.perf_counter()
        torch.manual_seed(seed)
        model = VelocityMLP()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
        generator = torch.Generator().manual_seed(seed+100000)
        logs = [{"update":0,**evaluate(model,validation)}]
        first = None
        for update in range(1,args.updates+1):
            clean,noise,tau,condition,candidate,label = batch(generator,256)
            # P3: only current position, generation time and available condition enter the model.
            prediction = model(candidate,tau,condition)
            # P4: supervise with the sampled endpoint difference; no analytic teacher or ODE unroll.
            loss = (prediction-label).square().mean()
            optimizer.zero_grad()
            loss.backward()
            if update == 1:
                previous = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
                first = {"source_noise":float(noise[0]),"clean_action":float(clean[0]),
                         "tau":float(tau[0]),"candidate":float(candidate[0]),
                         "velocity_label":float(label[0]),"prediction_before":float(prediction[0].detach()),
                         "output_gradient_for_row":float(2*(prediction[0]-label[0]).detach()/256),
                         "batch_loss":float(loss.detach()),
                         "parameter_gradient_l2":math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))}
            # P5: update parameters, then draw a fresh batch.
            optimizer.step()
            if update == 1:
                now = torch.cat([p.detach().flatten() for p in model.parameters()])
                first["parameter_change_l2"] = float((now-previous).norm())
            if update%500 == 0 or update == args.updates:
                logs.append({"update":update,**evaluate(model,validation)})
        model.eval()

        @torch.no_grad()
        def field(x,tau_value):
            candidate = torch.from_numpy(np.asarray(x,np.float32).reshape(-1,1))
            tau = torch.full_like(candidate,tau_value)
            return model(candidate,tau,torch.zeros_like(candidate)).numpy().reshape(np.asarray(x).shape)

        reports,arrays = {},{}
        for method,counts in [("euler",[1,16,64,256]),("heun",[32])]:
            for steps in counts:
                generated,snapshots,_ = integrate(initial,steps,field,method)
                key = f"{method}_{steps}"
                arrays[key] = generated
                reports[key] = {"steps":steps,"field_calls":steps*(2 if method=="heun" else 1),
                                **metrics(generated,target)}
                if method == "euler" and steps == 64:
                    arrays.update({f"snapshot_{i}":value for i,value in snapshots.items()})
        probe = np.array([-1.5,-0.5,-0.1,0.,0.1,0.5,1.5])
        final,_,trace = integrate(probe,64,field,record=True)
        result = {"seed":seed,"updates":args.updates,"batch_size":256,"learning_rate":0.002,
                  "parameter_count":sum(p.numel() for p in model.parameters()),
                  "torch":torch.__version__,"validation_seed":90901,
                  "validation_used_for_gradient_or_selection":False,
                  "analytic_reference_used_for_training":False,
                  "scope":"synthetic scalar two-mode data, one fixed condition; no robot execution",
                  "first_update":first,"learning_log":logs,"generation":reports,
                  "probe_initial":probe.tolist(),"probe_final":final.tolist(),"probe_trace":trace,
                  "seconds":time.perf_counter()-started}
        directory = ROOT/"results"/f"learned_seed_{seed}"
        directory.mkdir(parents=True,exist_ok=True)
        (directory/"training.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        np.savez_compressed(directory/"arrays.npz",**arrays)
        torch.save(model.state_dict(),directory/"model.pt")
        summary.append({"seed":seed,"validation_before":logs[0],"validation_after":logs[-1],
                        "generation":reports,"seconds":result["seconds"]})
        print(json.dumps(summary[-1],ensure_ascii=False),flush=True)
    (ROOT/"results/learning_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
