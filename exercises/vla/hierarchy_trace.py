"""Trace a small hierarchical distribution and two supervised targets.

The trainable objects are a two-class logit vector and a conditional table of
flow outputs. They isolate label roles, not the architecture of a pretrained VLA.
"""
from pathlib import Path
import json
import torch

torch.set_default_dtype(torch.float64)
OUT=Path(__file__).resolve().parent/"results"

def main():
    # H1: sum over latent/subtask choices; compare joint MAP with greedy decoding.
    high=torch.tensor([.55,.45])
    low=torch.tensor([[.5,.5,0.],[0.,0.,1.]])
    joint=high[:,None]*low
    marginal=joint.sum(0)
    assert torch.allclose(marginal,torch.tensor([.275,.275,.45]))
    assert joint.argmax().item()==5  # row1, col2: the non-greedy subtask
    # H2: training supplies a subtask label and future demonstrated actions.
    action=torch.tensor([[.2,.4],[.4,.6]])
    noise=torch.tensor([[-.2,.2],[.6,-.4]])
    tau=.25
    noisy=(1-tau)*noise+tau*action
    target=action-noise
    logits=torch.nn.Parameter(torch.zeros(2))
    flow_table=torch.nn.Parameter(torch.zeros(2,2,2))
    optimizer=torch.optim.SGD([logits,flow_table],lr=.2)
    trace=[]
    # H3: teacher subtask0 selects the conditional table during training.
    # CE moves high-level logits; FM moves only table0 in this simplified model.
    for step in range(4):
        probability=logits.softmax(0)
        ce=-probability[0].log()
        prediction=flow_table[0]
        fm=.5*(prediction-target).square().mean()
        trace.append({"updates":step,"probability":probability.detach().tolist(),
                      "flow_prediction":prediction.detach().tolist(),
                      "ce":ce.item(),"fm":fm.item()})
        if step==3: break
        optimizer.zero_grad()
        (ce+fm).backward()
        assert torch.equal(flow_table.grad[1],torch.zeros(2,2))
        optimizer.step()
    assert trace[-1]["ce"]<trace[0]["ce"]
    assert trace[-1]["fm"]<trace[0]["fm"]
    # H4: the label is not an inference input; generate from a NEW noise sample.
    chosen=logits.argmax().item()
    candidate=torch.tensor([[.1,-.3],[-.2,.2]])
    inference=[]
    for internal_step in range(4):
        # A constant velocity table is deliberately too small to fit the real field.
        # Its integral demonstrates information availability, not a good policy.
        candidate=candidate+.25*flow_table[chosen].detach()
        inference.append(candidate.tolist())
    # Even a zero training error table fails for other noise samples:
    # omega_new + (A - omega_train) is not generally A.
    ideal_table_endpoint=torch.tensor([[.1,-.3],[-.2,.2]])+target
    assert not torch.allclose(ideal_table_endpoint,action)
    result={"scope":"synthetic labels and lookup-table SGD; no images/VLA/robot",
            "high":high.tolist(),"conditional_actions":low.tolist(),
            "joint":joint.tolist(),"action_marginal":marginal.tolist(),
            "greedy_subtask":high.argmax().item(),"joint_map_subtask":1,
            "noisy":noisy.tolist(),"target":target.tolist(),
            "trace":trace,"new_noise_inference":inference,
            "inference_has_demonstration_action":False,
            "zero_training_loss_table_new_noise_endpoint":ideal_table_endpoint.tolist(),
            "demonstration_action":action.tolist()}
    OUT.mkdir(exist_ok=True)
    (OUT/"hierarchy_trace.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"marginal":marginal.tolist(),"initial_CE":trace[0]["ce"],
                      "final_CE":trace[-1]["ce"],"initial_FM":trace[0]["fm"],
                      "final_FM":trace[-1]["fm"]}))

if __name__=="__main__": main()
