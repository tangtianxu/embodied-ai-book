"""Trace a residual input gradient, then update the branch parameters on one teaching sample."""
from pathlib import Path
import json
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
EPS=1e-5


def norm(x):
    return F.layer_norm(x,(x.numel(),),eps=EPS)


def norm_jacobian(x):
    centered=x-x.mean()
    width=x.numel()
    scale=torch.sqrt(centered.square().mean()+EPS)
    return (torch.eye(width)-torch.ones(width,width)/width
            -torch.outer(centered,centered)/(width*scale.square()))/scale


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    # G1: choose a token, a branch and an incoming gradient from a downstream loss.
    x=torch.tensor([1.,3.,5.],requires_grad=True)
    branch=torch.diag(torch.tensor([.2,-.1,.3]))
    incoming=torch.tensor([1.,-2.,.5])
    identity=torch.eye(3)
    # G2: compute the two different orders, keeping the same chosen branch matrix.
    pre=lambda z:z+branch@norm(z)
    post=lambda z:norm(z+branch@z)
    # G3: backpropagate the incoming output gradient through each graph.
    pre_grad=torch.autograd.grad((pre(x)*incoming).sum(),x)[0]
    post_grad=torch.autograd.grad((post(x)*incoming).sum(),x)[0]
    # G4: compare automatic derivatives with the explicit chain-rule formulas.
    jn=norm_jacobian(x.detach())
    jn_post=norm_jacobian((x+branch@x).detach())
    jp=torch.autograd.functional.jacobian(pre,x)
    jo=torch.autograd.functional.jacobian(post,x)
    torch.testing.assert_close(jn,torch.autograd.functional.jacobian(norm,x),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(jp,identity+branch@jn,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(jo,jn_post@(identity+branch),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(pre_grad,incoming+jn.T@branch.T@incoming)
    torch.testing.assert_close(post_grad,(identity+branch).T@jn_post.T@incoming)

    # A local task-gradient cancellation can occur even with Pre-LN.
    fixed_scale=float(torch.sqrt(x.detach().var(correction=0)+EPS))
    cancelling_pre=lambda z:z-fixed_scale*norm(z)
    cancellation_incoming=torch.tensor([1.,-2.,1.])
    cancellation_grad=torch.autograd.grad((cancelling_pre(x)*cancellation_incoming).sum(),x)[0]
    torch.testing.assert_close(cancellation_grad,torch.zeros_like(x),rtol=0,atol=1e-12)

    gradient_modes=[]
    for mode in ["train_branch","frozen_branch_parameters","branch_in_no_grad"]:
        probe_x=x.detach().clone().requires_grad_()
        probe_w=branch.clone().requires_grad_(mode!="frozen_branch_parameters")
        if mode=="branch_in_no_grad":
            with torch.no_grad():
                delta=probe_w@norm(probe_x)
        else:
            delta=probe_w@norm(probe_x)
        probe_y=probe_x+delta
        (probe_y*incoming).sum().backward()
        gradient_modes.append({"mode":mode,"input_gradient":probe_x.grad.tolist(),
                               "branch_parameter_gradient":None if probe_w.grad is None
                                   else probe_w.grad.tolist()})
    torch.testing.assert_close(torch.tensor(gradient_modes[0]["input_gradient"]),
                               torch.tensor(gradient_modes[1]["input_gradient"]))
    torch.testing.assert_close(torch.tensor(gradient_modes[2]["input_gradient"]),incoming)
    assert gradient_modes[0]["branch_parameter_gradient"] is not None
    assert all(r["branch_parameter_gradient"] is None for r in gradient_modes[1:])

    # T1: learned_branch is a full trainable W, initialized diagonally; input/target are fixed.
    token=x.detach()
    normalized=norm(token)
    learned_branch=torch.nn.Parameter(branch.clone())
    target=torch.zeros(3)
    learning_rate=.1
    updates=[]
    for step in range(4):
        prediction=token+learned_branch@normalized
        loss=.5*(prediction-target).square().sum()
        row={"step":step,"prediction":prediction.detach().tolist(),
             "loss":float(loss.detach()),"branch_matrix":learned_branch.detach().tolist()}
        updates.append(row)
        if step==3:
            break
        # T2: task error reaches W through the branch, not through an identity parameter.
        loss.backward()
        expected=torch.outer(prediction.detach()-target,normalized)
        torch.testing.assert_close(learned_branch.grad,expected,rtol=1e-12,atol=1e-12)
        row["branch_gradient"]=learned_branch.grad.tolist()
        # T3: change weights; input and normalization statistics remain fixed.
        with torch.no_grad():
            learned_branch.add_(learned_branch.grad,alpha=-learning_rate)
        learned_branch.grad=None
    assert all(updates[i+1]["loss"]<updates[i]["loss"] for i in range(3))

    result={"status":"passed","torch":torch.__version__,"epsilon":EPS,
            "scope":"single-token mathematical sublayer and three actual one-sample parameter updates; not a trained Transformer or VLA",
            "x":token.tolist(),"normalized":normalized.tolist(),"initial_branch":branch.tolist(),
            "incoming_gradient":incoming.tolist(),"norm_jacobian":jn.tolist(),
            "pre_output":pre(x).tolist(),"post_output":post(x).tolist(),
            "pre_jacobian":jp.tolist(),"post_jacobian":jo.tolist(),
            "pre_input_gradient":pre_grad.tolist(),"post_input_gradient":post_grad.tolist(),
            "pre_direct_contribution":incoming.tolist(),
            "pre_branch_contribution":(pre_grad-incoming).tolist(),
            "same_forward_different_gradient_modes":gradient_modes,
            "local_preLN_cancellation":{"fixed_branch_scale":fixed_scale,
                "incoming":cancellation_incoming.tolist(),"input_gradient":cancellation_grad.tolist(),
                "meaning":"cancellation of this incoming task-gradient direction at this token, not a zero full Jacobian"},
            "branch_training":{"target":target.tolist(),"loss":"0.5 * sum((prediction-target)^2)",
                               "learning_rate":learning_rate,"updates":updates}}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/gradient_case.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                                                encoding="utf-8")
    print(json.dumps({"status":result["status"],"pre_input_gradient":pre_grad.tolist(),
                      "post_input_gradient":post_grad.tolist(),"training_losses":[r["loss"] for r in updates]},
                     ensure_ascii=False))


if __name__=="__main__":
    main()
