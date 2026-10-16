"""An explicit spatial/temporal reading example, not a trained MEM encoder."""
from pathlib import Path
import json
import torch

torch.set_default_dtype(torch.float64)
OUT=Path(__file__).resolve().parent/"results"

def encode(frames):
    # M1: each frame has two scalar patch values; all spatial scores are zero.
    # softmax therefore averages the two patches in that frame.
    spatial=frames.mean(1,keepdim=True).expand_as(frames)
    # M2: same-position temporal queries read only current/past frames.
    n=len(frames)
    allowed=torch.ones(n,n,dtype=torch.bool).tril()
    scores=torch.zeros(n,n).masked_fill(~allowed,float("-inf"))
    weight=scores.softmax(-1)
    temporal=weight@spatial
    # M3: retain current-frame tokens AFTER they have read past values.
    return spatial,temporal,temporal[-1]

def main():
    frames=torch.tensor([[0.,2.],[4.,6.],[8.,10.]])
    spatial,temporal,current=encode(frames)
    assert torch.allclose(current,torch.tensor([5.,5.]))
    _,_,dropped_early=encode(frames[-1:])
    assert torch.allclose(dropped_early,torch.tensor([9.,9.]))
    changed=frames.clone();changed[0]+=6
    _,_,past_changed=encode(changed)
    assert torch.allclose(past_changed,torch.tensor([7.,7.]))
    # M4: appending a future frame cannot change earlier causal outputs.
    with_future=torch.cat([frames,torch.tensor([[100.,100.]])])
    _,future_encoded,_=encode(with_future)
    assert torch.equal(future_encoded[:3],temporal)
    # M5: clear an episode buffer; stale observations would change a new episode.
    new=torch.tensor([[20.,22.]])
    reset=encode(new)[2]
    stale=encode(torch.cat([frames[-2:],new]))[2]
    assert torch.allclose(reset,torch.tensor([21.,21.]))
    assert torch.allclose(stale,torch.tensor([35/3,35/3]))
    # A count-only summary merges two histories that require opposite actions.
    # Balanced histories, identical current view/summary => max expected success .5.
    blind_success=max(.5,.5)
    informative_success=1.
    result={"scope":"uniform scalar reading and finite-state example; no MEM weights",
            "frames":frames.tolist(),"spatial":spatial.tolist(),
            "temporal":temporal.tolist(),"current":current.tolist(),
            "drop_before_reading":dropped_early.tolist(),
            "past_changed":past_changed.tolist(),"causal_future_invariant":True,
            "new_episode_reset":reset.tolist(),"new_episode_stale":stale.tolist(),
            "count_only_memory_max_success":blind_success,
            "location_preserving_memory_max_success":informative_success}
    OUT.mkdir(exist_ok=True)
    (OUT/"memory_routes.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"current":current.tolist(),"early_drop":dropped_early.tolist(),
                     "reset":reset.tolist(),"stale":stale.tolist()}))

if __name__=="__main__": main()
