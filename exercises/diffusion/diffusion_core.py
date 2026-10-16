"""Known forward process, analytic two-mode reference, and deterministic DDIM.

The analytic mixture and DDIM update refactor the book's existing
unified_practice/foundations/generation_paths.py; traces, common evaluation
quantiles and a shared interface for trained denoisers are added here.
"""
from pathlib import Path
from statistics import NormalDist
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent
TRAIN_LEVELS=1000
THETA_MAX=math.acos(.01)
DATA_STD=.15
MODE_MEANS=np.array([-2.,2.])


def alpha_bar(index):
    """Index 0 is clean data; index 1000 retains signal amplitude .01."""
    return math.cos(THETA_MAX*index/TRAIN_LEVELS)**2


def analytic_epsilon(x,index):
    """Exact E[epsilon | x_k=x] for a known Gaussian mixture, not a trained network."""
    original_shape=np.asarray(x).shape
    x=np.asarray(x,dtype=float).reshape(-1)
    keep=alpha_bar(index)
    noise_std=math.sqrt(1-keep)
    variance=keep*DATA_STD**2+1-keep
    residual=x[:,None]-math.sqrt(keep)*MODE_MEANS
    logits=-.5*residual**2/variance
    weights=np.exp(logits-logits.max(axis=1,keepdims=True))
    weights/=weights.sum(axis=1,keepdims=True)
    score=(-residual/variance*weights).sum(axis=1)
    return (-noise_std*score).reshape(original_shape)


def estimate_clean(x,epsilon,index):
    keep=alpha_bar(index)
    return (x-math.sqrt(1-keep)*epsilon)/math.sqrt(keep)


def ddim(initial,steps,predict_epsilon,record=False):
    if not 1<=steps<=TRAIN_LEVELS:
        raise ValueError("Use 1..1000 distinct reverse steps.")
    # D1: keep an actual subsequence of the 1000 training noise levels.
    indices=np.rint(np.linspace(TRAIN_LEVELS,0,steps+1)).astype(int)
    assert np.all(np.diff(indices)<0)
    x=np.array(initial,dtype=float,copy=True)
    snapshots={0:x.copy()}
    records=[]
    for iteration,(current,following) in enumerate(zip(indices[:-1],indices[1:]),1):
        # D2: prediction uses only the current candidate and its noise level.
        epsilon=np.asarray(predict_epsilon(x,int(current)),dtype=float)
        if epsilon.shape!=x.shape:
            raise ValueError("The predicted noise must have the candidate's shape.")
        # D3: estimate the clean action, but do not execute that intermediate mean.
        clean=estimate_clean(x,epsilon,int(current))
        next_keep=alpha_bar(int(following))
        # D4: eta=0 DDIM carries the predicted residual to the next noise scale.
        next_x=math.sqrt(next_keep)*clean+math.sqrt(1-next_keep)*epsilon
        if record:
            records.append({"iteration":iteration,"current_index":int(current),
                            "next_index":int(following),"alpha_bar":alpha_bar(int(current)),
                            "next_alpha_bar":next_keep,"candidate":x.tolist(),
                            "epsilon_prediction":epsilon.tolist(),"clean_estimate":clean.tolist(),
                            "next_candidate":next_x.tolist()})
        x=next_x
        if iteration in {steps//4,steps//2,3*steps//4,steps}:
            snapshots[iteration]=x.copy()
    # D5: only the final x is returned as a generated action.
    return x,snapshots,records


def equal_probability_noise(count=4096):
    """Deterministic Gaussian quantiles remove random-count fluctuations from sampler comparisons."""
    q=(np.arange(count)+.5)/count
    normal=NormalDist()
    return np.array([normal.inv_cdf(float(value)) for value in q])


def target_quantiles(count=4096):
    """Numerically invert the exact two-Gaussian CDF, including its negligible overlap."""
    q=(np.arange(count)+.5)/count
    lo=np.full(count,-5.)
    hi=np.full(count,5.)
    erf=np.vectorize(math.erf,otypes=[float])
    for _ in range(65):
        midpoint=(lo+hi)/2
        cdf=sum(.25*(1+erf((midpoint-mean)/(DATA_STD*math.sqrt(2))))
                for mean in MODE_MEANS)
        go_right=cdf<q
        lo=np.where(go_right,midpoint,lo)
        hi=np.where(go_right,hi,midpoint)
    return (lo+hi)/2


def distribution_metrics(samples,reference):
    samples=np.asarray(samples).reshape(-1)
    left=samples[samples<0]
    right=samples[samples>=0]
    return {"left_share":float(np.mean(samples<0)),
            "abs_over_one_share":float(np.mean(np.abs(samples)>1)),
            "mean":float(samples.mean()),"std":float(samples.std()),
            "left_mean":float(left.mean()) if len(left) else None,
            "left_std":float(left.std()) if len(left) else None,
            "right_mean":float(right.mean()) if len(right) else None,
            "right_std":float(right.std()) if len(right) else None,
            "mean_absolute_quantile_error":float(np.mean(np.abs(np.sort(samples)-reference)))}


def posterior_check():
    # Expand the two Gaussian factors in the unknown previous state.
    previous_keep,alpha,beta=.64,.8,.2
    keep=alpha*previous_keep
    clean,epsilon=2.,.5
    current=math.sqrt(keep)*clean+math.sqrt(1-keep)*epsilon
    precision=alpha/beta+1/(1-previous_keep)
    posterior_variance=1/precision
    linear=math.sqrt(alpha)*current/beta+math.sqrt(previous_keep)*clean/(1-previous_keep)
    mean_from_completion=linear/precision
    mean_from_noise=(current-beta/math.sqrt(1-keep)*epsilon)/math.sqrt(alpha)
    assert math.isclose(posterior_variance,beta*(1-previous_keep)/(1-keep))
    assert math.isclose(mean_from_completion,mean_from_noise,abs_tol=1e-12)
    epsilon_hat=.3
    predicted_mean=(current-beta/math.sqrt(1-keep)*epsilon_hat)/math.sqrt(alpha)
    # Here p and q have identical variance; the KL has only the mean error term.
    kl=(mean_from_completion-predicted_mean)**2/(2*posterior_variance)
    weight=beta**2/(2*posterior_variance*alpha*(1-keep))
    weighted_mse=weight*(epsilon-epsilon_hat)**2
    assert math.isclose(kl,weighted_mse,abs_tol=1e-12)
    return dict(previous_alpha_bar=previous_keep,alpha=alpha,beta=beta,alpha_bar=keep,
                x0=clean,epsilon=epsilon,x_k=current,precision=precision,
                posterior_variance=posterior_variance,
                posterior_mean=mean_from_completion,mean_noise_form=mean_from_noise,
                epsilon_prediction=epsilon_hat,model_mean=predicted_mean,
                kl=kl,mse_weight=weight,weighted_noise_mse=weighted_mse)


def run():
    initial=equal_probability_noise()
    reference=target_quantiles()
    reports={}
    arrays={"initial":initial,"target_quantiles":reference}
    for steps in [16,64,256]:
        final,snapshots,_=ddim(initial,steps,analytic_epsilon)
        reports[str(steps)]=distribution_metrics(final,reference)
        arrays[f"oracle_{steps}"]=final
        if steps==64:
            arrays.update({f"oracle64_snapshot_{k}":v for k,v in snapshots.items()})
    one_call=estimate_clean(initial,analytic_epsilon(initial,1000),1000)
    arrays["one_call_clean_estimate"]=one_call
    probe=np.array([-1.5,-.5,-.1,0.,.1,.5,1.5])
    final,_,trace=ddim(probe,64,analytic_epsilon,record=True)
    short_final,_,short_trace=ddim(np.array([.5]),8,analytic_epsilon,record=True)
    # The exactly symmetric zero is a stationary point in this deterministic ideal example.
    assert final[3]==0
    assert reports["256"]["mean_absolute_quantile_error"]<reports["16"]["mean_absolute_quantile_error"]
    result={"scope":"known-density analytic reference; no denoiser training or robot execution",
            "numpy":np.__version__,"data_means":MODE_MEANS.tolist(),"data_std":DATA_STD,
            "training_noise_levels":TRAIN_LEVELS,"terminal_alpha_bar":alpha_bar(1000),
            "initialization":"4096 deterministic standard-normal quantiles; not exact finite-level q_K",
            "comparison_initials_shared":True,"posterior_example":posterior_check(),
            "oracle_ddim":reports,"one_call":distribution_metrics(one_call,reference),
            "probe_initial":probe.tolist(),"probe_final":final.tolist(),"probe_trace":trace,
            "eight_step_initial":.5,"eight_step_final":float(short_final[0]),
            "eight_step_trace":short_trace}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/analytic_reference.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                                       encoding="utf-8")
    np.savez_compressed(ROOT/"results/analytic_arrays.npz",**arrays)
    print(json.dumps({"posterior_example":result["posterior_example"],
                      "oracle_ddim":reports,"one_call":result["one_call"]},ensure_ascii=False))


if __name__=="__main__":
    run()
