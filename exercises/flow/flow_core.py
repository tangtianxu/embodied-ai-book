"""Analytic probability paths, posterior-averaged velocity and explicit integration.

The scalar data/evaluation protocol is shared with the preceding diffusion
lesson. This reference knows the data density; it is not a trained network.
"""
from pathlib import Path
from statistics import NormalDist
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
MODE_MEANS = np.array([-2.0, 2.0])
DATA_STD = 0.15


def mixture_parts(x, tau):
    x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    variance = (1-tau)**2 + DATA_STD**2 * tau**2
    covariance = DATA_STD**2*tau - (1-tau)
    residual = x[..., None] - tau[..., None]*MODE_MEANS
    logits = -0.5*residual**2/variance[..., None]
    weight = np.exp(logits-logits.max(axis=-1, keepdims=True))
    weight /= weight.sum(axis=-1, keepdims=True)
    return variance, covariance, residual, weight


def analytic_field(x, tau):
    """E[A-Z | X_tau=x] for independent Z~N(0,1), A~the two-mode mixture."""
    variance, covariance, residual, weight = mixture_parts(x, tau)
    per_mode = MODE_MEANS + (covariance/variance)[..., None]*residual
    return np.sum(weight*per_mode, axis=-1)


def path_density(x, tau):
    variance, _, residual, _ = mixture_parts(x, tau)
    return np.mean(np.exp(-0.5*residual**2/variance[..., None])
                   /np.sqrt(2*math.pi*variance[..., None]), axis=-1)


def path_cdf(x, tau):
    variance = (1-tau)**2 + DATA_STD**2*tau**2
    return sum(0.25*(1+math.erf((x-tau*mean)/math.sqrt(2*variance))) for mean in MODE_MEANS)


def interval_flux(tau, left=-0.5, right=0.5):
    # An independently evaluated CDF gives the change of mass in a fixed interval.
    h = 1e-5
    mass = lambda t: path_cdf(right, t)-path_cdf(left, t)
    derivative = (mass(tau+h)-mass(tau-h))/(2*h)
    influx_minus_outflux = (path_density(left, tau)*analytic_field(left, tau)
                           -path_density(right, tau)*analytic_field(right, tau))
    assert math.isclose(derivative, float(influx_minus_outflux), rel_tol=2e-7, abs_tol=1e-8)
    return {"tau":tau, "interval":[left, right], "mass":mass(tau),
            "cdf_time_derivative":derivative, "left_flux":float(path_density(left,tau)*analytic_field(left,tau)),
            "right_flux":float(path_density(right,tau)*analytic_field(right,tau)),
            "influx_minus_outflux":float(influx_minus_outflux)}


def integrate(initial, steps, field, method="euler", record=False):
    if steps < 1 or method not in {"euler", "heun"}:
        raise ValueError("Use a positive step count and euler/heun.")
    # G1: initialize candidates and a generation-time grid, without changing weights.
    x = np.array(initial, dtype=float, copy=True)
    dt = 1.0/steps
    snapshots = {0:x.copy()}
    trace = []
    for index in range(steps):
        tau = index/steps
        # G2: ask the field at the current candidate and time.
        velocity = np.asarray(field(x,tau), float)
        assert velocity.shape == x.shape
        predicted = x+dt*velocity
        if method == "heun":
            # G3: a second query at the predicted endpoint corrects the average slope.
            second = np.asarray(field(predicted,(index+1)/steps), float)
            following = x+0.5*dt*(velocity+second)
        else:
            second = None
            following = predicted
        if record:
            trace.append({"iteration":index+1, "tau":tau, "next_tau":(index+1)/steps,
                          "candidate":x.tolist(), "velocity":velocity.tolist(),
                          "euler_prediction":predicted.tolist(),
                          "second_velocity":None if second is None else second.tolist(),
                          "next_candidate":following.tolist()})
        x = following
        if index+1 in {steps//4, steps//2, 3*steps//4, steps}:
            snapshots[index+1] = x.copy()
    # G4: return the final generated action; intermediate candidates are not executed.
    return x, snapshots, trace


def gaussian_field(x, tau, mean=2.0, std=1.0):
    variance = (1-tau)**2+std**2*tau**2
    covariance = std**2*tau-(1-tau)
    return mean+covariance/variance*(np.asarray(x)-tau*mean)


def gaussian_exact(initial, tau, mean=2.0, std=1.0):
    variance = (1-tau)**2+std**2*tau**2
    return tau*mean+math.sqrt(variance)*np.asarray(initial)


def gaussian_reflow_field(x, tau, mean=2.0, std=1.0):
    scale = 1-tau+std*tau
    return mean+(std-1)/scale*(np.asarray(x)-tau*mean)


def equal_probability_noise(count=4096):
    normal = NormalDist()
    return np.array([normal.inv_cdf((i+0.5)/count) for i in range(count)])


def target_quantiles(count=4096):
    q = (np.arange(count)+0.5)/count
    lo, hi = np.full(count,-5.0), np.full(count,5.0)
    erf = np.vectorize(math.erf, otypes=[float])
    for _ in range(65):
        middle = (lo+hi)/2
        cdf = sum(0.25*(1+erf((middle-mean)/(DATA_STD*math.sqrt(2)))) for mean in MODE_MEANS)
        lo, hi = np.where(cdf<q,middle,lo), np.where(cdf<q,hi,middle)
    return (lo+hi)/2


def metrics(x, target):
    x = np.asarray(x).reshape(-1)
    left, right = x[x<0], x[x>=0]
    return {"left_share":float(np.mean(x<0)), "mean":float(x.mean()), "std":float(x.std()),
            "left_mean":float(left.mean()) if len(left) else None,
            "left_std":float(left.std()) if len(left) else None,
            "right_mean":float(right.mean()) if len(right) else None,
            "right_std":float(right.std()) if len(right) else None,
            "mean_absolute_quantile_error":float(np.mean(np.abs(np.sort(x)-target)))}


def main():
    initial, target = equal_probability_noise(), target_quantiles()
    arrays = {"initial":initial, "target_quantiles":target}
    reports = {}
    for method, step_counts in [("euler",[1,8,16,64,256]), ("heun",[8,32,128])]:
        for steps in step_counts:
            final, snapshots, _ = integrate(initial,steps,analytic_field,method)
            key = f"{method}_{steps}"
            arrays[key] = final
            reports[key] = {"steps":steps,"field_calls":steps*(2 if method=="heun" else 1),
                            **metrics(final,target)}
            if method=="euler" and steps==64:
                arrays.update({f"snapshot_{i}":v for i,v in snapshots.items()})
    _, _, short_trace = integrate(np.array([0.5]),8,analytic_field,record=True)
    probe = np.array([-1.5,-0.5,-0.1,0.0,0.1,0.5,1.5])
    probe_final, _, probe_trace = integrate(probe,64,analytic_field,record=True)
    assert probe_final[3] == 0
    gaussian_reports = {}
    for steps in [1,4,16,64,256]:
        generated,_,_ = integrate(initial,steps,gaussian_field)
        gaussian_reports[str(steps)] = float(np.sqrt(np.mean((generated-(2+initial))**2)))
    reflow,_,_ = integrate(initial,1,gaussian_reflow_field)
    np.testing.assert_allclose(reflow,2+initial,atol=1e-14)
    assert np.max(np.abs(arrays["euler_1"])) < 1e-12
    result = {"scope":"known-density analytic fields, not learned networks or robot motion",
              "numpy":np.__version__, "data":{"means":[-2,2],"std":DATA_STD,"condition":0},
              "reports":reports, "eight_step_trace":short_trace, "probe_initial":probe.tolist(),
              "probe_final":probe_final.tolist(), "probe_trace":probe_trace,
              "interval_flux":[interval_flux(t) for t in [0.1,0.25,0.5,0.75]],
              "gaussian_independent_pairing_rmse":gaussian_reports,
              "gaussian_reflow_one_step_max_error":float(np.max(np.abs(reflow-(2+initial))))}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/analytic_flow.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                                encoding="utf-8")
    np.savez_compressed(ROOT/"results/analytic_arrays.npz",**arrays)
    print(json.dumps({k:result[k] for k in ["reports","interval_flux","gaussian_independent_pairing_rmse",
                                          "gaussian_reflow_one_step_max_error"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
