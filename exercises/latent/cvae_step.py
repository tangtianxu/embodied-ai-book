"""One CVAE sample: forward values, split gradients, joint update and detach counterexample."""
from pathlib import Path
import json
import math
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DTYPE = torch.float64


class TinyCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Two input features [condition, demonstrated action] produce [mu, log variance].
        self.posterior = nn.Linear(2, 2, bias=False, dtype=DTYPE)
        self.decoder = nn.Linear(2, 1, dtype=DTYPE)
        with torch.no_grad():
            self.posterior.weight.copy_(torch.tensor([[.2, .4], [0., 0.]], dtype=DTYPE))
            self.decoder.weight.copy_(torch.tensor([[.5, .8]], dtype=DTYPE))
            self.decoder.bias.zero_()

    def forward(self, condition, action_label, epsilon, *, detach_z=False):
        # P1: infer the two posterior parameters from this labelled example.
        mu, logvar = self.posterior(torch.stack([condition, action_label])).unbind()
        # P2: external noise stays fixed in a gradient calculation; the distribution still varies.
        std = torch.exp(.5 * logvar)
        z = mu + std * epsilon
        # A counterexample: identical numerical input, but reconstruction cannot train the encoder.
        decoder_z = z.detach() if detach_z else z
        prediction = self.decoder(torch.stack([condition, decoder_z])).squeeze()
        # P3: this teaching Gaussian uses observation variance 1; omit its constant log term.
        reconstruction = .5 * (prediction - action_label).square()
        kl = .5 * (mu.square() + logvar.exp() - 1 - logvar)
        return dict(mu=mu, logvar=logvar, std=std, z=z, prediction=prediction,
                    reconstruction=reconstruction, kl=kl)


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values])


def summarize(values):
    return {name: float(value.detach()) for name, value in values.items()}


def run():
    model = TinyCVAE()
    condition, label, epsilon = [torch.tensor(x, dtype=DTYPE) for x in (1., 2., .5)]
    beta, learning_rate = .1, .1
    parameters = tuple(model.parameters())
    # Order: mu weights (2), log-variance weights (2), decoder weights (2), bias (1).
    parameter_names = ["mu_c", "mu_A", "logvar_c", "logvar_A", "decoder_c", "decoder_z", "bias"]
    initial_parameters = flatten(parameters).detach().clone()
    before = model(condition, label, epsilon)
    # P4: split the gradients before adding them, to show the two training signals.
    g_rec = flatten(torch.autograd.grad(before["reconstruction"], parameters,
                                      retain_graph=True, allow_unused=True,
                                      materialize_grads=True))
    g_kl = flatten(torch.autograd.grad(beta * before["kl"], parameters,
                                     retain_graph=True, allow_unused=True,
                                     materialize_grads=True))
    loss = before["reconstruction"] + beta * before["kl"]
    g_total = flatten(torch.autograd.grad(loss, parameters))
    analytic = torch.tensor([-.14, -.28, -.06, -.12, -.3, -.45, -.3], dtype=DTYPE)
    assert torch.allclose(g_total, analytic, atol=1e-12, rtol=0)
    assert torch.allclose(g_total, g_rec+g_kl, atol=1e-12, rtol=0)

    # A central finite difference independently checks the derivative of this fixed-noise objective.
    finite_difference = []
    step = 1e-5
    for parameter in parameters:
        for index in range(parameter.numel()):
            view = parameter.reshape(-1)
            old = float(view[index].detach())
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    view[index] = old + sign*step
                value = model(condition, label, epsilon)
                values.append(float((value["reconstruction"]+beta*value["kl"]).detach()))
            with torch.no_grad():
                view[index] = old
            finite_difference.append((values[0]-values[1])/(2*step))
    finite_difference = torch.tensor(finite_difference, dtype=DTYPE)
    assert torch.allclose(g_total, finite_difference, atol=1e-8, rtol=0)

    detached = model(condition, label, epsilon, detach_z=True)
    detached_loss = detached["reconstruction"]+beta*detached["kl"]
    detached_gradient = flatten(torch.autograd.grad(detached_loss, parameters))
    assert torch.equal(loss, detached_loss)
    expected_detached = torch.tensor([.1, .2, 0., 0., -.3, -.45, -.3], dtype=DTYPE)
    assert torch.allclose(detached_gradient, expected_detached, atol=1e-12, rtol=0)

    wrong_model = TinyCVAE()
    wrong_optimizer = torch.optim.SGD(wrong_model.parameters(), lr=learning_rate)
    wrong_before = wrong_model(condition, label, epsilon, detach_z=True)
    wrong_optimizer.zero_grad()
    (wrong_before["reconstruction"]+beta*wrong_before["kl"]).backward()
    wrong_optimizer.step()
    wrong_after = wrong_model(condition, label, epsilon, detach_z=True)
    wrong_after_loss = float((wrong_after["reconstruction"]+beta*wrong_after["kl"]).detach())
    assert math.isclose(wrong_after_loss, .06818378125, abs_tol=1e-12)

    # P4: calculate the joint gradient on a fresh graph.
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    optimizer.zero_grad()
    fresh = model(condition, label, epsilon)
    (fresh["reconstruction"]+beta*fresh["kl"]).backward()
    # P5: update both networks simultaneously using the chosen optimizer.
    optimizer.step()
    after = model(condition, label, epsilon)
    new_parameters = flatten(parameters).detach()
    assert torch.allclose(new_parameters, initial_parameters-learning_rate*g_total,
                          atol=1e-12, rtol=0)
    # A smaller loss here is a local observation with the same noise, not a dataset guarantee.
    def pushed_density(y):
        latent = math.atanh(y/2)/5
        standard_normal_density = math.exp(-.5*latent**2)/math.sqrt(2*math.pi)
        return standard_normal_density/(10*(1-(y/2)**2))
    assert pushed_density(1.9) > pushed_density(0.)
    result = {
        "scope": "scalar teaching CVAE; one labelled sample, one parameter update; not ACT training",
        "torch": torch.__version__, "beta": beta, "learning_rate": learning_rate,
        "condition": 1., "label": 2., "epsilon": .5, "before": summarize(before),
        "loss_before": float(loss.detach()), "after": summarize(after),
        "loss_after_same_noise": float((after["reconstruction"]+beta*after["kl"]).detach()),
        "loss_after_detached_update_same_noise": wrong_after_loss,
        "gradients": {name: {"reconstruction": float(g_rec[i]), "weighted_kl": float(g_kl[i]),
                             "total": float(g_total[i]), "if_z_detached": float(detached_gradient[i]),
                             "finite_difference": float(finite_difference[i])}
                      for i,name in enumerate(parameter_names)},
        "updated_parameters": dict(zip(parameter_names,new_parameters.tolist())),
        "finite_difference_max_error": float((g_total-finite_difference).abs().max()),
        "nonlinear_prior_probe": {
            "definition": "f(z)=2*tanh(5*z), constructed example, not a trained model",
            "z": [-1.,0.,1.], "action": [2*math.tanh(5*z) for z in (-1.,0.,1.)],
            "output_density_y_0": pushed_density(0.),
            "output_density_y_1_9": pushed_density(1.9),
            "squared_decoder_at_prior_mean": 0., "squared_decoder_mean_output": 1.},
    }
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/cvae_step.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["loss_before","loss_after_same_noise",
                                         "finite_difference_max_error"]},ensure_ascii=False))


if __name__ == "__main__":
    run()
