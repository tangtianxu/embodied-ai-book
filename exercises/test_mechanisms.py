"""Run: python -m unittest discover -s . -p test_mechanisms.py -v
Each test checks a mechanism property or a known failure boundary.
"""
import json
import platform
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from attention_numpy import (masked_softmax, attention, attention_backward,
                             layer_norm, rms_norm, rope)
from attention_torch import HandMHA, PreNormBlock, LoRALinear
from generative_actions import masked_mse, ddpm_pair, clean_from_noise, flow_pair, euler_sample
from rl_estimators import gae, ppo_surrogate, expectile_loss, squash_log_prob, polyak
from robotics_algorithms import (inverse_transform, fk2, jacobian2, dls_ik,
                                 ransac_line, segment_free, rrt)
from systems_debug import causal_indices, ddp_loss_scaling, ReplayBuffer
from efficient_attention import online_attention, quantize, integer_dot, kv_bytes

torch.set_num_threads(1)
METRICS = {}


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        torch.manual_seed(2)

    def test_softmax_extreme_logits(self):
        p = masked_softmax([[1000.,1001.]])
        np.testing.assert_allclose(p, [[.26894142137,.73105857863]])

    def test_softmax_all_masked_row(self):
        p = masked_softmax([[1,2],[3,4]], [[False,False],[True,False]])
        np.testing.assert_array_equal(p, [[0,0],[1,0]])

    def test_attention_hand_calculation(self):
        q,k,v = np.array([[1.,0.]]),np.eye(2),np.array([[2.,0.],[0.,4.]])
        out,p = attention(q,k,v)
        expected = np.exp(1/np.sqrt(2))/(1+np.exp(1/np.sqrt(2)))
        np.testing.assert_allclose(out, [[2*expected,4*(1-expected)]])
        METRICS['attention_example_output'] = out.tolist()

    def test_manual_backward_finite_difference(self):
        q,k,v,g = [self.rng.normal(size=s) for s in [(2,3),(4,3),(4,2),(2,2)]]
        mask = np.array([[True,True,False,False],[True,True,True,False]])
        _,p = attention(q,k,v,mask)
        analytic = attention_backward(q,k,v,p,g)
        errors = []
        for variable,grad in zip((q,k,v),analytic):
            numerical = np.zeros_like(variable)
            for index in np.ndindex(variable.shape):
                original = variable[index]
                variable[index] = original+1e-6
                positive = (attention(q,k,v,mask)[0]*g).sum()
                variable[index] = original-1e-6
                negative = (attention(q,k,v,mask)[0]*g).sum()
                variable[index] = original
                numerical[index] = (positive-negative)/2e-6
            errors.append(float(np.abs(numerical-grad).max()))
            np.testing.assert_allclose(grad,numerical,atol=1e-8)
        METRICS['attention_gradient_max_errors'] = errors

    def test_mha_matches_official_sdpa(self):
        net = HandMHA(8,2).double().eval()
        x = torch.randn(2,5,8,dtype=torch.float64)
        out,_,_ = net(x,causal=True)
        q,k,v = [net.split_heads(layer(x)) for layer in (net.q,net.k,net.v)]
        core = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        expected = net.out(core.transpose(1,2).contiguous().reshape(2,5,8))
        torch.testing.assert_close(out,expected,atol=1e-12,rtol=1e-12)
        METRICS['sdpa_max_error'] = float((out-expected).abs().max().detach())

    def test_future_perturbation_cannot_leak(self):
        net = HandMHA(8,2).double().eval()
        x = torch.randn(1,6,8,dtype=torch.float64)
        changed = x.clone();changed[:,3:] += 100
        a,b = net(x,causal=True)[0],net(changed,causal=True)[0]
        torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-12,rtol=1e-12)
        self.assertGreater(float((a[:,3:]-b[:,3:]).abs().max().detach()),1.)

    def test_padding_perturbation_cannot_leak(self):
        net = HandMHA(8,2).double().eval()
        x = torch.randn(1,5,8,dtype=torch.float64)
        valid = torch.tensor([[True,True,True,False,False]])
        changed = x.clone();changed[:,3:] += 1000
        torch.testing.assert_close(net(x,key_valid=valid)[0][:,:3],
                                   net(changed,key_valid=valid)[0][:,:3])

    def test_all_masked_output_and_gradient_finite(self):
        net = HandMHA(8,2).double()
        x = torch.randn(1,3,8,dtype=torch.float64,requires_grad=True)
        out,p,_ = net(x,key_valid=torch.zeros(1,3,dtype=torch.bool))
        self.assertEqual(float(out.abs().sum().detach()),0.)
        self.assertEqual(float(p.abs().sum().detach()),0.)
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cross_attention_different_lengths(self):
        net = HandMHA(12,3)
        out,p,_ = net(torch.randn(2,3,12),torch.randn(2,7,12))
        self.assertEqual(tuple(out.shape),(2,3,12))
        self.assertEqual(tuple(p.shape),(2,3,3,7))

    def test_cache_matches_full_causal_forward(self):
        net = HandMHA(8,2).double().eval()
        x = torch.randn(1,7,8,dtype=torch.float64)
        with torch.no_grad():
            full = net(x,causal=True)[0]
            prefix,_,cache = net(x[:,:4],causal=True,use_cache=True)
            suffix,_,_ = net(x[:,4:],causal=True,cache=cache,use_cache=True)
        joined = torch.cat([prefix,suffix],dim=1)
        torch.testing.assert_close(full,joined,atol=1e-12,rtol=1e-12)
        METRICS['cache_max_error'] = float((full-joined).abs().max())

    def test_invalid_head_divisibility(self):
        with self.assertRaises(ValueError):HandMHA(7,2)

    def test_cache_valid_mask_length_checked(self):
        net = HandMHA(8,2)
        _,_,cache = net(torch.randn(1,3,8),use_cache=True)
        with self.assertRaises(ValueError):
            net(torch.randn(1,1,8),cache=cache,key_valid=torch.ones(1,1,dtype=torch.bool))

    def test_transformer_block_gradcheck(self):
        net = PreNormBlock(4,2,2).double()
        x = torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:net(z,causal=True),(x,),
                                                eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_eval_disables_dropout(self):
        net = HandMHA(8,2,.5).eval();x = torch.randn(1,4,8)
        torch.testing.assert_close(net(x)[0],net(x)[0])

    def test_lora_freezes_weights_but_not_input_gradient(self):
        net = LoRALinear(torch.nn.Linear(4,3,bias=False))
        x = torch.randn(2,4,requires_grad=True)
        net(x).square().sum().backward()
        self.assertIsNone(net.base.weight.grad)
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertGreater(float(net.b.grad.abs().sum()),0.)
        self.assertEqual(float(net.a.grad.abs().sum()),0.)

    def test_layer_norm_matches_framework(self):
        x = self.rng.normal(size=(2,3,4))
        np.testing.assert_allclose(layer_norm(x),F.layer_norm(torch.tensor(x),(4,)).numpy(),atol=1e-12)

    def test_rms_norm_does_not_remove_mean(self):
        x = np.array([[2.,2.]])
        np.testing.assert_allclose(layer_norm(x),0.)
        self.assertGreater(rms_norm(x).mean(),.99)

    def test_rope_preserves_norm_and_relative_dot(self):
        q,k = self.rng.normal(size=(1,2,4,6)),self.rng.normal(size=(1,2,4,6))
        a,b = rope(q,np.arange(4)),rope(k,np.arange(4))
        np.testing.assert_allclose(np.linalg.norm(a,axis=-1),np.linalg.norm(q,axis=-1))
        shifted_a,shifted_b = rope(q,np.arange(4)+10),rope(k,np.arange(4)+10)
        np.testing.assert_allclose(a@b.swapaxes(-1,-2),shifted_a@shifted_b.swapaxes(-1,-2),atol=1e-12)


class LearningTests(unittest.TestCase):
    def test_masked_loss_ignores_invalid_nan(self):
        pred = np.array([[[1.,2.],[np.nan,np.nan]]])
        target = np.array([[[0.,0.],[np.nan,np.nan]]])
        loss,grad = masked_mse(pred,target,[[[True],[False]]])
        self.assertEqual(loss,2.5);np.testing.assert_array_equal(grad,[[[1,2],[0,0]]])

    def test_empty_supervision_is_explicit(self):
        with self.assertRaises(ValueError):masked_mse(np.zeros(2),np.zeros(2),[False,False])

    def test_ddpm_recovers_clean_from_known_noise(self):
        x,noise = np.array([2.,-3.]),np.array([.5,-.7])
        noisy,eps = ddpm_pair(x,noise,.64)
        np.testing.assert_allclose(clean_from_noise(noisy,eps,.64),x)
        METRICS['ddpm_noisy_example'] = noisy.tolist()

    def test_flow_endpoints_and_direction(self):
        x,z = np.array([2.]),np.array([-1.])
        np.testing.assert_array_equal(flow_pair(x,z,0)[0],z)
        np.testing.assert_array_equal(flow_pair(x,z,1)[0],x)
        self.assertEqual(flow_pair(x,z,.25)[1][0],3.)

    def test_euler_constant_field(self):
        np.testing.assert_allclose(euler_sample(lambda x,t,c:np.full_like(x,3.),[-1.],None,7),[2.])

    def test_gae_hand_values(self):
        adv,target = gae([0,0,10],[2,4,6],[4,6,0],[False,False,True],[False]*3)
        np.testing.assert_allclose(adv,[4.6816,4.28,4.])
        np.testing.assert_allclose(target,[6.6816,8.28,10.])
        METRICS['gae_example'] = adv.tolist()

    def test_gae_truncation_bootstraps_but_stops_trace(self):
        adv,_ = gae([1,100],[2,0],[5,0],[False,True],[True,False])
        self.assertEqual(adv[0],3.5)

    def test_gae_terminal_removes_bootstrap(self):
        adv,_ = gae([1],[2],[999],[True],[False])
        self.assertEqual(adv[0],-1.)

    def test_gae_parallel_env_shape(self):
        a,_ = gae(np.ones((3,2)),np.zeros((3,2)),np.zeros((3,2)),
                  np.zeros((3,2),bool),np.zeros((3,2),bool))
        self.assertEqual(a.shape,(3,2));np.testing.assert_allclose(a[:,0],a[:,1])

    def test_ppo_four_gradient_cases(self):
        ratios,adv = np.array([1.5,.5,1.5,.5]),np.array([2.,2.,-2.,-2.])
        loss,grad = ppo_surrogate(np.log(ratios),np.zeros(4),adv)
        np.testing.assert_allclose(grad,[0,-.25,.75,0])
        self.assertAlmostEqual(loss,.3)
        METRICS['ppo_four_case_loss_and_gradient'] = [loss,grad.tolist()]

    def test_ppo_gradient_finite_difference(self):
        ln,lo,adv = np.log([1.3,.6,1.1,.9]),np.zeros(4),np.array([1.,1.,-1.,-1.])
        _,g = ppo_surrogate(ln,lo,adv)
        for i in range(4):
            positive,negative = ln.copy(),ln.copy()
            positive[i] += 1e-6;negative[i] -= 1e-6
            estimated = (ppo_surrogate(positive,lo,adv)[0]-ppo_surrogate(negative,lo,adv)[0])/2e-6
            self.assertAlmostEqual(g[i],estimated,places=7)

    def test_expectile_weights_and_gradient(self):
        loss,grad = expectile_loss([0.,0.],[2.,-2.],.7)
        self.assertEqual(loss,2.);np.testing.assert_allclose(grad,[-1.4,.6])

    def test_tanh_log_prob_large_input_is_finite(self):
        out = squash_log_prob(np.array([[30.,-30.]]),0.,0.)
        self.assertTrue(np.isfinite(out).all())

    def test_tanh_density_matches_framework(self):
        raw = np.array([[.2,-.4]])
        actual = squash_log_prob(raw,0.,0.)
        dist = torch.distributions.Normal(torch.zeros(2,dtype=torch.float64),torch.ones(2,dtype=torch.float64))
        z = torch.tensor(raw)
        expected = (dist.log_prob(z)-torch.log1p(-torch.tanh(z).square())).sum(-1).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_polyak_direction(self):
        np.testing.assert_allclose(polyak([10.],[2.],.1),[9.2])


class RoboticsSystemsTests(unittest.TestCase):
    def test_rigid_transform_inverse(self):
        t = np.eye(4);t[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];t[:3,3]=[1,2,3]
        np.testing.assert_allclose(inverse_transform(t)@t,np.eye(4))

    def test_fk_jacobian_finite_difference(self):
        q = np.array([.4,-.7]);numerical = np.zeros((2,2))
        for i in range(2):
            d = np.zeros(2);d[i]=1e-6
            numerical[:,i]=(fk2(q+d)-fk2(q-d))/2e-6
        error = float(np.abs(numerical-jacobian2(q)).max())
        self.assertLess(error,1e-8);METRICS['jacobian_max_error']=error

    def test_dls_reaches_reachable_target(self):
        q,error = dls_ik([1.,1.],[.2,.1])
        self.assertLess(error,1e-6);METRICS['ik_reachable_error']=error

    def test_dls_reports_unreachable_residual(self):
        _,error = dls_ik([3.,0.],[.1,.1])
        self.assertGreater(error,.99);METRICS['ik_unreachable_error']=error

    def test_ransac_vertical_line_with_outliers(self):
        rng = np.random.default_rng(3)
        clean = np.column_stack([.5+.002*rng.normal(size=60),np.linspace(-1,1,60)])
        points = np.vstack([clean,rng.uniform(-1,1,(25,2))])
        n,offset,mask = ransac_line(points,threshold=.012)
        self.assertGreaterEqual(mask[:60].sum(),58)
        self.assertLess(np.abs(clean@n+offset).mean(),.005)
        METRICS['ransac_inliers']=int(mask.sum())

    def test_segment_collision_even_when_endpoints_clear(self):
        obstacles = [(np.array([.4,.4]),np.array([.6,.6]))]
        self.assertFalse(segment_free([.1,.5],[.9,.5],obstacles))
        self.assertTrue(segment_free([.1,.1],[.9,.1],obstacles))

    def test_rrt_path_edges_are_collision_free(self):
        obstacles = [(np.array([.4,.2]),np.array([.6,.8]))]
        path = rrt([.1,.5],[.9,.5],obstacles)
        self.assertIsNotNone(path)
        self.assertTrue(all(segment_free(a,b,obstacles) for a,b in zip(path[:-1],path[1:])))
        np.testing.assert_allclose(path[0],[.1,.5]);np.testing.assert_allclose(path[-1],[.9,.5])
        METRICS['rrt_path_nodes']=len(path)

    def test_rrt_rejects_goal_inside_obstacle(self):
        self.assertIsNone(rrt([0,0],[.5,.5],[(np.array([.4,.4]),np.array([.6,.6]))]))

    def test_causal_alignment_does_not_read_future(self):
        indices = causal_indices([0,.1,.2],[-.1,.09,.15,.5],.12)
        np.testing.assert_array_equal(indices,[-1,0,1,-1])

    def test_unsorted_timestamps_rejected(self):
        with self.assertRaises(ValueError):causal_indices([.2,.1],[.3],.2)

    def test_ddp_global_mean_differs_from_local_means(self):
        scaled,global_mean = ddp_loss_scaling([2.,90.],[2,30])
        self.assertAlmostEqual(global_mean,2.875)
        self.assertAlmostEqual(scaled.mean(),global_mean)
        self.assertNotEqual(np.mean([1,3]),global_mean)

    def test_replay_overwrite_and_sampling(self):
        buf = ReplayBuffer(3)
        for i in range(5):buf.add(i)
        self.assertEqual(set(buf.data),{2,3,4})
        self.assertTrue(set(buf.sample(100,np.random.default_rng(0))) <= {2,3,4})

    def test_replay_empty_rejected(self):
        with self.assertRaises(ValueError):ReplayBuffer(3).sample(1,np.random.default_rng(0))


class DeploymentTests(unittest.TestCase):
    def test_online_softmax_matches_full_for_multiple_tiles(self):
        rng = np.random.default_rng(4)
        q,k,v = [rng.normal(size=s) for s in [(7,5),(13,5),(13,4)]]
        reference = attention(q,k,v)[0]
        errors=[]
        for block in [1,3,8,32]:
            actual = online_attention(q,k,v,block)
            np.testing.assert_allclose(actual,reference,atol=1e-12)
            errors.append(float(np.abs(actual-reference).max()))
        METRICS['online_softmax_max_errors']=errors

    def test_online_softmax_masked_prefix_and_empty_row(self):
        q,k,v = np.ones((2,2)),np.ones((6,2)),np.arange(6.)[:,None]
        mask = np.array([[False,False,False,False,True,True],[False]*6])
        np.testing.assert_allclose(online_attention(q,k,v,2,mask),[[4.5],[0.]])

    def test_online_softmax_handles_extreme_scores(self):
        q,k,v = np.array([[1000.,0.]]),np.array([[1.,0.],[2.,0.]]),np.array([[3.],[5.]])
        np.testing.assert_allclose(online_attention(q,k,v,1),[[5.]])

    def test_quantization_saturation_and_integer_dot(self):
        x,w = np.array([-2.,0.,.5,2.,4.]),np.array([.25,.5,-.5,.75,-1.])
        qx,qw = quantize(x,.5),quantize(w,.25)
        result,acc = integer_dot(qx,qw,.5,.25)
        expected = float(np.dot(qx*.5,qw*.25))
        self.assertEqual(result,expected);self.assertEqual(qx[-1],7)
        METRICS['integer_dot']={'accumulator':acc,'dequantized':result,'float':float(x@w)}

    def test_nonzero_zero_points(self):
        result,_ = integer_dot([3,5],[4,2],.2,.3,zx=3,zw=2)
        self.assertEqual(result,0.)

    def test_gqa_kv_budget(self):
        mha = kv_bytes(32,1,2048,32,128)
        gqa = kv_bytes(32,1,2048,8,128)
        self.assertEqual(mha,4*gqa);self.assertEqual(gqa,268435456)
        METRICS['kv_cache_bytes']={'MHA':mha,'GQA':gqa}


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'python':platform.python_version(),'numpy':np.__version__,
              'torch':torch.__version__,'device':'CPU','tests':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),
              'metrics':METRICS}
    Path(__file__).with_name('verification_results.json').write_text(
        json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
