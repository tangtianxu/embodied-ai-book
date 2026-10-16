"""Compare the same attention/blocks with full and cached execution, including stale-cache failures."""
from pathlib import Path
import json
import math
import torch
from legacy.attention_torch import HandMHA,PreNormBlock,safe_softmax

ROOT=Path(__file__).resolve().parent


def cached_block(block,x,cache=None,causal=True):
    # K1: normalize the new token representations for this particular layer.
    attention_input=block.norm1(x)
    # K2: append this layer's new K,V, and query all allowed history/new positions.
    delta,_,new_cache=block.attn(attention_input,cache=cache,use_cache=True,causal=causal)
    # K3: finish both residual additions; FFN and LayerNorm act token by token.
    x=x+delta
    x=x+block.ffn(block.norm2(x))
    # K4: keep a separate cache for this layer, and pass new outputs to the next.
    return x,new_cache


def full_stack(blocks,tokens,causal=True):
    x=tokens
    for block in blocks:
        x=block(x,causal=causal)
    return x


def cached_stack(blocks,tokens,parts,causal=True):
    assert sum(parts)==tokens.shape[1]
    caches=[None for _ in blocks]
    outputs,shapes=[],[]
    start=0
    for size in parts:
        x=tokens[:,start:start+size]
        for i,block in enumerate(blocks):
            x,caches[i]=cached_block(block,x,caches[i],causal)
        outputs.append(x)
        shapes.append([list(cache[0].shape) for cache in caches])
        start+=size
    return torch.cat(outputs,dim=1),caches,shapes


def attention_with_allow(module,new,cache,allow):
    q=module.split_heads(module.q(new))
    keys=torch.cat([cache[0],module.split_heads(module.k(new))],dim=-2)
    values=torch.cat([cache[1],module.split_heads(module.v(new))],dim=-2)
    weights=safe_softmax(q@keys.transpose(-2,-1)/math.sqrt(module.d_head),allow)
    per_head=weights@values
    merged=per_head.transpose(1,2).reshape(new.shape[0],new.shape[1],-1)
    return module.out(merged)


def rope_pair(vector,position,frequency):
    angle=position*frequency
    rotation=torch.tensor([[math.cos(angle),-math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    return rotation@vector


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(37)
    tokens=torch.randn(2,7,12)
    attention=HandMHA(12,3).eval()
    full,_,_=attention(tokens,causal=True)
    first,_,cache=attention(tokens[:,:4],causal=True,use_cache=True)
    following,_,final_cache=attention(tokens[:,4:],causal=True,cache=cache,use_cache=True)
    attention_assembled=torch.cat([first,following],dim=1)
    torch.testing.assert_close(attention_assembled,full,rtol=1e-12,atol=1e-12)
    # M1: new queries have absolute positions after the four cached positions.
    query_position=4+torch.arange(3)
    # M2: keys include both the history and the new positions.
    key_position=torch.arange(7)
    # M3: allow exactly keys that are not later than each query.
    allow=key_position[None,:]<=query_position[:,None]
    explicit=attention_with_allow(attention,tokens[:,4:],cache,allow)
    wrong_allow=torch.ones(3,7,dtype=torch.bool).tril()
    wrong=attention_with_allow(attention,tokens[:,4:],cache,wrong_allow)
    torch.testing.assert_close(explicit,following,rtol=1e-12,atol=1e-12)
    assert float((wrong-following).abs().max())>1e-5

    blocks=torch.nn.ModuleList([PreNormBlock(12,3) for _ in range(2)]).eval()
    full_blocks=full_stack(blocks,tokens)
    partitions=[]
    for parts in [[4,3],[1,1,1,1,1,1,1],[2,1,4]]:
        assembled,caches,shapes=cached_stack(blocks,tokens,parts)
        difference=float((assembled-full_blocks).abs().max())
        torch.testing.assert_close(assembled,full_blocks,rtol=1e-12,atol=1e-12)
        partitions.append({"parts":parts,"max_full_difference":difference,
                           "key_cache_shapes_after_each_chunk":shapes})

    # Reusing old keys after a feature in the prefix changes is a different computation.
    changed=tokens.clone()
    changed[0,0,0]+=2
    changed_full=full_stack(blocks,changed)
    _,old_caches,_=cached_stack(blocks,tokens[:,:4],[4])
    suffix=changed[:,4:]
    for i,block in enumerate(blocks):
        suffix,_=cached_block(block,suffix,old_caches[i])
    stale_prefix_gap=float((suffix-changed_full[:,4:]).abs().max())
    assert stale_prefix_gap>1e-5
    fresh_changed,_,_=cached_stack(blocks,changed,[4,3])
    torch.testing.assert_close(fresh_changed,changed_full,rtol=1e-12,atol=1e-12)

    # With bidirectional layers, adding future tokens changes old lower-layer outputs.
    full_bi=full_stack(blocks,tokens,causal=False)
    cached_bi,_,_=cached_stack(blocks,tokens,[4,3],causal=False)
    bi_suffix_gap=float((cached_bi[:,4:]-full_bi[:,4:]).abs().max())
    assert bi_suffix_gap>1e-5
    # In a position-free, unmasked block, a permutation only permutes the outputs.
    order=torch.tensor([6,2,0,5,1,4,3])
    permuted=full_stack(blocks,tokens[:,order],causal=False)
    permutation_gap=float((permuted-full_bi[:,order]).abs().max())
    torch.testing.assert_close(permuted,full_bi[:,order],rtol=1e-12,atol=1e-12)

    q,k=torch.tensor([1.,0.]),torch.tensor([1.,1.])
    score=float(rope_pair(q,0,math.pi/2)@rope_pair(k,1,math.pi/2))
    shifted=float(rope_pair(q,2,math.pi/2)@rope_pair(k,3,math.pi/2))
    assert abs(score+1)<1e-12 and abs(shifted-score)<1e-12
    result={"status":"passed","torch":torch.__version__,"seed":37,
            "scope":"random fixed weights, mathematical attention and two complete PreNorm blocks; no positional encoding inside the cached blocks, no action head or robot",
            "tokens_shape":list(tokens.shape),"attention_full_cached_max_difference":float((full-attention_assembled).abs().max()),
            "attention_cache_shapes":[list(cache[0].shape),list(final_cache[0].shape)],
            "correct_allow":allow.tolist(),"wrong_local_tril_allow":wrong_allow.tolist(),
            "wrong_mask_output_max_difference":float((wrong-following).abs().max()),
            "two_block_partitions":partitions,
            "stale_changed_prefix_suffix_max_difference":stale_prefix_gap,
            "rebuilt_changed_prefix_full_max_difference":float((fresh_changed-changed_full).abs().max()),
            "bidirectional_naive_cached_suffix_max_difference":bi_suffix_gap,
            "unmasked_position_free_permutation_max_difference":permutation_gap,
            "rope_separate_two_dimensional_check":{"q":q.tolist(),"k":k.tolist(),"frequency":math.pi/2,
                                                "positions":[0,1],"score":score,"shifted_positions":[2,3],
                                                "shifted_score":shifted,"same_content_vectors":True}}
    (ROOT/"results").mkdir(exist_ok=True)
    torch.save({"attention":attention.state_dict(),"blocks":blocks.state_dict(),"tokens":tokens},
               ROOT/"results/reference_weights.pt")
    torch.save({"attention_full":full,"two_blocks_full":full_blocks,"bidirectional_full":full_bi},
               ROOT/"results/reference_outputs.pt")
    (ROOT/"results/cache_execution.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                                                  encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
