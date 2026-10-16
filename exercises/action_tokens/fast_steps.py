"""FAST-style arithmetic and a deliberately small symbol-pair BPE lesson.

The real released byte-level tokenizer is exercised by fast_official.py.
This file does not train a VLA or learn a robot controller.
"""
from collections import Counter
from pathlib import Path
import json
import math
import numpy as np

HOME = Path(__file__).resolve().parent


def dct_matrix(horizon):
    """F1: each row is one orthonormal temporal cosine basis."""
    k = np.arange(horizon)[:, None]
    t = np.arange(horizon)[None, :]
    basis = np.cos(np.pi / horizon * (t + 0.5) * k)
    basis[0] *= math.sqrt(1 / horizon)
    basis[1:] *= math.sqrt(2 / horizon)
    return basis


def apply_merge(sequence, left, right):
    """F3: replace disjoint adjacent pairs; do not merge across chunks."""
    output, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
            output.append(left + right)
            i += 2
        else:
            output.append(sequence[i])
            i += 1
    return output


def learn_symbol_merges(corpus, count):
    """Teaching BPE over integer symbols, not FAST's Unicode/byte frontend."""
    sequences = [[(int(x),) for x in chunk] for chunk in corpus]
    rules, history = [], []
    for _ in range(count):
        pairs = Counter((a, b) for seq in sequences for a, b in zip(seq, seq[1:]))
        # Equal frequencies are resolved by expanded integer sequences.
        pair = min(pairs, key=lambda p: (-pairs[p], p))
        rules.append(pair)
        sequences = [apply_merge(seq, *pair) for seq in sequences]
        history.append({"left": pair[0], "right": pair[1], "frequency": pairs[pair],
                        "first_chunk": sequences[0]})
    return rules, history


def main():
    action = np.array([[0, .2], [.2, .4], [.4, .6], [.6, .8]], dtype=np.float64)
    basis = dct_matrix(4)
    assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-14)
    coeff = basis @ action                         # F1: time -> frequency.
    quantized = np.rint(10 * coeff).astype(int)     # F2: rounding loses precision.
    flat = quantized.ravel().tolist()              # frequency first, then channel.
    assert flat == [6, 10, -4, -4, 0, 0, 0, 0]
    corpus = [flat, [8, 12, -4, -4, 0, 0, 0, 0],
              [10, 14, -4, -4, 0, 0, 0, 0]]
    rules, history = learn_symbol_merges(corpus, 3)
    encoded = [(x,) for x in flat]
    for pair in rules:                              # F3: apply learned ranks in order.
        encoded = apply_merge(encoded, *pair)
    decoded = [x for token in encoded for x in token]
    assert decoded == flat and len(encoded) == 4    # F4: BPE preserves integers.
    restored = basis.T @ (np.array(decoded).reshape(4, 2) / 10)
    squared_error = float(np.square(restored - action).sum())
    coeff_error = float(np.square(quantized / 10 - coeff).sum())
    assert abs(squared_error - coeff_error) < 1e-14  # Orthogonality preserves L2 error.
    assert math.sqrt(squared_error) <= math.sqrt(action.size) / 20 + 1e-14

    # A discontinuity needs several frequencies. This is an extra truncation
    # experiment; the FAST encoder does not simply discard the upper half.
    step = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    basis8 = dct_matrix(8)
    step_coeff = basis8 @ step
    low_only = step_coeff.copy()
    low_only[4:] = 0
    truncated_step = basis8.T @ low_only
    assert not np.allclose(truncated_step, step)
    result = {
        "scope": "synthetic normalized two-channel arrays; symbol BPE is a teaching abstraction",
        "action": action.tolist(), "orthonormal_basis": basis.tolist(),
        "coefficients": coeff.tolist(), "scale": 10, "quantized": quantized.tolist(),
        "flattened_frequency_major": flat, "merge_history": history,
        "symbol_tokens": encoded, "decoded_integers": decoded,
        "reconstructed": restored.tolist(), "action_squared_error": squared_error,
        "coefficient_squared_error": coeff_error,
        "max_absolute_error": float(abs(restored - action).max()),
        "l2_error_bound": math.sqrt(action.size) / 20,
        "extra_truncation": {"input": step.tolist(), "coefficients": step_coeff.tolist(),
                             "reconstruction": truncated_step.tolist(),
                             "squared_error": float(np.square(truncated_step - step).sum())},
    }
    (HOME / "results").mkdir(exist_ok=True)
    (HOME / "results/fast_steps.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integer_symbols": len(flat), "teaching_tokens": len(encoded),
                      "max_absolute_error": result["max_absolute_error"]}))


if __name__ == "__main__":
    main()
