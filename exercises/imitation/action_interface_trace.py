"""Dimensionless learning probes and metric coordinate/command conversions.

No policy checkpoint, kinematics solver, controller or robot is executed.
"""
from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).resolve().parent


def rotation_z(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0.],
                     [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])


def to_base(points_task, rotation_task_from_base, translation_task_from_base):
    # P2: column formula R.T @ (p-t); samples here are stored as rows.
    return (points_task - translation_task_from_base) @ rotation_task_from_base


def to_task(points_base, rotation_task_from_base, translation_task_from_base):
    return points_base @ rotation_task_from_base.T + translation_task_from_base


def masked_mean_square(prediction, target, valid):
    # M1: select real dimensions before averaging; a legal zero is still real.
    errors = (prediction - target)[valid]
    return None if errors.size == 0 else float(np.mean(errors ** 2))


def parameter_probe():
    # G1: shared scalar representation, body-specific scalar output heads.
    phi, wa, wb = 1., 1., -1.
    target_a, target_b = 2., -3.
    ea, eb = wa * phi - target_a, wb * phi - target_b
    gradients = np.array([ea * wa + eb * wb, ea * phi, eb * phi])
    def loss(values):
        p, a, b = values
        return .5 * ((a * p - target_a) ** 2 + (b * p - target_b) ** 2)
    values = np.array([phi, wa, wb])
    eps = 1e-6
    numerical = np.array([
        (loss(values + eps * np.eye(3)[i]) - loss(values - eps * np.eye(3)[i])) / (2 * eps)
        for i in range(3)])
    assert np.allclose(gradients, numerical, atol=1e-8)
    # G2: update all parameters together, or freeze the shared scalar.
    full = values - .1 * gradients
    heads_only = values - .1 * gradients * np.array([0., 1., 1.])
    # G3: a separate conflict case shows the effect of body sampling weights.
    theta, pa, pb = 0., .9, .1
    gradient_conflict = pa * (theta - 1) + pb * (theta + 1)
    theta_new = theta - .1 * gradient_conflict
    return {"initial_parameters": values.tolist(), "gradient": gradients.tolist(),
            "initial_loss": loss(values), "full_parameters_after": full.tolist(),
            "full_loss_after": loss(full), "heads_only_parameters_after": heads_only.tolist(),
            "heads_only_loss_after": loss(heads_only),
            "conflict": {"weights": [pa, pb], "theta_before": theta, "theta_after": theta_new,
                         "body_squared_errors_before": [1., 1.],
                         "body_squared_errors_after": [(theta_new - 1) ** 2, (theta_new + 1) ** 2],
                         "weighted_half_loss_before": .5,
                         "weighted_half_loss_after": .5 * (pa * (theta_new - 1) ** 2 + pb * (theta_new + 1) ** 2)}}


def main():
    anchor_task = np.array([.4, .1, .2])
    # P1: undo normalization in the defined common representation [relative xyz,width].
    means = np.zeros(4)
    scales = np.array([.02, .02, .02, .04])
    normalized = np.array([[.5, 0., 0., 1.], [1.5, 1., 0., .5], [2., 1., 1., 0.]])
    common = normalized * scales + means
    targets_task = anchor_task + common[:, :3]
    widths_m = common[:, 3]
    ra, ta = np.eye(3), np.zeros(3)
    rb, tb = rotation_z(90), np.array([.5, 0., 0.])
    points_a = to_base(targets_task, ra, ta)
    points_b = to_base(targets_task, rb, tb)
    anchor_b = to_base(anchor_task, rb, tb)
    # P3: A uses absolute metric positions and open=1 for a maximum width of .08m.
    command_a = np.column_stack([points_a, widths_m / .08])
    # P3: B uses sequential position increments in mm and closed=1 at zero width.
    deltas_b = np.diff(np.vstack([anchor_b, points_b]), axis=0)
    command_b = np.column_stack([1000 * deltas_b, 1 - widths_m / .10])
    # P4: dimensions/width ranges are necessary; they do not prove reachability.
    assert command_a.shape == command_b.shape == (3, 4)
    assert np.all((widths_m >= 0) & (widths_m <= .08))
    # P5: invert each contract and compare the physical targets.
    recovered_a = to_task(command_a[:, :3], ra, ta)
    recovered_b = to_task(anchor_b + np.cumsum(command_b[:, :3] / 1000, axis=0), rb, tb)
    assert np.allclose(recovered_a, targets_task)
    assert np.allclose(recovered_b, targets_task)
    assert np.allclose(command_a[:, 3] * .08, widths_m)
    assert np.allclose((1 - command_b[:, 3]) * .10, widths_m)
    # F1: feeding common-origin relative positions into a delta interface double-counts.
    relative_b = points_b - anchor_b
    wrong_delta_targets = to_task(anchor_b + np.cumsum(relative_b, axis=0), rb, tb)
    # F2: changing the anchor after executing target1 corrupts target2's old chunk.
    moving_anchor_wrong_second = targets_task[0] + common[1, :3]
    # F3: using R instead of inverse rotation reverses this 90-degree example.
    wrong_first_delta_base = rb @ common[0, :3]
    wrong_first_delta_task = rb @ wrong_first_delta_base
    # M2: padded dimension is absent; genuine zero-valued labels remain supervised.
    target_a = np.array([1., -1., 0., 0., 0., 0., 0.])
    pred_a = np.array([.8, -.8, .1, .1, .1, .1, .1])
    target_b = target_a.copy()
    pred_b = np.array([.8, -.8, .1, .1, .1, .1, 9.])
    mask_a = np.ones(7, dtype=bool)
    mask_b = np.array([1, 1, 1, 1, 1, 1, 0], dtype=bool)
    # R1: local-axis and task-axis orientation increments have different multiplication order.
    angle = np.deg2rad(30)
    rx = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                   [0., np.sin(angle), np.cos(angle)]])
    r0 = rotation_z(90)
    ez = np.array([0., 0., 1.])
    result = {
        "scope": "three Cartesian targets, no IK/dynamics/robot; scalar learning probes are separate",
        "anchor_task_m": anchor_task.tolist(), "desired_target_times_from_anchor_s": [.1, .2, .3],
        "normalization_mean": means.tolist(),
        "normalization_scale": scales.tolist(), "normalized_chunk": normalized.tolist(),
        "common_relative_xyz_and_absolute_width_m": common.tolist(),
        "targets_task_m": targets_task.tolist(), "rotation_task_from_base_B": rb.tolist(),
        "translation_task_from_base_B_m": tb.tolist(), "anchor_base_B_m": anchor_b.tolist(),
        "command_A_absolute_xyz_m_and_open_fraction": command_a.tolist(),
        "command_B_sequential_xyz_mm_and_closed_fraction": command_b.tolist(),
        "round_trip_max_error_m": float(max(np.max(np.abs(recovered_a - targets_task)),
                                            np.max(np.abs(recovered_b - targets_task)))),
        "wrong_relative_as_delta_targets_task_m": wrong_delta_targets.tolist(),
        "wrong_delta_final_error_m": (wrong_delta_targets[-1] - targets_task[-1]).tolist(),
        "wrong_moving_anchor_second_target_m": moving_anchor_wrong_second.tolist(),
        "wrong_rotation_first_delta_task_m": wrong_first_delta_task.tolist(),
        "normalization_counterexample": {
            "current_absolute_m": .4, "target_absolute_m": .41,
            "correct_relative_normalized": (.41 - .4) / .02,
            "wrong_absolute_stats_on_relative": ((.41 - .4) - .5) / .2,
            "difference_of_absolute_standardized_values": ((.41 - .5) / .2) - ((.4 - .5) / .2)},
        "mask_probe": {"body_A_valid_mse": masked_mean_square(pred_a, target_a, mask_a),
                       "body_B_valid_mse": masked_mean_square(pred_b, target_b, mask_b),
                       "body_B_masked_sum_divided_by7": float(np.sum((pred_b[mask_b] - target_b[mask_b]) ** 2) / 7),
                       "body_B_unmasked_mse": float(np.mean((pred_b - target_b) ** 2)),
                       "no_valid_labels": masked_mean_square(pred_b, target_b, np.zeros(7, dtype=bool))},
        "missing_view_probe": {"valid_values": [2., 4.], "placeholder": 0.,
                               "masked_average": 3., "treat_missing_as_real_zero": 2.},
        "rotation_probe": {"right_local_increment_z_axis": (r0 @ rx @ ez).tolist(),
                           "left_task_increment_z_axis": (rx @ r0 @ ez).tolist()},
        "parameter_probe": parameter_probe(),
    }
    out = BASE / "results/interface_trace.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"round_trip_max_error_m": result["round_trip_max_error_m"],
                      "command_B": command_b.tolist(), "full_update_loss": result["parameter_probe"]["full_loss_after"],
                      "body_B_valid_mse": result["mask_probe"]["body_B_valid_mse"]}))


if __name__ == "__main__":
    main()
