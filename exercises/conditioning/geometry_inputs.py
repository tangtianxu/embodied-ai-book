"""Pinhole/rigid-transform interface example, not a camera calibration or robot run."""
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent


def camera_point(intrinsic, pixel, camera_z):
    # G1：像素已去畸变，深度是相机Z轴坐标，K与当前图像尺寸相符。
    return camera_z * torch.linalg.solve(intrinsic, pixel)


def point_to_base(point, rotation, translation):
    # G2：R、t明确把相机坐标映射到基座坐标。
    return rotation @ point + translation


def main():
    torch.set_default_dtype(torch.float64)
    intrinsic = torch.tensor([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    pixel = torch.tensor([380., 240., 1.])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = torch.tensor([1., 0., 0.])
    near = camera_point(intrinsic, pixel, camera_z=1.)
    far = camera_point(intrinsic, pixel, camera_z=2.)
    base_near = point_to_base(near, rotation, translation)
    torch.testing.assert_close(near, torch.tensor([0.1, 0., 1.]))
    torch.testing.assert_close(base_near, torch.tensor([1., 0.1, 1.]))
    # G3：逆变换、重新投影均应恢复原来的量。
    recovered = rotation.T @ (base_near - translation)
    torch.testing.assert_close(recovered, near)
    projections = []
    for point in [near, far]:
        projected = intrinsic @ point
        projected = projected / projected[2]
        torch.testing.assert_close(projected, pixel)
        projections.append(projected.tolist())
    # G4：两点差中的平移抵消；位移向量只需要旋转。
    delta_camera = torch.tensor([0.1, 0., 0.])
    moved = point_to_base(near + delta_camera, rotation, translation)
    delta_base = moved - base_near
    torch.testing.assert_close(delta_base, rotation @ delta_camera)
    torch.testing.assert_close(delta_base, torch.tensor([0., 0.1, 0.]))
    wrong_delta = point_to_base(delta_camera, rotation, translation)
    assert float(torch.linalg.vector_norm(wrong_delta - delta_base)) > 0.9
    record = {
        "status": "passed", "torch": torch.__version__,
        "scope": "given ideal intrinsics/depth and camera-to-base transform; no actual calibration, visual detection, VLA inference or robot action",
        "intrinsic": intrinsic.tolist(), "pixel": pixel.tolist(),
        "camera_axis_depths_m": [1., 2.], "camera_points_m": [near.tolist(), far.tolist()],
        "camera_to_base_rotation": rotation.tolist(),
        "camera_to_base_translation_m": translation.tolist(),
        "base_near_m": base_near.tolist(), "inverse_recovered_m": recovered.tolist(),
        "reprojected_pixels": projections, "delta_camera_m": delta_camera.tolist(),
        "delta_base_m": delta_base.tolist(), "incorrect_delta_with_translation_m": wrong_delta.tolist(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/geometry_inputs.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
