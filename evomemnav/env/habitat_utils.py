"""Habitat helpers: coordinate frames, simulator configuration and image utilities."""

import numpy as np
import quaternion
import habitat_sim
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_coeffs,
    quat_to_angle_axis,
    quat_to_coeffs,
)
from PIL import Image


def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )


def pose_habitat_to_tsdf(pose):
    return pose_normal_to_tsdf(pose_habitat_to_normal(pose))


def make_sim_cfg(settings):
    """Simulator with an RGB and a depth camera. No semantic sensor is attached."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    gpu_id = int(settings.get("gpu_device_id", 0))
    if hasattr(sim_cfg, "gpu_device_id"):
        sim_cfg.gpu_device_id = gpu_id

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    specs = []
    for uuid, sensor_type in (
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [settings["height"], settings["width"]]
        spec.position = habitat_sim.geo.UP * settings["sensor_height"]
        spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
        spec.hfov = settings["hfov"]
        specs.append(spec)
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_quaternion(angle, camera_tilt):
    normalized_angle = angle % (2 * np.pi)
    if np.abs(normalized_angle - np.pi) < 1e-6:
        return quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    return quat_to_coeffs(
        quat_from_angle_axis(angle, np.array([0, 1, 0]))
        * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
    ).tolist()


def start_pose(init_pos, init_rot):
    """Episode start position and yaw from a GOAT-Bench start state."""
    pts = np.asarray(init_pos)
    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])
    return pts, angle


def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype="float32") / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype="uint8")
