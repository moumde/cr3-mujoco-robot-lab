"""Oracle scene built on the P0 CR3 and parallel gripper."""
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from cr3_lab.core.model import build_mjcf


class Orientation(StrEnum):
    PROTRUSION = "protrusion_facing_camera"
    FLAT = "flat_facing_camera"


class SpawnRange(StrEnum):
    FIXED = "fixed"
    SMALL = "small"
    LARGE = "large"

    @property
    def half_width_m(self):
        return {self.FIXED: 0.0, self.SMALL: 0.010, self.LARGE: 0.025}[self]


@dataclass(frozen=True)
class MotorDimensions:
    """Nominal dimensions and masses shared by motor-task scenes."""
    radius_m: float = 0.026
    length_m: float = 0.076
    mass_kg: float = 0.18
    boss_radius_m: float = 0.012
    boss_length_m: float = 0.015
    boss_mass_kg: float = 0.018


@dataclass(frozen=True)
class SceneConfig:
    table_xy_m: tuple = (0.47, 0.0)
    table_half_size_m: tuple = (0.20, 0.20, 0.022)
    table_height_m: float = 0.12
    initial_xy_m: tuple = (0.45, 0.09)
    target_xy_m: tuple = (0.45, -0.09)
    target_half_size_m: tuple = (0.062, 0.043)
    inspection_world_m: tuple = (0.45, 0.39, 0.235)
    inspection_lookat_m: tuple = (0.45, 0.09, 0.146)
    localization_world_m: tuple = (0.72, 0.37, 0.49)
    localization_lookat_m: tuple = (0.45, 0.09, 0.146)
    overview_world_m: tuple = (0.83, -0.94, 0.72)
    overview_lookat_m: tuple = (0.31, 0.0, 0.25)
    motor: MotorDimensions = MotorDimensions()
    table_friction: tuple = (0.45, 0.005, 0.0001)
    motor_friction: tuple = (0.45, 0.005, 0.0001)
    pad_friction: tuple = (0.8, 0.005, 0.0001)
    pad_axial_half_width_m: float = 0.031

    @property
    def rest_z_m(self):
        return self.table_height_m + self.motor.radius_m

    @property
    def target_normal_world(self):
        return np.array([-1., 0., 0.])


DEFAULT_SCENE = SceneConfig()
PAD_NAMES = ("left_pad", "right_pad")
MOTOR_GEOMS = ("motor_housing", "motor_boss")


def xml_vector(values):
    return " ".join(map(str, values))


def load_scene(config=DEFAULT_SCENE):
    root = ET.fromstring(build_mjcf(gripper=True))
    world = root.find("worldbody")
    root.find("option").attrib.update(cone="elliptic", impratio="10", tolerance="1e-10")
    half_thickness = config.table_half_size_m[2]
    ET.SubElement(world, "geom", name="table", type="box",
                  pos=xml_vector((*config.table_xy_m, config.table_height_m - half_thickness)),
                  size=xml_vector(config.table_half_size_m), rgba="0.4 0.43 0.47 1",
                  contype="1", conaffinity="14", friction=xml_vector(config.table_friction))
    ET.SubElement(world, "site", name="target", type="box",
                  pos=xml_vector((*config.target_xy_m, config.table_height_m)),
                  size=xml_vector((*config.target_half_size_m, 0.001)),
                  rgba="0.25 0.8 0.4 0.55")
    tx, ty = config.target_xy_m
    z = config.table_height_m + 0.002
    for name, start, end in (
        ("shaft", (tx + 0.04, ty, z), (tx - 0.055, ty, z)),
        ("left", (tx - 0.055, ty, z), (tx - 0.038, ty + 0.014, z)),
        ("right", (tx - 0.055, ty, z), (tx - 0.038, ty - 0.014, z)),
    ):
        ET.SubElement(world, "site", name=f"target_arrow_{name}", type="capsule",
                      fromto=xml_vector((*start, *end)), size="0.002",
                      rgba="0.96 0.96 0.75 1")
    for name, eye, lookat, fovy in (
        ("inspection", config.inspection_world_m, config.inspection_lookat_m, 30),
        ("localization", config.localization_world_m, config.localization_lookat_m, 31),
        ("overview", config.overview_world_m, config.overview_lookat_m, 39),
    ):
        backward = np.asarray(eye) - lookat
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        ET.SubElement(world, "camera", name=name, pos=xml_vector(eye),
                      xyaxes=xml_vector(np.r_[right, up]), fovy=str(fovy))
    motor = ET.SubElement(world, "body", name="motor",
                          pos=xml_vector((*config.initial_xy_m, config.rest_z_m)))
    ET.SubElement(motor, "freejoint", name="motor_free")
    for name, radius, length, mass, z, color in (
        ("motor_housing", config.motor.radius_m, config.motor.length_m,
         config.motor.mass_kg, 0, "0.83 0.54 0.17 1"),
        ("motor_boss", config.motor.boss_radius_m, config.motor.boss_length_m,
         config.motor.boss_mass_kg, (config.motor.length_m + config.motor.boss_length_m) / 2,
         "0.9 0.25 0.08 1"),
    ):
        ET.SubElement(motor, "geom", name=name, type="cylinder", mass=str(mass),
                      pos=xml_vector((0, 0, z)), size=xml_vector((radius, length / 2)),
                      rgba=color, contype="8", conaffinity="7", condim="4",
                      friction=xml_vector(config.motor_friction))
    for name in PAD_NAMES:
        pad = world.find(f".//geom[@name='{name}']")
        pad.set("friction", xml_vector(config.pad_friction))
        pad.set("condim", "4")
        size = pad.get("size").split()
        size[1] = str(config.pad_axial_half_width_m)
        pad.set("size", " ".join(size))
    # Adjacent arm meshes overlap at their joints by design. Check all other
    # arm mesh pairs, and the gripper against upstream links.
    contact = ET.SubElement(root, "contact")
    arm = ["base_link_collision", *(f"Link{i}_collision" for i in range(1, 7))]
    for (i, first), (j, second) in combinations(enumerate(arm), 2):
        if j - i > 1:
            ET.SubElement(contact, "pair", geom1=first, geom2=second)
    for upstream in arm[:-1]:
        for tool in ("palm", *PAD_NAMES):
            ET.SubElement(contact, "pair", geom1=upstream, geom2=tool)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def sample_initial_xy(config, seed, spawn_range):
    """Sample one XY without rejection or coupling to the reset roll."""
    width = SpawnRange(spawn_range).half_width_m
    center = np.asarray(config.initial_xy_m)
    extent = np.array([config.motor.radius_m,
                       config.motor.length_m / 2 + config.motor.boss_length_m])
    if (np.any(abs(center - config.table_xy_m) + width + extent > config.table_half_size_m[:2])
            or center[1] - width - extent[1] <= 0):
        raise ValueError("Spawn range must fit the motor footprint on the left half of the table")
    if width == 0:
        return center.copy()
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    return center + rng.uniform(-width, width, size=2)


def reset_scene(model, data, orientation, seed, config=DEFAULT_SCENE, *, spawn_range=SpawnRange.FIXED):
    """Reset the motor center, axial roll and the requested end direction."""
    orientation = Orientation(orientation)
    xy = sample_initial_xy(config, seed, spawn_range)
    mujoco.mj_resetData(model, data)
    direction = 1 if orientation is Orientation.PROTRUSION else -1
    roll = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    horizontal = np.empty(4)
    axial = np.empty(4)
    mujoco.mju_axisAngle2Quat(horizontal, np.array([-1., 0., 0.]), direction * np.pi / 2)
    mujoco.mju_axisAngle2Quat(axial, np.array([0., 0., 1.]), roll)
    pose = data.joint("motor_free").qpos
    pose[:3] = (*xy, config.rest_z_m)
    mujoco.mju_mulQuat(pose[3:], horizontal, axial)
    mujoco.mj_forward(model, data)


def oracle_pose(data):
    """Return the housing origin and local-to-world rotation directly from MuJoCo."""
    body = data.body("motor")
    return body.xpos.copy(), body.xmat.reshape(3, 3).copy()
