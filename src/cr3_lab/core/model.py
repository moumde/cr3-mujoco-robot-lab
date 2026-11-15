"""Convert the pinned, macro-free CR3 description into MuJoCo at load time."""
from math import cos, sin
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "outputs" / "dobot_cr3"
DESCRIPTION = Path("cra_description/urdf/cr3_robot.xacro")
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
TIMESTEP_S = 0.002
CONTROL_HZ = 50
FLANGE = "flange"
TCP = "tcp"


def _quaternion(rpy):
    """URDF fixed-axis RPY (Rz Ry Rx) to MuJoCo wxyz."""
    roll, pitch, yaw = (float(angle) / 2 for angle in rpy.split())
    cr, cp, cy = cos(roll), cos(pitch), cos(yaw)
    sr, sp, sy = sin(roll), sin(pitch), sin(yaw)
    return " ".join(map(str, (
        cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
    )))


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return {}
    return {"pos": origin.get("xyz", "0 0 0"),
            "quat": _quaternion(origin.get("rpy", "0 0 0"))}


def build_mjcf(source=DEFAULT_SOURCE, *, gripper=False):
    """Return MJCF text. The public arm-only TCP coincides with the flange."""
    source = Path(source).resolve()
    try:
        urdf = ET.parse(source / DESCRIPTION).getroot()
    except FileNotFoundError as exc:
        raise FileNotFoundError("CR3 source missing. Run python scripts/fetch_cr3.py.") from exc
    xml = ET.Element("mujoco", model="dobot_cr3_p0")
    ET.SubElement(xml, "compiler", angle="radian", autolimits="true")
    ET.SubElement(xml, "option", timestep=str(TIMESTEP_S), gravity="0 0 -9.81",
                  integrator="implicitfast")
    assets = ET.SubElement(xml, "asset")
    world = ET.SubElement(xml, "worldbody")
    actuators = ET.SubElement(xml, "actuator")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05",
                  rgba="0.23 0.26 0.29 1", contype="1", conaffinity="6")

    # Keep the upstream world → dummy_link → base_link transform intact.
    bodies = {"world": world}
    chain = ("world_joint", "dummy_joint", *JOINT_NAMES)
    for name in chain:
        joint = urdf.find(f"joint[@name='{name}']")
        link_name = joint.find("child").attrib["link"]
        parent = bodies[joint.find("parent").attrib["link"]]
        body = bodies[link_name] = ET.SubElement(parent, "body", name=link_name,
                                                 **_origin(joint))
        if joint.attrib["type"] == "revolute":
            limit = joint.find("limit").attrib
            limits = f"{limit['lower']} {limit['upper']}"
            ET.SubElement(body, "joint", name=name, type="hinge",
                          axis=joint.find("axis").attrib["xyz"], range=limits,
                          damping="1", armature="0.015")
            ET.SubElement(actuators, "position", name=f"{name}_position",
                          joint=name, ctrlrange=limits, kp="600", kv="30",
                          forcerange="-60 60")

        link = urdf.find(f"link[@name='{link_name}']")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia").attrib
        origin = _origin(inertial)
        if origin.pop("quat") != "1.0 0.0 0.0 0.0":
            raise ValueError(f"Unexpected rotated inertia frame for {link_name}")
        ET.SubElement(body, "inertial", **origin, mass=inertial.find("mass").attrib["value"],
                      fullinertia=" ".join(inertia[key] for key in
                                           ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for kind in ("visual", "collision"):
            shape = link.find(kind)
            mesh = shape.find("geometry/mesh").attrib["filename"]
            filename = mesh.removeprefix("file://$(find cra_description)/")
            asset_name = f"{link_name}_{kind}"
            ET.SubElement(assets, "mesh", name=asset_name,
                          file=str(source / "cra_description" / filename))
            rgba = (shape.find("material/color").attrib["rgba"] if kind == "visual"
                    else "0 0 0 0")
            ET.SubElement(body, "geom", **_origin(shape), name=asset_name, type="mesh",
                          mesh=asset_name, rgba=rgba, group="0" if kind == "visual" else "3",
                          contype="0" if kind == "visual" else "2", conaffinity="0")

    flange = ET.SubElement(bodies["Link6"], "body", name=FLANGE)
    ET.SubElement(flange, "site", name=FLANGE, size="0.004")
    if gripper:
        tool = ET.parse(ROOT / "assets/robots/gripper/parallel.xml").getroot()
        flange.append(tool.find("worldbody/body"))
        actuators.extend(tool.find("actuator"))
        xml.append(tool.find("equality"))
    else:
        ET.SubElement(flange, "site", name=TCP, size="0.004")
    return ET.tostring(xml, encoding="unicode")


def load_model(source=DEFAULT_SOURCE, *, gripper=False):
    return mujoco.MjModel.from_xml_string(build_mjcf(source, gripper=gripper))
