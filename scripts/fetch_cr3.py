"""Fetch the pinned Dobot model into the ignored outputs directory."""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

COMMIT = "7807ef8fb31e8f86a76cd474a4a9cddffef6eca9"
URL = f"https://raw.githubusercontent.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/{COMMIT}"
DEST = Path(__file__).resolve().parents[1] / "outputs" / "dobot_cr3"
FILES = (
    "LICENSE",
    "cra_description/package.xml",
    "cra_description/urdf/cr3_robot.xacro",
    "cr3_moveit/config/initial_positions.yaml",
    "cr3_moveit/config/joint_limits.yaml",
    "cr3_moveit/config/cr3_robot.srdf",
    *(f"cra_description/meshes/cr3/{name}.STL"
      for name in ("base_link", "J1", "J2", "J3", "J4", "J5", "J6")),
)


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DEST.parent) as stage:
        for name in FILES:
            path = Path(stage) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(f"{URL}/{name}", path)
        for name in FILES:
            target = DEST / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (Path(stage) / name).replace(target)
    print(f"Dobot CR3 {COMMIT} → {DEST}")


if __name__ == "__main__":
    main()
