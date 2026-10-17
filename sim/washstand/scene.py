"""Build a physical SO101 tabletop scene from the pinned Menagerie asset.

All dimensions are nominal prototype values, not measured SO101 task hardware.
Only reset/planning may modify object states. No grasp equality or actuator is
added to the free toothbrushes/cups.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "assets" / "so101"
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
)
TASKS = {
    "red_left": "Put the red toothbrush into the left cup.",
    "red_right": "Put the red toothbrush into the right cup.",
    "blue_left": "Put the blue toothbrush into the left cup.",
    "blue_right": "Put the blue toothbrush into the right cup.",
}


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or HERE / "scene_config.json").read_text())


def _vec(values) -> str:
    return " ".join(f"{float(x):.12g}" for x in values)


def _add_brush(world, color: str, cfg: dict, xy_offset=(0.0, 0.0)) -> None:
    pos = list(cfg["centers"][color])
    pos[0] += xy_offset[0]
    pos[1] += xy_offset[1]
    body = ET.SubElement(world, "body", name=f"brush_{color}", pos=_vec(pos),
                         euler=f"0 0 {cfg['yaw_rad']}")
    ET.SubElement(body, "freejoint", name=f"brush_{color}_free")
    rgba = "0.86 0.07 0.055 1" if color == "red" else "0.04 0.21 0.88 1"
    common = {"friction": "0.8 0.005 0.0001", "condim": "4", "solref": "0.008 1"}
    # One free body, physically colliding compound parts with masses summing to
    # the specified nominal mass (there is no duplicate visual mass).
    ET.SubElement(body, "geom", name=f"brush_{color}_handle", type="capsule",
                  fromto="-0.070 0 0 0.049 0 0", size=str(cfg["handle_radius"]),
                  mass=str(cfg["mass_kg"] * 0.65), rgba=rgba, **common)
    ET.SubElement(body, "geom", name=f"brush_{color}_head", type="box",
                  pos="0.064 0 0", size="0.015 0.008 0.006",
                  mass=str(cfg["mass_kg"] * 0.35), rgba=rgba, **common)
    ET.SubElement(body, "geom", name=f"brush_{color}_bristles_visual", type="box",
                  pos="0.064 0 0.007", size="0.012 0.006 0.001", mass="0",
                  rgba="0.94 0.94 0.9 1", contype="0", conaffinity="0", group="2")
    for suffix, local in (("tip", cfg["handle_tip_local"]),
                          ("grasp", cfg["grasp_local"]), ("head", [0.064, 0, 0])):
        ET.SubElement(body, "site", name=f"brush_{color}_{suffix}", pos=_vec(local),
                      size="0.001", group="3", rgba="0 0 0 0")


def _add_cup(world, name: str, cfg: dict, xy_offset=(0.0, 0.0)) -> None:
    pos = list(cfg["centers"][name])
    pos[0] += xy_offset[0]
    pos[1] += xy_offset[1]
    body = ET.SubElement(world, "body", name=f"cup_{name}", pos=_vec(pos))
    ET.SubElement(body, "freejoint", name=f"cup_{name}_free")
    inner, thickness, height, bottom, segments = (
        cfg[k] for k in ("inner_radius", "wall_thickness", "height", "bottom_thickness", "wall_segments")
    )
    rgba = "0.84 0.89 0.80 1" if name == "left" else "0.89 0.84 0.72 1"
    common = {"friction": "0.8 0.005 0.0001", "condim": "4", "solref": "0.008 1", "rgba": rgba}
    ET.SubElement(body, "geom", name=f"cup_{name}_bottom", type="cylinder",
                  pos=_vec([0, 0, bottom / 2]), size=_vec([inner + thickness, bottom / 2]),
                  mass=str(cfg["mass_kg"] * .4), **common)
    # Individual convex wall segments preserve the empty interior. Tangential
    # overlap avoids the slit left by butt-jointed radial rectangles.
    radius = inner + thickness / 2
    half_tangent = radius * math.tan(math.pi / segments) * 1.03
    for i in range(segments):
        angle = i * math.tau / segments
        ET.SubElement(body, "geom", name=f"cup_{name}_wall_{i:02d}", type="box",
                      pos=_vec([radius * math.cos(angle), radius * math.sin(angle), (height + bottom) / 2]),
                      euler=_vec([0, 0, angle]),
                      size=_vec([thickness / 2, half_tangent, (height - bottom) / 2]),
                      mass=str(cfg["mass_kg"] * .6 / segments), **common)
    ET.SubElement(body, "site", name=f"cup_{name}_center", pos=_vec([0, 0, height]),
                  size="0.001", group="3", rgba="0 0 0 0")


def build_xml(config: dict | None = None, *, offsets: dict | None = None) -> str:
    """Return complete MJCF; mesh paths resolve solely inside this checkout."""
    cfg = copy.deepcopy(config or load_config())
    root = ET.parse(ASSET_DIR / "so101.xml").getroot()
    root.set("model", "so101_washstand_nominal_v1")
    root.find("compiler").set("meshdir", str(ASSET_DIR / "assets"))
    option = root.find("option")
    option.set("timestep", str(cfg["physics_timestep"]))
    option.set("iterations", "50")
    option.set("gravity", "0 0 -9.81")
    visual = root.find("visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960", azimuth="120", elevation="-35")
    ET.SubElement(visual, "headlight", ambient="0.15 0.15 0.15", diffuse="0.25 0.25 0.25", specular="0.05 0.05 0.05")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="table_checker", type="2d", builtin="checker",
                  rgb1="0.75 0.77 0.75", rgb2="0.70 0.72 0.70", width="128", height="128")
    ET.SubElement(asset, "material", name="table_material", texture="table_checker",
                  texrepeat="6 6", reflectance="0.05")
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="overhead", pos="0.2 -0.2 1.0", dir="0 0 -1", diffuse="0.6 0.6 0.6")
    ET.SubElement(world, "geom", name="table", type="plane", size="0.65 0.55 0.02",
                  material="table_material", friction="0.8 0.005 0.0001")
    ET.SubElement(world, "camera", name="top", pos="0.235 0 0.72", xyaxes="0 -1 0 1 0 0", fovy="48")
    wrist = root.find(".//camera[@name='wrist_cam']")
    wrist.set("name", "wrist")
    gripper = root.find(".//body[@name='gripper']")
    ET.SubElement(gripper, "site", name="pinch", pos="0 0 -0.098", quat="1 0 1 0",
                  size="0.001", group="3", rgba="0 0 0 0")
    # Stable geom names make physical contact diagnostics independent of IDs.
    for i, geom in enumerate(world.iter("geom")):
        if geom.get("name") is None:
            geom.set("name", f"robot_geom_{i:03d}")
    for color in ("red", "blue"):
        _add_brush(world, color, cfg["brush"], (offsets or {}).get(f"brush_{color}", (0, 0)))
    for cup in ("left", "right"):
        _add_cup(world, cup, cfg["cup"], (offsets or {}).get(f"cup_{cup}", (0, 0)))
    return ET.tostring(root, encoding="unicode")


def asset_manifest() -> dict:
    """Content hashes allow replay to identify the exact vendored model/assets."""
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ASSET_DIR.rglob("*")) if p.is_file()}
