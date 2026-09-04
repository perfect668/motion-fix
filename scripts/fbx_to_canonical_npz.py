"""Extract global joint positions from a binary FBX with Blender."""

import argparse
from pathlib import Path
import numpy as np

# The OptiTrack FBX used by the sample is centimetres in a Z-up world.  Keep
# this explicit: applying the BVH Y-up conversion here rotates the legs and
# puts the floor on the wrong axis.
FBX_UNIT_SCALE = 0.01


def rest_foot_forward(rest_head, rest_tail, pose_delta):
    """Propagate the bind-pose head-to-tail anatomical foot axis."""
    import numpy as _np
    axis = _np.asarray(rest_tail, dtype=float) - _np.asarray(rest_head, dtype=float)
    if _np.linalg.norm(axis) < 1e-8:
        raise ValueError("Foot rest axis has zero length")
    axis = _np.asarray(pose_delta, dtype=float) @ axis
    norm = _np.linalg.norm(axis)
    if norm < 1e-8:
        raise ValueError("Foot pose produced a zero-length forward axis")
    return axis / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import bpy
    import mathutils

    bpy.ops.import_scene.fbx(filepath=str(args.input), use_anim=True)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one FBX armature, found {len(armatures)}")
    armature = armatures[0]
    mapping = {
        "hip": "Hips", "abdomen": "Spine", "chest": "Spine1", "neck": "Neck", "head": "Head",
        "lCollar": "LeftShoulder", "lShldr": "LeftArm", "lForeArm": "LeftForeArm", "lHand": "LeftHand",
        "rCollar": "RightShoulder", "rShldr": "RightArm", "rForeArm": "RightForeArm", "rHand": "RightHand",
        "lThigh": "LeftUpLeg", "lShin": "LeftLeg", "lFoot": "LeftFoot",
        "rThigh": "RightUpLeg", "rShin": "RightLeg", "rFoot": "RightFoot",
    }
    available = {bone.name for bone in armature.data.bones}
    missing = sorted(set(mapping) - available)
    if missing:
        raise RuntimeError(f"FBX armature is missing mapped bones: {missing}")
    source_names = list(mapping.values())
    bone_names = list(mapping)
    scene = bpy.context.scene
    start, end = scene.frame_start, scene.frame_end
    positions = []
    orientations = []
    for frame in range(start, end + 1):
        scene.frame_set(frame)
        current = []
        current_rotations = []
        for bone_name in bone_names:
            pose_bone = armature.pose.bones[bone_name]
            point = armature.matrix_world @ pose_bone.head
            current.append((float(point.x) * FBX_UNIT_SCALE,
                            float(point.y) * FBX_UNIT_SCALE,
                            float(point.z) * FBX_UNIT_SCALE))
            rotation = (armature.matrix_world.to_3x3() @ pose_bone.matrix.to_3x3()).to_quaternion()
            current_rotations.append((float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)))
        # Add stable proxies required by the generic V4 semantic mapping.
        point_by_name = {name: current[index] for index, name in enumerate(source_names)}
        for side in ("Left", "Right"):
            foot_bone = "lFoot" if side == "Left" else "rFoot"
            foot_pose = armature.pose.bones[foot_bone]
            foot_data = armature.data.bones[foot_bone]
            toe_children = [child for child in foot_data.children if "toe" in child.name.lower() or "foot" in child.name.lower()]
            if toe_children:
                child_pose = armature.pose.bones[toe_children[0].name]
                toe_world = armature.matrix_world @ child_pose.head
                point_by_name[f"{side}ToeBase"] = tuple(float(v) * FBX_UNIT_SCALE for v in toe_world)
                continue
            rest_forward = foot_data.tail_local - foot_data.head_local
            if rest_forward.length < 1e-8 or abs(rest_forward.normalized().z) > 0.9:
                raise RuntimeError(f"Cannot determine anatomical rest foot axis for {foot_bone}")
            pose_delta = foot_pose.matrix.to_3x3() @ foot_data.matrix_local.to_3x3().inverted()
            forward = armature.matrix_world.to_3x3() @ mathutils.Vector(
                rest_foot_forward(rest_forward * 0.0, rest_forward, pose_delta)
            )
            forward.normalize()
            point_by_name[f"{side}ToeBase"] = tuple(
                np.asarray(point_by_name[f"{side}Foot"], dtype=float) + 0.12 * np.asarray(forward, dtype=float)
            )
            point_by_name[f"{side}HandMiddle3"] = point_by_name[f"{side}Hand"]
        positions.append([point_by_name[name] for name in source_names + ["LeftToeBase", "RightToeBase", "LeftHandMiddle3", "RightHandMiddle3"]])
        rotation_by_name = {name: current_rotations[index] for index, name in enumerate(source_names)}
        for side in ("Left", "Right"):
            rotation_by_name[f"{side}ToeBase"] = rotation_by_name[f"{side}Foot"]
            rotation_by_name[f"{side}HandMiddle3"] = rotation_by_name[f"{side}Hand"]
        orientations.append([rotation_by_name[name] for name in source_names + ["LeftToeBase", "RightToeBase", "LeftHandMiddle3", "RightHandMiddle3"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_joint_positions=np.asarray(positions, dtype=np.float32),
        global_joint_orientations=np.asarray(orientations, dtype=np.float32),
        joint_names=np.asarray(source_names + ["LeftToeBase", "RightToeBase", "LeftHandMiddle3", "RightHandMiddle3"]),
        fps=np.asarray(float(scene.render.fps)),
        source_file=np.asarray(str(args.input)),
        source_format=np.asarray("fbx_binary_blender_global_positions_meters_zup"),
        source_unit=np.asarray("centimeter"),
        coordinate_transform=np.asarray("scale_0.01_identity_zup"),
    )
    print(f"Saved canonical FBX positions: {args.output} ({len(positions)} frames, {len(source_names)} joints, {scene.render.fps:g} Hz)")


if __name__ == "__main__":
    main()
