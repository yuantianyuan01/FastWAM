"""
Script to generate a grasp index for a user-provided grasp library.
Assumes the following conventions:
    - Nonarticulated grasp files are of the form <uid>/<robot_name>/grasps.npz
    - Articulated grasp files are of the form <uid>/<robot_name>/joint_grasps_<joint_name>.npz

Where:
    - <uid> is the asset ID of the object the grasp is for
    - <robot_name> is the name of the robot/gripper the grasp is for (i.e. what it was collected with)
    - <joint_name> is the name of the joint the grasp is for, for articulated grasps

These conventions are required for this indexing script, but not by the rest of MolmoSpaces.
"""

import argparse
from pathlib import Path

from molmo_spaces.utils.lazy_loading_utils import UserGraspLibraryIndex


def get_grasp_paths(source_dir: Path) -> dict[str, dict[str, Path]]:
    grasp_paths: dict[str, dict[str, Path]] = {}

    for grasp_file in sorted(source_dir.rglob("grasps.npz")):
        robot = grasp_file.parent.name
        uid = grasp_file.parent.parent.name

        rel_path = grasp_file.relative_to(source_dir)
        grasp_paths.setdefault(robot, {})[uid] = rel_path

    # Keep output stable for easier diffs/review.
    sorted_grasp_paths = {
        robot: dict(sorted(uid_to_path.items()))
        for robot, uid_to_path in sorted(grasp_paths.items())
    }
    return sorted_grasp_paths


def get_articulated_grasp_paths(source_dir: Path) -> dict[str, dict[str, dict[str, Path]]]:
    articulated_grasp_paths: dict[str, dict[str, dict[str, Path]]] = {}

    for grasp_file in sorted(source_dir.rglob("joint_grasps_*.npz")):
        robot = grasp_file.parent.name
        uid = grasp_file.parent.parent.name
        joint_name = grasp_file.stem.removeprefix("joint_grasps_")
        assert joint_name

        rel_path = grasp_file.relative_to(source_dir)
        robot_dict = articulated_grasp_paths.setdefault(robot, {})
        uid_dict = robot_dict.setdefault(uid, {})
        uid_dict[joint_name] = rel_path

    # Keep output stable for easier diffs/review.
    sorted_articulated_grasp_paths = {
        robot: dict(sorted(uid_to_path.items()))
        for robot, uid_to_path in sorted(articulated_grasp_paths.items())
    }
    return sorted_articulated_grasp_paths


def build_grasp_index(source_dir: Path) -> UserGraspLibraryIndex:
    grasp_paths = get_grasp_paths(source_dir)
    articulated_grasp_paths = get_articulated_grasp_paths(source_dir)
    return UserGraspLibraryIndex(
        grasp_paths=grasp_paths,
        articulated_grasp_paths=articulated_grasp_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Grasp library directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for grasps_index.json. Defaults to <source_dir>/grasps_index.json.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file if it exists."
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_path: Path = (
        args.output.resolve() if args.output is not None else source_dir / "grasps_index.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to overwrite.")

    grasp_index = build_grasp_index(source_dir=source_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(grasp_index.model_dump_json(indent=4))

    num_robots = len(grasp_index.grasp_paths)
    num_uids = sum(len(uid_to_path) for uid_to_path in grasp_index.grasp_paths.values())
    print(f"Wrote {output_path} with {num_robots} robots / {num_uids} grasp files.")


if __name__ == "__main__":
    main()
