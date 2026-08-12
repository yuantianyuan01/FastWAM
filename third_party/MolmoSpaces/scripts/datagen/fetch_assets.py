import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        dest="asset_type", help="Type of asset to fetch.", required=True
    )

    scene_parser = subparser.add_parser("scene", help="Fetch a scene")
    scene_parser.add_argument(
        "scene_type", help="Type of scene to fetch (ithor, procthor-10k, procthor-objaverse, etc.)"
    )
    scene_parser.add_argument("index", type=int, help="Scene index")
    scene_parser.add_argument(
        "--split", default="train", help="Dataset split to fetch the scene from, defaults to train"
    )
    scene_parser.add_argument(
        "--variant",
        default="base",
        help="Variant of the scene to fetch if applicable, defaults to base",
    )
    scene_parser.set_defaults(func=fetch_scene)

    defaults_parser = subparser.add_parser("default", help="Install default sources (robots, etc.)")
    defaults_parser.set_defaults(func=fetch_defaults)

    return parser.parse_args()


def fetch_scene(args):
    from molmo_spaces.molmo_spaces_constants import get_scenes
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scenes = get_scenes(args.scene_type)
    scene_path_or_dict = scenes[args.split][args.index]
    if isinstance(scene_path_or_dict, dict):
        scene_path = scene_path_or_dict[args.variant]
    else:
        scene_path = scene_path_or_dict
    scene_path = Path(scene_path)

    install_scene_with_objects_and_grasps_from_path(scene_path)


def fetch_defaults(args):
    os.environ["MLSPACES_FORCE_INSTALL"] = "True"
    from molmo_spaces.molmo_spaces_constants import get_resource_manager

    get_resource_manager()


def main():
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
