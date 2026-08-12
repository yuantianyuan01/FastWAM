from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from p_tqdm import p_uimap
from tqdm import tqdm

from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

SKIP_SUBSTR = ("orig", "fix", "non_settled", "ceiling")

Datasets = Literal["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"]
Splits = Literal["train", "val", "test"]


@dataclass
class Args:
    dataset: Datasets
    split: Splits

    scenes_dir: Path

    max_workers: int = 1


def generate_map(scene_xml: Path) -> bool:
    success = False
    try:
        thormap: ProcTHORMap | iTHORMap | None = None
        if "FloorPlan" in scene_xml.stem:
            thormap = iTHORMap.from_mj_model_path(
                scene_xml.as_posix(), px_per_m=200, agent_radius=None
            )
        else:
            thormap = ProcTHORMap.from_mj_model_path(
                scene_xml.as_posix(), px_per_m=200, agent_radius=None
            )

        map_filepath = scene_xml.parent.resolve() / f"{scene_xml.stem}_map.png"
        thormap.save(map_filepath.as_posix())
        success = True
    except Exception as e:
        print(f"[ERROR]: got an error while processing {scene_xml.stem} : {e}")

    return success


def main() -> int:
    args = tyro.cli(Args)

    if not args.scenes_dir.is_dir():
        return 1

    dataset_id = "ithor" if args.dataset == "ithor" else f"{args.dataset}-{args.split}"
    dataset_scenes_dir = args.scenes_dir / dataset_id

    if not dataset_scenes_dir.is_dir():
        return 1

    f_pattern = "FloorPlan*_physics.xml" if args.dataset == "ithor" else f"{args.split}*.xml"

    def is_valid_scene(path: Path) -> bool:
        return all(substr not in path.stem for substr in SKIP_SUBSTR)

    scenes_xmls = [path for path in dataset_scenes_dir.glob(f_pattern) if is_valid_scene(path)]

    if args.max_workers > 1:
        futures = p_uimap(generate_map, scenes_xmls, num_cpus=args.max_workers)
        for _ in futures:
            pass
    else:
        for scene_xml in tqdm(scenes_xmls):
            generate_map(scene_xml)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
