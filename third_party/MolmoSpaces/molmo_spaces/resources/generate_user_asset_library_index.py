"""
Script to generate an asset index for a user-provided asset library.
Optionally, the script can also compute some unprovided metadata for the assets.

The asset library can have any directory structure, but asset XMLs should be named <uid>.xml.
Their corresponding metadata JSONs should be named <uid>.json.
If these assets have array-valued metadata (e.g. clip features), they should be stored in an npz file named <uid>.npz,
    keys are forward-slash-delimited representing the hierarchy of the metadata.
"""

import argparse
import json
from pathlib import Path

import mujoco
from mujoco import MjData, MjModel

from molmo_spaces.utils.lazy_loading_utils import UserAssetLibraryIndex, UserAssetLibraryIndexEntry
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb


def hydrate_asset_metadata(asset_path: Path, metadata: dict):
    model = MjModel.from_xml_path(str(asset_path))
    data = MjData(model)
    mujoco.mj_fwdPosition(model, data)

    assert model.nbody == 2, "Expected a single body in the asset (other than worldbody)"
    body = model.body(1)

    if "mass" not in metadata:
        metadata["mass"] = body.mass.item()

    if "boundingBox" not in metadata:
        aabb_center, aabb_size = body_aabb(model, data, body.id, visible_only=False)
        # metadata bbox is centered at origin, so grow noncentered AABB to make centered
        aabb = 2 * (aabb_center + aabb_size / 2)
        metadata["boundingBox"] = {
            "x": aabb[0].item(),
            "y": aabb[1].item(),
            "z": aabb[2].item(),
        }


def build_asset_index(source_dir: Path, args) -> dict[str, UserAssetLibraryIndexEntry]:
    asset_index: dict[str, UserAssetLibraryIndexEntry] = {}

    for object_xml in sorted(source_dir.rglob("*.xml")):
        asset_id = object_xml.stem
        metadata_json = object_xml.with_suffix(".json")
        metadata_npz = object_xml.with_suffix(".npz")
        if not metadata_json.exists():
            if args.suppress_missing_metadata:
                print(
                    f"Warning: Expected metadata JSON for {object_xml} at {metadata_json}, but it was not found. Skipping."
                )
                continue
            raise FileNotFoundError(
                f"Expected metadata JSON for {object_xml} at {metadata_json}, but it was not found."
            )

        if args.compute_metadata:
            computed_metadata_json = metadata_json.with_name(f"{asset_id}_computed.json")
            if computed_metadata_json.exists() and not args.overwrite:
                print(
                    f"Computed metadata JSON for {object_xml} already exists at {computed_metadata_json}, skipping. Pass --overwrite to recompute."
                )
                continue

            with metadata_json.open() as f:
                metadata = json.load(f)

            hydrate_asset_metadata(object_xml, metadata)

            with computed_metadata_json.open("w") as f:
                json.dump(metadata, f, indent=4)
            metadata_path = computed_metadata_json
        else:
            metadata_path = metadata_json

        if asset_id in asset_index:
            prev = asset_index[asset_id].object_path
            raise ValueError(
                f"Duplicate asset id '{asset_id}' found at "
                f"{object_xml.relative_to(source_dir)} and {prev}."
            )

        asset_index[asset_id] = UserAssetLibraryIndexEntry(
            uid=asset_id,
            object_path=object_xml.relative_to(source_dir),
            metadata_path=metadata_path.relative_to(source_dir),
            metadata_npz_path=(
                metadata_npz.relative_to(source_dir) if metadata_npz.exists() else None
            ),
        )

    return dict(sorted(asset_index.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Asset library directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for assets_index.json. Defaults to <source_dir>/assets_index.json.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file if it exists."
    )
    parser.add_argument(
        "--compute-metadata",
        action="store_true",
        help="Compute metadata for the assets, write to a new JSON file next to the original.",
    )
    parser.add_argument(
        "--suppress-missing-metadata",
        action="store_true",
        help="Suppress missing metadata errors, skipping instead.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_path: Path = (
        args.output.resolve() if args.output is not None else source_dir / "assets_index.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to overwrite.")

    asset_index = build_asset_index(source_dir, args)
    output_json = UserAssetLibraryIndex.dump_json(asset_index, indent=4).decode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_json)

    print(f"Wrote {output_path} with {len(asset_index)} assets.")


if __name__ == "__main__":
    main()
