#!/usr/bin/env python3
"""Update task_description in benchmark JSON files.

For each task, resolves the pickup object's asset_id from the object name hash,
then uses ObjectMeta.short_descriptions (1/2/3 word name) to set the
task_description using the appropriate template based on task_cls.

This matches the runtime behavior of PromptSampler.
"""

import argparse
import hashlib
import json
import random
import re

import numpy as np

from molmo_spaces.utils.object_metadata import ObjectMeta

COLORS = [
    ("red", np.array([1.0, 0.0, 0.0, 1.0])),
    ("blue", np.array([0.0, 0.0, 1.0, 1.0])),
    ("green", np.array([0.0, 1.0, 0.0, 1.0])),
    ("yellow", np.array([1.0, 1.0, 0.0, 1.0])),
    ("purple", np.array([0.5, 0.0, 0.5, 1.0])),
    ("orange", np.array([1.0, 0.5, 0.0, 1.0])),
    ("black", np.array([0.0, 0.0, 0.0, 1.0])),
    ("white", np.array([1.0, 1.0, 1.0, 1.0])),
    ("brown", np.array([0.55, 0.35, 0.15, 1.0])),
    ("tan", np.array([0.82, 0.71, 0.55, 1.0])),
    ("light blue", np.array([0.68, 0.85, 0.90, 1.0])),
    ("light green", np.array([0.56, 0.93, 0.56, 1.0])),
    ("light yellow", np.array([1.0, 1.0, 0.88, 1.0])),
]


def rgba_to_color_name(target_rgba):
    target_rgba = np.array(target_rgba)
    for color_name, color_rgba in COLORS:
        if np.allclose(target_rgba, color_rgba, atol=0.01):
            return color_name
    return "colored"


def build_hash_to_uid():
    """Build reverse lookup: md5(asset_id) -> asset_id."""
    mapping = {}
    for uid in ObjectMeta.all_uids():
        h = hashlib.md5(uid.encode()).hexdigest()
        mapping[h] = uid
        mapping[uid] = uid
    return mapping


def get_object_name(asset_uid, fallback, word_num=1):
    """Get name from metadata at given word count, falling back to the provided string."""
    if asset_uid is None:
        return fallback
    short_descs = ObjectMeta.short_descriptions(asset_uid)
    if not short_descs:
        return fallback
    # word_num is 1-indexed: 1->index 0, 2->index 1, 3->index 2
    idx = word_num - 1
    if idx < len(short_descs):
        return short_descs[idx].lower()
    # If requested word count exceeds available, use the longest available
    return short_descs[-1].lower()


def extract_hash_from_obj_name(obj_name: str) -> str:
    """Extract the hash from a pickup_obj_name.

    Handles both formats:
      - Thor objects: 'boiler_335af...69_1_0_2' (md5 hex hash)
      - Objaverse objects: 'obja..._l7EUZXch..._1_0_2' (base64-like uid)
    """
    # Match: {lemma}_{hash}_{count}_{body}_{room}
    # Hash is everything between first _ and the _\d+_\d+_\d+ suffix
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9]*_(.+)_\d+_\d+_\d+$", obj_name)
    if match:
        return match.group(1)
    return None


def sample_word_num(rng):
    """Sample word count: 50% -> 1, 30% -> 2, 20% -> 3."""
    r = rng.random()
    if r < 0.5:
        return 1
    elif r < 0.8:
        return 2
    else:
        return 3


def main():
    parser = argparse.ArgumentParser(description="Update task descriptions in benchmark JSON.")
    parser.add_argument("json_file", help="Path to the benchmark JSON file")
    parser.add_argument("--task-type", choices=["pick", "open", "close", "pick_and_place_next_to", "pick_and_place_color", "pick_and_place"], default=None,
                        help="Override task type (auto-detected from task_cls if not set)")
    parser.add_argument("--word-num", type=int, default=None,
                        help="Fixed word count (1/2/3). If not set, randomly samples 50%%/30%%/20%%.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.json_file, "r") as f:
        data = json.load(f)

    hash_to_uid = build_hash_to_uid()

    changed = 0
    errors = 0
    for i, task in enumerate(data):
        task_config = task.get("task", {})
        pickup_obj_name = task_config.get("pickup_obj_name")
        if not pickup_obj_name:
            continue

        obj_hash = extract_hash_from_obj_name(pickup_obj_name)
        if obj_hash is None:
            print(f"WARNING: task {i}: could not extract hash from '{pickup_obj_name}'")
            errors += 1
            continue

        asset_uid = hash_to_uid.get(obj_hash)
        target_category = "_".join(pickup_obj_name.split("_")[0:1])

        word_num = args.word_num if args.word_num else sample_word_num(rng)
        one_word_name = get_object_name(asset_uid, target_category, word_num)

        # Determine template from --task-type or task_cls
        if args.task_type:
            task_type = args.task_type
        else:
            task_cls = task_config.get("task_cls", "")
            if "PickAndPlaceNextTo" in task_cls:
                task_type = "pick_and_place_next_to"
            elif "PickAndPlaceColor" in task_cls:
                task_type = "pick_and_place_color"
            elif "PickAndPlace" in task_cls:
                task_type = "pick_and_place"
            elif "OpeningTask" in task_cls:
                task_type = "open"
            else:
                task_type = "pick"

        if task_type == "open":
            new_desc = f"open the {one_word_name}."
        elif task_type == "close":
            new_desc = f"close the {one_word_name}."
        elif task_type == "pick_and_place_color":
            receptacle_full = task_config.get("place_receptacle_name", "")
            place_word_num = args.word_num if args.word_num else sample_word_num(rng)
            # Format: place_receptacle/{idx}_{idx}/{uid}
            receptacle_uid = receptacle_full.split("/")[-1] if receptacle_full else None
            if receptacle_uid and receptacle_uid in ObjectMeta.annotation():
                receptacle_name = get_object_name(receptacle_uid, "receptacle", place_word_num)
            else:
                receptacle_name = "receptacle"
            # Get color from object_colors
            object_colors = task_config.get("object_colors", {})
            color_rgba = object_colors.get(receptacle_full)
            color_name = rgba_to_color_name(color_rgba) if color_rgba else "colored"
            new_desc = f"pick up the {one_word_name} and place it in or on the {color_name} {receptacle_name}."
        elif task_type in ("pick_and_place", "pick_and_place_next_to"):
            receptacle_full = task_config.get("place_receptacle_name", "")
            place_word_num = args.word_num if args.word_num else sample_word_num(rng)
            # Receptacle can be "place_receptacle/<uid>" or an object name like "pillow_<hash>_1_0_4"
            if "/" in receptacle_full:
                receptacle_uid = receptacle_full.split("/")[-1]
            else:
                rec_hash = extract_hash_from_obj_name(receptacle_full)
                receptacle_uid = hash_to_uid.get(rec_hash) if rec_hash else None
            rec_category = "_".join(receptacle_full.split("_")[0:1]) if receptacle_full else "receptacle"
            if receptacle_uid and receptacle_uid in ObjectMeta.annotation():
                receptacle_name = get_object_name(receptacle_uid, rec_category, place_word_num)
            else:
                receptacle_name = rec_category
            if task_type == "pick_and_place_next_to":
                new_desc = f"pick up the {one_word_name} and place it next to the {receptacle_name}."
            else:
                new_desc = f"pick up the {one_word_name} and place it on the {receptacle_name}."
        else:
            new_desc = f"pick up the {one_word_name}."

        lang = task.get("language", {})
        old_desc = lang.get("task_description", "")
        if old_desc != new_desc:
            if args.dry_run:
                print(f"Task {i} ({word_num}w): '{old_desc}' -> '{new_desc}'")
            lang["task_description"] = new_desc
            changed += 1

    if not args.dry_run:
        with open(args.json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {changed} task descriptions in {args.json_file}")
    else:
        print(f"\nDry run: {changed} tasks would be updated")

    if errors:
        print(f"{errors} tasks had errors (see warnings above)")


if __name__ == "__main__":
    main()
