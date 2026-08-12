import json
import os
from pathlib import Path
import pickle

import yaml


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pkl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        s = f.read().strip()
        if s == "":
            return {}
        return json.loads(s)


def load_object_metadata(modeldir, index):
    obj_path = Path(modeldir) / f"{index:05d}"
    output = dict()

    # 1. load trajectory json
    p = obj_path / "metadata.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {p}: {e}")
        return None

    output["physics"] = data.get("physics", {})
    output["visual"] = data.get("visual", {})
    output["geometry"] = data.get("geometry", {})
    output["active"] = data.get("active", {})
    output["passive"] = data.get("passive", {})
    required_keys = ["geometry"]

    for key in required_keys:
        if output.get(key) == {}:
            return None

    return output


def load_desc_info(modeldir, index, key="Rigid"):
    obj_path = Path(modeldir) / f"{index:05d}"
    p = obj_path / "description.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {p}: {e}")
        return None
    output = dict()
    desc = data.get("description", [])
    if isinstance(desc, list):
        output["description"] = desc
    elif isinstance(desc, str) and desc:
        output["description"] = [desc]
    else:
        output["description"] = []
    output["caption"] = data.get("caption", "")
    return output
