import h5py
import yaml

def load_hdf5(path: str) -> dict:

    def _read(obj):
        # Dataset -> numpy / scalar / bytes->str
        if isinstance(obj, h5py.Dataset):
            v = obj[()]
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            # numpy scalar -> python scalar
            try:
                return v.item()
            except Exception:
                return v

        # Group -> dict
        out = {}
        for k, v in obj.items():
            out[k] = _read(v)
        return out

    with h5py.File(path, "r") as f:
        data = _read(f)

        if len(f.attrs) > 0:
            data["_attrs"] = {k: f.attrs[k] for k in f.attrs.keys()}

        return data

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_json(path: str) -> dict:
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data