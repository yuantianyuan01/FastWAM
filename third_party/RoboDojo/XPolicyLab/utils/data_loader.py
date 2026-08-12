# Load data for data conversion
import h5py, json, argparse
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit

def load_xspark_v1(hdf5_path, decode_images=True):
    def h5_to_dict(obj):
        d = {}
        for key, item in obj.items():
            if isinstance(item, h5py.Group):
                d[key] = h5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                val = item[()]
                
                if key == "colors" and isinstance(val, np.ndarray):
                    d[key] = decode_image_bit(val) if decode_images else val
                    continue

                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        decoded_str = val.decode("utf-8")
                        try:
                            d[key] = json.loads(decoded_str)
                        except json.JSONDecodeError:
                            d[key] = decoded_str
                    except Exception:
                        d[key] = val
                elif isinstance(val, np.ndarray) and val.dtype.kind in ["S", "U", "O"]:
                    try:
                        val_item = val.item() if val.size == 1 else val
                        if isinstance(val_item, (bytes, np.bytes_)):
                            decoded_str = val_item.decode("utf-8")
                        elif isinstance(val_item, str):
                            decoded_str = val_item
                        else:
                            d[key] = val
                            continue
                        try:
                            d[key] = json.loads(decoded_str)
                        except json.JSONDecodeError:
                            d[key] = decoded_str
                    except Exception:
                        d[key] = val
                else:
                    d[key] = val
        return d

    with h5py.File(hdf5_path, "r") as f:
        return h5_to_dict(f)

def load(data_path, data_type="xspark", data_version="v1.0"):
    # RoboDojo sim_cloud HDF5 uses the same xspark v1.0 layout.
    if data_type in {"xspark", "RoboDojo"}:
        if data_version == "v1.0":
            return load_xspark_v1(data_path, decode_images=True)
        raise NotImplementedError(f"{data_version} is not valid in {data_type} .")
    raise NotImplementedError(f"{data_type} is not valid data type. ")