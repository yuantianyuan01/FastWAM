import json
import numpy as np
import base64
from typing import Any, Mapping

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def _to_numpy(obj: Any) -> Any:
    """递归将 torch.Tensor / numpy 转成可 JSON 序列化类型"""
    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        # Handle CUDA and half precision by moving to CPU before converting to NumPy
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj  # Let NumpyEncoder handle this
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        # Encode bytes as base64 so the payload stays JSON-serializable
        return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Mapping):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(v) for v in obj]
    # Try direct serialization first; fall back to str on failure
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class NumpyEncoder(json.JSONEncoder):
    """支持 numpy array 序列化"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype)
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": dtype,
                "shape": obj.shape,
            }
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """将任意嵌套结构转换为 JSON 字符串"""
    cleaned = _to_numpy(data)
    return json.dumps(cleaned, cls=NumpyEncoder, ensure_ascii=False)

def json_to_numpy(json_str: str) -> Any:
    """从 JSON 字符串反序列化 numpy array 和 bytes"""

    def object_hook(dct):
        if "__numpy__" in dct:
            data = base64.b64decode(dct["__numpy__"])
            return np.frombuffer(data, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
        if "__numpy_array__" in dct:
            data = base64.b64decode(dct["data"])
            return np.frombuffer(data, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
        if "__bytes__" in dct:
            return base64.b64decode(dct["data"])
        return dct

    return json.loads(json_str, object_hook=object_hook)
