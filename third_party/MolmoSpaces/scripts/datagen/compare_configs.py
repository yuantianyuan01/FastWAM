import pickle
import argparse
import sys
from typing import Any, get_origin, get_args, Union
from pydantic import BaseModel
import types

try:
    import numpy as np
except ImportError:
    np = None

NOT_DEFINED = "NOT-DEFINED"

def are_equal(v1: Any, v2: Any) -> bool:
    """Safely checks equality for standard types and multi-element arrays."""
    if v1 is NOT_DEFINED or v2 is NOT_DEFINED:
        return v1 is v2
    if np is not None and (isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray)):
        return np.array_equal(v1, v2)
    try:
        # Standard equality check
        comparison = (v1 == v2)
        if hasattr(comparison, "all"): # Handle cases like torch.Tensor or pandas
            return comparison.all()
        return bool(comparison)
    except (ValueError, TypeError):
        return False

def _is_pydantic_list(annotation: Any) -> bool:
    """Checks if a type annotation is a list of Pydantic models (handles Union and |)."""
    origin = get_origin(annotation)

    # 1. Check if the top-level origin is a list
    if origin is list or origin is list:
        args = get_args(annotation)
        if not args:
            return False

        inner_type = args[0]
        inner_origin = get_origin(inner_type)

        # 2. Check if inner type is a Union (e.g., list[MyModel | None])
        # types.UnionType covers the new 'A | B' syntax
        if inner_origin in (Union, types.UnionType):
            return any(
                isinstance(t, type) and issubclass(t, BaseModel)
                for t in get_args(inner_type)
            )

        # 3. Check if inner type is directly a BaseModel subclass
        return isinstance(inner_type, type) and issubclass(inner_type, BaseModel)

    return False

def _schema_aware_compare(inst1: Any, inst2: Any, path: str = "root", fn1="File1", fn2="File2"):
    """Recursively compares models. Recursively enters lists ONLY if they contain Pydantic models."""
    if isinstance(inst1, BaseModel) and isinstance(inst2, BaseModel):
        for name, field in inst1.model_fields.items():
            val1 = getattr(inst1, name, NOT_DEFINED)
            val2 = getattr(inst2, name, NOT_DEFINED)
            new_path = f"{path}.{name}"

            # 1. Check if it's a list of Pydantic Models
            if _is_pydantic_list(field.annotation):
                # Compare list lengths first
                l1 = 0 if val1 is NOT_DEFINED else len(val1)
                l2 = 0 if val2 is NOT_DEFINED else len(val2)

                if l1 != l2:
                    print(f"{new_path} (List Length): {l1} != {l2} in {fn1}|{fn2}")

                # Iterate through the items and recurse
                max_len = max(l1, l2)
                for i in range(max_len):
                    item_path = f"{new_path}[{i}]"
                    v1 = val1[i] if i < l1 else NOT_DEFINED
                    v2 = val2[i] if i < l2 else NOT_DEFINED

                    if isinstance(v1, BaseModel) and isinstance(v2, BaseModel):
                        _schema_aware_compare(v1, v2, item_path, fn1, fn2)
                    elif not are_equal(v1, v2):
                        print(f"{item_path}: {v1!r} != {v2!r} in {fn1}|{fn2}")
                continue

            # 2. Standard List or Atomic Type
            if get_origin(field.annotation) is list:
                if not are_equal(val1, val2):
                    print(f"{new_path} (List): {val1!r} != {val2!r} in {fn1}|{fn2}")
                continue

            # 3. Handle Nested Pydantic Models (Non-list)
            if isinstance(val1, BaseModel) and isinstance(val2, BaseModel):
                _schema_aware_compare(val1, val2, new_path, fn1, fn2)

            # 4. Direct Value Comparison
            elif not are_equal(val1, val2):
                print(f"{new_path}: {val1!r} != {val2!r} in {fn1}|{fn2}")

    elif not are_equal(inst1, inst2):
        print(f"{path}: {inst1!r} != {inst2!r}")

def load_pickle(file_path: str) -> BaseModel:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        if not isinstance(data, BaseModel):
            print(f"Error: {file_path} is not a Pydantic model.")
            sys.exit(1)
        return data

def main():
    parser = argparse.ArgumentParser(description="Compare two pickled Pydantic models.")
    parser.add_argument("file1", help="Path to first pickle")
    parser.add_argument("file2", help="Path to second pickle")
    args = parser.parse_args()

    model1 = load_pickle(args.file1)
    model2 = load_pickle(args.file2)

    print(f"Comparing {args.file1} vs {args.file2}...\n")
    _schema_aware_compare(model1, model2, fn1=args.file1, fn2=args.file2)

if __name__ == "__main__":
    main()
