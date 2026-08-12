import argparse
import h5py
import numpy as np
import json
import re
import pydoc
import yaml
from io import StringIO


np.set_printoptions(linewidth=200)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("prefix", nargs="?")
    parser.add_argument("--auto-page", choices=["auto", "never", "always"], default="auto")
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Do not print values of datasets when printing groups",
    )

    return parser.parse_args()


def auto_print(s: str):
    if len(s) > 300 or s.count("\n") >= 20:
        pydoc.pager(s)
    else:
        print(s, end="")


def natural_sort(l):
    # taken from https://stackoverflow.com/a/4836734
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def load_dicts_data(data: h5py.Dataset) -> list[dict]:
    ret = []
    for i in range(data.shape[0]):
        d = json.loads(data[i].tobytes().decode("utf-8").rstrip("\x00"))
        ret.append(d)
    return ret


def build_group_tree(args, group: h5py.Group | h5py.File):
    ret = {}
    for key in natural_sort(group.keys()):
        if isinstance(group[key], h5py.Group):
            ret[key] = build_group_tree(args, group[key])
        elif isinstance(group[key], h5py.Dataset):
            ret[key] = {
                "shape": str(group[key].shape),
                "dtype": str(group[key].dtype),
            }
            if group[key].dtype == np.object_ and not args.no_values:
                try:
                    ret[key]["value"] = group[key].astype("T")[()]
                except Exception:
                    pass
    return ret


def print_group(args, file: h5py.File):
    file_or_group = file[args.prefix] if args.prefix else file
    tree = build_group_tree(args, file_or_group)
    args.print_or_page(yaml.dump(tree, indent=2, sort_keys=False))


def print_dataset(args, file: h5py.File):
    dataset = file[args.prefix]
    buf = StringIO()
    buf.write(f"{args.data_path}:{args.prefix}\n")

    if dataset.dtype == np.uint8:
        if dataset.ndim == 1:
            buf.write(dataset[()].tobytes().decode("utf-8").rstrip("\x00"))
            buf.write("\n")
        elif dataset.ndim == 2:
            try:
                dicts = load_dicts_data(dataset)
                buf.write(f"{len(dicts)} elements\n")
                for i, d in enumerate(dicts):
                    buf.write(f"Element {i}:\n")
                    buf.write(json.dumps(d, indent=2))
                    buf.write("\n")
            except:
                for i in range(dataset.shape[0]):
                    buf.write(dataset[i].tobytes().decode("utf-8").rstrip("\x00"))
                    buf.write("\n")
        else:
            buf.write(
                f"Unsure how to print dataset of shape {dataset.shape} with dtype {dataset.dtype}\n"
            )
    else:
        buf.write(str(dataset[()]))
        buf.write("\n")
    args.print_or_page(buf.getvalue())


def main() -> None:
    args = get_args()

    if args.auto_page == "auto":
        args.print_or_page = auto_print
    elif args.auto_page == "never":
        args.print_or_page = lambda s: print(s, end="")
    elif args.auto_page == "always":
        args.print_or_page = pydoc.pager
    else:
        raise ValueError(f"Invalid auto-page value: {args.auto_page}")

    with h5py.File(args.data_path, "r") as file:
        if args.prefix is not None and args.prefix not in file:
            raise ValueError(f"Key '{args.prefix}' not found in {args.data_path}")
        if args.prefix is None or isinstance(file[args.prefix], h5py.Group):
            print_group(args, file)
        elif isinstance(file[args.prefix], h5py.Dataset):
            print_dataset(args, file)
        else:
            raise ValueError(f"Unknown type: {type(file[args.prefix])}")


if __name__ == "__main__":
    main()
