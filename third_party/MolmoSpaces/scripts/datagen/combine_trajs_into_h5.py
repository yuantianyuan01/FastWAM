"""
Command-line script to combine trajectory files using thor_analysis module.

Usage:
    python combine_trajectories.py <folder_path> [--output OUTPUT] [--first-n N]

Examples:
    python combine_trajectories.py /weka/prior-default/aguru/datasets/pi_hard_bench_0108
    python combine_trajectories.py /weka/prior-default/aguru/datasets/pi_hard_bench_0108 --output combined.h5
    python combine_trajectories.py /weka/prior-default/aguru/datasets/pi_hard_bench_0108 --output combined.h5 --first-n 100
"""

import argparse
import sys
from pathlib import Path
import importlib.util

script_dir = Path(__file__).parent
thor_analysis_path = script_dir.parent / "paper_plots" / "thor_analysis.py"

if not thor_analysis_path.exists():
    print(f"Error: Could not find thor_analysis.py at {thor_analysis_path}")
    sys.exit(1)

# Load the module dynamically
spec = importlib.util.spec_from_file_location("thor_analysis", thor_analysis_path)
thor_analysis = importlib.util.module_from_spec(spec)
sys.modules["thor_analysis"] = thor_analysis
spec.loader.exec_module(thor_analysis)

def main():
    parser = argparse.ArgumentParser(
        description="Combine trajectory files from a folder into a single HDF5 file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the folder containing trajectory files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='combined_trajectories.h5',
        help='Output HDF5 file name (default: combined_trajectories.h5)'
    )

    parser.add_argument(
        '-n', '--first-n',
        type=int,
        default=None,
        help='Only combine the first N trajectories (default: combine all)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Validate folder path
    folder = Path(args.folder_path)
    if not folder.exists():
        print(f"Error: Folder '{args.folder_path}' does not exist.")
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: '{args.folder_path}' is not a directory.")
        sys.exit(1)

    # Print configuration if verbose
    if args.verbose:
        print(f"Configuration:")
        print(f"  Input folder: {args.folder_path}")
        print(f"  Output file:  {args.output}")
        print(f"  First N:      {args.first_n if args.first_n else 'All'}")
        print()

    # Call the combine function
    try:
        output_path = thor_analysis.combine_all_trajectories(
            folder_path=args.folder_path,
            output_file=args.output,
            first_n=args.first_n
        )

        if output_path:
            print(f"\nSuccess! Combined trajectories saved to: {output_path}")
        else:
            print("\nWarning: combine_all_trajectories returned None")

    except Exception as e:
        print(f"\nError during trajectory combination: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
