"""
Main entry point for Experiment 1: Data Stream Statistics Algorithms.

Usage (quick start with the sample data):
    cd <repo_root>
    python code/main.py --input data.txt

Usage (full pipeline with sorting):
    python code/main.py --input Gowalla_totalCheckins.txt --do-sort \
        --sorted-output data_sorted.txt --grid-step 0.001 --topk 10 \
        --checkpoint-every 100000 --output-dir output

Run `python code/main.py --help` for all options.
"""

import argparse
import os
import sys

# Ensure the `code/` directory is on the Python path so sibling modules
# (sketches/, grid.py, sort_external.py, stream_processor.py) are importable
# whether the script is run as `python code/main.py` or `python main.py`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sort_external import external_sort
from stream_processor import StreamProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Gowalla check-in data stream statistics using "
            "Bloom Filter, Count-Min Sketch, and Misra-Gries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument(
        "--input", default="data.txt",
        help="Path to the raw (or pre-sorted) check-in data file.",
    )
    p.add_argument(
        "--sorted-output", default=None,
        help=(
            "Path for the sorted output file. "
            "Required when --do-sort is set; if omitted, defaults to "
            "<input>.sorted."
        ),
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory for checkpoint CSV and any other output files.",
    )

    # Preprocessing
    p.add_argument(
        "--do-sort", action="store_true",
        help=(
            "Sort the input file by timestamp before processing. "
            "Uses external merge sort so the full file never enters RAM."
        ),
    )

    # Grid
    p.add_argument(
        "--grid-step", type=float, default=0.001,
        help="Grid cell size in degrees (0.001 ≈ 111 m at the equator).",
    )

    # Query / output
    p.add_argument("--topk", type=int, default=10, help="K for Top-K queries.")
    p.add_argument(
        "--print-all-users", action="store_true",
        help="Print the complete (uid, count) table to stdout.",
    )

    # Checkpoint
    p.add_argument(
        "--checkpoint-every", type=int, default=100_000,
        help="Trigger a sketch evaluation checkpoint every N records.",
    )
    p.add_argument(
        "--sample-size", type=int, default=200,
        help="Number of keys sampled per checkpoint evaluation.",
    )

    # Sketch parameters
    p.add_argument("--bf-capacity", type=int, default=300_000,
                   help="Bloom Filter expected capacity.")
    p.add_argument("--bf-fpr", type=float, default=0.01,
                   help="Bloom Filter target false positive rate.")
    p.add_argument("--cms-width", type=int, default=20_000,
                   help="Count-Min Sketch width (columns).")
    p.add_argument("--cms-depth", type=int, default=7,
                   help="Count-Min Sketch depth (rows / hash functions).")
    p.add_argument("--mg-k", type=int, default=500,
                   help="Misra-Gries counter table size.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    input_path = os.path.abspath(args.input)

    # -----------------------------------------------------------------------
    # Step 0: Pre-processing – sort by timestamp
    # -----------------------------------------------------------------------
    if args.do_sort:
        sorted_path = args.sorted_output or (input_path + ".sorted")
        sorted_path = os.path.abspath(sorted_path)
        print(f"[Sort] {input_path} → {sorted_path}")
        external_sort(input_path, sorted_path)
        print("[Sort] Done.")
        input_path = sorted_path
    elif args.sorted_output and os.path.exists(args.sorted_output):
        # Use a previously sorted file if it exists
        input_path = os.path.abspath(args.sorted_output)
        print(f"[Info] Using existing sorted file: {input_path}")

    # -----------------------------------------------------------------------
    # Step 1 & 2: Stream processing
    # -----------------------------------------------------------------------
    print(f"\n[Stream] Processing {input_path} …")
    processor = StreamProcessor(
        grid_step=args.grid_step,
        topk=args.topk,
        checkpoint_every=args.checkpoint_every,
        sample_size=args.sample_size,
        output_dir=os.path.abspath(args.output_dir),
        bf_capacity=args.bf_capacity,
        bf_fpr=args.bf_fpr,
        cms_width=args.cms_width,
        cms_depth=args.cms_depth,
        mg_k=args.mg_k,
    )
    processor.process_file(input_path)

    # Always run a final checkpoint so small datasets produce at least one row
    if processor.total_records % args.checkpoint_every != 0:
        print("\n[Checkpoint] Running final evaluation …")
        processor.run_final_checkpoint()

    # -----------------------------------------------------------------------
    # Output: User statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("=== Task 1: User Check-in Statistics (Exact) ===")
    stats = processor.get_user_stats()
    print(f"  Distinct users    : {stats['n_users']:,}")
    print(f"  Max check-in count: {stats['max_count']:,}")
    print(f"  Top user(s)       : {stats['top_users']}")

    if args.print_all_users:
        print("\n  --- All users (uid, count) sorted by count desc ---")
        for uid, cnt in processor.get_all_user_counts():
            print(f"    {uid}\t{cnt}")

    # -----------------------------------------------------------------------
    # Output: Grid Top-K (Exact)
    # -----------------------------------------------------------------------
    print(f"\n=== Task 2: Top-{args.topk} Grid Cells (Exact, step={args.grid_step}) ===")
    for i, g in enumerate(processor.get_grid_topk(), 1):
        print(
            f"  #{i:3d}  lat{g['lat_range']}  lon{g['lon_range']}  "
            f"count={g['count']:,}"
        )

    # -----------------------------------------------------------------------
    # Output: Sketch Top-K
    # -----------------------------------------------------------------------
    print(f"\n=== Task 1 Sketch: Top-{args.topk} Users (MG + CMS) ===")
    for i, (uid, est) in enumerate(processor.get_sketch_topk_user(), 1):
        true_cnt = processor.user_count.get(uid, 0)
        print(f"  #{i:3d}  user={uid:>8}  est={est:,}  true={true_cnt:,}")

    print(f"\n=== Task 2 Sketch: Top-{args.topk} Grids (MG + CMS) ===")
    for i, (gk, est) in enumerate(processor.get_sketch_topk_grid(), 1):
        true_cnt = processor.grid_count.get(gk, 0)
        print(f"  #{i:3d}  grid={gk}  est={est:,}  true={true_cnt:,}")

    # -----------------------------------------------------------------------
    # Output: Sketch structure summary
    # -----------------------------------------------------------------------
    print("\n=== Sketch Structure Summary ===")
    print(f"  {processor.bf_user}")
    print(f"  {processor.bf_grid}")
    print(f"  {processor.cms_user}")
    print(f"  {processor.cms_grid}")
    print(f"  {processor.mg_user}")
    print(f"  {processor.mg_grid}")

    csv_path = os.path.join(args.output_dir, "checkpoints.csv")
    print(f"\n[Done] Checkpoint CSV: {csv_path}")
    print(f"[Done] Total records processed: {processor.total_records:,}")


if __name__ == "__main__":
    main()
