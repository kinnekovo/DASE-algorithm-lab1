"""
External merge sort for large check-in files.

Sorts records by the ISO 8601 timestamp field (column index 1, tab-separated)
in ascending order without loading the entire file into memory.

Algorithm:
  Phase 1 – produce sorted "runs":
      Read the input in chunks of `chunk_size` lines, sort each chunk in
      memory, and write it to a temporary file.
  Phase 2 – k-way merge:
      Use a min-heap to merge all temporary run files, emitting lines in
      timestamp order to the final output file.

Complexity:
  Time:   O(N log N)  (dominated by the final k-way merge)
  Memory: O(chunk_size + num_runs)
"""

import heapq
import os
import tempfile
from typing import List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _timestamp(line: str) -> str:
    """
    Extract the timestamp from a tab-separated check-in line.

    ISO 8601 timestamps ("2009-02-04T00:09:19Z") sort lexicographically in
    the same order as chronologically, so a plain string comparison suffices.
    """
    parts = line.split("\t", 2)
    return parts[1] if len(parts) >= 2 else ""


def _write_sorted_run(lines: List[str], tmp_dir: str) -> str:
    """Sort *lines* by timestamp, write to a temp file, return its path."""
    lines.sort(key=_timestamp)
    fd, path = tempfile.mkstemp(dir=tmp_dir, suffix=".run")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def external_sort(
    input_path: str,
    output_path: str,
    chunk_size: int = 200_000,
    tmp_dir: Optional[str] = None,
) -> None:
    """
    Sort *input_path* by timestamp and write the result to *output_path*.

    Args:
        input_path:  path to the raw (unsorted) check-in file.
        output_path: path where the sorted file will be written.
        chunk_size:  number of lines per in-memory sort chunk.
                     Increase for faster sorting on machines with more RAM.
        tmp_dir:     directory for temporary run files (default: system temp).
    """
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()

    # Ensure the output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    run_paths: List[str] = []

    # -----------------------------------------------------------------------
    # Phase 1: read in chunks, sort each chunk, write run files
    # -----------------------------------------------------------------------
    with open(input_path, "r", encoding="utf-8") as fin:
        chunk: List[str] = []
        for raw_line in fin:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            chunk.append(line)
            if len(chunk) >= chunk_size:
                run_paths.append(_write_sorted_run(chunk, tmp_dir))
                chunk = []
        if chunk:
            run_paths.append(_write_sorted_run(chunk, tmp_dir))

    if not run_paths:
        # Empty input – create empty output file
        open(output_path, "w").close()
        return

    # -----------------------------------------------------------------------
    # Phase 2: k-way merge using a min-heap
    # -----------------------------------------------------------------------
    file_handles = [open(p, "r", encoding="utf-8") for p in run_paths]
    try:
        # Heap entries: (timestamp_str, line_str, file_index)
        heap: list = []
        for i, fh in enumerate(file_handles):
            line = fh.readline().rstrip("\n")
            if line:
                heapq.heappush(heap, (_timestamp(line), line, i))

        with open(output_path, "w", encoding="utf-8") as fout:
            while heap:
                _, line, i = heapq.heappop(heap)
                fout.write(line + "\n")
                next_raw = file_handles[i].readline()
                if next_raw:
                    next_line = next_raw.rstrip("\n")
                    heapq.heappush(heap, (_timestamp(next_line), next_line, i))
    finally:
        for fh in file_handles:
            fh.close()
        for path in run_paths:
            try:
                os.remove(path)
            except OSError:
                pass
