"""
Count-Min Sketch (CMS) for frequency estimation in data streams.

Properties:
- Estimates freq(x) with an upper-bound guarantee:
      est(x) >= true_freq(x)   always
      est(x) <= true_freq(x) + eps * N   with probability >= 1 - delta
  where N is the total stream length, eps = e/width, delta = 2^(-depth).
- Updates and queries both run in O(depth) time.
- Space: O(width * depth) integer counters.

Recommended parameters for Gowalla (N ≈ 6.4M, eps=1e-4, delta=1e-5):
    width = ceil(e / 1e-4) ≈ 27 183
    depth = ceil(ln(1 / 1e-5)) ≈ 12
"""

import math
import hashlib
from typing import Any


class CountMinSketch:
    """Count-Min Sketch for frequency estimation."""

    def __init__(self, width: int = 20_000, depth: int = 7):
        """
        Args:
            width: number of columns (w). Larger → smaller additive error.
            depth: number of rows / hash functions (d). Larger → higher
                   confidence.  Failure probability ≈ 2^(-depth).
        """
        self.width = width
        self.depth = depth
        self.table: list[list[int]] = [[0] * width for _ in range(depth)]
        self._seeds: list[int] = [self._make_seed(i) for i in range(depth)]
        self.total: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_seed(i: int) -> int:
        digest = hashlib.md5(f"cms_row_{i}".encode()).hexdigest()
        return int(digest, 16)

    def _col(self, key: str, row: int) -> int:
        """Return column index for *key* in *row*."""
        # Built-in hash combined with a per-row seed – fast and uniform enough
        return (hash(key) ^ self._seeds[row]) % self.width

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, item: Any, count: int = 1) -> None:
        """Increment counters for *item* by *count*."""
        key = str(item)
        for row in range(self.depth):
            self.table[row][self._col(key, row)] += count
        self.total += count

    def query(self, item: Any) -> int:
        """Return the estimated frequency of *item* (upper bound)."""
        key = str(item)
        return min(self.table[row][self._col(key, row)] for row in range(self.depth))

    def memory_bytes(self) -> int:
        """Approximate memory usage of the counter table (8 bytes per int)."""
        return self.width * self.depth * 8

    @classmethod
    def from_error_params(cls, epsilon: float = 1e-4, delta: float = 1e-5) -> "CountMinSketch":
        """
        Construct a CMS from error parameters.

        Args:
            epsilon: additive error fraction of total count N.
            delta:   failure probability.
        """
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1.0 / delta))
        return cls(width=width, depth=depth)

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self.width}, depth={self.depth}, "
            f"total={self.total}, mem≈{self.memory_bytes() // 1024}KB)"
        )
