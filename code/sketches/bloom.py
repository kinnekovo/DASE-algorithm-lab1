"""
Bloom Filter for approximate membership queries.

Properties:
- No false negatives (if item was added, query always returns True).
- Possible false positives (item not added may still return True).
- Space-efficient: uses a bit array of size m with k hash functions.

Parameter selection:
    m = ceil(-n * ln(p) / ln(2)^2)   # optimal bit array size
    k = ceil((m / n) * ln(2))         # optimal number of hash functions
where n = expected number of elements, p = desired false positive rate.
"""

import math
import hashlib
from typing import Any


class BloomFilter:
    """Bloom Filter for approximate membership queries."""

    def __init__(self, capacity: int = 100_000, fpr: float = 0.01):
        """
        Args:
            capacity: expected number of distinct elements to insert.
            fpr:      desired false positive rate (e.g. 0.01 = 1 %).
        """
        self.capacity = capacity
        self.fpr = fpr

        # Optimal bit-array size and number of hash functions
        self.m = max(1, math.ceil(-capacity * math.log(fpr) / (math.log(2) ** 2)))
        self.k = max(1, math.ceil((self.m / capacity) * math.log(2)))

        # Bit array stored as a bytearray (1 byte = 8 bits)
        self._bits = bytearray(math.ceil(self.m / 8))
        self._count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hashes(self, item: str):
        """Yield k bit positions for *item* using double hashing."""
        h1 = int(hashlib.md5(item.encode("utf-8", errors="replace")).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode("utf-8", errors="replace")).hexdigest(), 16)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, item: Any) -> None:
        """Insert *item* into the filter."""
        key = str(item)
        for pos in self._hashes(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def __contains__(self, item: Any) -> bool:
        """Return True if *item* is (probably) in the filter."""
        key = str(item)
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._hashes(key)
        )

    @property
    def count(self) -> int:
        """Number of items that have been added (may include duplicates)."""
        return self._count

    def memory_bytes(self) -> int:
        """Size of the underlying bit array in bytes."""
        return len(self._bits)

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self.capacity}, fpr={self.fpr}, "
            f"m={self.m}, k={self.k}, mem={self.memory_bytes()}B)"
        )
