"""
Misra-Gries (Frequent) algorithm for Top-K heavy-hitter detection.

Properties:
- Maintains a summary of at most k counters.
- Guarantees: every item with true frequency > N / (k + 1) appears in the
  summary (where N is the stream length).
- Counter values are underestimates; the true frequency is in the range
  [counter, counter + N / (k + 1)].
- For Top-K output, set k >= 2 * K (larger k → better recall, more memory).
- Update and query both run in O(k) time worst-case, O(1) amortised.
"""

from typing import Any, Dict, List, Tuple


class MisraGries:
    """Misra-Gries algorithm for Top-K candidate detection."""

    def __init__(self, k: int = 200):
        """
        Args:
            k: maximum number of counters to maintain.
               To reliably find Top-K items, set k >= 2 * K.
        """
        self.k = k
        self.counters: Dict[str, int] = {}
        self.total: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, item: Any) -> None:
        """Process one item from the stream."""
        key = str(item)
        self.total += 1

        if key in self.counters:
            self.counters[key] += 1
        elif len(self.counters) < self.k:
            self.counters[key] = 1
        else:
            # Decrement all counters by 1 and remove zero-count entries
            to_delete = [k for k, v in self.counters.items() if v <= 1]
            for dk in to_delete:
                del self.counters[dk]
            for ck in list(self.counters):
                self.counters[ck] -= 1
            # The new item is effectively discarded in this reduction step

    def top_k(self, k: int = 10) -> List[Tuple[str, int]]:
        """
        Return the top-k candidates sorted by counter value (descending).

        Note: counter values are lower bounds; use CMS for better estimates.
        """
        return sorted(self.counters.items(), key=lambda x: -x[1])[:k]

    def candidates(self) -> List[str]:
        """Return all candidate keys currently maintained."""
        return list(self.counters.keys())

    def memory_entries(self) -> int:
        """Number of entries currently in the summary."""
        return len(self.counters)

    def __repr__(self) -> str:
        return (
            f"MisraGries(k={self.k}, entries={len(self.counters)}, "
            f"total={self.total})"
        )
