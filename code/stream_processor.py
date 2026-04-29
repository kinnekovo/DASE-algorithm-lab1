"""
Stream processor for the Gowalla check-in data stream.

Maintains two parallel sets of structures:

  Exact (ground truth):
    - user_count  : dict[uid  -> int]  – precise per-user check-in count
    - grid_count  : dict[grid -> int]  – precise per-cell check-in count

  Sketch (data-stream algorithms):
    - BF_user  / BF_grid  : Bloom Filter  – membership queries
    - CMS_user / CMS_grid : Count-Min Sketch – frequency estimates
    - MG_user  / MG_grid  : Misra-Gries   – Top-K candidates

Checkpoint evaluation (every `checkpoint_every` records):
  Bloom  → FPR  (false-positive rate on sampled negative keys)
  CMS    → MAE, MRE  (mean absolute / relative error on sampled keys)
  Top-K  → Recall@K, Precision@K, Jaccard@K
           (exact Top-K vs. MG candidates re-ranked by CMS estimates)
Results are appended to `output_dir/checkpoints.csv`.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from sketches.bloom import BloomFilter
from sketches.cms import CountMinSketch
from sketches.misra_gries import MisraGries
from grid import compute_grid_id, grid_id_to_str, grid_str_to_id, grid_id_to_range


# ---------------------------------------------------------------------------
# Line parser
# ---------------------------------------------------------------------------

def _parse_line(line: str) -> Optional[Tuple[str, str, float, float, str]]:
    """
    Parse one tab-separated check-in record.

    Returns (uid, timestamp, lat, lon, loc_id) or None if malformed.
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    uid, ts, lat_s, lon_s, loc_id = parts[:5]
    try:
        lat = float(lat_s)
        lon = float(lon_s)
    except ValueError:
        return None
    return uid, ts, lat, lon, loc_id


# ---------------------------------------------------------------------------
# StreamProcessor
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "checkpoint",
    "records_processed",
    "bf_user_fpr",
    "bf_grid_fpr",
    "cms_user_mae",
    "cms_user_mre",
    "cms_grid_mae",
    "cms_grid_mre",
    "topk_user_recall",
    "topk_user_precision",
    "topk_user_jaccard",
    "topk_grid_recall",
    "topk_grid_precision",
    "topk_grid_jaccard",
]


class StreamProcessor:
    """
    Processes a sorted check-in data stream and evaluates sketch accuracy.
    """

    def __init__(
        self,
        grid_step: float = 0.001,
        topk: int = 10,
        checkpoint_every: int = 100_000,
        sample_size: int = 200,
        output_dir: str = "output",
        # Bloom Filter parameters
        bf_capacity: int = 300_000,
        bf_fpr: float = 0.01,
        # Count-Min Sketch parameters
        cms_width: int = 20_000,
        cms_depth: int = 7,
        # Misra-Gries parameters
        mg_k: int = 500,
    ):
        self.grid_step = grid_step
        self.topk = topk
        self.checkpoint_every = checkpoint_every
        self.sample_size = sample_size
        self.output_dir = output_dir

        # ---- Exact structures (ground truth) ----
        self.user_count: Dict[str, int] = {}
        self.grid_count: Dict[str, int] = {}

        # ---- Sketch structures ----
        self.bf_user = BloomFilter(capacity=bf_capacity, fpr=bf_fpr)
        # Grid cells are far more numerous than distinct users (up to ~N cells
        # for fine-grained steps), so allocate a proportionally larger filter.
        self.bf_grid = BloomFilter(capacity=bf_capacity * 10, fpr=bf_fpr)
        self.cms_user = CountMinSketch(width=cms_width, depth=cms_depth)
        self.cms_grid = CountMinSketch(width=cms_width, depth=cms_depth)
        self.mg_user = MisraGries(k=mg_k)
        self.mg_grid = MisraGries(k=mg_k)

        # ---- Runtime state ----
        self.total_records: int = 0
        self._checkpoint_index: int = 0

        # ---- Output ----
        os.makedirs(output_dir, exist_ok=True)
        self._csv_path = os.path.join(output_dir, "checkpoints.csv")
        self._init_csv()

    # ------------------------------------------------------------------
    # CSV initialisation
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_FIELDS)

    # ------------------------------------------------------------------
    # Stream processing
    # ------------------------------------------------------------------

    def process_line(self, line: str) -> None:
        """Process one raw text line from the stream."""
        parsed = _parse_line(line)
        if parsed is None:
            return
        uid, _ts, lat, lon, _loc = parsed

        grid_tuple = compute_grid_id(lat, lon, self.grid_step)
        grid_key = grid_id_to_str(grid_tuple)

        # --- Exact update ---
        self.user_count[uid] = self.user_count.get(uid, 0) + 1
        self.grid_count[grid_key] = self.grid_count.get(grid_key, 0) + 1

        # --- Sketch update ---
        self.bf_user.add(uid)
        self.bf_grid.add(grid_key)
        self.cms_user.add(uid)
        self.cms_grid.add(grid_key)
        self.mg_user.add(uid)
        self.mg_grid.add(grid_key)

        self.total_records += 1

        # --- Periodic checkpoint ---
        if self.total_records % self.checkpoint_every == 0:
            self._run_checkpoint()

    def process_file(self, path: str) -> None:
        """Read a sorted check-in file line by line and process every record."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.process_line(line)

    # ------------------------------------------------------------------
    # Checkpoint evaluation
    # ------------------------------------------------------------------

    def run_final_checkpoint(self) -> None:
        """
        Force a checkpoint evaluation after all records have been processed.

        This is useful when the total record count is not a multiple of
        `checkpoint_every` (e.g., for the small sample data.txt).
        """
        self._run_checkpoint()

    def _run_checkpoint(self) -> None:
        """Evaluate all sketches against exact structures; append to CSV."""
        self._checkpoint_index += 1
        idx = self._checkpoint_index

        bf_user_fpr = self._eval_bloom_fpr(self.bf_user, self.user_count)
        bf_grid_fpr = self._eval_bloom_fpr(self.bf_grid, self.grid_count)

        cms_user_mae, cms_user_mre = self._eval_cms(self.cms_user, self.user_count)
        cms_grid_mae, cms_grid_mre = self._eval_cms(self.cms_grid, self.grid_count)

        tu_recall, tu_prec, tu_jacc = self._eval_topk(
            self.cms_user, self.mg_user, self.user_count
        )
        tg_recall, tg_prec, tg_jacc = self._eval_topk(
            self.cms_grid, self.mg_grid, self.grid_count
        )

        row = {
            "checkpoint": idx,
            "records_processed": self.total_records,
            "bf_user_fpr": round(bf_user_fpr, 6),
            "bf_grid_fpr": round(bf_grid_fpr, 6),
            "cms_user_mae": round(cms_user_mae, 4),
            "cms_user_mre": round(cms_user_mre, 6),
            "cms_grid_mae": round(cms_grid_mae, 4),
            "cms_grid_mre": round(cms_grid_mre, 6),
            "topk_user_recall": round(tu_recall, 4),
            "topk_user_precision": round(tu_prec, 4),
            "topk_user_jaccard": round(tu_jacc, 4),
            "topk_grid_recall": round(tg_recall, 4),
            "topk_grid_precision": round(tg_prec, 4),
            "topk_grid_jaccard": round(tg_jacc, 4),
        }

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

        print(
            f"[Checkpoint {idx}] records={self.total_records:,}  "
            f"bf_user_fpr={bf_user_fpr:.4f}  "
            f"cms_user_mae={cms_user_mae:.2f}  cms_user_mre={cms_user_mre:.4f}  "
            f"topk_user_recall={tu_recall:.4f}  topk_grid_recall={tg_recall:.4f}"
        )

    # ------------------------------------------------------------------
    # Bloom Filter evaluation: False Positive Rate
    # ------------------------------------------------------------------

    def _eval_bloom_fpr(
        self, bf: BloomFilter, exact: Dict[str, int]
    ) -> float:
        """
        Estimate the false positive rate of a Bloom Filter.

        Positive set  : keys sampled from `exact` (should have 0 FN).
        Negative set  : synthetic absent keys (used to measure FPR).

        The FPR is FP / (FP + TN) over the negative sample.
        """
        n = min(self.sample_size, max(1, len(exact)))

        # Verify no false negatives (for diagnostic purposes)
        pos_keys = random.sample(sorted(exact.keys()), min(n, len(exact)))
        fn_count = sum(1 for k in pos_keys if k not in bf)
        if fn_count > 0:
            print(
                f"  [Warning] Bloom Filter has {fn_count} false negatives "
                f"(unexpected — check implementation)."
            )

        # Generate absent keys and count false positives.
        # n * 200 attempts is generous: even if 50 % of the candidate space
        # collides with existing keys, we will still collect n samples quickly.
        neg_keys: List[str] = []
        rng = random.Random(42)
        attempts = 0
        while len(neg_keys) < n and attempts < n * 200:
            candidate = f"__absent_{rng.randint(0, 10**9)}__"
            if candidate not in exact:
                neg_keys.append(candidate)
            attempts += 1

        if not neg_keys:
            return 0.0

        fp = sum(1 for k in neg_keys if k in bf)
        return fp / len(neg_keys)

    # ------------------------------------------------------------------
    # CMS evaluation: MAE and MRE
    # ------------------------------------------------------------------

    def _eval_cms(
        self, cms: CountMinSketch, exact: Dict[str, int]
    ) -> Tuple[float, float]:
        """
        Evaluate CMS accuracy on a random sample of known keys.

        Returns:
            (MAE, MRE) – mean absolute error and mean relative error.
            MRE is computed only for keys with true count > 0.
        """
        if not exact:
            return 0.0, 0.0

        keys = random.sample(sorted(exact.keys()), min(self.sample_size, len(exact)))
        abs_errors: List[float] = []
        rel_errors: List[float] = []

        for k in keys:
            true_val = exact[k]
            est_val = cms.query(k)
            err = abs(est_val - true_val)
            abs_errors.append(float(err))
            if true_val > 0:
                rel_errors.append(err / true_val)

        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        mre = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0
        return mae, mre

    # ------------------------------------------------------------------
    # Top-K evaluation: Recall@K, Precision@K, Jaccard@K
    # ------------------------------------------------------------------

    def _eval_topk(
        self,
        cms: CountMinSketch,
        mg: MisraGries,
        exact: Dict[str, int],
    ) -> Tuple[float, float, float]:
        """
        Compare sketch Top-K against exact Top-K.

        Exact Top-K  : top K keys by true count in `exact`.
        Sketch Top-K : MG candidates re-ranked by CMS estimates, top K.

        Returns:
            (Recall@K, Precision@K, Jaccard@K)
        """
        K = min(self.topk, len(exact))
        if K == 0:
            return 0.0, 0.0, 0.0

        # Ground-truth Top-K
        exact_topk = set(
            k for k, _ in sorted(exact.items(), key=lambda x: -x[1])[:K]
        )

        # Sketch Top-K: use MG candidate keys, score with CMS, take top K
        candidates = mg.candidates()
        if not candidates:
            return 0.0, 0.0, 0.0

        scored = sorted(candidates, key=lambda k: -cms.query(k))
        sketch_topk = set(scored[:K])

        intersection = exact_topk & sketch_topk
        recall = len(intersection) / len(exact_topk) if exact_topk else 0.0
        precision = len(intersection) / len(sketch_topk) if sketch_topk else 0.0
        union = exact_topk | sketch_topk
        jaccard = len(intersection) / len(union) if union else 0.0
        return recall, precision, jaccard

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_user_stats(self) -> dict:
        """Return exact user statistics (distinct users, max user(s))."""
        if not self.user_count:
            return {"n_users": 0, "max_count": 0, "top_users": []}
        max_count = max(self.user_count.values())
        top_users = sorted(u for u, c in self.user_count.items() if c == max_count)
        return {
            "n_users": len(self.user_count),
            "max_count": max_count,
            "top_users": top_users,
        }

    def get_all_user_counts(self) -> List[Tuple[str, int]]:
        """Return all (uid, count) pairs sorted by count descending."""
        return sorted(self.user_count.items(), key=lambda x: -x[1])

    def get_grid_topk(self) -> List[dict]:
        """Return exact Top-K grid cells with coordinate ranges and counts."""
        K = self.topk
        sorted_grids = sorted(self.grid_count.items(), key=lambda x: -x[1])[:K]
        result = []
        for grid_key, count in sorted_grids:
            try:
                lat_idx, lon_idx = grid_str_to_id(grid_key)
            except (ValueError, IndexError):
                continue
            rng = grid_id_to_range((lat_idx, lon_idx), self.grid_step)
            result.append(
                {
                    "grid_key": grid_key,
                    "lat_range": f"[{rng['lat_min']:.6f}, {rng['lat_max']:.6f})",
                    "lon_range": f"[{rng['lon_min']:.6f}, {rng['lon_max']:.6f})",
                    "count": count,
                }
            )
        return result

    def get_sketch_topk_user(self) -> List[Tuple[str, int]]:
        """Return sketch Top-K users (MG candidates scored by CMS)."""
        candidates = self.mg_user.candidates()
        scored = sorted(candidates, key=lambda k: -self.cms_user.query(k))
        return [(k, self.cms_user.query(k)) for k in scored[: self.topk]]

    def get_sketch_topk_grid(self) -> List[Tuple[str, int]]:
        """Return sketch Top-K grid cells (MG candidates scored by CMS)."""
        candidates = self.mg_grid.candidates()
        scored = sorted(candidates, key=lambda k: -self.cms_grid.query(k))
        return [(k, self.cms_grid.query(k)) for k in scored[: self.topk]]
