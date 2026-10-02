"""Pure-standard-library source proxies for one-level Strassen product orders.

There are no timings or hardware-utilization estimates here. Product IDs are
one-based, matching p1..p7 in kernels_v001. Quadrants are zero-based row-major.
An order changes FP32 accumulation order; exact algebra does not imply identical
floating-point results. Keep this version immutable when adding new schedules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
import json
from pathlib import Path
from types import MappingProxyType


VERSION = "strassen_schedule_v001"
PRODUCT_IDS = (1, 2, 3, 4, 5, 6, 7)
CANONICAL_ORDER = PRODUCT_IDS
CURRENT_ORDER = (4, 6, 5, 2, 7, 3, 1)
INTERPRETATION = (
    "Source-order hypotheses only. Lower proxy scores do not establish lower "
    "latency, compiler scheduling, physical concurrency, or numerical safety. "
    "All permutations retain seven dots, five A and five B combinations, and "
    "twelve quadrant accumulator updates per tile-panel."
)


@dataclass(frozen=True)
class Product:
    product_id: int
    lhs: tuple[tuple[int, int], ...]
    rhs: tuple[tuple[int, int], ...]
    updates: tuple[tuple[int, int], ...]

    @property
    def targets(self):
        return frozenset(q for q, _ in self.updates)

    @property
    def preadd_count(self):
        return len(self.lhs) + len(self.rhs) - 2


# Operand terms are (quadrant, sign); output updates are (quadrant, sign).
# Each two-term input expression is explicitly rounded to BF16 in the kernel.
PRODUCTS = MappingProxyType({
    1: Product(1, ((0, 1), (3, 1)), ((0, 1), (3, 1)), ((0, 1), (3, 1))),
    2: Product(2, ((2, 1), (3, 1)), ((0, 1),), ((2, 1), (3, -1))),
    3: Product(3, ((0, 1),), ((1, 1), (3, -1)), ((1, 1), (3, 1))),
    4: Product(4, ((3, 1),), ((2, 1), (0, -1)), ((0, 1), (2, 1))),
    5: Product(5, ((0, 1), (1, 1)), ((3, 1),), ((0, -1), (1, 1))),
    6: Product(6, ((2, 1), (0, -1)), ((0, 1), (1, 1)), ((3, 1),)),
    7: Product(7, ((1, 1), (3, -1)), ((2, 1), (3, 1)), ((0, 1),)),
})


def validate_order(order):
    """Return a tuple containing every one-based product ID exactly once."""
    try:
        result = tuple(order)
    except TypeError as exc:
        raise ValueError("Order must be a permutation of integer IDs 1..7") from exc
    if (len(result) != 7 or any(type(p) is not int for p in result)
            or tuple(sorted(result)) != PRODUCT_IDS):
        raise ValueError("Order must be a permutation of integer IDs 1..7")
    return result


def schedule_id(order):
    return VERSION + "_p" + "".join(map(str, validate_order(order)))


@dataclass(frozen=True)
class Schedule:
    alias: str
    order: tuple[int, ...]
    rationale: str

    def __post_init__(self):
        object.__setattr__(self, "order", validate_order(self.order))

    @property
    def schedule_id(self):
        return schedule_id(self.order)


# Exact orders are fixed constants, not recomputed from a mutable search policy.
REGISTRY = (
    Schedule("canonical", CANONICAL_ORDER, "Canonical source p1..p7 control."),
    Schedule("current", CURRENT_ORDER, "Previously measured interleaved source order."),
    Schedule("proxy_lexicographic", (1, 2, 7, 3, 4, 6, 5),
             "Lexicographically first order minimizing proxy_key over all 5040."),
    Schedule("greedy_from_p1", (1, 2, 5, 6, 4, 3, 7),
             "Deterministic local dependency greedy search starting at p1."),
    Schedule("reverse_current", (1, 3, 7, 2, 5, 6, 4),
             "Reverse current order: same cyclic proxies, different accumulation order."),
    Schedule("diverse_proxy_tie", (6, 4, 1, 3, 7, 2, 5),
             "Among proxy minima, maximizes minimum Kendall distance from the first "
             "five registry orders; lexicographic tie break."),
)
SCHEDULES = MappingProxyType({s.schedule_id: s for s in REGISTRY})
SCHEDULE_ALIASES = MappingProxyType({s.alias: s.schedule_id for s in REGISTRY})
DEFAULT_SCHEDULE_ID = schedule_id(CURRENT_ORDER)


def get_schedule(name_or_id):
    """Resolve a registered alias or immutable ID; never silently fall back."""
    try:
        return SCHEDULES[SCHEDULE_ALIASES.get(name_or_id, name_or_id)]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown registered schedule: {name_or_id!r}") from exc


def all_orders():
    """Yield all 7! distinct orders in lexicographic order."""
    return permutations(PRODUCT_IDS)


def _fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def score_order(order):
    """Count source dependency spacing and pre-add bursts, without a time model.

    A gap is distance in product positions between consecutive updates to one
    accumulator quadrant. Cyclic gaps also include the last-to-first update
    across repeated K panels. Inverse-gap sums penalize closely spaced updates.
    Pre-add burst windows count expressions, not equally costly instructions:
    A and B expression sizes differ when BM != BN. No order changes their total.
    """
    order = validate_order(order)
    shared = lambda a, b: len(PRODUCTS[a].targets & PRODUCTS[b].targets)
    linear_overlap = sum(shared(a, b) for a, b in zip(order, order[1:]))
    boundary_overlap = shared(order[-1], order[0])
    linear_gaps, cyclic_gaps = {}, {}
    for q in range(4):
        positions = [i for i, p in enumerate(order) if q in PRODUCTS[p].targets]
        gaps = [b - a for a, b in zip(positions, positions[1:])]
        linear_gaps[str(q)] = gaps
        cyclic_gaps[str(q)] = gaps + [7 + positions[0] - positions[-1]]
    penalty = sum((Fraction(1, gap) for gaps in cyclic_gaps.values()
                   for gap in gaps), Fraction())
    preadds = [PRODUCTS[p].preadd_count for p in order]
    burst = lambda width: max(sum(preadds[(i + j) % 7] for j in range(width))
                             for i in range(7))
    return {
        "schedule_id": schedule_id(order), "order": list(order),
        "linear_adjacent_shared_quadrants": linear_overlap,
        "panel_boundary_shared_quadrants": boundary_overlap,
        "cyclic_adjacent_shared_quadrants": linear_overlap + boundary_overlap,
        "linear_update_gaps_by_quadrant": linear_gaps,
        "cyclic_update_gaps_by_quadrant": cyclic_gaps,
        "cyclic_inverse_gap_penalty": _fraction_record(penalty),
        "cyclic_preadd_burst_2_products": burst(2),
        "cyclic_preadd_burst_3_products": burst(3),
        "preadd_expressions_by_product_position": preadds,
        "a_preadd_expressions": 5, "b_preadd_expressions": 5,
        "quadrant_accumulator_updates": 12, "dots_per_panel": 7,
    }


def _key_from_score(score):
    inverse = score["cyclic_inverse_gap_penalty"]
    return (score["cyclic_adjacent_shared_quadrants"],
            Fraction(inverse["numerator"], inverse["denominator"]),
            score["cyclic_preadd_burst_2_products"],
            score["cyclic_preadd_burst_3_products"])


def proxy_key(order):
    """Lexicographic hypothesis score; this is not a predicted latency."""
    return _key_from_score(score_order(order))


def greedy_order(start=1):
    """Choose least shared targets with last, then penultimate product.

    Next tie breaks are expression count and product ID. This local heuristic
    does not optimize the panel boundary and is not a hardware scheduler.
    """
    if type(start) is not int or start not in PRODUCT_IDS:
        raise ValueError("start must be one integer product ID in 1..7")
    result = [start]
    while len(result) < 7:
        def key(p):
            return (len(PRODUCTS[p].targets & PRODUCTS[result[-1]].targets),
                    len(PRODUCTS[p].targets & PRODUCTS[result[-2]].targets)
                    if len(result) > 1 else 0,
                    PRODUCTS[p].preadd_count, p)
        result.append(min(set(PRODUCT_IDS) - set(result), key=key))
    return tuple(result)


def kendall_distance(first, second):
    """Number of unordered product pairs appearing in opposite relative order."""
    first, second = validate_order(first), validate_order(second)
    positions = {p: i for i, p in enumerate(second)}
    return sum(positions[first[i]] > positions[first[j]]
               for i, j in combinations(range(7), 2))


def source_costs(tile):
    """Exact operation counts for any positive even algebraic (BM,BN,BK).

    This is not a hardware tile validator or a memory-traffic estimator. The
    explicit BF16 cast count does not imply a separately executed instruction.
    """
    try:
        bm, bn, bk = tile
    except (TypeError, ValueError) as exc:
        raise ValueError("Tile must contain three positive even integers") from exc
    if any(type(b) is not int or b <= 0 or b % 2 for b in (bm, bn, bk)):
        raise ValueError("Tile must contain three positive even integers")
    a = 5 * (bm // 2) * (bk // 2)
    b = 5 * (bk // 2) * (bn // 2)
    updates = 12 * (bm // 2) * (bn // 2)
    cubic_updates = bm * bn
    dot = 7 * 2 * (bm // 2) * (bn // 2) * (bk // 2)
    cubic_dot = 2 * bm * bn * bk
    ratio = Fraction(a + b + updates - cubic_updates, cubic_dot - dot)
    return {
        "tile_bm_bn_bk": [bm, bn, bk],
        "a_preadd_element_operations": a, "b_preadd_element_operations": b,
        "explicit_bf16_preadd_result_elements": a + b,
        "strassen_accumulator_element_operations": updates,
        "cubic_accumulator_element_operations": cubic_updates,
        "strassen_dot_flops": dot, "cubic_dot_flops": cubic_dot,
        "extra_vector_element_operations": a + b + updates - cubic_updates,
        "saved_dot_flops": cubic_dot - dot,
        "extra_vector_per_saved_dot": _fraction_record(ratio),
        "estimand": "Source element-operation counts per padded tile-panel; "
                    "excludes initialization, copies, stores, compiler instructions "
                    "and scheduling. Invariant under all product permutations.",
    }


def enumeration_report(include_orders=False):
    """Enumerate 5040 orders once; produce a reproducible non-timing ledger."""
    scores = [score_order(order) for order in all_orders()]
    keys = [_key_from_score(score) for score in scores]
    best = min(keys)
    overlap_counts = Counter(s["cyclic_adjacent_shared_quadrants"] for s in scores)
    result = {
        "version": VERSION, "interpretation": INTERPRETATION,
        "product_id_convention": "one-based p1..p7; quadrants zero-based row-major",
        "permutation_count": len(scores),
        "cyclic_overlap_distribution": dict(sorted(overlap_counts.items())),
        "minimum_cyclic_adjacent_shared_quadrants": best[0],
        "orders_attaining_minimum_cyclic_overlap": overlap_counts[best[0]],
        "minimum_inverse_gap_penalty": _fraction_record(min(k[1] for k in keys)),
        "orders_tied_on_full_proxy_key": sum(k == best for k in keys),
        "current_attains_full_proxy_minimum": proxy_key(CURRENT_ORDER) == best,
        "registry": [{"alias": s.alias, "rationale": s.rationale,
                      **score_order(s.order)} for s in REGISTRY],
        "new_device_measurements": 0,
    }
    if include_orders:
        result["all_orders"] = scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-orders", action="store_true",
                        help="Include every order's source scores in JSON")
    parser.add_argument("--output", type=Path,
                        help="Create a new JSON file exclusively; otherwise stdout")
    args = parser.parse_args()
    report = enumeration_report(include_orders=args.all_orders)
    if args.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
