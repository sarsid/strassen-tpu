"""Exact algebra and combinatorial checks, not benchmark evidence."""
import ast
from collections import defaultdict
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from strassen_mm import strassen_schedule_v001 as schedules


class ScheduleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orders = list(schedules.all_orders())
        cls.keys = {o: schedules.proxy_key(o) for o in cls.orders}

    def test_enumeration_has_every_distinct_permutation(self):
        self.assertEqual(len(self.orders), 5040)
        self.assertEqual(len(set(self.orders)), 5040)
        self.assertEqual(self.orders, sorted(self.orders))
        self.assertEqual(len({schedules.schedule_id(o) for o in self.orders}), 5040)

    def test_invalid_orders_and_unknown_names_are_rejected(self):
        bad = (None, "1234567", (1, 2), (0, 1, 2, 3, 4, 5, 6),
               (1, 2, 3, 4, 5, 6, 6), (True, 2, 3, 4, 5, 6, 7),
               (1., 2, 3, 4, 5, 6, 7))
        for order in bad:
            with self.subTest(order=order), self.assertRaises(ValueError):
                schedules.validate_order(order)
        with self.assertRaises(ValueError):
            schedules.get_schedule("best_measured")
        for start in (0, 8, True, 1.0):
            with self.assertRaises(ValueError):
                schedules.greedy_order(start)

    def test_registry_is_immutable_and_ids_encode_exact_order(self):
        self.assertEqual(len(schedules.REGISTRY), 6)
        self.assertEqual(len(schedules.SCHEDULES), 6)
        for s in schedules.REGISTRY:
            self.assertIs(schedules.get_schedule(s.alias), s)
            self.assertIs(schedules.get_schedule(s.schedule_id), s)
            self.assertEqual(s.schedule_id.rsplit("_p", 1)[1],
                             "".join(map(str, s.order)))
        with self.assertRaises(TypeError):
            schedules.SCHEDULES["new"] = schedules.REGISTRY[0]
        with self.assertRaises(FrozenInstanceError):
            schedules.REGISTRY[0].order = schedules.CURRENT_ORDER
        self.assertEqual(schedules.get_schedule(schedules.DEFAULT_SCHEDULE_ID).order,
                         schedules.CURRENT_ORDER)

    def test_product_signs_expand_to_exact_classical_block_algebra(self):
        # Formal noncommuting monomials A_i B_j avoid dependence on random data
        # or numeric roundoff. Check all orders, without assuming BF16 equivalence.
        expected = ({(0, 0): 1, (1, 2): 1}, {(0, 1): 1, (1, 3): 1},
                    {(2, 0): 1, (3, 2): 1}, {(2, 1): 1, (3, 3): 1})
        for order in self.orders:
            got = [defaultdict(int) for _ in range(4)]
            for p in order:
                product = schedules.PRODUCTS[p]
                for q, sign in product.updates:
                    for a, a_sign in product.lhs:
                        for b, b_sign in product.rhs:
                            got[q][a, b] += sign * a_sign * b_sign
            cleaned = tuple({key: value for key, value in q.items() if value}
                            for q in got)
            self.assertEqual(cleaned, expected)

    def test_bf16_combination_and_accumulator_counts_match_source(self):
        products = schedules.PRODUCTS.values()
        self.assertEqual(sum(len(p.lhs) - 1 for p in products), 5)
        self.assertEqual(sum(len(p.rhs) - 1 for p in products), 5)
        self.assertEqual(sum(len(p.updates) for p in products), 12)
        self.assertEqual([sum(q in p.targets for p in products) for q in range(4)],
                         [4, 2, 2, 4])

    def test_optimized_kernel_updates_match_formal_metadata_without_import(self):
        # Read literals rather than import the JAX-dependent kernel. This keeps
        # the exhaustive schedule/algebra suite usable with the stdlib alone.
        path = Path(__file__).resolve().parents[1] / "src/strassen_mm/strassen_optimized.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("UPDATES", "DEFAULT_ORDER"):
                        self.assertNotIn(target.id, literals)
                        literals[target.id] = ast.literal_eval(node.value)
        self.assertEqual(literals["UPDATES"],
                         {p: spec.updates for p, spec in schedules.PRODUCTS.items()})
        self.assertEqual(literals["DEFAULT_ORDER"], schedules.CURRENT_ORDER)

    def test_dependency_scores_include_repeated_panel_boundary(self):
        canonical = schedules.score_order(schedules.CANONICAL_ORDER)
        current = schedules.score_order(schedules.CURRENT_ORDER)
        self.assertEqual((canonical["linear_adjacent_shared_quadrants"],
                          canonical["panel_boundary_shared_quadrants"],
                          canonical["cyclic_adjacent_shared_quadrants"]), (3, 1, 4))
        self.assertEqual((current["linear_adjacent_shared_quadrants"],
                          current["panel_boundary_shared_quadrants"],
                          current["cyclic_adjacent_shared_quadrants"]), (1, 1, 2))
        for score in (canonical, current):
            self.assertEqual(sum(len(g) for g in score["cyclic_update_gaps_by_quadrant"].values()), 12)
            self.assertTrue(all(sum(g) == 7 for g in score["cyclic_update_gaps_by_quadrant"].values()))

    def test_existing_order_already_attains_provable_proxy_minima(self):
        best = min(self.keys.values())
        self.assertEqual(best, (2, Fraction(37, 6), 3, 5))
        self.assertEqual(self.keys[schedules.CURRENT_ORDER], best)
        self.assertEqual(sum(k[0] == 2 for k in self.keys.values()), 112)
        self.assertEqual(sum(k == best for k in self.keys.values()), 28)
        for key in self.keys.values():
            self.assertGreaterEqual(key[0], 2)
            self.assertGreaterEqual(key[1], Fraction(37, 6))
            self.assertGreaterEqual(key[2], 3)
            self.assertGreaterEqual(key[3], 5)

    def test_registry_search_recipes_reproduce_frozen_choices(self):
        best = min(self.keys.values())
        minimizers = [o for o in self.orders if self.keys[o] == best]
        self.assertEqual(min(minimizers), schedules.get_schedule("proxy_lexicographic").order)
        self.assertEqual(schedules.greedy_order(1), schedules.get_schedule("greedy_from_p1").order)
        self.assertEqual(tuple(reversed(schedules.CURRENT_ORDER)),
                         schedules.get_schedule("reverse_current").order)
        previous = [s.order for s in schedules.REGISTRY[:5]]
        distant = min((o for o in minimizers if o not in previous),
                      key=lambda o: (-min(schedules.kendall_distance(o, p) for p in previous), o))
        self.assertEqual(distant, schedules.get_schedule("diverse_proxy_tie").order)
        self.assertEqual(min(schedules.kendall_distance(distant, p) for p in previous), 10)

    def test_element_counts_give_asymmetric_tile_ratio(self):
        for bm, bn, bk in ((2, 4, 6), (128, 512, 256), (2048, 1024, 512)):
            counts = schedules.source_costs((bm, bn, bk))
            ratio = counts["extra_vector_per_saved_dot"]
            self.assertEqual(Fraction(ratio["numerator"], ratio["denominator"]),
                             Fraction(5, bm) + Fraction(5, bn) + Fraction(8, bk))
            self.assertEqual(counts["strassen_dot_flops"], 7 * bm * bn * bk // 4)
            self.assertEqual(counts["strassen_accumulator_element_operations"], 3 * bm * bn)
        for tile in (None, (2, 2), (2, 2, 2, 2), (1, 2, 2), (0, 2, 2), (True, 2, 2)):
            with self.assertRaises(ValueError):
                schedules.source_costs(tile)

    def test_cyclic_proxies_are_rotation_and_reversal_invariant(self):
        order = schedules.CURRENT_ORDER
        for shift in range(7):
            rotated = order[shift:] + order[:shift]
            self.assertEqual(schedules.proxy_key(rotated), schedules.proxy_key(order))
            self.assertEqual(schedules.proxy_key(rotated[::-1]), schedules.proxy_key(order))
        self.assertEqual(schedules.kendall_distance(order, order), 0)
        self.assertEqual(schedules.kendall_distance(order, order[::-1]), 21)

    def test_export_labels_hypotheses_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_schedule_unit_") as temporary:
            path = Path(temporary) / "source_proxy.json"
            with patch("sys.argv", ["schedule", "--output", str(path)]):
                schedules.main()
            initial = path.read_bytes()
            report = json.loads(initial)
            self.assertEqual(report["permutation_count"], 5040)
            self.assertEqual(report["new_device_measurements"], 0)
            self.assertTrue(report["current_attains_full_proxy_minimum"])
            self.assertNotIn("all_orders", report)
            self.assertIn("do not establish lower latency", report["interpretation"])
            with patch("sys.argv", ["schedule", "--output", str(path)]):
                with self.assertRaises(FileExistsError):
                    schedules.main()
            self.assertEqual(path.read_bytes(), initial)


if __name__ == "__main__":
    unittest.main()
