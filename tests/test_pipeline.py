"""
test_pipeline.py
----------------
Unit tests for the claims → triangle pipeline, focused on the paid vs. incurred
feature (incurred = paid + case reserve).

Run with:
    python -m unittest tests/test_pipeline.py -v
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import (
    build_triangle, build_paid_incurred,
    suggest_mapping, infer_date_format, check_date_format,
)
from app.methods import chain_ladder


def _claims():
    """
    Small claim-level dataset (annual grain, one payment per claim-year).
    case_reserve is the OUTSTANDING balance (>= 0) at each valuation, declining
    as the claim is paid.
    """
    return pd.DataFrame({
        "incurred_date": ["01/15/2020", "03/05/2020", "02/20/2021", "08/01/2021"],
        "paid_date":     ["09/15/2020", "06/10/2021", "12/20/2021", "08/01/2022"],
        "paid_amount":   [1000.0, 300.0, 1100.0, 400.0],
        "case_reserve":  [200.0,  150.0, 300.0,  80.0],
    })


def _example_claim():
    """
    The user's example: a 5000 claim (AY 2020) paying 1000/yr, RSP 4000→3000→2000.
    Two filler claims (AY 2021, 2022) set the latest valuation to 2022 so the
    future-cell masking keeps AY 2020's three observed development cells.
    """
    return pd.DataFrame({
        "incurred_date": ["01/01/2020", "01/01/2020", "01/01/2020", "01/01/2021", "01/01/2022"],
        "paid_date":     ["06/01/2020", "06/01/2021", "06/01/2022", "06/01/2021", "06/01/2022"],
        "paid_amount":   [1000.0, 1000.0, 1000.0, 700.0, 600.0],
        "case_reserve":  [4000.0, 3000.0, 2000.0, 900.0, 1200.0],
    })


class TestBuildPaidIncurred(unittest.TestCase):

    def setUp(self):
        self.df = _claims()
        self.base_kw = dict(
            incurred_col="incurred_date", paid_col="paid_date",
            date_format="MM/DD/YYYY", grain="annual",
        )
        self.kw = {**self.base_kw, "amount_col": "paid_amount"}

    def test_incurred_none_without_reserve(self):
        tp, ti = build_paid_incurred(self.df, **self.kw)
        self.assertIsNotNone(tp)
        self.assertIsNone(ti, "incurred triangle should be None when no reserve mapped")

    def test_incurred_equals_paid_when_reserve_zero(self):
        df = self.df.copy()
        df["case_reserve"] = 0.0
        tp, ti = build_paid_incurred(df, reserve_col="case_reserve", **self.kw)
        pd.testing.assert_frame_equal(tp, ti)

    def test_incurred_is_cum_paid_plus_reserve_snapshot(self):
        tp, ti = build_paid_incurred(self.df, reserve_col="case_reserve", **self.kw)
        # Expected: cumulative paid + reserve snapshot (reserve NOT cumulated).
        reserve_snap = build_triangle(self.df, amount_col="case_reserve",
                                      cumulate=False, **self.base_kw)
        reserve_snap = reserve_snap.reindex(index=tp.index, columns=tp.columns)
        expected = (tp + reserve_snap.fillna(0)).where(~tp.isna())
        pd.testing.assert_frame_equal(ti, expected)

    def test_example_incurred_is_flat_at_ultimate(self):
        # 5000 claim: incurred should be 5000 across all observed dev periods.
        df = _example_claim()
        tp, ti = build_paid_incurred(
            df, incurred_col="incurred_date", paid_col="paid_date",
            amount_col="paid_amount", reserve_col="case_reserve",
            date_format="MM/DD/YYYY", grain="annual",
        )
        row = ti.loc[2020].dropna().tolist()
        self.assertEqual(len(row), 3)
        for v in row:
            self.assertAlmostEqual(v, 5000.0, places=2)
        # paid develops 1000 -> 2000 -> 3000
        self.assertEqual(tp.loc[2020].dropna().tolist(), [1000.0, 2000.0, 3000.0])

    def test_reserve_nan_treated_as_zero(self):
        df = self.df.copy()
        df.loc[0, "case_reserve"] = np.nan
        # Should not raise and should equal paid+reserve with that cell as 0.
        tp, ti = build_paid_incurred(df, reserve_col="case_reserve", **self.kw)
        self.assertIsNotNone(ti)
        self.assertFalse(ti.isna().all().all())


class TestSampleReconciliation(unittest.TestCase):
    """End-to-end check on the generated sample with case reserves."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "samples" / "sample1_claims.csv"
        cls.available = path.exists() and "case_reserve" in pd.read_csv(path, nrows=1).columns
        if cls.available:
            cls.df = pd.read_csv(path)

    def test_paid_and_incurred_build(self):
        if not self.available:
            self.skipTest("sample with case_reserve not available")
        tp, ti = build_paid_incurred(
            self.df, incurred_col="incurred_date", paid_col="paid_date",
            amount_col="paid_amount", reserve_col="case_reserve",
            date_format="MM/DD/YYYY", grain="annual",
        )
        self.assertEqual(tp.shape, ti.shape)

    def test_ultimates_converge(self):
        # Reserves run off to ~0 at maturity ⇒ ultimate(paid) ≈ ultimate(incurred).
        if not self.available:
            self.skipTest("sample with case_reserve not available")
        tp, ti = build_paid_incurred(
            self.df, incurred_col="incurred_date", paid_col="paid_date",
            amount_col="paid_amount", reserve_col="case_reserve",
            date_format="MM/DD/YYYY", grain="annual",
        )
        up = chain_ladder(tp).total_ultimate
        ui = chain_ladder(ti).total_ultimate
        self.assertLess(abs(up - ui) / up, 0.05,
                        msg=f"ultimates should be within 5% (paid={up:.0f}, inc={ui:.0f})")

    def test_incurred_develops_faster(self):
        # Incurred CDFs should be lower than paid CDFs (faster development).
        if not self.available:
            self.skipTest("sample with case_reserve not available")
        tp, ti = build_paid_incurred(
            self.df, incurred_col="incurred_date", paid_col="paid_date",
            amount_col="paid_amount", reserve_col="case_reserve",
            date_format="MM/DD/YYYY", grain="annual",
        )
        cdf_paid = chain_ladder(tp).cdfs[-1]     # CDF for the most immature year
        cdf_inc  = chain_ladder(ti).cdfs[-1]
        self.assertLess(cdf_inc, cdf_paid)


class TestCumulativeAmounts(unittest.TestCase):
    """Cumulative-per-claim amounts must be differenced per claim (A1)."""

    def _df(self):
        # Claim A: cumulative 100 -> 300; claim B: 50 -> 80 (same AY 2020).
        return pd.DataFrame({
            "claim":         ["A", "B", "A", "B", "C"],
            "incurred_date": ["01/01/2020"] * 4 + ["01/01/2021"],
            "paid_date":     ["06/01/2020", "06/01/2020", "06/01/2021",
                              "06/01/2021", "06/01/2021"],
            "paid_amount":   [100.0, 50.0, 300.0, 80.0, 10.0],
        })

    def test_per_claim_diff_with_claim_id(self):
        tri = build_triangle(
            self._df(), "incurred_date", "paid_date", "paid_amount",
            amount_type="cumulative", grain="annual", claim_col="claim",
        )
        self.assertAlmostEqual(tri.loc[2020, 12], 150.0)   # 100 + 50
        self.assertAlmostEqual(tri.loc[2020, 24], 380.0)   # 300 + 80

    def test_duplicates_without_claim_id_raise(self):
        with self.assertRaises(ValueError):
            build_triangle(
                self._df(), "incurred_date", "paid_date", "paid_amount",
                amount_type="cumulative", grain="annual",
            )

    def test_one_row_per_period_without_claim_id_still_works(self):
        df = pd.DataFrame({
            "incurred_date": ["01/01/2020", "01/01/2020", "01/01/2021"],
            "paid_date":     ["06/01/2020", "06/01/2021", "06/01/2021"],
            "paid_amount":   [100.0, 300.0, 40.0],   # cumulative
        })
        tri = build_triangle(
            df, "incurred_date", "paid_date", "paid_amount",
            amount_type="cumulative", grain="annual",
        )
        self.assertAlmostEqual(tri.loc[2020, 12], 100.0)
        self.assertAlmostEqual(tri.loc[2020, 24], 300.0)


class TestReserveSnapshotPerClaim(unittest.TestCase):
    """RSP 'level' read per claim: last value per period, carried forward (A2)."""

    def _kw(self):
        return dict(
            incurred_col="incurred_date", paid_col="paid_date",
            amount_col="paid_amount", reserve_col="case_reserve",
            date_format="MM/DD/YYYY", grain="annual", claim_col="claim",
        )

    def test_two_payments_same_period_no_double_count(self):
        # RSP is an outstanding balance: the latest valuation (4000) counts,
        # not the sum of both rows (8500).
        df = pd.DataFrame({
            "claim":         ["A", "A", "B"],
            "incurred_date": ["01/01/2020", "01/01/2020", "01/01/2021"],
            "paid_date":     ["03/01/2020", "09/01/2020", "06/01/2021"],
            "paid_amount":   [500.0, 500.0, 700.0],
            "case_reserve":  [4500.0, 4000.0, 900.0],
        })
        _, ti = build_paid_incurred(df, **self._kw())
        self.assertAlmostEqual(ti.loc[2020, 12], 5000.0)   # 1000 paid + 4000 RSP

    def test_reserve_carried_through_inactive_period(self):
        # Claim A pays in 2020 (RSP 4000) and 2022 (RSP 3000), nothing in 2021:
        # the 4000 balance must persist at dev 24, keeping incurred flat at 5000.
        df = pd.DataFrame({
            "claim":         ["A", "A", "B", "C"],
            "incurred_date": ["01/01/2020", "01/01/2020", "01/01/2021", "01/01/2022"],
            "paid_date":     ["06/01/2020", "06/01/2022", "06/01/2021", "06/01/2022"],
            "paid_amount":   [1000.0, 1000.0, 500.0, 600.0],
            "case_reserve":  [4000.0, 3000.0, 900.0, 1200.0],
        })
        tp, ti = build_paid_incurred(df, **self._kw())
        self.assertEqual(tp.loc[2020].dropna().tolist(), [1000.0, 1000.0, 2000.0])
        self.assertEqual(ti.loc[2020].dropna().tolist(), [5000.0, 5000.0, 5000.0])


class TestTriangleGridAndMask(unittest.TestCase):
    """Interior zero-fill, grid completion and data-driven valuation date."""

    def test_grid_completed_and_cumulative_flat_across_inactive_dev(self):
        # No payment anywhere at dev 24 → the column must still exist and the
        # cumulative must carry flat (1000), not leave a NaN hole.
        df = pd.DataFrame({
            "incurred_date": ["01/01/2020", "01/01/2020", "01/01/2022"],
            "paid_date":     ["06/01/2020", "06/01/2022", "06/01/2022"],
            "paid_amount":   [1000.0, 1000.0, 600.0],
        })
        tri = build_triangle(df, "incurred_date", "paid_date", "paid_amount",
                             grain="annual")
        self.assertEqual(tri.columns.tolist(), [12, 24, 36])
        self.assertEqual(tri.loc[2020].tolist(), [1000.0, 1000.0, 2000.0])

    def test_valuation_date_from_data_not_newest_ay(self):
        # Latest payment is in 2022 (AY2021 at dev 24) and there is no AY2022:
        # that real payment must not be masked away.
        df = pd.DataFrame({
            "incurred_date": ["01/01/2020", "01/01/2021", "01/01/2021"],
            "paid_date":     ["06/01/2020", "06/01/2021", "08/01/2022"],
            "paid_amount":   [1000.0, 1100.0, 400.0],
        })
        tri = build_triangle(df, "incurred_date", "paid_date", "paid_amount",
                             grain="annual")
        self.assertAlmostEqual(tri.loc[2021, 24], 1500.0)
        # AY2020 dev 36 is exactly on the diagonal → observed (flat at 1000).
        self.assertAlmostEqual(tri.loc[2020, 36], 1000.0)


class TestSuggestMapping(unittest.TestCase):

    def test_matches_snake_case_columns(self):
        cols = ["claim_id", "incurred_date", "paid_date", "paid_amount",
                "case_reserve", "line_of_business"]
        m = suggest_mapping(cols)
        self.assertEqual(m.get("incurred_date"), "incurred_date")
        self.assertEqual(m.get("paid_date"), "paid_date")
        self.assertEqual(m.get("paid_amount"), "paid_amount")
        self.assertEqual(m.get("reserve"), "case_reserve")
        self.assertEqual(m.get("segment"), "line_of_business")
        self.assertEqual(m.get("claim_id"), "claim_id")

    def test_matches_spanish_claim_number(self):
        m = suggest_mapping(["Siniestro", "Fecha Siniestro", "Fecha de pago", "Pago"])
        self.assertEqual(m.get("claim_id"), "Siniestro")
        self.assertEqual(m.get("incurred_date"), "Fecha Siniestro")

    def test_matches_spanish_and_spaced(self):
        cols = ["Fecha de incurrido", "Fecha de pago", "Monto pagado", "Reserva", "Ramo"]
        m = suggest_mapping(cols)
        self.assertEqual(m.get("incurred_date"), "Fecha de incurrido")
        self.assertEqual(m.get("paid_date"), "Fecha de pago")
        self.assertEqual(m.get("paid_amount"), "Monto pagado")
        self.assertEqual(m.get("reserve"), "Reserva")
        self.assertEqual(m.get("segment"), "Ramo")

    def test_no_column_used_twice(self):
        m = suggest_mapping(["incurred_date", "paid_date", "paid_amount", "case_reserve"])
        self.assertEqual(len(set(m.values())), len(m.values()))


class TestDateFormat(unittest.TestCase):

    def test_detects_day_first_when_day_gt_12(self):
        # 28 can't be a month -> must be DD/MM/YYYY
        ok, sugg = check_date_format(["28/02/2020", "01/03/2020"], "MM/DD/YYYY")
        self.assertFalse(ok)
        self.assertEqual(sugg, "DD/MM/YYYY")

    def test_ok_when_format_matches(self):
        ok, sugg = check_date_format(["02/28/2020", "03/01/2020"], "MM/DD/YYYY")
        self.assertTrue(ok)
        self.assertIsNone(sugg)

    def test_infer_picks_unambiguous_format(self):
        self.assertEqual(infer_date_format(["28/02/2020", "15/06/2021"]), "DD/MM/YYYY")
        self.assertEqual(infer_date_format(["2020-02-28", "2021-06-15"]), "YYYY-MM-DD")

    def test_infer_keeps_default_when_ambiguous(self):
        # all days <= 12 -> both MM/DD and DD/MM parse; keep default
        self.assertEqual(infer_date_format(["01/02/2020", "03/04/2020"],
                                           default="MM/DD/YYYY"), "MM/DD/YYYY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
