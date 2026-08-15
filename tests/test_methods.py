"""
test_methods.py
---------------
Unit tests for IBNR methods using Python's built-in unittest.

Includes validation against Ejercicio N°4 (verified manually):
    Chain Ladder total IBNR  = 8,897.4
    Cape Cod total IBNR      = 7,194.7  (Stanard-Bühlmann)

Run with:
    python -m unittest tests/test_methods.py -v
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.methods import (
    chain_ladder,
    bornhuetter_ferguson,
    cape_cod,
    compute_fdi,
    compute_fdi_avg,
    compute_cdfs,
    detect_anomalies,
    align_premium,
    reserve_decomposition,
    credibility_weighted_result,
    weighted_selection,
    default_maturity_weights,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def make_simple_triangle():
    """Small 4x4 cumulative triangle for unit tests."""
    data = np.array([
        [1000, 1800, 2300, 2500],
        [1100, 1950, 2450, np.nan],
        [1200, 2100, np.nan, np.nan],
        [1300, np.nan, np.nan, np.nan],
    ], dtype=float)
    return pd.DataFrame(data, index=[2020, 2021, 2022, 2023], columns=[12, 24, 36, 48])


def make_simple_premium():
    return pd.Series([5000, 5500, 6000, 6500], index=[2020, 2021, 2022, 2023])


# Ejercicio N°4 data
# EJ4_LATEST[i] is the latest paid for AY EJ4_AYS[i]; EJ4_FDA[i] is the CDF
# applied to it. AY 2008 is fully developed (CDF 1.0); AY 2013 is the newest.
EJ4_LATEST  = np.array([3200, 3400, 3500, 2800, 2100, 1600], dtype=float)
EJ4_FDA     = np.array([1.0000, 1.0530, 1.1760, 1.4290, 2.0000, 4.0000])
EJ4_PREMIUM = np.array([6200, 6500, 7500, 7800, 7800, 7000], dtype=float)
EJ4_AYS     = [2008, 2009, 2010, 2011, 2012, 2013]


def make_ej4_triangle():
    """
    Reconstruct a cumulative triangle from the exercise's latest diagonal and
    CDFs, so the *real* methods (chain_ladder / cape_cod) can be validated
    end-to-end rather than a re-implementation of the formulas.

    Every accident year is given the same multiplicative development pattern,
    so each column's simple-average link ratio is exact and build_metrics
    reproduces EJ4_FDA. AY i is observed up to development column (5 - i),
    where its cumulative value equals EJ4_LATEST[i].
    """
    n = len(EJ4_AYS)
    # C[k] = CDF from development column k (tail = 1.0 at the last column).
    # Row i's latest sits at column (n-1-i) with CDF EJ4_FDA[i], so C[n-1-i] = EJ4_FDA[i].
    C = np.array([EJ4_FDA[n - 1 - k] for k in range(n)])  # [4.0, 2.0, 1.429, 1.176, 1.053, 1.0]

    values = np.full((n, n), np.nan)
    for i in range(n):
        latest_col = n - 1 - i
        ultimate   = EJ4_LATEST[i] * C[latest_col]
        for j in range(latest_col + 1):
            values[i, j] = ultimate / C[j]

    dev_cols = [12 * (k + 1) for k in range(n)]
    return pd.DataFrame(values, index=EJ4_AYS, columns=dev_cols)


def make_ej4_premium():
    return pd.Series(EJ4_PREMIUM, index=EJ4_AYS)


# ---------------------------------------------------------------------------
# Test: Ejercicio N°4 validation — drives the real methods
# ---------------------------------------------------------------------------

class TestEjercicio4(unittest.TestCase):
    """Validate the actual chain_ladder / cape_cod functions against manually
    verified exercise results."""

    TOL = 0.5  # ±0.5 rounding tolerance

    def setUp(self):
        self.tri  = make_ej4_triangle()
        self.prem = make_ej4_premium()

    def test_triangle_reproduces_cdfs(self):
        # Sanity check: the reconstructed triangle yields the exercise CDFs.
        res = chain_ladder(self.tri)
        np.testing.assert_allclose(res.cdfs, EJ4_FDA, rtol=1e-6)

    def test_chain_ladder_total(self):
        res = chain_ladder(self.tri)
        self.assertAlmostEqual(res.total_ibnr, 8897.4, delta=self.TOL,
            msg=f"CL total IBNR: expected 8897.4, got {res.total_ibnr:.1f}")

    def test_chain_ladder_by_year(self):
        res      = chain_ladder(self.tri)
        expected = [0.0, 180.2, 616.0, 1201.2, 2100.0, 4800.0]
        for i, exp in enumerate(expected):
            self.assertAlmostEqual(res.ibnr[i], exp, delta=self.TOL,
                msg=f"AY {EJ4_AYS[i]}: expected {exp}, got {res.ibnr[i]:.1f}")

    def test_cape_cod_total(self):
        res = cape_cod(self.tri, self.prem)
        self.assertAlmostEqual(res.total_ibnr, 7194.7, delta=self.TOL,
            msg=f"Cape Cod total IBNR: expected 7194.7, got {res.total_ibnr:.1f}")

    def test_cape_cod_elr(self):
        res = cape_cod(self.tri, self.prem)
        self.assertAlmostEqual(res.elr, 0.5560, delta=0.001,
            msg=f"Cape Cod ELR: expected 0.5560, got {res.elr:.4f}")

    def test_cape_cod_lower_than_cl_for_immature_years(self):
        cc = cape_cod(self.tri, self.prem)
        cl = chain_ladder(self.tri)
        self.assertLess(cc.ibnr[-1], cl.ibnr[-1],
            msg="Cape Cod IBNR for AY 2013 should be lower than Chain Ladder "
                f"(CC={cc.ibnr[-1]:.1f}, CL={cl.ibnr[-1]:.1f})")


# ---------------------------------------------------------------------------
# Test: FDI and CDF calculations
# ---------------------------------------------------------------------------

class TestFDI(unittest.TestCase):

    def setUp(self):
        self.tri = make_simple_triangle()
        self.fdi = compute_fdi(self.tri)

    def test_fdi_shape(self):
        self.assertEqual(self.fdi.shape, (4, 3))

    def test_fdi_first_row_first_col(self):
        self.assertAlmostEqual(self.fdi.iloc[0, 0], 1.8, places=4)

    def test_fdi_first_row_second_col(self):
        self.assertAlmostEqual(self.fdi.iloc[0, 1], 2300 / 1800, places=4)

    def test_fdi_nan_for_missing_cells(self):
        # Row 2023 has only one known value → all FDIs are NaN
        self.assertTrue(self.fdi.iloc[3].isna().all())

    def test_fdi_avg_is_simple_average(self):
        avg      = compute_fdi_avg(self.fdi)
        col      = self.fdi.columns[0]
        expected = self.fdi[col].dropna().mean()
        self.assertAlmostEqual(avg.iloc[0], expected, places=6)

    def test_fdi_exclusion_changes_average(self):
        col          = self.fdi.columns[0]
        exclusions   = {(2020, col): "test exclusion"}
        avg_excl     = compute_fdi_avg(self.fdi, exclusions)
        avg_full     = compute_fdi_avg(self.fdi)
        self.assertNotAlmostEqual(avg_excl.iloc[0], avg_full.iloc[0], places=6)

    def test_cdfs_last_equals_tail(self):
        avg  = compute_fdi_avg(self.fdi)
        cdfs = compute_cdfs(avg, tail=1.0)
        self.assertEqual(cdfs[-1], 1.0)

    def test_cdfs_are_decreasing(self):
        avg  = compute_fdi_avg(self.fdi)
        cdfs = compute_cdfs(avg, tail=1.0)
        for i in range(len(cdfs) - 1):
            self.assertGreaterEqual(cdfs[i], cdfs[i + 1])


# ---------------------------------------------------------------------------
# Test: Chain Ladder method
# ---------------------------------------------------------------------------

class TestChainLadder(unittest.TestCase):

    def setUp(self):
        self.tri = make_simple_triangle()
        self.res = chain_ladder(self.tri)

    def test_method_name(self):
        self.assertEqual(self.res.method, "Chain Ladder")

    def test_output_length(self):
        self.assertEqual(len(self.res.ibnr), 4)
        self.assertEqual(len(self.res.ultimates), 4)

    def test_most_mature_year_zero_ibnr(self):
        # AY 2020 has all 4 columns → CDF = 1.0 → IBNR = 0
        self.assertAlmostEqual(self.res.ibnr[0], 0.0, delta=0.01)

    def test_ibnr_non_negative(self):
        self.assertTrue(all(self.res.ibnr >= 0))

    def test_ultimate_equals_latest_plus_ibnr(self):
        np.testing.assert_allclose(
            self.res.ultimates,
            self.res.latest_paid + self.res.ibnr,
            rtol=1e-6,
        )

    def test_to_dataframe_has_total_row(self):
        df = self.res.to_dataframe()
        self.assertIn("TOTAL", df.index)

    def test_ibnr_increases_for_less_mature_years(self):
        # Later accident years should have higher IBNR
        self.assertGreater(self.res.ibnr[3], self.res.ibnr[0])


# ---------------------------------------------------------------------------
# Test: Bornhuetter-Ferguson
# ---------------------------------------------------------------------------

class TestBornhuetterFerguson(unittest.TestCase):

    def setUp(self):
        self.tri  = make_simple_triangle()
        self.prem = make_simple_premium()
        self.res  = bornhuetter_ferguson(self.tri, self.prem)

    def test_elr_is_derived(self):
        self.assertIsNotNone(self.res.elr)
        self.assertGreater(self.res.elr, 0)
        self.assertLess(self.res.elr, 10)

    def test_custom_elr_is_used(self):
        res = bornhuetter_ferguson(self.tri, self.prem, elr=0.70)
        self.assertAlmostEqual(res.elr, 0.70, places=9)

    def test_most_mature_year_zero_ibnr(self):
        self.assertAlmostEqual(self.res.ibnr[0], 0.0, delta=0.01)

    def test_ibnr_non_negative(self):
        self.assertTrue(all(self.res.ibnr >= 0))

    def test_method_name(self):
        self.assertIn("Bornhuetter", self.res.method)


# ---------------------------------------------------------------------------
# Test: Cape Cod
# ---------------------------------------------------------------------------

class TestCapeCod(unittest.TestCase):

    def setUp(self):
        self.tri  = make_simple_triangle()
        self.prem = make_simple_premium()
        self.res  = cape_cod(self.tri, self.prem)

    def test_elr_derived_from_data(self):
        self.assertIsNotNone(self.res.elr)
        self.assertGreater(self.res.elr, 0)

    def test_most_mature_year_zero_ibnr(self):
        self.assertAlmostEqual(self.res.ibnr[0], 0.0, delta=0.01)

    def test_ibnr_non_negative(self):
        self.assertTrue(all(self.res.ibnr >= 0))

    def test_method_name_includes_stanard_buhlmann(self):
        self.assertIn("Stanard", self.res.method)

    def test_ibnr_dampened_vs_chain_ladder_for_immature_year(self):
        cl  = chain_ladder(self.tri)
        # Most immature year (index 3) — Cape Cod should be ≤ Chain Ladder
        self.assertLessEqual(self.res.ibnr[3], cl.ibnr[3] * 1.1)


# ---------------------------------------------------------------------------
# Test: premium alignment (A3)
# ---------------------------------------------------------------------------

class TestAlignPremium(unittest.TestCase):

    def test_int_years_match_str_and_float_keys(self):
        prem = pd.Series([100.0, 200.0], index=["2020", 2021.0])
        aligned, missing = align_premium(prem, [2020, 2021])
        self.assertEqual(aligned.tolist(), [100.0, 200.0])
        self.assertEqual(missing, [])

    def test_annual_premium_prorated_to_quarters(self):
        prem = pd.Series([1000.0], index=[2020])
        aligned, missing = align_premium(prem, ["2020Q1", "2020Q2"])
        self.assertEqual(aligned.tolist(), [250.0, 250.0])
        self.assertEqual(missing, [])

    def test_missing_period_reported(self):
        prem = pd.Series([100.0], index=[2020])
        aligned, missing = align_premium(prem, [2020, 2021])
        self.assertTrue(np.isnan(aligned.iloc[1]))
        self.assertEqual(missing, [2021])

    def test_no_match_raises(self):
        prem = pd.Series([100.0], index=[1999])
        with self.assertRaises(ValueError):
            align_premium(prem, [2020, 2021])

    def test_bf_on_quarterly_triangle_with_annual_premium(self):
        # Previously produced ELR = inf and NaN IBNR with no warning.
        tri = pd.DataFrame(
            [[100.0, 300.0], [150.0, np.nan]],
            index=["2020Q1", "2020Q2"], columns=[3, 6],
        )
        res = bornhuetter_ferguson(tri, pd.Series([1000.0], index=[2020]))
        self.assertTrue(np.isfinite(res.elr))
        self.assertFalse(np.isnan(res.ibnr).any())

    def test_elr_uses_only_years_with_premium(self):
        # AY 2021 has no premium: it must not inflate the derived ELR numerator.
        tri = pd.DataFrame(
            [[100.0, 300.0], [150.0, np.nan]],
            index=[2020, 2021], columns=[12, 24],
        )
        prem = pd.Series([600.0], index=[2020])
        res = bornhuetter_ferguson(tri, prem)
        self.assertAlmostEqual(res.elr, 300.0 / 600.0, places=6)


# ---------------------------------------------------------------------------
# Test: CDF guard against emptied transitions (A4)
# ---------------------------------------------------------------------------

class TestComputeCdfsGuard(unittest.TestCase):

    def test_nan_average_raises(self):
        avg = pd.Series({"12-24m": 1.5, "24-36m": np.nan})
        with self.assertRaises(ValueError):
            compute_cdfs(avg)

    def test_excluding_only_factor_raises_via_chain_ladder(self):
        tri = pd.DataFrame(
            [[1000, 1800, 2300], [1100, 1950, np.nan], [1200, np.nan, np.nan]],
            index=[2020, 2021, 2022], columns=[12, 24, 36], dtype=float,
        )
        with self.assertRaises(ValueError):
            chain_ladder(tri, exclusions={(2020, "24-36m"): "test"})


# ---------------------------------------------------------------------------
# Test: % reported column (C4)
# ---------------------------------------------------------------------------

class TestPctReported(unittest.TestCase):

    def test_pct_reported_is_inverse_cdf(self):
        res = chain_ladder(make_simple_triangle())
        df = res.to_dataframe()
        self.assertIn("pct_reported", df.columns)
        body = df.drop(index="TOTAL")
        np.testing.assert_allclose(
            body["pct_reported"].values.astype(float),
            np.round(1.0 / res.cdfs, 4),
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# Test: maturity-weighted (Benktander) best estimate
# ---------------------------------------------------------------------------

class TestCredibilityWeighted(unittest.TestCase):

    def setUp(self):
        self.tri  = make_simple_triangle()
        self.prem = make_simple_premium()
        self.cl   = chain_ladder(self.tri)
        self.bf   = bornhuetter_ferguson(self.tri, self.prem)

    def test_blend_equals_completion_factor_weighting(self):
        sel = credibility_weighted_result([self.cl, self.bf])
        z = 1.0 / self.cl.cdfs  # completion factor per period
        expected = z * self.cl.ibnr + (1 - z) * self.bf.ibnr
        np.testing.assert_allclose(sel.ibnr, expected, rtol=1e-9)

    def test_mature_year_all_chain_ladder(self):
        # AY 2020 is fully developed (CDF=1 → Z=1): selection == Chain Ladder.
        sel = credibility_weighted_result([self.cl, self.bf])
        self.assertAlmostEqual(sel.ibnr[0], self.cl.ibnr[0], delta=0.01)

    def test_immature_year_between_cl_and_bf(self):
        # Newest year (immature): the blend sits between CL and BF.
        sel = credibility_weighted_result([self.cl, self.bf])
        lo, hi = sorted([self.cl.ibnr[-1], self.bf.ibnr[-1]])
        self.assertGreaterEqual(sel.ibnr[-1], lo - 0.01)
        self.assertLessEqual(sel.ibnr[-1], hi + 0.01)

    def test_ultimate_is_latest_plus_ibnr(self):
        sel = credibility_weighted_result([self.cl, self.bf])
        np.testing.assert_allclose(sel.ultimates, sel.latest_paid + sel.ibnr, rtol=1e-9)

    def test_fallback_to_average_without_premium_anchor(self):
        # Only Chain Ladder → no expected anchor → returns CL itself.
        sel = credibility_weighted_result([self.cl])
        np.testing.assert_allclose(sel.ibnr, self.cl.ibnr, rtol=1e-9)


# ---------------------------------------------------------------------------
# Test: manual per-year weighted selection (no normalisation)
# ---------------------------------------------------------------------------

class TestWeightedSelection(unittest.TestCase):

    def setUp(self):
        self.tri  = make_simple_triangle()
        self.prem = make_simple_premium()
        self.cl   = chain_ladder(self.tri)
        self.bf   = bornhuetter_ferguson(self.tri, self.prem)

    def test_per_period_weights_applied_as_entered(self):
        results = [self.cl, self.bf]
        n = len(self.cl.ibnr)
        # Row 0 all CL, all other rows all BF.
        W = np.zeros((n, 2))
        W[0] = [1.0, 0.0]
        W[1:] = [0.0, 1.0]
        sel = weighted_selection(results, W)
        self.assertAlmostEqual(sel.ibnr[0], self.cl.ibnr[0], delta=0.01)
        for k in range(1, n):
            self.assertAlmostEqual(sel.ibnr[k], self.bf.ibnr[k], delta=0.01)

    def test_no_normalisation_half_weights_halve_result(self):
        # Weights that sum to 0.5 per row are used as-is (not rescaled to 1).
        results = [self.cl, self.bf]
        n = len(self.cl.ibnr)
        W = np.full((n, 2), 0.25)   # rows sum to 0.5
        sel = weighted_selection(results, W)
        expected = 0.25 * self.cl.ibnr + 0.25 * self.bf.ibnr
        np.testing.assert_allclose(sel.ibnr, expected, rtol=1e-9)

    def test_ultimate_is_latest_plus_ibnr(self):
        W = np.full((len(self.cl.ibnr), 2), 0.5)
        sel = weighted_selection([self.cl, self.bf], W)
        np.testing.assert_allclose(sel.ultimates, sel.latest_paid + sel.ibnr, rtol=1e-9)


class TestDefaultMaturityWeights(unittest.TestCase):

    def test_every_row_sums_to_100(self):
        cl = chain_ladder(make_simple_triangle())
        bf = bornhuetter_ferguson(make_simple_triangle(), make_simple_premium())
        w = default_maturity_weights([cl, bf])
        for _, row in w.iterrows():
            self.assertEqual(int(row.sum()), 100)

    def test_values_are_integers_and_columns_match_methods(self):
        cl = chain_ladder(make_simple_triangle())
        bf = bornhuetter_ferguson(make_simple_triangle(), make_simple_premium())
        w = default_maturity_weights([cl, bf])
        self.assertEqual(list(w.columns), [cl.method, bf.method])
        self.assertTrue((w.to_numpy() == w.to_numpy().astype(int)).all())

    def test_mature_year_favours_chain_ladder(self):
        # AY 2020 fully developed (Z=1) → CL weight 100, BF weight 0.
        cl = chain_ladder(make_simple_triangle())
        bf = bornhuetter_ferguson(make_simple_triangle(), make_simple_premium())
        w = default_maturity_weights([cl, bf])
        self.assertEqual(int(w.loc[2020, cl.method]), 100)


# ---------------------------------------------------------------------------
# Test: reserve decomposition (Reconcile view)
# ---------------------------------------------------------------------------

class TestReserveDecomposition(unittest.TestCase):
    """Paid + RSP + Pure IBNR must add up to the incurred-base ultimate."""

    def _paid_incurred(self):
        # Paid triangle and an incurred triangle = paid + a known RSP snapshot.
        paid = pd.DataFrame([
            [1000, 1800, 2300, 2500],
            [1100, 1950, 2450, np.nan],
            [1200, 2100, np.nan, np.nan],
            [1300, np.nan, np.nan, np.nan],
        ], index=[2020, 2021, 2022, 2023], columns=[12, 24, 36, 48], dtype=float)
        # RSP on the latest diagonal: 0, 200, 500, 900 for AY 2020..2023.
        rsp_latest = {2020: 0.0, 2021: 200.0, 2022: 500.0, 2023: 900.0}
        incurred = paid.copy()
        for ay, add in rsp_latest.items():
            row = incurred.loc[ay]
            last = row.last_valid_index()
            incurred.loc[ay, last] = row[last] + add
        return chain_ladder(paid), chain_ladder(incurred), rsp_latest

    def test_components_sum_to_ultimate(self):
        p0, i0, _ = self._paid_incurred()
        dec = reserve_decomposition(p0, i0)
        body = dec.drop(index="TOTAL")
        np.testing.assert_allclose(
            (body["paid"] + body["rsp"] + body["pure_ibnr"]).to_numpy(float),
            body["ultimate"].to_numpy(float),
            rtol=1e-6,
        )

    def test_rsp_is_incurred_minus_paid_diagonal(self):
        p0, i0, rsp_latest = self._paid_incurred()
        dec = reserve_decomposition(p0, i0)
        for ay, expected in rsp_latest.items():
            self.assertAlmostEqual(dec.loc[ay, "rsp"], expected, delta=0.01)

    def test_total_row_sums_columns(self):
        p0, i0, _ = self._paid_incurred()
        dec = reserve_decomposition(p0, i0)
        body = dec.drop(index="TOTAL")
        for col in ["paid", "rsp", "pure_ibnr", "ultimate"]:
            self.assertAlmostEqual(dec.loc["TOTAL", col], body[col].sum(), delta=0.01)

    def test_pure_ibnr_matches_incurred_base_ibnr(self):
        p0, i0, _ = self._paid_incurred()
        dec = reserve_decomposition(p0, i0)
        self.assertAlmostEqual(dec.loc["TOTAL", "pure_ibnr"], i0.total_ibnr, delta=0.01)


# ---------------------------------------------------------------------------
# Test: Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection(unittest.TestCase):

    def test_no_anomalies_in_stable_triangle(self):
        fdi   = compute_fdi(make_simple_triangle())
        flags = detect_anomalies(fdi)
        vals  = flags.stack().dropna().tolist()
        self.assertEqual(len(vals), 0)

    def test_detects_extreme_outlier(self):
        # Use more rows so IQR is not inflated by the outlier itself
        data = np.array([
            [1000, 1800, np.nan],
            [1100, 1950, np.nan],
            [1200, 1980, np.nan],
            [1050, 1820, np.nan],
            [1080, 1890, np.nan],
            [1150, 9999, np.nan],   # extreme outlier
            [1300, np.nan, np.nan],
        ], dtype=float)
        tri   = pd.DataFrame(data, index=range(7), columns=[12, 24, 36])
        fdi   = compute_fdi(tri)
        flags = detect_anomalies(fdi)
        # Only one severity now: everything outside the fence is a "warning".
        self.assertEqual(flags.iloc[5, 0], "warning",
            msg="Extreme outlier (9999) should be flagged as a warning")

    def test_detects_warning_level(self):
        data = np.array([
            [1000, 1800],
            [1000, 1820],
            [1000, 1810],
            [1000, 1815],
            [1000, 2200],   # moderate outlier
        ], dtype=float)
        tri   = pd.DataFrame(data, index=range(5), columns=[12, 24])
        fdi   = compute_fdi(tri)
        flags = detect_anomalies(fdi)
        self.assertEqual(flags.iloc[4, 0], "warning")

    def test_never_flags_anomaly_tier(self):
        # No matter how extreme, the flag is "warning" (no strong/red tier).
        data = np.array([
            [1000, 1800, np.nan],
            [1100, 1950, np.nan],
            [1200, 1980, np.nan],
            [1050, 1820, np.nan],
            [1080, 1890, np.nan],
            [1150, 99999, np.nan],   # absurd outlier
            [1300, np.nan, np.nan],
        ], dtype=float)
        fdi   = compute_fdi(pd.DataFrame(data, index=range(7), columns=[12, 24, 36]))
        flags = detect_anomalies(fdi)
        self.assertNotIn("anomaly", flags.stack().dropna().tolist())

    def test_flags_same_shape_as_fdi(self):
        fdi   = compute_fdi(make_simple_triangle())
        flags = detect_anomalies(fdi)
        self.assertEqual(flags.shape, fdi.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
