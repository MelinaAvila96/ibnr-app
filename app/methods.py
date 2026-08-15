"""
methods.py
----------
IBNR estimation methods applied to a cumulative loss triangle.

All methods receive a triangle (pd.DataFrame) and return an IBNRResult.

Methods
-------
- chain_ladder()            : projects latest diagonal using simple average FDI
- bornhuetter_ferguson()    : blends development pattern with a priori ELR
- cape_cod()                : derives ELR from used-up premium (Stanard-Bühlmann)

FDI calculation
---------------
All methods use SIMPLE AVERAGE link ratios (equal weight per accident year).
This makes individual deviations visible for anomaly detection.

Anomaly detection
-----------------
detect_anomalies() flags individual FDIs that fall outside
Q1 - 1.5*IQR or Q3 + 1.5*IQR per development column.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IBNRResult:
    """Holds the output of a single IBNR method."""
    method: str
    accident_periods: list
    latest_paid: np.ndarray
    ultimates: np.ndarray
    ibnr: np.ndarray
    cdfs: np.ndarray          # CDF applied to each accident period
    elr: float | None = None  # only for BF and Cape Cod

    @property
    def total_ibnr(self) -> float:
        return float(np.nansum(self.ibnr))

    @property
    def total_ultimate(self) -> float:
        return float(np.nansum(self.ultimates))

    def to_dataframe(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_reported = np.where(self.cdfs != 0, 1.0 / self.cdfs, np.nan)
        df = pd.DataFrame({
            "accident_period": self.accident_periods,
            "latest_paid":     np.round(self.latest_paid, 2),
            "cdf":             np.round(self.cdfs, 4),
            "pct_reported":    np.round(pct_reported, 4),
            "ultimate":        np.round(self.ultimates, 2),
            "ibnr":            np.round(self.ibnr, 2),
        })
        if self.elr is not None:
            df["elr"] = round(self.elr, 4)
        df = df.set_index("accident_period")
        totals = df[["latest_paid", "ultimate", "ibnr"]].sum()
        totals.name = "TOTAL"
        return pd.concat([df, totals.to_frame().T])


@dataclass
class TriangleMetrics:
    """FDI table and derived CDFs — shared across methods."""
    fdi_table: pd.DataFrame        # individual link ratios (rows=AY, cols=dev transitions)
    fdi_avg: pd.Series             # simple average FDI per transition
    cdfs: np.ndarray               # cumulative development factors (one per column)
    tail: float = 1.0
    exclusions: dict = field(default_factory=dict)  # {(ay, col_idx): reason}


# ---------------------------------------------------------------------------
# Core triangle calculations
# ---------------------------------------------------------------------------

def compute_fdi(triangle: pd.DataFrame) -> pd.DataFrame:
    """
    Compute individual age-to-age (link ratio) factors.

    Returns a DataFrame with the same index as triangle and columns
    named '{dev_j}-{dev_j+1}m' for each transition.

    Cells whose starting cumulative is <= 0 (nothing paid yet, or a net
    recovery) yield NaN — a ratio over a non-positive base is not a
    meaningful development factor, so those transitions simply don't
    contribute to the averages.
    """
    values = triangle.values.astype(float)
    periods = triangle.index.tolist()
    devs = triangle.columns.tolist()
    n_rows, n_cols = values.shape

    col_names = [f"{devs[j]}-{devs[j+1]}m" for j in range(n_cols - 1)]
    fdi_matrix = np.full((n_rows, n_cols - 1), np.nan)

    for j in range(n_cols - 1):
        for i in range(n_rows):
            c = values[i, j]
            n = values[i, j + 1]
            if not np.isnan(c) and not np.isnan(n) and c > 0:
                fdi_matrix[i, j] = n / c

    return pd.DataFrame(fdi_matrix, index=periods, columns=col_names)


def compute_fdi_avg(
    fdi_table: pd.DataFrame,
    exclusions: dict | None = None,
) -> pd.Series:
    """
    Compute simple average FDI per column, respecting user exclusions.

    Parameters
    ----------
    fdi_table : pd.DataFrame
        Output of compute_fdi().
    exclusions : dict, optional
        {(accident_period, col_name): reason_string}
        Cells in this dict are excluded from the average.

    Returns
    -------
    pd.Series
        Simple average FDI per development transition.
    """
    exclusions = exclusions or {}
    avgs = {}

    for col in fdi_table.columns:
        series = fdi_table[col].copy()
        for (period, excl_col), _ in exclusions.items():
            if excl_col == col and period in series.index:
                series[period] = np.nan
        avgs[col] = series.mean(skipna=True)

    return pd.Series(avgs)


def compute_cdfs(fdi_avg: pd.Series, tail: float = 1.0) -> np.ndarray:
    """
    Compute cumulative development factors from simple average FDIs.

    CDF[j] = fdi[j] * fdi[j+1] * ... * tail

    Returns an array of length n_cols (one CDF per development column,
    including the last column which equals the tail).

    Raises
    ------
    ValueError
        If any average FDI is NaN (e.g. every factor in a transition was
        excluded) — a NaN would silently propagate into every CDF and the
        IBNR totals would understate.
    """
    nan_cols = [str(c) for c, v in fdi_avg.items() if pd.isna(v)]
    if nan_cols:
        raise ValueError(
            "No development factor available for transition(s): "
            + ", ".join(nan_cols)
            + ". Review the exclusions or the triangle data."
        )
    factors = list(fdi_avg.values) + [tail]
    n = len(factors)
    cdfs = np.ones(n)
    cdfs[-1] = tail
    for i in range(n - 2, -1, -1):
        cdfs[i] = factors[i] * cdfs[i + 1]
    return cdfs


def _latest_diagonal(triangle: pd.DataFrame):
    """
    Extract the latest known value and its column index for each row.

    Returns
    -------
    latest_values : np.ndarray
    latest_col_idx : np.ndarray (int)
    """
    values = triangle.values.astype(float)
    n_rows = values.shape[0]
    latest_values  = np.full(n_rows, np.nan)
    latest_col_idx = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        known = np.where(~np.isnan(values[i]))[0]
        if len(known) > 0:
            j = known[-1]
            latest_values[i]  = values[i, j]
            latest_col_idx[i] = j

    return latest_values, latest_col_idx


def build_metrics(
    triangle: pd.DataFrame,
    tail: float = 1.0,
    exclusions: dict | None = None,
) -> TriangleMetrics:
    """
    Compute FDI table, average FDIs, and CDFs for a triangle.
    Central entry point used by all methods.
    """
    fdi_table = compute_fdi(triangle)
    fdi_avg   = compute_fdi_avg(fdi_table, exclusions)
    cdfs      = compute_cdfs(fdi_avg, tail)

    return TriangleMetrics(
        fdi_table=fdi_table,
        fdi_avg=fdi_avg,
        cdfs=cdfs,
        tail=tail,
        exclusions=exclusions or {},
    )


# ---------------------------------------------------------------------------
# Premium alignment
# ---------------------------------------------------------------------------

def _norm_period(label) -> str:
    """'2020', 2020, 2020.0 → '2020'; '2020Q3' stays as is."""
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label).strip()


def align_premium(premium: pd.Series, index) -> tuple[pd.Series, list]:
    """
    Align an earned-premium series to a triangle's accident periods.

    Handles type mismatches (int vs. str years) and annual premium against a
    quarterly triangle: the year's premium is prorated equally (1/4 per
    quarter). Periods with no premium get NaN.

    Returns
    -------
    (aligned, missing)
        aligned : pd.Series indexed like the triangle.
        missing : list of triangle periods with no premium found.

    Raises
    ------
    ValueError
        If no triangle period could be matched to a premium at all —
        premium-based methods would silently produce NaN/inf otherwise.
    """
    prem_by_key = {}
    for k, v in premium.items():
        prem_by_key[_norm_period(k)] = float(v)

    values, missing = [], []
    for p in index:
        key = _norm_period(p)
        if key in prem_by_key:
            values.append(prem_by_key[key])
        elif "Q" in key and key.split("Q")[0] in prem_by_key:
            values.append(prem_by_key[key.split("Q")[0]] / 4.0)
        else:
            values.append(np.nan)
            missing.append(p)

    aligned = pd.Series(values, index=index, dtype=float)
    if aligned.isna().all():
        raise ValueError(
            "The earned premium could not be matched to any accident period "
            f"of the triangle (triangle periods: {list(index)}; premium "
            f"periods: {list(premium.index)}). Check the premium file."
        )
    return aligned, missing


# ---------------------------------------------------------------------------
# IBNR Methods
# ---------------------------------------------------------------------------

def chain_ladder(
    triangle: pd.DataFrame,
    tail: float = 1.0,
    exclusions: dict | None = None,
) -> IBNRResult:
    """
    Chain Ladder method using simple average FDI.

    Ultimate_i = Latest_i * CDF_i
    IBNR_i     = Ultimate_i - Latest_i
    """
    metrics = build_metrics(triangle, tail, exclusions)
    latest, latest_col = _latest_diagonal(triangle)
    cdfs_per_row = metrics.cdfs[latest_col]

    ultimates = latest * cdfs_per_row
    ibnr      = ultimates - latest

    return IBNRResult(
        method="Chain Ladder",
        accident_periods=triangle.index.tolist(),
        latest_paid=latest,
        ultimates=ultimates,
        ibnr=ibnr,
        cdfs=cdfs_per_row,
    )


def bornhuetter_ferguson(
    triangle: pd.DataFrame,
    premium: pd.Series,
    elr: float | None = None,
    tail: float = 1.0,
    exclusions: dict | None = None,
) -> IBNRResult:
    """
    Bornhuetter-Ferguson method.

    IBNR_i = (1 - 1/CDF_i) * ELR * Premium_i

    If elr is None, it is derived from Chain Ladder ultimates:
        ELR = sum(CL ultimates) / sum(premiums)

    Parameters
    ----------
    premium : pd.Series
        Earned premium indexed by accident period.
    elr : float, optional
        A priori expected loss ratio. Derived from data if not provided.
    """
    metrics = build_metrics(triangle, tail, exclusions)
    latest, latest_col = _latest_diagonal(triangle)
    cdfs_per_row = metrics.cdfs[latest_col]

    # align premium to triangle index (validates the match)
    prem = align_premium(premium, triangle.index)[0].values

    # derive ELR from Chain Ladder if not supplied — only over periods that
    # have premium, so a missing year cannot bias the ratio
    if elr is None:
        cl_ultimates = latest * cdfs_per_row
        has_prem = ~np.isnan(prem)
        elr = np.nansum(cl_ultimates[has_prem]) / np.nansum(prem[has_prem])

    unreported_pct = 1 - 1 / cdfs_per_row
    ibnr           = unreported_pct * elr * prem
    ultimates      = latest + ibnr

    return IBNRResult(
        method="Bornhuetter-Ferguson",
        accident_periods=triangle.index.tolist(),
        latest_paid=latest,
        ultimates=ultimates,
        ibnr=ibnr,
        cdfs=cdfs_per_row,
        elr=elr,
    )


def cape_cod(
    triangle: pd.DataFrame,
    premium: pd.Series,
    tail: float = 1.0,
    exclusions: dict | None = None,
) -> IBNRResult:
    """
    Cape Cod method (also known as Stanard-Bühlmann).

    ELR is derived from the data itself using used-up premium:
        ELR = sum(Latest_i) / sum(Premium_i / CDF_i)

    IBNR_i = (1 - 1/CDF_i) * ELR * Premium_i
    """
    metrics = build_metrics(triangle, tail, exclusions)
    latest, latest_col = _latest_diagonal(triangle)
    cdfs_per_row = metrics.cdfs[latest_col]

    prem = align_premium(premium, triangle.index)[0].values

    # ELR from used-up premium, restricted to periods that have premium so
    # the numerator and denominator cover the same years
    has_prem        = ~np.isnan(prem)
    used_up_premium = prem / cdfs_per_row
    elr             = np.nansum(latest[has_prem]) / np.nansum(used_up_premium[has_prem])

    unreported_pct = 1 - 1 / cdfs_per_row
    ibnr           = unreported_pct * elr * prem
    ultimates      = latest + ibnr

    return IBNRResult(
        method="Cape Cod (Stanard-Bühlmann)",
        accident_periods=triangle.index.tolist(),
        latest_paid=latest,
        ultimates=ultimates,
        ibnr=ibnr,
        cdfs=cdfs_per_row,
        elr=elr,
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    fdi_table: pd.DataFrame,
    warning_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag individual FDI values that fall outside the IQR fence, as a single
    "warning" severity (there is no separate strong-anomaly tier).

    For each development column with at least 3 factors, a factor is flagged
    "warning" when it lies below Q1 - warning_multiplier*IQR or above
    Q3 + warning_multiplier*IQR. The analyst reviews the warnings and decides
    what, if anything, to exclude.

    Parameters
    ----------
    warning_multiplier : float
        IQR multiplier for the fence. Default 1.5 (the usual Tukey fence).

    Returns
    -------
    pd.DataFrame
        Same shape as fdi_table. Values: None (normal) or 'warning'.
    """
    flags = pd.DataFrame(None, index=fdi_table.index, columns=fdi_table.columns)

    for col in fdi_table.columns:
        series = fdi_table[col].dropna()
        if len(series) < 3:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        fence_low  = q1 - warning_multiplier * iqr
        fence_high = q3 + warning_multiplier * iqr

        for period in fdi_table.index:
            val = fdi_table.loc[period, col]
            if pd.isna(val):
                continue
            if val < fence_low or val > fence_high:
                flags.loc[period, col] = "warning"

    return flags


def reserve_decomposition(
    paid_result: IBNRResult,
    incurred_result: IBNRResult,
) -> pd.DataFrame:
    """
    Decompose each accident period's ultimate into its three additive parts:

        ultimate = paid_to_date + case_reserves(RSP) + pure_IBNR

    This reframes the paid-vs-incurred picture as a *decomposition* rather than a
    comparison: the paid base's IBNR (= RSP + pure IBNR, the total reserve) and the
    incurred base's IBNR (= pure IBNR) are simply different slices of the same total,
    not competing estimates.

    Components
    ----------
    paid      : cumulative paid to date  = paid_result.latest_paid  (data, method-free)
    rsp       : outstanding case reserve = incurred.latest_paid − paid.latest_paid
                (the incurred diagonal already includes RSP; data, method-free)
    pure_ibnr : incurred_result.ibnr     (depends on the incurred-base method)
    ultimate  : paid + rsp + pure_ibnr   (= incurred_result.ultimates)

    Values are kept raw (RSP may be ~0, or slightly negative under net recoveries).

    Returns
    -------
    pd.DataFrame
        Indexed by accident period with a TOTAL row; columns
        ['paid', 'rsp', 'pure_ibnr', 'ultimate'].
    """
    paid = np.asarray(paid_result.latest_paid, dtype=float)
    inc  = np.asarray(incurred_result.latest_paid, dtype=float)
    rsp  = inc - paid
    pure_ibnr = np.asarray(incurred_result.ibnr, dtype=float)
    ultimate  = paid + rsp + pure_ibnr

    df = pd.DataFrame(
        {
            "paid":      np.round(paid, 2),
            "rsp":       np.round(rsp, 2),
            "pure_ibnr": np.round(pure_ibnr, 2),
            "ultimate":  np.round(ultimate, 2),
        },
        index=incurred_result.accident_periods,
    )
    totals = df.sum()
    totals.name = "TOTAL"
    return pd.concat([df, totals.to_frame().T])


def credibility_weighted_result(results: list[IBNRResult], label: str = "Best estimate") -> IBNRResult:
    """
    Benktander / credibility-weighted best estimate (Gunnar-Benktander,
    a.k.a. twice-iterated Bornhuetter-Ferguson).

    Per accident period, blend the development method (Chain Ladder) with the
    a priori method (Bornhuetter-Ferguson) using the completion factor
    Z = 1/CDF (the % already reported) as the credibility weight:

        IBNR_selected = Z · IBNR_CL + (1 − Z) · IBNR_BF

    which is equivalent to Ultimate = Z · Ultimate_CL + (1 − Z) · Ultimate_BF.
    Mature periods (Z → 1) follow Chain Ladder; immature periods (Z → 0) lean
    on the a priori (BF) — the standard actuarial response to the low
    credibility of link-ratio projections for recent, immature periods.

    The a priori anchor is Bornhuetter-Ferguson (the classic choice); Cape Cod
    is used only when BF was not configured. With no development anchor or no
    premium-based method it falls back to the simple average of the available
    results (no maturity blend is possible).

    All results must share the same accident periods and latest diagonal (i.e.
    come from the same base and triangle).
    """
    dev = next((r for r in results if r.method.startswith("Chain")), None)
    # A priori anchor: Bornhuetter-Ferguson; fall back to Cape Cod only if BF
    # was not configured.
    bf  = next((r for r in results if r.method.startswith("Bornhuetter")), None)
    cc  = next((r for r in results if r.method.startswith("Cape")), None)
    exp = bf if bf is not None else cc

    latest = np.asarray(results[0].latest_paid, dtype=float)
    cdfs   = np.asarray(results[0].cdfs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(cdfs > 0, 1.0 / cdfs, 1.0)   # completion factor (% reported)
    z = np.clip(z, 0.0, 1.0)

    if dev is not None and exp is not None:
        ibnr = (z * np.asarray(dev.ibnr, dtype=float)
                + (1.0 - z) * np.asarray(exp.ibnr, dtype=float))
    else:
        ibnr = np.mean([np.asarray(r.ibnr, dtype=float) for r in results], axis=0)

    ultimates = latest + ibnr
    with np.errstate(divide="ignore", invalid="ignore"):
        out_cdfs = np.where(latest != 0, ultimates / latest, np.nan)

    return IBNRResult(
        method=label,
        accident_periods=list(results[0].accident_periods),
        latest_paid=latest,
        ultimates=ultimates,
        ibnr=ibnr,
        cdfs=out_cdfs,
    )


def _round_to_100(vals) -> np.ndarray:
    """Round a percentage vector to integers summing to exactly 100 (largest
    remainder). Assumes the input already sums to ~100."""
    vals = np.asarray(vals, dtype=float)
    floors = np.floor(vals).astype(int)
    deficit = int(round(vals.sum())) - int(floors.sum())
    if deficit > 0:
        order = np.argsort(-(vals - floors))      # largest fractional parts first
        for i in order[:deficit]:
            floors[i] += 1
    return floors


def default_maturity_weights(results: list[IBNRResult]) -> pd.DataFrame:
    """
    Integer percentage weights (accident period × method) that sum to 100 per
    row, seeded from the maturity blend: Chain Ladder gets Z = 1/CDF (the
    completion factor) and the remainder is split evenly among the premium-based
    methods. A sensible starting point for the manual per-year weight grid.

    Returns a DataFrame indexed by accident period, columns = method names.
    """
    methods = [r.method for r in results]
    periods = list(results[0].accident_periods)
    cdfs = np.asarray(results[0].cdfs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(cdfs > 0, 1.0 / cdfs, 1.0)
    z = np.clip(z, 0.0, 1.0)

    dev_idx = [i for i, m in enumerate(methods) if m.startswith("Chain")]
    exp_idx = [i for i, m in enumerate(methods) if m.startswith(("Bornhuetter", "Cape"))]

    rows = []
    for p in range(len(periods)):
        w = np.zeros(len(methods))
        if dev_idx and exp_idx:
            w[dev_idx[0]] = z[p] * 100
            for i in exp_idx:
                w[i] = (1 - z[p]) * 100 / len(exp_idx)
        else:
            w[:] = 100.0 / len(methods)
        rows.append(_round_to_100(w))
    return pd.DataFrame(rows, index=periods, columns=methods)


def weighted_selection(results: list[IBNRResult], weight_matrix, label: str = "Best estimate") -> IBNRResult:
    """
    Combine methods into a best estimate using explicit per-accident-period
    weights — no normalisation. The caller supplies weights that already sum to
    1 per period (rows of ``weight_matrix``).

    Parameters
    ----------
    weight_matrix : array-like, shape (n_periods, n_methods)
        Aligned to the order of ``results``. Row p, column m is the weight of
        method m for accident period p.
    """
    W = np.asarray(weight_matrix, dtype=float)                       # (P, M)
    ibnr_by_method = np.column_stack(
        [np.asarray(r.ibnr, dtype=float) for r in results]
    )                                                                # (P, M)
    ibnr = np.nansum(W * ibnr_by_method, axis=1)                     # (P,)
    latest = np.asarray(results[0].latest_paid, dtype=float)
    ultimates = latest + ibnr
    with np.errstate(divide="ignore", invalid="ignore"):
        cdfs = np.where(latest != 0, ultimates / latest, np.nan)
    return IBNRResult(
        method=label,
        accident_periods=list(results[0].accident_periods),
        latest_paid=latest,
        ultimates=ultimates,
        ibnr=ibnr,
        cdfs=cdfs,
    )


def summarize_results(results: list[IBNRResult], premium: pd.Series | None = None) -> pd.DataFrame:
    """
    Build a side-by-side IBNR comparison table for all methods.

    Parameters
    ----------
    results : list of IBNRResult
    premium : pd.Series, optional
        If provided, adds an ultimate loss ratio column per method.

    Returns
    -------
    pd.DataFrame
        Columns: method names. Rows: accident periods + TOTAL.
    """
    ibnr_cols = {}
    for r in results:
        col = pd.Series(
            np.round(r.ibnr, 2),
            index=r.accident_periods,
            name=r.method,
        )
        ibnr_cols[r.method] = col

    df = pd.DataFrame(ibnr_cols)
    total_row = df.sum()
    total_row.name = "TOTAL"
    df = pd.concat([df, total_row.to_frame().T])

    return df
