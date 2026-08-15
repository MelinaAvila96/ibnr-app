"""
pipeline.py
-----------
Transforms raw claims data (CSV or Excel) into a cumulative loss triangle.

Flow:
    load_file() → validate_columns() → build_triangle()

The triangle returned is a pandas DataFrame where:
    - rows    = accident periods (e.g. 2015, 2016 ...)
    - columns = development ages in months (e.g. 12, 24, 36 ...)
    - values  = cumulative paid losses
"""

import re
import difflib
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Date format mapping
# ---------------------------------------------------------------------------
DATE_FORMATS = {
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD/MM/YYYY": "%d/%m/%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
}


# ---------------------------------------------------------------------------
# Column auto-mapping (suggest which file column matches each required field)
# ---------------------------------------------------------------------------
FIELD_PATTERNS = {
    "claim_id":      ["claim id", "claim number", "claim no", "claim nbr", "claim",
                      "siniestro", "nro siniestro", "numero de siniestro",
                      "nro de siniestro", "id siniestro", "expediente"],
    "incurred_date": ["incurred date", "incurred", "accident date", "accident",
                      "loss date", "date of loss", "occurrence date", "fecha incurrido",
                      "fecha de incurrido", "fecha siniestro", "fecha ocurrencia", "ocurrencia"],
    "paid_date":     ["paid date", "payment date", "date paid", "fecha pago", "fecha de pago"],
    "paid_amount":   ["paid amount", "amount paid", "amount", "paid loss", "loss amount",
                      "payment amount", "monto pagado", "monto", "importe", "pago"],
    "reserve":       ["case reserve", "case reserves", "reserve", "outstanding", "o s",
                      "rsp", "reserva de caso", "reserva", "pendiente"],
    "segment":       ["line of business", "lob", "segment", "ramo", "region",
                      "linea de negocio", "linea", "producto", "clase"],
}


def _norm(s) -> str:
    """Lower-case and collapse separators so 'incurred_date' ~ 'incurred date'."""
    return re.sub(r"[\s_\-/.]+", " ", str(s).strip().lower()).strip()


def suggest_mapping(columns) -> dict:
    """
    Suggest a column for each required field by fuzzy name matching.

    Returns a dict {field: column} for the fields it could match with enough
    confidence. Each column is assigned to at most one field (greedy by score).
    Fields: incurred_date, paid_date, paid_amount, reserve, segment.
    """
    THRESHOLD = 0.6
    norm_cols = {c: _norm(c) for c in columns}
    pairs = []  # (score, field, column)
    for field, patterns in FIELD_PATTERNS.items():
        npats = [_norm(p) for p in patterns]
        for col, nc in norm_cols.items():
            score = 0.0
            for p in npats:
                if nc == p:
                    score = max(score, 1.0)
                elif p in nc or nc in p:
                    score = max(score, 0.9)
                else:
                    score = max(score, difflib.SequenceMatcher(None, nc, p).ratio())
            pairs.append((score, field, col))

    pairs.sort(key=lambda t: t[0], reverse=True)
    result, used_fields, used_cols = {}, set(), set()
    for score, field, col in pairs:
        if score < THRESHOLD:
            break
        if field in used_fields or col in used_cols:
            continue
        result[field] = col
        used_fields.add(field)
        used_cols.add(col)
    return result


# ---------------------------------------------------------------------------
# Date-format detection / validation
# ---------------------------------------------------------------------------

def _parses_all(values, fmt_key) -> bool:
    """True if every non-null value parses with the given DATE_FORMATS key."""
    s = pd.Series(list(values)).dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return True
    parsed = pd.to_datetime(s, format=DATE_FORMATS[fmt_key], errors="coerce")
    return bool(parsed.notna().all())


def infer_date_format(values, default="MM/DD/YYYY") -> str:
    """
    Best date format for these values. If exactly one supported format parses
    them all, return it; if several do (ambiguous, e.g. all days <= 12) keep
    `default`; if none parse, return `default`.
    """
    valid = [fmt for fmt in DATE_FORMATS if _parses_all(values, fmt)]
    if len(valid) == 1:
        return valid[0]
    if default in valid:
        return default
    return valid[0] if valid else default


def check_date_format(values, chosen_format):
    """
    Validate a chosen date format against actual values.

    Returns (ok, suggested):
      - ok=True, None        → chosen format parses every value.
      - ok=False, '<fmt>'    → it doesn't; this other supported format does
                               (e.g. 28/02/2020 fails MM/DD/YYYY → DD/MM/YYYY).
      - ok=False, None       → no supported format parses every value.
    """
    if _parses_all(values, chosen_format):
        return True, None
    for fmt in DATE_FORMATS:
        if fmt != chosen_format and _parses_all(values, fmt):
            return False, fmt
    return False, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the input file (.csv or .xlsx / .xls).

    Returns
    -------
    pd.DataFrame
        Raw data with original column names preserved.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload a .csv or .xlsx file."
        )

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    return df


def validate_columns(
    df: pd.DataFrame,
    incurred_col: str,
    paid_col: str,
    amount_col: str,
    segment_col: str | None = None,
    reserve_col: str | None = None,
) -> None:
    """
    Check that all mapped columns exist in the DataFrame.

    Raises
    ------
    ValueError
        With a descriptive message listing any missing columns.
    """
    required = {
        "incurred date": incurred_col,
        "paid date": paid_col,
        "paid amount": amount_col,
    }
    if segment_col:
        required["segment"] = segment_col
    if reserve_col:
        required["case reserve"] = reserve_col

    missing = [
        f"'{col}' (mapped to field '{field}')"
        for field, col in required.items()
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following columns were not found in the file:\n  "
            + "\n  ".join(missing)
        )


def _prepare_claims(
    df: pd.DataFrame,
    incurred_col: str,
    paid_col: str,
    date_format: str,
    grain: str,
    segment_col: str | None = None,
    segment_value: str | None = None,
) -> pd.DataFrame:
    """
    Shared preprocessing for triangle builders: segment filter, date parsing
    and validation, and the derived _accident_period / _dev_age columns.
    """
    data = df.copy()

    # --- optional segment filter -------------------------------------------
    if segment_col and segment_value:
        data = data[data[segment_col] == segment_value]
        if data.empty:
            raise ValueError(
                f"No rows found for segment '{segment_value}' "
                f"in column '{segment_col}'."
            )

    # --- parse dates -------------------------------------------------------
    fmt = DATE_FORMATS.get(date_format)
    if fmt is None:
        raise ValueError(
            f"Unknown date format '{date_format}'. "
            f"Choose from: {list(DATE_FORMATS.keys())}"
        )

    try:
        data[incurred_col] = pd.to_datetime(data[incurred_col], format=fmt)
        data[paid_col]     = pd.to_datetime(data[paid_col],     format=fmt)
    except Exception as e:
        raise ValueError(
            f"Could not parse dates with format '{date_format}'. "
            f"Check that your dates match the selected format.\nDetail: {e}"
        )

    # --- paid date must be >= incurred date --------------------------------
    bad_dates = (data[paid_col] < data[incurred_col]).sum()
    if bad_dates > 0:
        raise ValueError(
            f"{bad_dates} rows have a paid date earlier than the incurred date. "
            f"Check your date columns or date format selection."
        )

    # --- assign accident period and development age ------------------------
    data["_accident_period"] = _accident_period(data[incurred_col], grain)
    data["_dev_age"]         = _development_age(
        data[incurred_col], data[paid_col], grain
    )
    return data


def build_triangle(
    df: pd.DataFrame,
    incurred_col: str,
    paid_col: str,
    amount_col: str,
    date_format: str = "MM/DD/YYYY",
    amount_type: str = "incremental",
    grain: str = "annual",
    segment_col: str | None = None,
    segment_value: str | None = None,
    cumulate: bool = True,
    claim_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a cumulative loss triangle from claim-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw claims data.
    incurred_col : str
        Column name for the date the loss was incurred.
    paid_col : str
        Column name for the date the payment was made.
    amount_col : str
        Column name for the payment amount.
    date_format : str
        One of 'MM/DD/YYYY' (default), 'DD/MM/YYYY', 'YYYY-MM-DD'.
    amount_type : str
        'incremental' (default) — each row is a single payment.
        'cumulative'  — each row is cumulative paid to date per claim/period.
    grain : str
        'annual' (default) — accident years, development in 12m steps.
        'quarterly'        — accident quarters, development in 3m steps.
    segment_col : str, optional
        Column to filter by before building the triangle.
    segment_value : str, optional
        Value to filter on in segment_col.
    claim_col : str, optional
        Claim identifier column. Required to convert cumulative amounts
        correctly when an accident period holds more than one claim.

    Returns
    -------
    pd.DataFrame
        Cumulative loss triangle.
        Index   = accident periods.
        Columns = development ages (in months).
        Missing cells are NaN.
    """
    data = _prepare_claims(
        df, incurred_col, paid_col, date_format, grain,
        segment_col, segment_value,
    )

    # --- validate amount column --------------------------------------------
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")
    n_invalid = data[amount_col].isna().sum()
    if n_invalid > 0:
        raise ValueError(
            f"{n_invalid} rows in '{amount_col}' could not be converted to "
            f"numbers. Check for text or empty values."
        )

    # --- if cumulative, convert to incremental first -----------------------
    if amount_type == "cumulative":
        data = _cumulative_to_incremental(data, amount_col, claim_col, paid_col)

    # --- aggregate: sum incremental payments per (accident, dev) -----------
    grouped = (
        data.groupby(["_accident_period", "_dev_age"])[amount_col]
        .sum()
        .reset_index()
    )

    # --- pivot to triangle shape -------------------------------------------
    triangle_inc = grouped.pivot(
        index="_accident_period", columns="_dev_age", values=amount_col
    )
    triangle_inc = triangle_inc.sort_index(axis=0).sort_index(axis=1)

    # The valuation date implied by the data (latest cell with activity) —
    # used both to complete the grid and to mask future cells below.
    as_of = _latest_observed_ordinal(triangle_inc, grain)

    # Complete the development grid at regular steps: a dev age with no
    # activity anywhere would otherwise vanish from the grid (the link ratio
    # would span several real periods), and older accident periods must reach
    # the valuation date so their observed flat development counts.
    step = {"annual": 12, "quarterly": 3}[grain]
    if len(triangle_inc.columns) and as_of is not None:
        lo = int(min(triangle_inc.columns))
        hi = int(max(triangle_inc.columns))
        hi_diag = max(
            (as_of - _period_ordinal(p, grain) + 1) * step
            for p in triangle_inc.index
        )
        hi = max(hi, hi_diag)
        triangle_inc = triangle_inc.reindex(columns=range(lo, hi + step, step))

    triangle_inc.index.name   = "accident_period"
    triangle_inc.columns.name = "development_age"

    # --- cumulate across development ages ----------------------------------
    # Cells with no payment activity are 0 incremental, so the cumulative
    # carries flat across them (a NaN hole inside the observed region would
    # otherwise shift the latest diagonal and misassign CDFs). Future cells
    # are re-masked to NaN below.
    # When cumulate=False the per-cell sums are returned as-is (used for the
    # case-reserve snapshot, which is an outstanding balance, not something to
    # accumulate across development).
    triangle_out = triangle_inc.fillna(0).cumsum(axis=1) if cumulate else triangle_inc

    # --- mask future cells (upper-right of the diagonal) ------------------
    triangle_out = _mask_future(triangle_out, grain, as_of)

    return triangle_out


def build_paid_incurred(
    df: pd.DataFrame,
    incurred_col: str,
    paid_col: str,
    amount_col: str,
    reserve_col: str | None = None,
    date_format: str = "MM/DD/YYYY",
    amount_type: str = "incremental",
    reserve_amount_type: str = "level",
    grain: str = "annual",
    segment_col: str | None = None,
    segment_value: str | None = None,
    claim_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Build the paid triangle and (if a case-reserve column is given) the incurred
    triangle.

    The case reserve (RSP) is an **outstanding balance** (the amount still pending
    to be paid as of each valuation), not an increment — it is ≥ 0 and runs off to
    0 as the claim is paid. So the reserve forms a **snapshot** triangle that is NOT
    cumulated across development, and the incurred triangle is:

        incurred(AY, dev) = cumulative_paid(AY, dev) + outstanding_RSP(AY, dev)

    Equivalently, the incremental incurred in a period is the incremental paid plus
    the change in the reserve (RSP_t − RSP_(t-1)).

    Parameters
    ----------
    reserve_col : str, optional
        Column holding the case reserve (RSP). Missing/NaN values are treated as 0.
        If not provided, only the paid triangle is built and the incurred triangle
        is returned as ``None``.
    reserve_amount_type : str
        How the reserve column is expressed:
        'level'    (default) — outstanding RSP at each valuation (a snapshot; not
                   accumulated across development).
        'movement' — incremental change in reserve per period; cumulated across
                   development to reconstruct the outstanding balance.
    claim_col : str, optional
        Claim identifier. With it, the 'level' reserve is read per claim —
        latest RSP within each period, carried forward through periods with no
        payment activity — so several payments of one claim in a period no
        longer double-count the reserve, and the balance persists until it
        actually changes. Without it, each row's RSP is summed per cell, which
        assumes at most one row per claim and period.

    Returns
    -------
    (triangle_paid, triangle_incurred)
        ``triangle_incurred`` is ``None`` when ``reserve_col`` is not provided.
    """
    triangle_paid = build_triangle(
        df,
        incurred_col=incurred_col,
        paid_col=paid_col,
        amount_col=amount_col,
        date_format=date_format,
        amount_type=amount_type,
        grain=grain,
        segment_col=segment_col,
        segment_value=segment_value,
        claim_col=claim_col,
    )

    if not reserve_col:
        return triangle_paid, None

    reserve_is_movement = reserve_amount_type == "movement"
    data = df.copy()
    data[reserve_col] = pd.to_numeric(data[reserve_col], errors="coerce").fillna(0)

    if not reserve_is_movement and claim_col:
        # 'level' with claim id → per-claim outstanding snapshot, carried
        # forward across development until it changes.
        prepared = _prepare_claims(
            data, incurred_col, paid_col, date_format, grain,
            segment_col, segment_value,
        )
        reserve_tri = _reserve_snapshot_by_claim(
            prepared, claim_col, reserve_col, paid_col,
            dev_grid=triangle_paid.columns.tolist(),
        )
    else:
        # 'movement' → incremental reserve changes, cumulated across
        # development. 'level' without claim id → RSP summed per (AY, dev)
        # as-is (assumes one row per claim and period).
        reserve_tri = build_triangle(
            data,
            incurred_col=incurred_col,
            paid_col=paid_col,
            amount_col=reserve_col,
            date_format=date_format,
            amount_type="incremental",   # rows are per-cell values; sum them
            grain=grain,
            segment_col=segment_col,
            segment_value=segment_value,
            cumulate=reserve_is_movement,
        )

    # incurred = cumulative paid + outstanding reserve (aligned to the paid grid).
    reserve_aligned = reserve_tri.reindex(
        index=triangle_paid.index, columns=triangle_paid.columns
    )
    triangle_incurred = triangle_paid + reserve_aligned.fillna(0)
    # Keep future cells (NaN in the paid triangle) as NaN.
    triangle_incurred = triangle_incurred.where(~triangle_paid.isna())
    return triangle_paid, triangle_incurred


def _reserve_snapshot_by_claim(
    data: pd.DataFrame,
    claim_col: str,
    reserve_col: str,
    paid_col: str,
    dev_grid: list,
) -> pd.DataFrame:
    """
    Outstanding-reserve (RSP) snapshot triangle read per claim.

    For each (claim, accident period, development age) the RSP is the value on
    the claim's latest payment row within that period (an outstanding balance,
    so the last valuation wins — summing rows would double-count). The balance
    is then carried forward across later development ages with no activity,
    and claims are summed per (accident period, development age) cell.

    `dev_grid` is the paid triangle's development-age grid, so the snapshot
    aligns cell-for-cell with the paid triangle (future cells are masked by
    the caller via the paid triangle's NaNs).
    """
    d = data.sort_values([claim_col, "_dev_age", paid_col], kind="stable")
    last_rsp = d.groupby([claim_col, "_accident_period", "_dev_age"])[reserve_col].last()

    per_claim = last_rsp.unstack("_dev_age")
    all_devs  = sorted(set(dev_grid) | set(per_claim.columns))
    per_claim = per_claim.reindex(columns=all_devs)
    # Carry the outstanding balance forward through inactive periods. Leading
    # NaNs (before the claim's first valuation) stay NaN → count as 0.
    per_claim = per_claim.ffill(axis=1)

    snap = per_claim.groupby(level="_accident_period").sum()
    snap.index.name   = "accident_period"
    snap.columns.name = "development_age"
    return snap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _accident_period(incurred: pd.Series, grain: str) -> pd.Series:
    """Return the accident period label for each incurred date."""
    if grain == "annual":
        return incurred.dt.year
    elif grain == "quarterly":
        return incurred.dt.to_period("Q").astype(str)
    else:
        raise ValueError(f"Unknown grain '{grain}'. Choose: annual, quarterly.")


def _development_age(incurred: pd.Series, paid: pd.Series, grain: str) -> pd.Series:
    """
    Return development age in months for each row.

    For annual grain:
        dev_age = (year(paid) - year(incurred) + 1) * 12
        e.g. paid same year → 12m, paid 1 year later → 24m
    For quarterly grain:
        dev_age = (quarter(paid) - quarter(incurred) + 1) * 3
    """
    if grain == "annual":
        return (paid.dt.year - incurred.dt.year + 1) * 12
    elif grain == "quarterly":
        paid_q     = paid.dt.year * 4 + paid.dt.quarter
        incurred_q = incurred.dt.year * 4 + incurred.dt.quarter
        return (paid_q - incurred_q + 1) * 3
    else:
        raise ValueError(f"Unknown grain '{grain}'.")


def _cumulative_to_incremental(
    df: pd.DataFrame,
    amount_col: str,
    claim_col: str | None = None,
    paid_col: str | None = None,
) -> pd.DataFrame:
    """
    Convert cumulative amounts to incremental by differencing along the
    development axis.

    With `claim_col`, the difference is taken within each claim (correct when
    an accident period holds several claims). Without it, the difference is
    within the accident period, which is only valid when there is at most one
    row per (accident period, development age) — mixing rows from different
    claims would subtract unrelated cumulatives, so that case raises instead.
    """
    if claim_col:
        sort_keys = [claim_col, "_accident_period", "_dev_age"]
        if paid_col:
            sort_keys.append(paid_col)
        df = df.sort_values(sort_keys, kind="stable")
        group_keys = [claim_col, "_accident_period"]
    else:
        if df.duplicated(["_accident_period", "_dev_age"]).any():
            raise ValueError(
                "Cumulative amounts with several rows per accident period and "
                "development age cannot be converted safely: map a Claim ID "
                "column (so amounts are differenced per claim) or provide one "
                "row per period."
            )
        df = df.sort_values(["_accident_period", "_dev_age"], kind="stable")
        group_keys = ["_accident_period"]

    df[amount_col] = df.groupby(group_keys)[amount_col].diff().fillna(
        df[amount_col]
    )
    return df


def _period_ordinal(label, grain: str) -> int:
    """
    Convert an accident-period label to an integer calendar ordinal so that
    consecutive periods differ by 1, regardless of grain.

        annual    : 2015          -> 2015
        quarterly : '2015Q3'      -> 2015*4 + 3
    """
    if grain == "annual":
        return int(label)
    if grain == "quarterly":
        year, q = str(label).split("Q")
        return int(year) * 4 + int(q)
    raise ValueError(f"Unknown grain '{grain}'.")


def _dev_units(dev_age: int, grain: str) -> int:
    """Convert a development age in months to the number of grain periods."""
    step = {"annual": 12, "quarterly": 3}[grain]
    return dev_age // step


def _latest_observed_ordinal(triangle: pd.DataFrame, grain: str) -> int | None:
    """
    Calendar ordinal of the latest cell with actual activity — the valuation
    date implied by the data. A cell (period p, dev age d) is observed at
    ordinal ord(p) + dev_units(d) - 1.
    """
    best = None
    for period in triangle.index:
        row = triangle.loc[period]
        known = row[row.notna()]
        if known.empty:
            continue
        last_dev    = known.index[-1]
        observed_at = _period_ordinal(period, grain) + _dev_units(last_dev, grain) - 1
        best = observed_at if best is None else max(best, observed_at)
    return best


def _mask_future(triangle: pd.DataFrame, grain: str = "annual",
                 as_of: int | None = None) -> pd.DataFrame:
    """
    Set future cells (upper-right of the diagonal) to NaN, for any grain.

    A cell (accident period p, development age d) is observed at calendar
    ordinal  ord(p) + dev_units(d) - 1.  Cells observed beyond `as_of`
    (the valuation ordinal) are in the future. When `as_of` is not given it
    is derived from the triangle's own non-NaN cells.
    """
    periods  = triangle.index.tolist()
    dev_ages = triangle.columns.tolist()
    if not periods or not dev_ages:
        return triangle

    ordinals = {p: _period_ordinal(p, grain) for p in periods}
    if as_of is None:
        as_of = _latest_observed_ordinal(triangle, grain)
    if as_of is None:
        return triangle

    masked = triangle.copy()
    for period in periods:
        for dev in dev_ages:
            observed_at = ordinals[period] + _dev_units(dev, grain) - 1
            if observed_at > as_of:
                masked.loc[period, dev] = np.nan

    return masked


def get_segments(df: pd.DataFrame, segment_col: str) -> list:
    """Return sorted list of unique values in the segment column."""
    return sorted(df[segment_col].dropna().unique().tolist())
