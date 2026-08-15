# IBNR Reserve Estimation

Actuarial estimation of **Incurred But Not Reported (IBNR)** reserves using standard
development methods. The project ships as a **bilingual (English / Spanish) Streamlit web
app**, with the same methods also available as Python and R notebooks.

## 🚀 Live app — no install needed

### ▶ **[Open the app](https://ibnr-app.streamlit.app/)**

Just click the link — it runs in your browser, nothing to install.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ibnr-app.streamlit.app/)

## Methods

| Method | Description |
|---|---|
| **Chain Ladder** | Projects the latest diagonal using **simple-average** age-to-age (link ratio) factors |
| **Bornhuetter-Ferguson** | Blends the development pattern with an a priori expected loss ratio (ELR) |
| **Cape Cod (Stanard-Bühlmann)** | Like BF, but derives the ELR from used-up premium in the data |
| **Benktander (Gunnar-Benktander)** | Credibility blend of Chain Ladder and BF, weighted per accident year by the completion factor (% reported) |

> All methods use **simple-average** link ratios (equal weight per accident year). This keeps
> individual deviations visible for the anomaly-detection step.

## The app

A 6-screen wizard (`streamlit_app.py`):

1. **Upload** — load a claims file (CSV/Excel) or one of the built-in samples.
2. **Map columns** — map your columns to incurred date / paid date / paid amount, choose the
   date format, incremental vs. cumulative amounts, and the period grain (annual / quarterly).
   Optional mappings: a **case reserve (RSP)** column (enables the **incurred** triangle,
   incurred = paid + RSP), a **claim ID** (for correct per-claim handling of cumulative amounts
   and reserves), and a **segment** column. An earned-premium file (loaded on the Upload screen)
   unlocks Bornhuetter-Ferguson & Cape Cod.
3. **Review** — inspect the cumulative loss triangle; when case reserves were mapped, choose the
   **paid** or **incurred** base to continue with.
4. **Anomalies** — an interactive grid of individual development factors (FDIs) with IQR-based
   outlier flagging; exclude factors (with a required comment) to see the average update live.
5. **Configure** — choose methods, tail factor, and an optional ELR override.
6. **Results** — per-method IBNR by accident year, a **best estimate** (Benktander maturity blend
   or manual per-year weights), paid-vs-incurred **reconciliation**, maturity (% reported)
   highlighting, a segment filter, and CSV / Excel / PDF export.

## Project structure

```
IBNR/
├── streamlit_app.py        # Streamlit web app (6-screen wizard)
├── app/
│   ├── pipeline.py         # raw claims (CSV/Excel) → cumulative paid & incurred triangles
│   ├── methods.py          # Chain Ladder, BF, Cape Cod, Benktander; FDI/CDF math; IQR anomalies
│   ├── exports.py          # multi-sheet Excel (openpyxl) + PDF report (reportlab)
│   └── i18n.py             # English / Spanish translations
├── tests/                  # unittest suite (methods, pipeline, exports)
├── samples/                # sample claims / segmented / earned-premium CSVs
├── data/                   # standalone triangle + premium CSVs (used by the notebooks)
├── python/ibnr_analysis.ipynb   # Python notebook (pandas, numpy, matplotlib)
├── R/ibnr_analysis.Rmd          # R Markdown notebook (ggplot2, dplyr)
├── requirements.txt             # app dependencies (what Streamlit Cloud installs)
├── requirements-dev.txt         # + Jupyter/notebook extras for local development
└── README.md
```

## Getting started

### Streamlit app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Requires Python 3.10+ (the code uses `X | None` type hints).

### Tests

```bash
python -m unittest discover tests -v
```

### Notebooks

```bash
# Python (Jupyter comes from the dev extras)
pip install -r requirements-dev.txt
cd python && jupyter notebook ibnr_analysis.ipynb
```

```r
# R — open R/ibnr_analysis.Rmd in RStudio and Knit, or:
install.packages(c("ggplot2", "dplyr", "tidyr", "knitr", "scales"))
rmarkdown::render("R/ibnr_analysis.Rmd")
```

## Input data

**Claims file** (one row per payment): an incurred-date column, a paid-date column, and a
paid-amount column. Optionally a segment column (line of business, region, etc.). The app builds
the cumulative triangle from these.

**Earned-premium file** (optional, required for BF & Cape Cod): two columns —
`accident_year` and `earned_premium`.

The `data/` folder also holds a pre-built `paid_loss_triangle.csv` (10×10) and
`earned_premium.csv` used directly by the notebooks.

## Key concepts

**IBNR** — losses that have already occurred but have not yet been reported to the insurer.
Reserving for IBNR is a core actuarial function required for accurate financial statements.

**Triangle** — a matrix where rows are accident periods and columns are development ages. The
observed diagonal (latest data) is the starting point for projection.

**FDI / link ratio** — the age-to-age factor between two development columns for one accident
period. The simple average of these per column drives the projection.

**CDF (Cumulative Development Factor)** — the factor applied to the latest paid losses to
estimate ultimate losses. `CDF = 1.0` means fully developed.

**ELR (Expected Loss Ratio)** — used by BF and Cape Cod as a prior expectation.
`ELR = Expected Losses / Premium`.

**Completion factor / % reported** — the share of ultimate losses already reported,
`q = 1 / CDF`. It drives the **Benktander** best estimate: per accident year the reserve is
`q · Chain Ladder + (1 − q) · BF`, so mature years follow the data and immature years lean on the
a priori.
