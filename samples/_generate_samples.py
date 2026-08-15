"""
_generate_samples.py
--------------------
Deterministic generator for the IBNR demo sample files.

Model (one row per claim per development year, max one payment per year):
    Each claim has an ultimate U. It is paid down over the years following a
    development pattern, and carries an OUTSTANDING case reserve (RSP) =
    (U - cumulative_paid) * adequacy at the end of each year. The RSP is a
    pending balance (always >= 0) that runs off to 0 as the claim is paid:

        claim 5000 -> 2020 pay 1000, RSP 4000 | 2021 pay 1000, RSP 3000 | ...

    So  incurred(AY, dev) = cumulative_paid(AY, dev) + outstanding_RSP(AY, dev),
    and the reserve forms a SNAPSHOT triangle (not cumulated across development).

A mild above-trend 12->24 paid development is injected at AY 2022 so the IQR
anomaly detector flags it on the Anomalies screen.

Outputs (written next to this script):
    sample1_claims.csv      -> claim_id, incurred_date, paid_date, paid_amount, case_reserve
    sample2_segmented.csv   -> + line_of_business
    sample_earned_premium.csv is left untouched (2020-2024).

Run:  python3 samples/_generate_samples.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))

AYS = [2020, 2021, 2022, 2023, 2024]
# Development years observed per AY as of the 2024 year-end valuation.
N_OBS = {2020: 5, 2021: 4, 2022: 3, 2023: 2, 2024: 1}

LOB = [("Pharmacy", "PHA", 0.55), ("Medical", "MED", 0.32), ("Behavioral", "BEH", 0.13)]

CLAIMS_PER_AY = 60
MEAN_ULTIMATE = 22_000.0
RESERVE_ADEQUACY = 0.90        # case reserves cover ~90% of the true outstanding

# Cumulative paid % of ultimate by development year (dev 0..4).
BASE_CUM = [0.40, 0.66, 0.84, 0.94, 1.00]
# Per-AY cumulative-at-dev1, used to vary the 12->24 factor; AY 2022 is the anomaly.
CUM1_BY_AY = {2020: 0.66, 2021: 0.67, 2022: 0.80, 2023: 0.655, 2024: 0.66}


def cum_pattern(ay):
    c = list(BASE_CUM)
    c[1] = CUM1_BY_AY[ay]
    return c


def pick_lob():
    r = rng.random()
    acc = 0.0
    for name, code, w in LOB:
        acc += w
        if r <= acc:
            return name, code
    return LOB[-1][0], LOB[-1][1]


def generate():
    rows = []
    counters = {code: 0 for _, code, _ in LOB}

    for ay in AYS:
        cum = cum_pattern(ay)
        for _ in range(CLAIMS_PER_AY):
            name, code = pick_lob()
            counters[code] += 1
            claim_id = f"C{ay}-{code}-{counters[code]:05d}"
            ultimate = float(rng.lognormal(mean=np.log(MEAN_ULTIMATE) - 0.18, sigma=0.55))

            inc_month = int(rng.integers(1, 7))     # Jan-Jun
            inc_day = int(rng.integers(1, 28))
            incurred = f"{inc_month:02d}/{inc_day:02d}/{ay}"

            prev_cum = 0.0
            for j in range(N_OBS[ay]):
                cum_paid = ultimate * cum[j]
                inc_paid = cum_paid - prev_cum
                prev_cum = cum_paid
                rsp = (ultimate - cum_paid) * RESERVE_ADEQUACY   # outstanding, >= 0

                pay_year = ay + j
                # First year: pay in a later month than incurred (so paid >= incurred).
                pay_month = (int(rng.integers(inc_month + 1, 13)) if j == 0
                             else int(rng.integers(1, 13)))
                pay_day = int(rng.integers(1, 28))
                paid = f"{pay_month:02d}/{pay_day:02d}/{pay_year}"

                rows.append({
                    "claim_id": claim_id,
                    "incurred_date": incurred,
                    "paid_date": paid,
                    "paid_amount": round(inc_paid, 2),
                    "case_reserve": round(rsp, 2),
                    "line_of_business": name,
                })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    seg_cols = ["claim_id", "incurred_date", "paid_date", "paid_amount", "case_reserve", "line_of_business"]
    df[seg_cols].to_csv(os.path.join(HERE, "sample2_segmented.csv"), index=False)
    # sample1 = sample2 without the segment column (keeps claim_id).
    df[["claim_id", "incurred_date", "paid_date", "paid_amount", "case_reserve"]].to_csv(
        os.path.join(HERE, "sample1_claims.csv"), index=False
    )
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} claim-year rows ({df['claim_id'].nunique()} claims).")
    print(df["line_of_business"].value_counts())
