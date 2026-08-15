"""
streamlit_app.py
----------------
IBNR Reserve Estimation — Streamlit App

Run with:
    streamlit run streamlit_app.py

Screens
-------
1. Upload       — load claims file (CSV / Excel)
2. Map columns  — map file columns to required fields + settings
3. Review       — cumulative loss triangle
4. Anomalies    — FDI individual factors, anomaly detection + exclusions
5. Configure    — select methods, tail factor, ELR override
6. Results      — IBNR tables, charts, segment filter, export
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io

from app.pipeline import (
    load_file, validate_columns, build_triangle, build_paid_incurred,
    get_segments, DATE_FORMATS, suggest_mapping, infer_date_format, check_date_format,
)
from app.methods import (
    build_metrics, compute_fdi, detect_anomalies,
    chain_ladder, bornhuetter_ferguson, cape_cod,
    summarize_results, IBNRResult, align_premium, reserve_decomposition,
    credibility_weighted_result, weighted_selection, default_maturity_weights,
)
from app.i18n import t as _translate

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IBNR Estimation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 960px; }
    .stProgress > div > div { background-color: #1f77b4; }
    /* Metric cards use a fixed light background, so pin dark text on them —
       otherwise the value/label are white-on-light (invisible) in dark theme.
       The label is a <label> (not a div) and its text sits in a nested <p>, so
       target every descendant, not just a direct child div. */
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 0.75rem 1rem; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #1f2937 !important; }
    [data-testid="stMetricValue"] > div { font-size: 1.3rem !important; white-space: nowrap; overflow: visible; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #4b5563 !important; }
    [data-testid="stMetricLabel"] > div { font-size: 0.8rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* Radio option labels (Paid / Incurred base selector, etc.) — a touch larger */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p { font-size: 1.05rem; }

    /* FDI grid buttons — compact monospace */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"],
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        font-family: monospace;
        font-size: 11px !important;
        padding: 3px 1px !important;
        min-height: 0px !important;
        width: 100%;
        /* Keep each factor on a single line — many quarterly columns make the
           cells narrow, and without this the number breaks into 2-3 lines. */
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: hidden;
    }
    div[data-testid="stHorizontalBlock"] button p {
        white-space: nowrap !important;
        word-break: keep-all !important;
        margin: 0 !important;
    }
    /* FDI label/value cells rendered as plain divs (headers, AY, avg, NaN). */
    .fdi-lbl { white-space: nowrap; overflow: hidden; text-overflow: clip; }
    /* Excluded cell styling (static div) */
    .fdi-excluded {
        text-align: center;
        font-size: 11px;
        font-family: monospace;
        color: #aaa;
        text-decoration: line-through;
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px 1px;
        white-space: nowrap;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
DEFAULTS = {
    "screen":          1,
    "max_screen":      1,      # furthest step reached — drives the sidebar index
    "lang":            "en",   # "en" or "es"
    "raw_df":          None,
    "premium_df":      None,
    "incurred_col":    None,
    "paid_col":        None,
    "amount_col":      None,
    "reserve_col":     None,    # case reserve (RSP) column; enables incurred base
    "claim_col":       None,    # claim id; per-claim cumulative and RSP handling
    "segment_col":     None,
    "date_format":     "MM/DD/YYYY",
    "amount_type":     "Incremental",   # paid amount: Incremental | Cumulative
    "reserve_amount_type": "Level",      # RSP: Level (outstanding) | Movement
    "grain":           "Annual",
    # Per-base data: "paid" always present, "incurred" present only if reserve mapped.
    "triangle_paid":     None,
    "triangle_incurred": None,
    "fdi_paid":          None,
    "fdi_incurred":      None,
    "flags_paid":        None,
    "flags_incurred":    None,
    "exclusions_paid":     {},  # {(period, col): comment}
    "exclusions_incurred": {},
    "active_base":       "paid",  # base shown/edited on the anomaly screen
    # Legacy mirrors of the active base (kept so downstream code is unchanged).
    "triangle":        None,
    "fdi_table":       None,
    "anomaly_flags":   None,
    "exclusions":      {},     # {(period, col): comment}
    "selected_cell":   None,   # (period, col) currently selected in FDI grid
    "tail":            1.0,
    "elr_override":    None,
    "run_cl":          True,
    "run_bf":          True,
    "run_cc":          False,
    "results":         None,         # active-base results (back-compat)
    "results_by_base": None,         # {"paid": [...], "incurred": [...]}
    "segment_value":   None,
    "_sample_name":    "",
    "_premium_name":   "",
    "_dfmt_auto":      False,   # date format auto-detected once per dataset
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# i18n shortcut — always reads current lang from session state
# ---------------------------------------------------------------------------
def _(key: str, **kwargs) -> str:
    return _translate(key, lang=st.session_state.get("lang", "en"), **kwargs)

# ---------------------------------------------------------------------------
# Language selector (sidebar — accessible via hamburger / 3-dot menu)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 Language / Idioma")
    lang_choice = st.radio(
        label="language_selector",
        options=["English", "Español"],
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed",
    )
    new_lang = "en" if lang_choice == "English" else "es"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def go_to(screen: int):
    st.session_state.screen = screen
    st.session_state.max_screen = max(st.session_state.get("max_screen", 1), screen)

def scroll_to_top():
    """Reset the main container's scroll to the top. Streamlit keeps the scroll
    position across reruns, so without this a new step can open mid-page."""
    components.html(
        """
        <script>
        const d = window.parent.document;
        const main = d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main');
        if (main) main.scrollTo({top: 0, left: 0, behavior: 'auto'});
        </script>
        """,
        height=0,
    )

def reset_for_new_data():
    """Clear the column mapping and date auto-detect so a freshly loaded file
    gets fresh auto-suggestions."""
    for k in ["incurred_col", "paid_col", "amount_col", "reserve_col",
              "claim_col", "segment_col"]:
        st.session_state[k] = None
    st.session_state._dfmt_auto = False
    # New data invalidates downstream steps — collapse the sidebar index.
    st.session_state.max_screen = 1

def has_incurred() -> bool:
    """True when a case-reserve column was mapped and an incurred triangle exists."""
    return st.session_state.get("triangle_incurred") is not None

def sync_active_base(base: str | None = None):
    """
    Point the legacy mirrors (triangle / fdi_table / anomaly_flags / exclusions)
    at the chosen base so the rest of the app keeps working unchanged.
    """
    if base is not None:
        st.session_state.active_base = base
    b = st.session_state.active_base
    if b == "incurred" and not has_incurred():
        b = st.session_state.active_base = "paid"
    st.session_state.triangle      = st.session_state.get(f"triangle_{b}")
    st.session_state.fdi_table     = st.session_state.get(f"fdi_{b}")
    st.session_state.anomaly_flags = st.session_state.get(f"flags_{b}")
    st.session_state.exclusions    = st.session_state.get(f"exclusions_{b}")

def grain_key(grain: str) -> str:
    return {"Annual": "annual", "Quarterly": "quarterly",
            "Anual": "annual", "Trimestral": "quarterly"}.get(grain, "annual")

def fmt_number(v, decimals=0):
    if pd.isna(v): return "—"
    return f"{v:,.{decimals}f}"

def run_configured_methods(tri, exclusions=None):
    """
    Run the methods the user selected (with their tail / ELR / exclusions)
    against the given triangle. Returns a list of IBNRResult.

    Used on the Configure screen, per base when comparing paid vs incurred, and
    when re-running for a segment on the Results screen — so every path produces
    identical, consistent output.
    """
    tail = st.session_state.tail
    elr  = st.session_state.elr_override
    excl = st.session_state.exclusions if exclusions is None else exclusions

    prem_df = st.session_state.premium_df
    prem = None
    if prem_df is not None:
        prem = prem_df.set_index(prem_df.columns[0])[prem_df.columns[1]]

    results = []
    if st.session_state.run_cl:
        results.append(chain_ladder(tri, tail, excl))
    if st.session_state.run_bf and prem is not None:
        results.append(bornhuetter_ferguson(tri, prem, elr, tail, excl))
    if st.session_state.run_cc and prem is not None:
        results.append(cape_cod(tri, prem, tail, excl))
    return results

def step_bar(current: int):
    labels = [
        _("step_upload"),
        _("step_map"),
        _("step_review"),
        _("step_anomalies"),
        _("step_configure"),
        _("step_results"),
    ]
    cols = st.columns(len(labels))
    for i, (col, label) in enumerate(zip(cols, labels)):
        step = i + 1
        if step < current:
            col.markdown(f"<p style='text-align:center;color:#1f77b4;font-size:13px;'>✓ {label}</p>", unsafe_allow_html=True)
        elif step == current:
            col.markdown(f"<p style='text-align:center;font-weight:600;font-size:13px;'>● {label}</p>", unsafe_allow_html=True)
        else:
            col.markdown(f"<p style='text-align:center;color:#aaa;font-size:13px;'>○ {label}</p>", unsafe_allow_html=True)
    st.divider()

# ---------------------------------------------------------------------------
# Screen 1 — Upload
# ---------------------------------------------------------------------------
def screen_upload():
    step_bar(1)

    # Language can be switched anytime from the sidebar selector (🌐).
    st.caption(_("language_sidebar_hint"))

    st.header(_("upload_title"))
    st.caption(_("upload_caption"))

    # --- Two clear ways to load data, side by side ----------------------------
    opt_own, opt_sample = st.columns(2, gap="large")

    with opt_own:
        with st.container(border=True):
            st.markdown(f"#### {_('upload_opt_own')}")
            st.caption(_("upload_opt_own_hint"))
            uploaded = st.file_uploader(
                _("upload_label"), type=["csv", "xlsx", "xls"],
                label_visibility="collapsed",
            )

    with opt_sample:
        with st.container(border=True):
            st.markdown(f"#### {_('upload_opt_sample')}")
            st.caption(_("upload_opt_sample_hint"))
            for key, path in [
                ("sample1", "samples/sample1_claims.csv"),
                ("sample2", "samples/sample2_segmented.csv"),
            ]:
                try:
                    if st.button(_(key), use_container_width=True, key=f"btn_{key}"):
                        df_sample = pd.read_csv(path)
                        st.session_state.raw_df = df_sample
                        st.session_state._sample_name = path.split("/")[-1]
                        st.session_state.premium_df = None
                        st.session_state._premium_name = ""
                        st.session_state.run_cc = False
                        reset_for_new_data()
                        st.rerun()
                except FileNotFoundError:
                    pass

            try:
                if st.button(_("sample3"), use_container_width=True, key="btn_sample3"):
                    df_sample = pd.read_csv("samples/sample1_claims.csv")
                    st.session_state.raw_df = df_sample
                    st.session_state._sample_name = "sample1_claims.csv"
                    prem_sample = pd.read_csv("samples/sample_earned_premium.csv")
                    st.session_state.premium_df = prem_sample
                    st.session_state._premium_name = "sample_earned_premium.csv"
                    st.session_state.run_cc = True
                    reset_for_new_data()
                    st.rerun()
            except FileNotFoundError:
                pass
            st.caption(_("sample3_hint"))

    # Handle manual file upload
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.raw_df = df
            st.session_state._sample_name = uploaded.name
            st.session_state.premium_df = None
            st.session_state._premium_name = ""
            st.session_state.run_cc = False
            reset_for_new_data()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # --- Loaded confirmation: make it obvious the data is ready ----------------
    df_loaded = st.session_state.raw_df
    prem_loaded = st.session_state.premium_df
    st.divider()
    if df_loaded is not None:
        sample_name = st.session_state.get("_sample_name", "")
        two_files = prem_loaded is not None

        st.success(f"### ✓ {_('upload_ready_title')}")

        with st.container(border=True):
            # Claims summary line
            st.markdown(
                f"**{_('upload_claims_loaded')}** · `{sample_name}` — "
                + _("upload_success", rows=len(df_loaded), cols=len(df_loaded.columns))
            )
            st.dataframe(df_loaded.head(5), use_container_width=True)

            # Earned premium summary (Sample 3 / Cape Cod)
            if two_files:
                prem_name = st.session_state.get("_premium_name", "")
                st.markdown(
                    f"**{_('upload_premium_loaded')}** · `{prem_name}` — "
                    + _("upload_success", rows=len(prem_loaded), cols=len(prem_loaded.columns))
                    + f"  ·  {_('premium_methods_unlocked')}"
                )
                st.dataframe(prem_loaded.head(5), use_container_width=True)

        st.button(_("next"), type="primary", use_container_width=True, on_click=go_to, args=(2,))
    else:
        st.info(_("upload_empty"))

# ---------------------------------------------------------------------------
# Screen 2 — Map columns
# ---------------------------------------------------------------------------
def screen_map():
    step_bar(2)
    st.header(_("map_title"))
    st.caption(_("map_caption"))

    df   = st.session_state.raw_df
    none = _("map_none")
    sel  = _("map_select")
    cols = [sel] + df.columns.tolist()

    # Auto-suggest column mapping by name similarity.
    sugg = suggest_mapping(df.columns.tolist())
    if sugg:
        st.caption(_("map_autosuggest"))

    def _idx(stored, options, field=None):
        """Index for a selectbox: stored value, else auto-suggested column, else 0."""
        if stored in options:
            return options.index(stored)
        guess = sugg.get(field) if field else None
        return options.index(guess) if guess in options else 0

    # Auto-detect the date format once per dataset, from the suggested incurred column.
    if not st.session_state._dfmt_auto:
        guess_inc = sugg.get("incurred_date") or sugg.get("paid_date")
        if guess_inc is not None:
            st.session_state.date_format = infer_date_format(
                df[guess_inc].dropna().astype(str).head(300),
                default=st.session_state.date_format,
            )
        st.session_state._dfmt_auto = True

    c1, c2 = st.columns(2)

    opt_cols = [none] + df.columns.tolist()
    with c1:
        st.subheader(_("map_required"))
        inc = st.selectbox(_("map_incurred"), cols,
            index=_idx(st.session_state.incurred_col, cols, "incurred_date"),
            help=_("map_hint_incurred"))
        paid = st.selectbox(_("map_paid"), cols,
            index=_idx(st.session_state.paid_col, cols, "paid_date"),
            help=_("map_hint_paid"))
        amt = st.selectbox(_("map_amount"), cols,
            index=_idx(st.session_state.amount_col, cols, "paid_amount"))

        st.subheader(_("map_optional"))
        reserve = st.selectbox(_("map_reserve"), opt_cols,
            index=_idx(st.session_state.reserve_col, opt_cols, "reserve"),
            help=_("map_hint_reserve"))
        claim = st.selectbox(_("map_claim"), opt_cols,
            index=_idx(st.session_state.claim_col, opt_cols, "claim_id"),
            help=_("map_hint_claim"))
        seg = st.selectbox(_("map_segment"), opt_cols,
            index=_idx(st.session_state.segment_col, opt_cols, "segment"))

        # RSP read as an outstanding balance really needs the claim id to
        # avoid double counting — nudge when it's missing.
        if reserve != none and claim == none:
            st.info(_("map_claim_reco"))

    with c2:
        st.subheader(_("map_date_format"))
        date_fmt = st.radio(_("map_date_format"), list(DATE_FORMATS.keys()),
            index=list(DATE_FORMATS.keys()).index(st.session_state.date_format),
            label_visibility="collapsed",
            help="Select the format that matches your incurred and paid date columns.")

        st.subheader(_("map_amount_type"))
        amt_opts = [_("map_incremental"), _("map_cumulative")]
        amt_type_label = st.radio(_("map_amount_type_paid"), amt_opts,
            index=0 if st.session_state.amount_type == "Incremental" else 1,
            help=f'{_("map_hint_incremental")} / {_("map_hint_cumulative")}')
        # Normalise back to English key for storage
        amt_type = "Incremental" if amt_type_label == _("map_incremental") else "Cumulative"

        # Reserve (RSP) has its own nature — only ask when a reserve is mapped.
        res_type = "Level"
        if reserve != none:
            res_opts = [_("map_res_level"), _("map_res_movement")]
            res_type_label = st.radio(_("map_amount_type_res"), res_opts,
                index=0 if st.session_state.reserve_amount_type == "Level" else 1,
                help=f'{_("map_hint_res_level")} / {_("map_hint_res_movement")}')
            res_type = "Level" if res_type_label == _("map_res_level") else "Movement"

        st.subheader(_("map_grain"))
        grain_opts = [_("map_annual"), _("map_quarterly")]
        stored_grain = st.session_state.grain
        grain_idx = 1 if stored_grain in ("Quarterly", "Trimestral") else 0
        grain_label = st.radio(_("map_grain"), grain_opts, index=grain_idx,
                                label_visibility="collapsed")
        grain_map = {
            _("map_annual"):    "Annual",
            _("map_quarterly"): "Quarterly",
        }
        grain = grain_map.get(grain_label, "Annual")

    # Validate the chosen date format against the actual date columns.
    for col in [inc, paid]:
        if col != sel:
            ok, suggd = check_date_format(df[col].dropna().astype(str).head(300), date_fmt)
            if not ok and suggd:
                st.warning(_("map_date_mismatch", col=col, sugg=suggd))
            elif not ok:
                st.warning(_("map_date_unparseable", col=col))

    st.divider()
    if st.session_state.premium_df is not None:
        st.info(_("map_cc_unlocked"))
    else:
        st.warning(_("map_cc_locked") + " — " + _("map_cc_locked_hint"))

    st.divider()
    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        st.button(_("back"), on_click=go_to, args=(1,))
    with bcol2:
        if st.button(_("next"), type="primary"):
            errors = []
            if inc  == sel:  errors.append(_("map_err_incurred"))
            if paid == sel:  errors.append(_("map_err_paid"))
            if amt  == sel:  errors.append(_("map_err_amount"))
            if inc == paid:  errors.append(_("map_err_same_col"))

            reserve_col = None if reserve == none else reserve

            warnings = []
            if amt != sel and amt in (inc, paid):
                warnings.append(_("map_err_amount_dup"))
            elif amt != sel and pd.to_numeric(df[amt], errors="coerce").isna().all():
                warnings.append(_("map_err_amount_numeric", col=amt))
            if reserve_col and pd.to_numeric(df[reserve_col], errors="coerce").isna().all():
                warnings.append(_("map_err_reserve_numeric", col=reserve_col))

            if errors or warnings:
                for e in errors:   st.error(e)
                for w in warnings: st.warning(w)
            else:
                st.session_state.incurred_col = inc
                st.session_state.paid_col     = paid
                st.session_state.amount_col   = amt
                st.session_state.reserve_col  = reserve_col
                st.session_state.claim_col    = None if claim == none else claim
                st.session_state.segment_col  = None if seg == none else seg
                st.session_state.date_format  = date_fmt
                st.session_state.amount_type  = amt_type
                st.session_state.reserve_amount_type = res_type
                st.session_state.grain        = grain

                try:
                    tri_paid, tri_inc = build_paid_incurred(
                        df,
                        incurred_col  = inc,
                        paid_col      = paid,
                        amount_col    = amt,
                        reserve_col   = reserve_col,
                        date_format   = date_fmt,
                        amount_type   = amt_type.lower(),
                        reserve_amount_type = res_type.lower(),
                        grain         = grain_key(grain),
                        segment_col   = st.session_state.segment_col,
                        claim_col     = st.session_state.claim_col,
                    )

                    st.session_state.triangle_paid     = tri_paid
                    st.session_state.fdi_paid          = compute_fdi(tri_paid)
                    st.session_state.flags_paid        = detect_anomalies(st.session_state.fdi_paid)
                    st.session_state.exclusions_paid   = {}

                    if tri_inc is not None:
                        st.session_state.triangle_incurred   = tri_inc
                        st.session_state.fdi_incurred        = compute_fdi(tri_inc)
                        st.session_state.flags_incurred      = detect_anomalies(st.session_state.fdi_incurred)
                        st.session_state.exclusions_incurred = {}
                    else:
                        st.session_state.triangle_incurred = None

                    # Default to the incurred base when case reserves gave us one.
                    default_base = "incurred" if tri_inc is not None else "paid"
                    st.session_state.active_base = default_base
                    sync_active_base(default_base)
                    go_to(3)
                    st.rerun()

                except Exception as e:
                    st.error(_("map_err_triangle", e=e))

# ---------------------------------------------------------------------------
# Screen 3 — Cumulative triangle review
# ---------------------------------------------------------------------------
def _show_triangle(tri):
    def style_triangle(val):
        if pd.isna(val): return "color: #ccc"
        return ""
    st.dataframe(
        tri.style
           .format(lambda v: fmt_number(v) if not pd.isna(v) else "—")
           .map(style_triangle),
        use_container_width=True,
    )

def _suggestion_box():
    with st.expander(f"💡 {_('tri_suggest_title')}", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{_('tri_paid_title')}**")
            st.markdown(f"✅ {_('tri_paid_pros')}")
            st.markdown(f"⚠️ {_('tri_paid_cons')}")
        with c2:
            st.markdown(f"**{_('tri_incurred_title')}**")
            st.markdown(f"✅ {_('tri_incurred_pros')}")
            st.markdown(f"⚠️ {_('tri_incurred_cons')}")
        st.info(_("tri_suggestion"))

def screen_review():
    step_bar(3)
    st.header(_("review_title"))
    st.caption(_("review_caption"))

    if has_incurred():
        # Both triangles exist → let the user pick the base to continue with.
        # Incurred is the default (case reserves are available).
        st.caption(_("review_choose_base"))
        base_opts = ["incurred", "paid"]
        labels = {"paid": _("base_paid"), "incurred": _("base_incurred")}
        default_idx = base_opts.index(st.session_state.active_base) \
            if st.session_state.active_base in base_opts else 0
        choice = st.radio(
            _("base_label"), base_opts, format_func=lambda b: labels[b],
            index=default_idx, horizontal=True, key="review_base_radio",
        )
        if choice != st.session_state.active_base:
            sync_active_base(choice)
            st.rerun()
        _show_triangle(st.session_state.get(f"triangle_{choice}"))
        _suggestion_box()
    else:
        # Only paid was mapped → no incurred option, use the paid triangle.
        _show_triangle(st.session_state.triangle_paid)
        st.caption(_("base_no_incurred"))

    st.divider()
    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        st.button(_("back"), on_click=go_to, args=(2,))
    with bcol2:
        st.button(_("next"), type="primary", on_click=go_to, args=(4,))

# ---------------------------------------------------------------------------
# Screen 4 — Anomaly detection (FDI grid + exclusions)
# ---------------------------------------------------------------------------
def screen_anomalies():
    step_bar(4)
    st.header(_("anomaly_title"))
    st.caption(_("anomaly_caption"))

    # Base selector (only when an incurred triangle exists). Exclusions are
    # tracked independently per base.
    if has_incurred():
        base_opts = {"paid": _("base_paid"), "incurred": _("base_incurred")}
        choice = st.radio(
            _("base_label"),
            options=list(base_opts.keys()),
            format_func=lambda b: base_opts[b],
            index=0 if st.session_state.active_base == "paid" else 1,
            horizontal=True,
        )
        if choice != st.session_state.active_base:
            sync_active_base(choice)
            st.session_state.selected_cell = None
            st.rerun()

    fdi   = st.session_state.fdi_table
    flags = st.session_state.anomaly_flags
    excl  = st.session_state.exclusions

    n_warnings = int((flags == "warning").sum().sum())

    if n_warnings > 0:
        st.warning(_("review_anomaly_warn", w=n_warnings))
    else:
        st.success(_("anomaly_none"))

    from app.methods import compute_fdi_avg, compute_cdfs
    fdi_avg = compute_fdi_avg(fdi, excl)

    st.caption(_("review_fdi_caption"))

    # --- Selection action panel ---
    sel_cell = st.session_state.get("selected_cell")
    if sel_cell and sel_cell[0] in fdi.index and sel_cell[1] in fdi.columns \
            and not pd.isna(fdi.loc[sel_cell[0], sel_cell[1]]):
        period_s, col_s = sel_cell
        is_already_excl = (period_s, col_s) in excl
        val_s = fdi.loc[period_s, col_s]

        with st.container(border=True):
            st.markdown(_("review_selected", p=period_s, c=col_s, v=val_s))
            if is_already_excl:
                st.info(_("review_already_excl", r=excl[(period_s, col_s)]))
                ac1, ac2 = st.columns(2)
                if ac1.button(_("review_remove_excl"), type="primary"):
                    del st.session_state.exclusions[(period_s, col_s)]
                    st.session_state.selected_cell = None
                    st.rerun()
                if ac2.button(_("cancel")):
                    st.session_state.selected_cell = None
                    st.rerun()
            else:
                comment_input = st.text_input(
                    _("review_comment"),
                    placeholder=_("review_comment_ph"),
                    key="excl_comment_input",
                )
                # Excluding the last remaining factor of a transition would
                # leave the CDFs undefined — refuse it upfront.
                col_periods = fdi[col_s].dropna().index
                others_left = [p for p in col_periods
                               if p != period_s and (p, col_s) not in excl]

                ac1, ac2 = st.columns(2)
                if ac1.button(_("review_confirm_excl"), type="primary"):
                    if not others_left:
                        st.error(_("review_excl_last_err", c=col_s))
                    elif not comment_input.strip():
                        st.error(_("review_comment_err"))
                    else:
                        st.session_state.exclusions[(period_s, col_s)] = comment_input.strip()
                        st.session_state.selected_cell = None
                        st.rerun()
                if ac2.button(_("cancel")):
                    st.session_state.selected_cell = None
                    st.rerun()

    # --- FDI interactive grid ---
    n_dev = len(fdi.columns)

    # With many development columns (quarterly grain has ~19) the default 960px
    # page width squeezes each cell until the factors clip; widen the container
    # so every factor stays on one readable line. Annual / few-column grids keep
    # the standard centered width. This style is only emitted while the anomaly
    # screen renders, so other screens stay at 960px.
    if n_dev > 12:
        grid_w = min(1700, 80 + n_dev * 66)
        st.markdown(
            f"<style>.block-container {{ max-width: {grid_w}px !important; }}</style>",
            unsafe_allow_html=True,
        )

    # Accident-period labels need room for the quarter suffix (e.g. "2020Q1"),
    # so the first column is a bit wider than a factor cell.
    label_w = 1.2

    # Header row
    h_cols = st.columns([label_w] + [1] * n_dev)
    h_cols[0].markdown("<div class='fdi-lbl' style='font-size:11px;color:gray;text-align:right;'>AY</div>", unsafe_allow_html=True)
    for j, col in enumerate(fdi.columns, 1):
        h_cols[j].markdown(
            f"<div class='fdi-lbl' style='font-size:11px;color:gray;text-align:center;font-weight:600;'>{col}</div>",
            unsafe_allow_html=True,
        )

    # Data rows. Warning/anomaly cells are tinted via injected CSS (keyed on the
    # button) instead of an emoji prefix — the emoji widened the label and clipped
    # the factor in the narrow quarterly columns.
    flag_rules = []
    for period in fdi.index:
        r_cols = st.columns([label_w] + [1] * n_dev)
        r_cols[0].markdown(
            f"<div class='fdi-lbl' style='font-size:12px;color:gray;text-align:right;padding-top:6px;'>{period}</div>",
            unsafe_allow_html=True,
        )
        for j, col in enumerate(fdi.columns, 1):
            val = fdi.loc[period, col]
            if pd.isna(val):
                r_cols[j].markdown(
                    "<div class='fdi-lbl' style='text-align:center;color:#ddd;font-size:11px;padding-top:6px;'>—</div>",
                    unsafe_allow_html=True,
                )
                continue

            is_excl     = (period, col) in excl
            flag        = flags.loc[period, col]
            is_selected = st.session_state.get("selected_cell") == (period, col)

            if is_excl:
                r_cols[j].markdown(
                    f"<div class='fdi-excluded' title='{excl[(period, col)]}'>{val:.4f}</div>",
                    unsafe_allow_html=True,
                )
            else:
                label = f"{val:.4f}"   # flag shown as cell background, not an emoji
                if flag == "warning" and not is_selected:
                    flag_rules.append(f"fdi_{period}_{col}")

                btn_type = "primary" if is_selected else "secondary"
                if r_cols[j].button(label, key=f"fdi_{period}_{col}",
                                    type=btn_type, use_container_width=True):
                    if is_selected:
                        st.session_state.selected_cell = None
                    else:
                        st.session_state.selected_cell = (period, col)
                    st.rerun()

    # Tint the flagged (warning) cells yellow, matching the Excel/PDF export.
    if flag_rules:
        WARN_BG, WARN_BD = "#FFF3CD", "#F1D592"
        rules = []
        for key in flag_rules:
            # Dark text too: the tint is light, so on the dark theme the
            # default light button text would be unreadable over it.
            rules.append(
                f".st-key-{key} button {{ background:{WARN_BG} !important; "
                f"border-color:{WARN_BD} !important; }} "
                f".st-key-{key} button p {{ color:#333 !important; }}"
            )
        st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)

    # Average row
    avg_label = _("review_avg_label")
    avg_cols = st.columns([label_w] + [1] * n_dev)
    avg_cols[0].markdown(
        f"<div class='fdi-lbl' style='font-size:11px;font-weight:600;text-align:right;padding-top:6px;'>{avg_label}</div>",
        unsafe_allow_html=True,
    )
    for j, col in enumerate(fdi.columns, 1):
        avg_txt = "—" if pd.isna(fdi_avg[col]) else f"{fdi_avg[col]:.4f}"
        avg_cols[j].markdown(
            f"<div class='fdi-lbl' style='text-align:center;font-size:11px;font-family:monospace;"
            f"font-weight:600;color:#1f77b4;padding-top:6px;'>{avg_txt}</div>",
            unsafe_allow_html=True,
        )

    st.caption(_("review_avg_caption"))

    # --- Active exclusions log ---
    if excl:
        st.divider()
        st.subheader(_("review_excl_log", n=len(excl)))
        for (period, col), reason in excl.items():
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"- **AY {period} — {col}**: *\"{reason}\"*")
            if c2.button(_("review_remove_btn"), key=f"rm_{period}_{col}"):
                del st.session_state.exclusions[(period, col)]
                st.rerun()

    st.divider()
    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        st.button(_("back"), on_click=go_to, args=(3,))
    with bcol2:
        st.button(_("review_confirm_btn"), type="primary", on_click=go_to, args=(5,))

# ---------------------------------------------------------------------------
# Screen 5 — Configure
# ---------------------------------------------------------------------------
def screen_configure():
    step_bar(5)
    st.header(_("config_title"))

    c1, c2 = st.columns(2)

    with c1:
        tail = st.number_input(
            _("config_tail"),
            min_value=1.0, max_value=5.0,
            value=float(st.session_state.tail),
            step=0.001, format="%.3f",
            help=_("config_tail_help"),
        )

    with c2:
        elr_input = st.text_input(
            _("config_elr"),
            value="" if st.session_state.elr_override is None else str(st.session_state.elr_override),
            placeholder=_("config_elr_ph"),
            help=_("config_elr_help"),
        )

    st.divider()
    st.subheader(_("config_methods"))

    has_premium = st.session_state.premium_df is not None

    mc1, mc2 = st.columns(2)
    with mc1:
        run_cl = st.checkbox("Chain Ladder", value=st.session_state.run_cl)
        if has_premium:
            run_bf = st.checkbox("Bornhuetter-Ferguson", value=st.session_state.run_bf)
        else:
            st.checkbox(_("config_bf_locked"), value=False, disabled=True)
            run_bf = False
    with mc2:
        if has_premium:
            run_cc = st.checkbox("Cape Cod (Stanard-Bühlmann)", value=st.session_state.run_cc)
        else:
            st.checkbox(_("config_cc_locked"), value=False, disabled=True)
            run_cc = False

    if not any([run_cl, run_bf, run_cc]):
        st.warning(_("config_no_method"))

    st.divider()
    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        st.button(_("back"), on_click=go_to, args=(4,))
    with bcol2:
        if st.button(_("config_run"), type="primary"):
            elr_val = None
            if elr_input.strip():
                try:
                    elr_val = float(elr_input.strip())
                    if not (0 < elr_val < 10):
                        st.error(_("config_elr_err"))
                        return
                except ValueError:
                    st.error(_("config_elr_err2"))
                    return

            st.session_state.tail         = tail
            st.session_state.elr_override = elr_val
            st.session_state.run_cl       = run_cl
            st.session_state.run_bf       = run_bf
            st.session_state.run_cc       = run_cc

            if run_bf and st.session_state.premium_df is None:
                st.error(_("config_bf_no_prem"))
                return

            try:
                by_base = {"paid": run_configured_methods(
                    st.session_state.triangle_paid,
                    st.session_state.exclusions_paid,
                )}
                if has_incurred():
                    by_base["incurred"] = run_configured_methods(
                        st.session_state.triangle_incurred,
                        st.session_state.exclusions_incurred,
                    )
                st.session_state.results_by_base = by_base
                st.session_state.results = by_base[st.session_state.active_base]
                st.session_state.segment_value = None
                go_to(6)
                st.rerun()

            except Exception as e:
                st.error(_("config_run_err", e=e))

# ---------------------------------------------------------------------------
# Screen 6 — Results
# ---------------------------------------------------------------------------
CHART_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
SEL_COLOR = "#d62728"   # best-estimate accent


def _method_abbr(name: str) -> str:
    """Short tag for a method name, used in weight labels."""
    if name.startswith("Chain"):          return "CL"
    if name.startswith("Bornhuetter"):    return "BF"
    if name.startswith("Cape"):           return "CC"
    return name[:3]


def _filter_exclusions_for(excl, tri):
    """
    Keep only the exclusions whose cell exists (non-NaN) in this triangle's
    FDI table. Full-book exclusions may point at cells a segment doesn't have.
    Returns (filtered, n_dropped).
    """
    fdi = compute_fdi(tri)
    kept = {
        k: v for k, v in (excl or {}).items()
        if k[0] in fdi.index and k[1] in fdi.columns
        and not pd.isna(fdi.loc[k[0], k[1]])
    }
    return kept, len(excl or {}) - len(kept)


def _segment_results(seg_value):
    """Rebuild both triangles for a segment and recompute results per base."""
    tp, ti = build_paid_incurred(
        st.session_state.raw_df,
        incurred_col  = st.session_state.incurred_col,
        paid_col      = st.session_state.paid_col,
        amount_col    = st.session_state.amount_col,
        reserve_col   = st.session_state.reserve_col,
        date_format   = st.session_state.date_format,
        amount_type   = st.session_state.amount_type.lower(),
        reserve_amount_type = st.session_state.reserve_amount_type.lower(),
        grain         = grain_key(st.session_state.grain),
        segment_col   = st.session_state.segment_col,
        segment_value = seg_value,
        claim_col     = st.session_state.claim_col,
    )
    excl_p, dropped_p = _filter_exclusions_for(st.session_state.exclusions_paid, tp)
    by_base  = {"paid": run_configured_methods(tp, excl_p)}
    tri_by   = {"paid": tp}
    excl_by  = {"paid": (excl_p, dropped_p)}
    if ti is not None:
        excl_i, dropped_i = _filter_exclusions_for(st.session_state.exclusions_incurred, ti)
        by_base["incurred"] = run_configured_methods(ti, excl_i)
        tri_by["incurred"]  = ti
        excl_by["incurred"] = (excl_i, dropped_i)
    return by_base, tri_by, excl_by


def _warn_missing_estimates(results):
    """Surface accident periods whose IBNR is NaN (missing premium or factor):
    nansum would silently leave them out of every total."""
    for r in results:
        latest = np.asarray(r.latest_paid, dtype=float)
        ibnr   = np.asarray(r.ibnr, dtype=float)
        bad = [str(p) for p, l, i in zip(r.accident_periods, latest, ibnr)
               if not np.isnan(l) and np.isnan(i)]
        if bad:
            st.warning(_("results_nan_warn", m=r.method, periods=", ".join(bad)))


def _render_single_base(results, prem_total, base="paid"):
    """Summary cards (one per method) + IBNR table + charts for one base."""
    _warn_missing_estimates(results)
    st.subheader(_("results_summary"))

    # Latest diagonal on its own row, in a column wide enough for the full
    # label (paid alone, or paid + RSP for the incurred base) — the "(paid +
    # RSP)" detail lives in the tooltip so the label never gets truncated.
    latest_total = float(np.nansum(results[0].latest_paid))
    latest_label = _("results_latest_inc") if base == "incurred" else _("results_latest")
    latest_help  = _("results_latest_inc_help") if base == "incurred" else _("results_latest_help")
    st.columns([2, 3])[0].metric(latest_label, f"{latest_total:,.2f}", help=latest_help)

    # One clearly-labelled card per method: the method name heads the card and
    # each figure (IBNR, Ultimate, Loss ratio) is labelled underneath, so it is
    # unambiguous which number belongs to which method.
    st.caption(_("results_summary_hint"))
    method_cols = st.columns(len(results))
    for i, r in enumerate(results):
        with method_cols[i]:
            with st.container(border=True):
                st.markdown(f"**{r.method}**")
                st.metric(_("tbl_ibnr"), f"{r.total_ibnr:,.2f}",
                          help=_("results_ibnr_help"))
                st.metric(_("results_ultimate_label"), f"{r.total_ultimate:,.2f}")
                if prem_total:
                    st.caption(
                        f"**{_('tbl_lr')}:** {r.total_ultimate / prem_total * 100:.1f}%"
                    )
                else:
                    st.caption(_("results_lr_no_prem"))

    st.divider()
    st.subheader(_("results_ibnr_table"))
    comp = summarize_results(results)

    # C — surface maturity: % reported (= 1/CDF) per accident period, so the
    # user can see which years are immature (Chain Ladder less reliable there).
    IMMATURE = 0.75  # completion-factor credibility threshold (SOA 2009)
    pct_col = _("results_pct_reported")
    pct_by_period = {p: (1.0 / c if c else np.nan)
                     for p, c in zip(results[0].accident_periods, results[0].cdfs)}
    method_cols = list(comp.columns)
    # Pre-format the % column as text (the TOTAL row has no meaningful %), so the
    # display never falls back to a raw "None"/NaN.
    comp.insert(0, pct_col, [
        "—" if pd.isna(pct_by_period.get(idx, np.nan)) else f"{pct_by_period[idx]*100:.0f}%"
        for idx in comp.index
    ])

    def _highlight_immature(row):
        v = pct_by_period.get(row.name, np.nan)
        if not pd.isna(v) and v < IMMATURE:
            return ["background-color: #FFF3CD; color: #333"] * len(row)
        return [""] * len(row)

    styler = (
        comp.style
            .format("{:,.1f}", subset=method_cols, na_rep="—")
            .apply(_highlight_immature, axis=1)
    )
    st.dataframe(styler, use_container_width=True)
    st.caption(_("results_maturity_note", t=int(IMMATURE * 100)))

    st.divider()
    st.subheader(_("results_charts"))
    chart_tab1, chart_tab2 = st.tabs([_("results_chart1"), _("results_chart2")])
    with chart_tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        periods = results[0].accident_periods
        x = np.arange(len(periods))
        w = 0.8 / len(results)
        for i, r in enumerate(results):
            ax.bar(x + i * w, r.ibnr, w, label=r.method, color=CHART_COLORS[i], alpha=0.85)
        ax.set_xticks(x + w * (len(results) - 1) / 2)
        ax.set_xticklabels([str(p) for p in periods], rotation=45)
        ax.set_ylabel("IBNR")
        ax.set_title(_("results_chart1_title"))
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with chart_tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        method_names = [r.method for r in results]
        ibnr_totals  = [r.total_ibnr for r in results]
        bars = ax2.bar(method_names, ibnr_totals, color=CHART_COLORS[:len(results)], alpha=0.85)
        for bar, val in zip(bars, ibnr_totals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(ibnr_totals) * 0.01,
                     f"{val:,.0f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        ax2.set_ylabel("Total IBNR")
        ax2.set_title(_("results_chart2_title"))
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, __: f"{v:,.0f}"))
        ax2.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    return comp


# Colours for the reserve decomposition (Paid | RSP | Pure IBNR).
DECOMP_COLORS = {"paid": "#1f77b4", "rsp": "#ff7f0e", "pure_ibnr": "#2ca02c"}


def _render_compare(by_base, prem_total):
    """
    Reconciliation view. Instead of putting paid IBNR (= RSP + pure IBNR, the
    total reserve) next to incurred IBNR (= pure IBNR) as if they competed —
    which makes paid look alarmingly larger — this reframes them as a single
    decomposition: Paid to date + Case reserves (RSP) + Pure IBNR = Ultimate.
    The real cross-check is whether the two bases' ultimates converge.
    Uses the first configured method on each base for the pure-IBNR slice.
    """
    paid_r, inc_r = by_base["paid"], by_base["incurred"]
    _warn_missing_estimates(paid_r + inc_r)
    p0, i0 = paid_r[0], inc_r[0]

    # --- Convergence headline: the meaningful cross-check ------------------
    gap = abs(p0.total_ultimate - i0.total_ultimate) / p0.total_ultimate if p0.total_ultimate else 0
    if gap < 0.05:
        st.success(_("recon_converge_ok", g=gap))
    else:
        st.warning(_("recon_converge_warn", g=gap))
    st.caption(_("recon_ult_check", p=p0.total_ultimate, i=i0.total_ultimate, g=gap))

    # --- Decomposition figures (data + primary method) --------------------
    st.markdown(_("recon_intro"))
    decomp = reserve_decomposition(p0, i0)
    totals = decomp.loc["TOTAL"]
    rc = st.columns(4)
    rc[0].metric(_("decomp_paid"),      f"{totals['paid']:,.0f}")
    rc[1].metric(_("recon_rsp"),        f"{totals['rsp']:,.0f}")
    rc[2].metric(_("recon_pure_ibnr"),  f"{totals['pure_ibnr']:,.0f}")
    rc[3].metric(_("decomp_ultimate"),  f"{totals['ultimate']:,.0f}")
    st.caption(_("decomp_method_note", m=i0.method))

    # --- Stacked bar per accident year: Paid | RSP | Pure IBNR ------------
    body = decomp.drop(index="TOTAL")
    periods = body.index.tolist()
    x = np.arange(len(periods))
    paid_vals = body["paid"].to_numpy(dtype=float)
    rsp_vals  = np.clip(body["rsp"].to_numpy(dtype=float), 0, None)  # clip only for drawing
    ibnr_vals = np.clip(body["pure_ibnr"].to_numpy(dtype=float), 0, None)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, paid_vals, 0.62, label=_("decomp_paid"), color=DECOMP_COLORS["paid"], alpha=0.9)
    ax.bar(x, rsp_vals, 0.62, bottom=paid_vals, label=_("decomp_rsp"),
           color=DECOMP_COLORS["rsp"], alpha=0.9)
    ax.bar(x, ibnr_vals, 0.62, bottom=paid_vals + rsp_vals, label=_("decomp_ibnr"),
           color=DECOMP_COLORS["pure_ibnr"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in periods], rotation=45)
    ax.set_ylabel(_("decomp_ultimate"))
    ax.set_title(_("decomp_chart_title"))
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Decomposition table ---------------------------------------------
    st.subheader(_("decomp_table"))
    disp = decomp.rename(columns={
        "paid":      _("decomp_paid"),
        "rsp":       _("decomp_rsp"),
        "pure_ibnr": _("decomp_ibnr"),
        "ultimate":  _("decomp_ultimate"),
    })
    st.dataframe(
        disp.style.format(lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"),
        use_container_width=True,
    )
    return decomp


def _render_selected(results, prem_total, base_key):
    """
    Best-estimate section. Two blending modes:
      • Maturity-weighted (Benktander): per accident period, Chain Ladder for
        mature years and BF / Cape Cod for immature years, weighted by the
        completion factor — the actuarial default when both are available.
      • Manual (per year): the user sets each method's weight per accident year
        in an editable grid; every row must add up to exactly 100% (no
        normalisation) and the weights are applied as entered.

    Returns the selected IBNRResult, or None when the manual grid is incomplete
    (a row that does not add up to 100%).
    """
    st.subheader(_("sel_title"))
    st.caption(_("sel_caption"))

    n = len(results)
    has_dev = any(r.method.startswith("Chain") for r in results)
    has_exp = any(r.method.startswith(("Bornhuetter", "Cape")) for r in results)
    can_benktander = has_dev and has_exp

    # Build the selection per mode, and name it so it reads as one more method
    # column (e.g. "Benktander").
    if n == 1:
        st.info(_("sel_single_note"))
        sel_name = _("sel_method")
        sel = weighted_selection(results, np.ones((len(results[0].ibnr), 1)), sel_name)
    else:
        mode = "manual"
        if can_benktander:
            mode = st.radio(
                _("sel_mode_label"), ["benktander", "manual"],
                format_func=lambda m: _("sel_mode_benktander") if m == "benktander"
                else _("sel_mode_manual"),
                horizontal=True, key=f"selmode_{base_key}",
            )
        if mode == "benktander":
            st.caption(_("sel_benktander_explain"))
            with st.expander(f"💡 {_('benktander_when_title')}"):
                st.markdown("\n".join(
                    f"- {_(f'benktander_when_{k}')}" for k in range(1, 6)
                ))
            sel_name = _("sel_benktander_name")     # "Benktander" — it is one more method
            sel = credibility_weighted_result(results, sel_name)
        else:
            # Editable weight grid: rows = accident years, columns = methods (%).
            st.caption(_("sel_manual_grid_caption"))
            grid = st.data_editor(
                default_maturity_weights(results),
                key=f"wgrid_{base_key}",
                use_container_width=True,
                column_config={
                    m: st.column_config.NumberColumn(
                        min_value=0, max_value=100, step=1, format="%d%%")
                    for m in [r.method for r in results]
                },
            )
            row_sums = grid.sum(axis=1)
            bad = [f"{p} = {int(s) if not pd.isna(s) else 0}%"
                   for p, s in row_sums.items() if pd.isna(s) or int(round(s)) != 100]
            if bad:
                st.error(_("sel_manual_bad_rows", rows="; ".join(bad)))
                return None
            sel_name = _("sel_manual_name")
            sel = weighted_selection(results, grid.to_numpy(dtype=float) / 100.0, sel_name)

    # --- Per-year comparison: every method + the selection, IBNR by year ---
    st.subheader(_("sel_compare_table"))
    comp = summarize_results(results + [sel])   # cols = methods + selection; rows = years + TOTAL
    st.dataframe(
        comp.style.format(lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"),
        use_container_width=True,
    )

    # --- Per-year chart: IBNR by accident year, methods + selection ---
    series  = results + [sel]
    periods = sel.accident_periods
    x = np.arange(len(periods))
    bw = 0.8 / len(series)
    colors = CHART_COLORS[:len(results)] + [SEL_COLOR]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, r in enumerate(series):
        ax.bar(x + i * bw, r.ibnr, bw, label=r.method, color=colors[i], alpha=0.85)
    ax.set_xticks(x + bw * (len(series) - 1) / 2)
    ax.set_xticklabels([str(p) for p in periods], rotation=45)
    ax.set_ylabel("IBNR")
    ax.set_title(_("sel_chart_title"))
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, __: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    return sel


def screen_results():
    step_bar(6)
    st.header(_("results_title"))

    if not st.session_state.results_by_base:
        st.error(_("results_no_results"))
        st.button(_("back"), on_click=go_to, args=(5,))
        return

    by_base = st.session_state.results_by_base
    tri_by  = {"paid": st.session_state.triangle_paid}
    if has_incurred():
        tri_by["incurred"] = st.session_state.triangle_incurred

    prem_df = st.session_state.premium_df
    prem_series = None
    if prem_df is not None:
        prem_series = prem_df.set_index(prem_df.columns[0])[prem_df.columns[1]]

    def _prem_total_for(tri):
        """Earned premium over this triangle's accident periods only, so loss
        ratios aren't diluted by premium years outside the triangle."""
        if prem_series is None or tri is None:
            return None
        try:
            aligned, _missing = align_premium(prem_series, tri.index)
        except ValueError:
            return None
        return float(np.nansum(aligned))

    # --- Segment filter (rebuilds both bases for the segment) ---
    seg_col = st.session_state.segment_col
    seg_for_meta = None
    excl_by_seg = None
    if seg_col and st.session_state.raw_df is not None:
        all_label = _("results_all")
        segments = [all_label] + get_segments(st.session_state.raw_df, seg_col)
        selected_seg = st.selectbox(_("results_segment"), segments)
        if selected_seg != all_label:
            try:
                by_base, tri_by, excl_by_seg = _segment_results(selected_seg)
                seg_for_meta = selected_seg
                st.caption(_("results_seg_prem_note"))
            except Exception as e:
                st.error(_("results_seg_err", e=e))

    # --- Base selector: paid / incurred / compare ---
    sel = "paid"
    if has_incurred():
        opts = ["paid", "incurred", "compare"]
        labels = {"paid": _("base_paid"), "incurred": _("base_incurred"), "compare": _("base_compare")}
        default_idx = opts.index(st.session_state.active_base) if st.session_state.active_base in opts else 0
        sel = st.radio(_("base_label"), opts, format_func=lambda b: labels[b],
                       index=default_idx, horizontal=True)

    if sel == "compare":
        comp = _render_compare(by_base, _prem_total_for(tri_by["paid"]))
    else:
        comp = _render_single_base(by_base[sel], _prem_total_for(tri_by[sel]), sel)

    # Base used for the per-base export/exclusion sections. In compare mode
    # the user picks which base the files should document.
    if sel == "compare":
        base_labels = {"paid": _("base_paid"), "incurred": _("base_incurred")}
        export_base = st.radio(_("export_base_label"), ["paid", "incurred"],
                               format_func=lambda b: base_labels[b], horizontal=True)
    else:
        export_base = sel
    results = by_base[export_base]
    tri     = tri_by[export_base]
    if excl_by_seg is not None:
        excl, n_dropped = excl_by_seg[export_base]
        if excl or n_dropped:
            st.caption(_("results_seg_excl_note", n=len(excl), d=n_dropped))
        fdi_for_export = compute_fdi(tri)
    else:
        excl = st.session_state.get(f"exclusions_{export_base}")
        fdi_for_export = st.session_state.get(f"fdi_{export_base}")
    prem_total = _prem_total_for(tri)

    st.divider()

    # --- Exclusion log ---
    if excl:
        st.subheader(_("results_excl_log", n=len(excl)))
        for (period, col), reason in excl.items():
            st.markdown(f"- **AY {period} — {col}**: *\"{reason}\"*")
        st.divider()

    # --- Best estimate (weighted selection) ---
    # None when the manual per-year grid is incomplete; then the export offers
    # the methods without a best-estimate row until the weights add to 100%.
    sel_result = _render_selected(results, prem_total, export_base)
    all_results = results + ([sel_result] if sel_result is not None else [])
    st.divider()

    # --- Export ---
    st.subheader(_("results_export"))

    scope = st.radio(
        _("export_scope"), ["full", "specific"],
        format_func=lambda s: _("export_full") if s == "full" else _("export_specific"),
        horizontal=True,
    )
    if scope == "full":
        export_results = all_results
        file_stub = "ibnr_results"
    else:
        names = [r.method for r in all_results]
        pick = st.selectbox(_("export_pick"), names, index=len(names) - 1)
        export_results = [r for r in all_results if r.method == pick]
        file_stub = "ibnr_" + _method_abbr(pick).lower()

    # Run context documented inside the exported files.
    export_latest_label = ("Latest Incurred" if export_base == "incurred"
                           else "Latest Paid")
    export_meta = {
        "Base":         "Incurred (paid + RSP)" if export_base == "incurred" else "Paid",
        "Tail factor":  f"{st.session_state.tail:.3f}",
        "ELR override": (st.session_state.elr_override
                         if st.session_state.elr_override is not None else "—"),
        "Segment":      seg_for_meta or _("results_all"),
        "Grain":        st.session_state.grain,
    }

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_buf = io.StringIO()
        summarize_results(export_results).to_csv(csv_buf)
        st.download_button(
            _("results_csv"),
            data=csv_buf.getvalue(),
            file_name=f"{file_stub}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp2:
        try:
            from app.exports import build_excel
            excel_buf = build_excel(export_results, tri, fdi_for_export, excl,
                                    premium_total=prem_total, meta=export_meta,
                                    latest_label=export_latest_label)
            st.download_button(
                _("results_excel"),
                data=excel_buf,
                file_name=f"{file_stub}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.button(_("results_excel_soon"), disabled=True,
                      use_container_width=True)
            st.error(f"Excel export failed: {e}")

    with exp3:
        try:
            from app.exports import build_pdf
            pdf_buf = build_pdf(export_results, tri, fdi_for_export, excl,
                                premium_total=prem_total, meta=export_meta,
                                latest_label=export_latest_label)
            st.download_button(
                _("results_pdf"),
                data=pdf_buf,
                file_name=f"{file_stub}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.button(_("results_pdf_soon"), disabled=True,
                      use_container_width=True)
            st.error(f"PDF export failed: {e}")

    st.divider()
    col_nav1, col_nav2 = st.columns([1, 5])
    with col_nav1:
        st.button(_("back"), on_click=go_to, args=(5,))
    with col_nav2:
        if st.button(_("results_new"), type="secondary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.title(f"📊 {_('app_title')}")
st.caption(_("app_caption"))
st.markdown(
    "Created by **Melina Daniela Avila** · "
    "meeli.avila96@gmail.com · melina.daniela@outlook.com · "
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Melina_Daniela_Avila-0077B5?logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/melina-daniela-avila-842605146/)"
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar navigation index — sits under the language selector. Steps already
# reached become clickable links; steps not yet unlocked stay greyed out.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown(f"### 🧭 {_('nav_index')}")
    _nav_labels = [
        _("step_upload"), _("step_map"), _("step_review"),
        _("step_anomalies"), _("step_configure"), _("step_results"),
    ]
    _cur  = st.session_state.screen
    _maxr = st.session_state.get("max_screen", 1)
    for _i, _lbl in enumerate(_nav_labels, start=1):
        if _i == _cur:
            st.markdown(f"**▶ {_lbl}**")
        elif _i <= _maxr:
            if st.button(_lbl, key=f"nav_{_i}", use_container_width=True):
                go_to(_i)
                st.rerun()
        else:
            st.markdown(
                f"<span style='color:#bbb'>🔒 {_lbl}</span>",
                unsafe_allow_html=True,
            )
    if _maxr < len(_nav_labels):
        st.caption(_("nav_locked_hint"))

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
SCREENS = {
    1: screen_upload,
    2: screen_map,
    3: screen_review,
    4: screen_anomalies,
    5: screen_configure,
    6: screen_results,
}

# Jump back to the top of the page whenever the step changes (not on in-step
# reruns, so toggling a control doesn't yank the view around).
if st.session_state.get("_last_screen") != st.session_state.screen:
    st.session_state._last_screen = st.session_state.screen
    scroll_to_top()

SCREENS[st.session_state.screen]()

# ---------------------------------------------------------------------------
# Sources & references — footer shown only on the Results screen.
# ---------------------------------------------------------------------------
if st.session_state.screen == 6:
    st.divider()
    with st.expander(f"📚 {_('sources_title')}"):
        st.markdown(
            "- Chadick, C., Campbell, W., & Knox-Seith, F. (2009). "
            "*Comparison of Incurred But Not Reported (IBNR) Methods*. "
            "Society of Actuaries, Health Section. "
            "https://www.soa.org/globalassets/assets/files/research/projects/research-ibnr-report-2009.pdf\n"
            "- MetricGate. (2024). *Benktander Method Calculator* [Web application]. "
            "https://metricgate.com/docs/benktander-method/"
        )
