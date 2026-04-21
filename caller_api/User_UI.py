"""
streamlit_app.py
─────────────────────────────────────────────────────────────
Fraud Detection Dashboard
Handles:  CSV upload | manual form | JSON paste
Calls:    fraud API via score_transactions() from client.py
Displays: results table, metrics, CSV download
─────────────────────────────────────────────────────────────
Run:
    streamlit run streamlit_app.py

Requires fraud_api running on port 8000:
    uvicorn main:app --port 8000
"""

import streamlit as st
import pandas as pd
import json
import io
from fraud_client import score_transactions   # ← your caller API — only line that touches the fraud API

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon= "resources/cloud-upload.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# paste this near the top of your main area, after set_page_config
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px">
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
         xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V7L12 2z"
              fill="#185FA5" opacity="0.15" stroke="#185FA5" stroke-width="1.5"
              stroke-linejoin="round"/>
        <path d="M9 12l2 2 4-4" stroke="#185FA5" stroke-width="1.5"
              stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span style="font-size:22px; font-weight:600; color:#185FA5">Fraud Detection</span>
</div>
""", unsafe_allow_html=True)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* tighten default streamlit padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* metric cards */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* fraud row highlight — applied via df styling */
    .fraud-row { background-color: #fff0f0 !important; }

    /* sidebar header */
    .sidebar-header {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def parse_csv(file) -> list[dict]:
    """Read uploaded CSV file → list of dicts (one per row)."""
    df = pd.read_csv(file)
    return df.to_dict("records")


def parse_json_text(text: str) -> list[dict]:
    """Parse raw JSON string — accepts single object or list."""
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def build_results_df(predictions: list[dict]) -> pd.DataFrame:
    """Turn raw prediction dicts into a display-ready DataFrame."""
    df = pd.DataFrame(predictions)

    # reorder columns for readability
    preferred_cols = [
        "transaction_id", "fraud_probability",
        "prediction", "decision", "is_fraud"
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    rest     = [c for c in df.columns if c not in preferred_cols]
    df = df[existing + rest]

    # round probability for display
    if "fraud_probability" in df.columns:
        df["fraud_probability"] = df["fraud_probability"].round(4)

    return df


def highlight_fraud(row: pd.Series) -> list[str]:
    try:
        is_fraud = row["is_fraud"]
        # handle both bool True and string "True"
        if is_fraud is True or str(is_fraud).lower() == "true":
            return ["background-color: #ffcccc; color: #7f0000"] * len(row)
        decision = str(row.get("decision", "")).lower()
        if decision == "review":
            return ["background-color: #fffbea"] * len(row)
    except Exception:
        pass
    return [""] * len(row)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for st.download_button."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — input source + controls
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Fraud Detection")
    st.caption("Powered by your XGBoost fraud model")
    st.divider()

    # ── Input source selector ──────────────────────────────────
    st.markdown('<p class="sidebar-header">Input source</p>', unsafe_allow_html=True)
    source = st.radio(
        label="source",
        options=["CSV / Excel file", "Manual entry", "Paste JSON"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Source: CSV upload ─────────────────────────────────────
    transactions: list[dict] | None = None

    if source == "CSV / Excel file":
        st.markdown('<p class="sidebar-header">Upload file</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="Drop CSV or Excel here",
            type=["csv", "xlsx"],
            help="Columns must match your model's feature names",
        )

        if uploaded:
            try:
                if uploaded.name.endswith(".xlsx"):
                    df_upload = pd.read_excel(uploaded)
                else:
                    df_upload = pd.read_csv(uploaded)

                transactions = df_upload.to_dict("records")
                st.success(f"{len(transactions)} transactions loaded")

                # preview
                with st.expander("Preview first 5 rows"):
                    st.dataframe(df_upload.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── Source: Manual form ────────────────────────────────────
    elif source == "Manual entry":
        st.markdown('<p class="sidebar-header">Transaction details</p>', unsafe_allow_html=True)

        with st.form("manual_form"):
            txn_id = st.text_input("Transaction ID", value="txn_001")  
            time   = st.number_input("Time", value=0.0)
            v1     = st.number_input("V1", value=0.0, format="%.4f")
            v2     = st.number_input("V2", value=0.0, format="%.4f")
            v3     = st.number_input("V3", value=0.0, format="%.4f")
            v4     = st.number_input("V4", value=0.0, format="%.4f")
            v5     = st.number_input("V5", value=0.0, format="%.4f")
            v6     = st.number_input("V6", value=0.0, format="%.4f")
            v7     = st.number_input("V7", value=0.0, format="%.4f")
            v8     = st.number_input("V8", value=0.0, format="%.4f")
            v9     = st.number_input("V9", value=0.0, format="%.4f")
            v10    = st.number_input("V10", value=0.0, format="%.4f")
            v11    = st.number_input("V11", value=0.0, format="%.4f")
            v12    = st.number_input("V12", value=0.0, format="%.4f")
            v13    = st.number_input("V13", value=0.0, format="%.4f")
            v14    = st.number_input("V14", value=0.0, format="%.4f")
            v15    = st.number_input("V15", value=0.0, format="%.4f")
            v16    = st.number_input("V16", value=0.0, format="%.4f")
            v17    = st.number_input("V17", value=0.0, format="%.4f")
            v18    = st.number_input("V18", value=0.0, format="%.4f")
            v19    = st.number_input("V19", value=0.0, format="%.4f")
            v20    = st.number_input("V20", value=0.0, format="%.4f")
            v21    = st.number_input("V21", value=0.0, format="%.4f")
            v22    = st.number_input("V22", value=0.0, format="%.4f")
            v23    = st.number_input("V23", value=0.0, format="%.4f")
            v24    = st.number_input("V24", value=0.0, format="%.4f")
            v25    = st.number_input("V25", value=0.0, format="%.4f")
            v26    = st.number_input("V26", value=0.0, format="%.4f")
            v27    = st.number_input("V27", value=0.0, format="%.4f")
            v28    = st.number_input("V28", value=0.0, format="%.4f")
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
        
            form_submitted = st.form_submit_button("Load transaction", use_container_width=True)

        if form_submitted:
            transactions = [{
                "transaction_id": txn_id,
                
                "time":           time,
                "v1":             v1,
                "v2":             v2,
                "v3":             v3,
                "v4":             v4,
                "v5":             v5,
                "v6":             v6,
                "v7":             v7,
                "v8":             v8,
                "v9":             v9,
                "v10":            v10,
                "v11":            v11,
                "v12":            v12,
                "v13":            v13,
                "v14":            v14,
                "v15":            v15,
                "v16":            v16,
                "v17":            v17,
                "v18":            v18,
                "v19":            v19,
                "v20":            v20,
                "v21":            v21,
                "v22":            v22,
                "v23":            v23,
                "v24":            v24,
                "v25":            v25,
                "v26":            v26,
                "v27":            v27,
                "v28":            v28,

                "amount":         amount,
                
            }]
            st.success("1 transaction loaded")

    # ── Source: JSON paste ─────────────────────────────────────
    elif source == "Paste JSON":
        st.markdown('<p class="sidebar-header">Raw JSON</p>', unsafe_allow_html=True)
        json_text = st.text_area(
            label="Paste JSON",
            height=200,
            placeholder='[{"transaction_id": "txn_001", "amount": 120.5, "v1": -1.3, ...}]',
            label_visibility="collapsed",
        )
        if st.button("Load JSON", use_container_width=True):
            try:
                transactions = parse_json_text(json_text)
                st.success(f"{len(transactions)} transactions loaded")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()

    # ── Score button ───────────────────────────────────────────
    ready = transactions is not None and len(transactions) > 0
    score_clicked = st.button(
        "Run fraud detection",
        disabled=not ready,
        type="primary",
        use_container_width=True,
        help="Load transactions first, then click here",
    )

    if not ready:
        st.caption("Load transactions above to enable scoring")
    else:
        st.caption(f"{len(transactions)} transaction(s) ready to score")


# ══════════════════════════════════════════════════════════════
# MAIN AREA — results
# ══════════════════════════════════════════════════════════════

# Store results in session_state so they persist across reruns
if "results" not in st.session_state:
    st.session_state.results = None

# ── Call the fraud API via client.py ──────────────────────────
if score_clicked and transactions:
    with st.spinner(f"Scoring {len(transactions)} transaction(s) via fraud API..."):
        try:
            # score_transactions() is from client.py
            # it sends POST /predict and returns list[dict]
            predictions = score_transactions(transactions)
            st.session_state.results = predictions

        except ConnectionError:
            st.error(
                "Cannot reach the fraud API on port 8000. "
                "Make sure it is running:  `uvicorn main:app --port 8000`"
            )
            st.session_state.results = None

        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.session_state.results = None

# ── Display results ────────────────────────────────────────────
if st.session_state.results:
    results   = st.session_state.results
    df_result = build_results_df(results)

    # ── Metrics row ──────────────────────────────────────────
    total  = len(results)
    fraud  = sum(1 for r in results if r.get("is_fraud", False))
    review = sum(1 for r in results if str(r.get("decision","")).lower() == "review")
    safe   = total - fraud - review
    avg_prob = sum(r["fraud_probability"] for r in results) / total

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total scored",       total)
    col2.metric("Fraud detected",     fraud,  delta=f"{fraud/total:.1%}",  delta_color="inverse")
    col3.metric("Flagged for review", review, delta=f"{review/total:.1%}", delta_color="off")
    col4.metric("Legitimate",         safe,   delta=f"{safe/total:.1%}",   delta_color="normal")
    col5.metric("Avg fraud prob",     f"{avg_prob:.3f}")

    st.divider()

    # ── Results table ────────────────────────────────────────
    col_hdr, col_dl = st.columns([3, 1])
    col_hdr.subheader(f"Results — {total} transactions")
    col_dl.download_button(
        label="Download results CSV",
        data=df_to_csv_bytes(df_result),
        file_name="fraud_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # styled table — fraud rows red, review rows amber
    st.dataframe(
        df_result.style.apply(highlight_fraud, axis=1),
        use_container_width=True,
        height=min(400, 36 + 35 * len(df_result)),  # auto-size up to 400px
    )

    # ── Fraud-only filter ─────────────────────────────────────
    fraud_only = [r for r in results if r.get("is_fraud", False)]
    if fraud_only:
            with st.expander(f"Fraud detected — {len(fraud_only)} transaction(s)", expanded=True):

                # build display with row number from original file + transaction_id
                fraud_df = build_results_df(fraud_only).reset_index(drop=True)

                # row number = position in the original full results list (1-based)
                fraud_indices = [
                    i + 1                              # 1-based row number in the file
                    for i, r in enumerate(results)
                    if r.get("is_fraud", False)
                ]
                fraud_df.insert(0, "row #", fraud_indices)

                # make sure transaction_id is prominent — fill missing with row number
                if "transaction_id" in fraud_df.columns:
                    fraud_df["transaction_id"] = fraud_df["transaction_id"].fillna(
                        fraud_df["row #"].apply(lambda x: f"row_{x}")
                    )

                st.dataframe(
                    fraud_df.style.apply(highlight_fraud, axis=1),
                    use_container_width=True,
                    column_config={
                        "row #":          st.column_config.NumberColumn("Row #", width="small"),
                        "transaction_id": st.column_config.TextColumn("Transaction ID", width="medium"),
                        "fraud_probability": st.column_config.ProgressColumn(
                            "Fraud probability",
                            min_value=0, max_value=1, format="%.4f",
                        ),
                    }
                )

                # summary list — clean text callout of flagged IDs
                flagged_ids = [
                    r.get("transaction_id") or f"row {i+1}"
                    for i, r in enumerate(results)
                    if r.get("is_fraud", False)
                ]
                st.error(f"Flagged transactions: {', '.join(str(x) for x in flagged_ids)}")

    else:
        st.success("No fraudulent transactions detected in this batch.")

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown("""
    ### How to use

    1. Choose an input source in the sidebar — CSV upload, manual entry, or paste JSON
    2. Load your transactions
    3. Click **Run fraud detection**
    4. Results appear here with fraud highlighted in red

    The fraud API must be running on port 8000.
    """)