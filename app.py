import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.markdown("# 📊 Portfolio TER Calculator")

# ─── Session state setup ───
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = []
if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = None
if "current_errors" not in st.session_state:
    st.session_state.current_errors = []

# ─── Callbacks for Save/Compare ───
def save_as_I():
    st.session_state.saved_portfolios.append({
        "label": "Portfolio I",
        **st.session_state.current_portfolio
    })

def save_as_II():
    st.session_state.saved_portfolios.append({
        "label": "Portfolio II",
        **st.session_state.current_portfolio
    })

# ─── Step 1: Upload ───
uploaded_file = st.file_uploader("Upload Excel file with share classes", type=["xlsx"])
if not uploaded_file:
    st.stop()
df = pd.read_excel(uploaded_file, skiprows=2)
required = [
    "Family Name","Type of Share","Currency","Hedged",
    "Min. Initial","MiFID FH","Ongoing Charge","ISIN"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
st.success("File uploaded and validated.")

# ─── Clean Ongoing Charge ───
df["Ongoing Charge"] = (
    df["Ongoing Charge"].astype(str)
       .str.replace("%","")
       .str.replace(",",".")
       .astype(float)
)

# ─── Step 2: Global Filters ───
# Inject CSS so any dropdown option “NOT FOUND” is red
st.markdown(
    """
    <style>
      select option[value="NOT FOUND"] {
        color: red !important;
      }
      /* ensure selected “NOT FOUND” also shows red */
      div[data-baseweb="select"] [role="combobox"] div[aria-selected="true"] {
        color: red !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Step 2: Global Share Class Filters")
filter_cols = [
    "Type of Share","Currency","Hedged",
    "Min. Initial","MiFID FH"
]
opts = {col: sorted(df[col].dropna().unique()) for col in filter_cols}
c1,c2,c3,c4,c5 = st.columns(5)
global_filters = {}
with c1:
    global_filters["Type of Share"] = st.selectbox("Type of Share", opts["Type of Share"])
with c2:
    global_filters["Currency"] = st.selectbox("Currency", opts["Currency"])
with c3:
    global_filters["Hedged"] = st.selectbox("Hedged", opts["Hedged"])
with c4:
    global_filters["Min. Initial"] = st.selectbox("Min. Initial", opts["Min. Initial"])
with c5:
    global_filters["MiFID FH"] = st.selectbox("MiFID FH", opts["MiFID FH"])

# ─── Step 3: Per‑Fund Cascading Dropdowns ───
# Re-inject the same CSS so “NOT FOUND” is red here too
st.markdown(
    """
    <style>
      select option[value="NOT FOUND"] {
        color: red !important;
      }
      div[data-baseweb="select"] [role="combobox"] div[aria-selected="true"] {
        color: red !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Step 3: Customize Share Class per Fund")
st.write("🔽 Dropdowns only show valid combinations per fund")

edited = []
for idx, fam in enumerate(df["Family Name"].dropna().unique()):
    fund_df = df[df["Family Name"] == fam].copy()
    st.markdown(f"---\n#### {fam}")
    cols = st.columns([1.7,1.2,1.2,1.7,1.5,1.3])
    row = {"Family Name": fam, "Weight %": 0.0}
    context = fund_df

    def cascade(i, label, key, ctx):
        opts = sorted(ctx[key].dropna().unique().tolist())
        init = global_filters[key] if global_filters[key] in opts else "NOT FOUND"
        if init == "NOT FOUND":
            opts = ["NOT FOUND"] + opts
        sel = cols[i].selectbox(label, opts, index=opts.index(init), key=f"{key}_{idx}")
        new_ctx = ctx[ctx[key] == sel] if sel != "NOT FOUND" else ctx
        return sel, new_ctx

    row["Type of Share"], context = cascade(0, "Type of Share", "Type of Share", context)
    row["Currency"],     context = cascade(1, "Currency",     "Currency",     context)
    row["Hedged"],       context = cascade(2, "Hedged",       "Hedged",       context)
    row["Min. Initial"], context = cascade(3, "Min. Initial", "Min. Initial", context)
    row["MiFID FH"],     context = cascade(4, "MiFID FH",     "MiFID FH",     context)

    row["Weight %"] = cols[5].number_input(
        "Weight %",
        min_value=0.0, max_value=100.0, step=0.1,
        key=f"weight_{idx}"
    )
    edited.append(row)

# ─── Total Weight Summary ───
total_weight = sum(r["Weight %"] for r in edited)
st.markdown("---")
left, _ = st.columns([1,3])
with left:
    st.subheader("Total Weight")
    st.write(f"{total_weight:.2f}%")
    if abs(total_weight - 100.0) > 1e-6:
        st.warning("Total must sum to 100% before calculating TER")
st.divider()

# ─── Step 4: Calculate TER ───
st.subheader("Step 4: Calculate TER")
if st.button("Calculate TER", key="calc"):
    results, errors = [], []
    twc, tw = 0.0, 0.0
    for row in edited:
        if "NOT FOUND" in [row[c] for c in filter_cols]:
            errors.append((row["Family Name"], "Invalid selections"))
            continue
        match = df[
            (df["Family Name"] == row["Family Name"]) &
            (df["Type of Share"] == row["Type of Share"]) &
            (df["Currency"] == row["Currency"]) &
            (df["Hedged"] == row["Hedged"]) &
            (df["Min. Initial"] == row["Min. Initial"]) &
            (df["MiFID FH"] == row["MiFID FH"])
        ]
        if match.empty:
            errors.append((row["Family Name"], "No matching share class"))
            continue
        best = match.loc[match["Ongoing Charge"].idxmin()]
        charge, weight = best["Ongoing Charge"], row["Weight %"]
        twc += charge * (weight / 100)
        tw  += weight
        results.append({
            **row,
            "ISIN": best["ISIN"],
            "Ongoing Charge": charge
        })

    df_res = pd.DataFrame(results)
    if tw > 0 and not errors:
        st.session_state.current_portfolio = {"table": df_res, "ter": twc / (tw / 100)}
    else:
        st.session_state.current_portfolio = {"table": df_res, "ter": None}
    st.session_state.current_errors = errors

# ─── Step 5: Show current portfolio ───
if st.session_state.current_portfolio:
    cp = st.session_state.current_portfolio
    st.subheader("Step 5: Final Fund Table with ISINs and Charges")
    st.dataframe(cp["table"], use_container_width=True)
    if cp["ter"] is not None:
        st.metric("📊 Weighted Average TER", f"{cp['ter']:.2%}")
    if st.session_state.current_errors:
        st.subheader("⚠️ Issues Detected")
        for fam, msg in st.session_state.current_errors:
            st.error(f"{fam}: {msg}")

# ─── Step 6: Compare Portfolios ───
if (
    st.session_state.current_portfolio
    and st.session_state.current_portfolio["ter"] is not None
    and not st.session_state.current_errors
):
    st.subheader("Step 6: Compare Portfolios")

    # render saved portfolios
    for p in st.session_state.saved_portfolios:
        st.markdown(f"#### {p['label']}")
        st.metric("Weighted Average TER", f"{p['ter']:.2%}")
        st.dataframe(p["table"], use_container_width=True)

    # next action
    if len(st.session_state.saved_portfolios) == 0:
        st.button("Save for Comparison", on_click=save_as_I, key="save1")
    elif len(st.session_state.saved_portfolios) == 1:
        st.button("Compare with Current", on_click=save_as_II, key="compare")
    else:
        p1, p2 = st.session_state.saved_portfolios
        diff = p2["ter"] - p1["ter"]
        st.markdown("---")
        st.subheader("TER Difference (II − I)")
        st.metric("Difference", f"{diff:.2%}")


