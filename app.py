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

# ─── Step 1: Upload master share‐class file ───
uploaded_file = st.file_uploader("Upload Excel file with all share-class data", type=["xlsx"])
if not uploaded_file:
    st.stop()
df = pd.read_excel(uploaded_file, skiprows=2)

required = [
    "Family Name","Type of Share","Currency","Hedged",
    "Min. Initial","MiFID FH","Ongoing Charge","ISIN","Prospectus AF"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in master file: {missing}")
    st.stop()
st.success("Master share-class file loaded.")

# clean Ongoing Charge
df["Ongoing Charge"] = (
    df["Ongoing Charge"].astype(str)
       .str.replace("%","")
       .str.replace(",",".")
       .astype(float)
)

# ─── Choose input mode ───
mode = st.radio(
    "How would you like to build your portfolio?",
    ("Choose weights & share-classes manually",
     "Import Excel with existing ISINs & weights")
)

edited = []
if mode == "Import Excel with existing ISINs & weights":
    # ─── Step 1b: Upload existing portfolio file ───
    weights_file = st.file_uploader(
        "Upload Excel file with columns 'ISIN' and 'Peso %'", type=["xlsx"], key="weights"
    )
    if weights_file:
        wdf = pd.read_excel(weights_file)
        req2 = ["ISIN","Peso %"]
        miss2 = [c for c in req2 if c not in wdf.columns]
        if miss2:
            st.error(f"Missing in portfolio file: {miss2}")
        else:
            st.success("Existing portfolio loaded.")
            # merge on ISIN
            merged = pd.merge(wdf, df, on="ISIN", how="left", validate="one_to_many")
            if merged["Family Name"].isnull().any():
                bad = merged[merged["Family Name"].isnull()]["ISIN"].tolist()
                st.error(f"No share-class data for ISIN(s): {bad}")
            else:
                # build edited list
                for _, row in merged.iterrows():
                    edited.append({
                        "Family Name":    row["Family Name"],
                        "Type of Share":  row["Type of Share"],
                        "Currency":       row["Currency"],
                        "Hedged":         row["Hedged"],
                        "Min. Initial":   row["Min. Initial"],
                        "MiFID FH":       row["MiFID FH"],
                        "Prospectus AF":  row["Prospectus AF"],
                        "Weight %":       float(row["Peso %"])
                    })
                st.markdown("**Step 2 & 3 skipped: portfolio pre-filled from import.**")
else:
    # ─── Step 2: Global Filters ───
    st.markdown(
        """
        <style>
          select option[value="NOT FOUND"] { color: red !important; }
          div[data-baseweb="select"] [role="combobox"] div[aria-selected="true"] {
            color: red !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Step 2: Global Share Class Filters")
    filter_cols = ["Type of Share","Currency","Hedged","Min. Initial","MiFID FH"]
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

    # ─── Step 3: Manual per-fund selection ───
    st.markdown(
        """
        <style>
          select option[value="NOT FOUND"] { color: red !important; }
          div[data-baseweb="select"] [role="combobox"] div[aria-selected="true"] {
            color: red !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Step 3: Customize Share Class per Fund")
    st.write("🔽 Dropdowns only show valid combinations per fund")

    for idx, fam in enumerate(df["Family Name"].dropna().unique()):
        fund_df = df[df["Family Name"] == fam].copy()
        st.markdown(f"---\n#### {fam}")
        cols = st.columns([1.5,1.1,1.1,1.5,1.3,1.3,1.2])
        row = {"Family Name": fam}
        context = fund_df

        def cascade(i, label, key, ctx):
            opts = sorted(ctx[key].dropna().unique().tolist())
            init = global_filters[key] if key in global_filters and global_filters[key] in opts else "NOT FOUND"
            if init == "NOT FOUND":
                opts = ["NOT FOUND"] + opts
            sel = cols[i].selectbox(label, opts, index=opts.index(init), key=f"{key}_{idx}")
            new_ctx = ctx[ctx[key] == sel] if sel != "NOT FOUND" else ctx
            return sel, new_ctx

        # five globals
        row["Type of Share"], context = cascade(0, "Type of Share", "Type of Share", context)
        row["Currency"],     context = cascade(1, "Currency",     "Currency",     context)
        row["Hedged"],       context = cascade(2, "Hedged",       "Hedged",       context)
        row["Min. Initial"], context = cascade(3, "Min. Initial", "Min. Initial", context)
        row["MiFID FH"],     context = cascade(4, "MiFID FH",     "MiFID FH",     context)
        # always include Prospectus AF
        row["Prospectus AF"], context = cascade(5, "Prospectus AF", "Prospectus AF", context)

        # weight entry
        row["Weight %"] = cols[6].number_input(
            "Weight %",
            min_value=0.0, max_value=100.0, step=0.1,
            key=f"weight_{idx}"
        )
        edited.append(row)

# ─── Now steps 4–6 (Compute TER, show, compare) ───

# Step 4: Calculate TER
st.subheader("Step 4: Calculate ISIN, Ongoing Charge & TER")
if st.button("Calculate TER"):
    results, errors = [], []
    total_weighted = 0.0
    total_w = 0.0
    for row in edited:
        # ensure no NOT FOUND
        if any(val == "NOT FOUND" for val in [row.get(k) for k in [
            "Type of Share","Currency","Hedged",
            "Min. Initial","MiFID FH","Prospectus AF"]]):
            errors.append((row["Family Name"], "Invalid selection"))
            continue
        # find match
        match = df[
            (df["Family Name"] == row["Family Name"]) &
            (df["Type of Share"] == row["Type of Share"]) &
            (df["Currency"] == row["Currency"]) &
            (df["Hedged"] == row["Hedged"]) &
            (df["Min. Initial"] == row["Min. Initial"]) &
            (df["MiFID FH"] == row["MiFID FH"]) &
            (df["Prospectus AF"] == row["Prospectus AF"])
        ]
        if match.empty:
            errors.append((row["Family Name"], "No matching share class"))
            continue
        best = match.loc[match["Ongoing Charge"].idxmin()]
        charge = best["Ongoing Charge"]
        w = row["Weight %"]
        total_weighted += charge * (w/100)
        total_w += w
        results.append({**row, "ISIN": best["ISIN"], "Ongoing Charge": charge})
    result_df = pd.DataFrame(results)
    if total_w > 0:
        ter = total_weighted / (total_w/100)
        st.session_state.current_portfolio = {"table": result_df, "ter": ter}
    else:
        st.session_state.current_portfolio = {"table": result_df, "ter": None}
    st.session_state.current_errors = errors

# Step 5: Display current portfolio
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

# Step 6: Compare portfolios
if (
    st.session_state.current_portfolio and
    st.session_state.current_portfolio["ter"] is not None and
    not st.session_state.current_errors
):
    st.subheader("Step 6: Compare Portfolios")
    for p in st.session_state.saved_portfolios:
        st.markdown(f"#### {p['label']}")
        st.metric("Weighted Average TER", f"{p['ter']:.2%}")
        st.dataframe(p["table"], use_container_width=True)
    if len(st.session_state.saved_portfolios) == 0:
        st.button("Save for Comparison", on_click=save_as_I)
    elif len(st.session_state.saved_portfolios) == 1:
        st.button("Compare with Current", on_click=save_as_II)
    else:
        p1, p2 = st.session_state.saved_portfolios
        diff = p2["ter"] - p1["ter"]
        st.markdown("---")
        st.subheader("TER Difference (II − I)")
        st.metric("Difference", f"{diff:.2%}")
