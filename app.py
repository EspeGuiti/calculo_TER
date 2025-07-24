import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.markdown("# 📊 Portfolio TER Calculator")

# Initialize session state for saved portfolios
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = []

# Step 1: Upload and Validate Excel
uploaded_file = st.file_uploader("Upload Excel file with share classes", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=2)

    required_columns = [
        "Family Name", "Type of Share", "Currency", "Hedged",
        "Min. Initial", "MiFID FH", "Ongoing Charge", "ISIN"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    st.success("File uploaded and validated.")

    # ─── Cleaning ───
    df["Ongoing Charge"] = (
        df["Ongoing Charge"]
          .astype(str)
          .str.replace("%", "")
          .str.replace(",", ".")
          .astype(float)
    )

    filter_cols = ["Type of Share","Currency","Hedged","Min. Initial","MiFID FH"]
    group_key = "Family Name"

    # Step 2: Global Filters
    st.subheader("Step 2: Global Share Class Filters")
    global_opts = {col: sorted(df[col].dropna().unique()) for col in filter_cols}
    c1,c2,c3,c4,c5 = st.columns(5)
    global_filters = {}
    with c1:
        global_filters["Type of Share"] = st.selectbox("Type of Share", global_opts["Type of Share"])
    with c2:
        global_filters["Currency"] = st.selectbox("Currency", global_opts["Currency"])
    with c3:
        global_filters["Hedged"] = st.selectbox("Hedged", global_opts["Hedged"])
    with c4:
        global_filters["Min. Initial"] = st.selectbox("Min. Initial", global_opts["Min. Initial"])
    with c5:
        global_filters["MiFID FH"] = st.selectbox("MiFID FH", global_opts["MiFID FH"])

    # Step 3: Per‑Fund Cascading Dropdowns
    st.subheader("Step 3: Customize Share Class per Fund")
    st.write("🔽 Dropdowns only show valid combinations per fund.")

    unique_families = df[group_key].dropna().unique()
    edited = []

    for idx, family in enumerate(unique_families):
        fund_df = df[df[group_key]==family].copy()
        st.markdown(f"---\n#### {family}")
        cols = st.columns([1.7,1.2,1.2,1.7,1.5,1.3])

        row = {"Family Name": family, "Weight %": 0.0}
        context = fund_df

        def cascade(col_idx, label, key, context_df):
            opts = sorted(context_df[key].dropna().unique().tolist())
            init = global_filters[key] if global_filters[key] in opts else "NOT FOUND"
            if init=="NOT FOUND":
                opts = ["NOT FOUND"] + opts
            sel = cols[col_idx].selectbox(label, opts, index=opts.index(init), key=f"{key}_{idx}")
            new_ctx = context_df[context_df[key]==sel] if sel!="NOT FOUND" else context_df
            return sel, new_ctx

        row["Type of Share"], context = cascade(0, "Type of Share", "Type of Share", context)
        row["Currency"],     context = cascade(1, "Currency",     "Currency",     context)
        row["Hedged"],       context = cascade(2, "Hedged",       "Hedged",       context)
        row["Min. Initial"], context = cascade(3, "Min. Initial", "Min. Initial", context)
        row["MiFID FH"],     context = cascade(4, "MiFID FH",     "MiFID FH",     context)

        row["Weight %"] = cols[5].number_input(
            "Weight %",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key=f"weight_{idx}"
        )

        edited.append(row)

    # Single total weight at end of step 3
    total_weight = sum(r["Weight %"] for r in edited)
    st.markdown("---")
    left, _ = st.columns([1,3])
    with left:
        st.subheader("Total Weight")
        st.write(f"{total_weight:.2f}%")
        if abs(total_weight-100.0)>1e-6:
            st.warning(f"Total is {total_weight:.2f}%, must be 100% before TER calculation.")

    st.divider()

    # Step 4: Calculate ISIN and TER
    st.subheader("Step 4: Calculate ISIN, Ongoing Charge, and TER")

    if st.button("Calculate TER"):
        results, errors = [], []
        twc, tw = 0.0, 0.0

        for row in edited:
            if "NOT FOUND" in [row[k] for k in filter_cols]:
                errors.append((row["Family Name"], "Invalid selections"))
                continue

            match = df[
                (df[group_key]==row[group_key]) &
                (df["Type of Share"]==row["Type of Share"]) &
                (df["Currency"]==row["Currency"]) &
                (df["Hedged"]==row["Hedged"]) &
                (df["Min. Initial"]==row["Min. Initial"]) &
                (df["MiFID FH"]==row["MiFID FH"])
            ]
            if match.empty:
                errors.append((row["Family Name"], "No matching class"))
                continue

            best = match.loc[match["Ongoing Charge"].idxmin()]
            charge, weight = best["Ongoing Charge"], row["Weight %"]
            twc += charge * (weight/100)
            tw  += weight

            results.append({
                **row,
                "ISIN": best["ISIN"],
                "Ongoing Charge": charge
            })

        res_df = pd.DataFrame(results)
        st.success("TER calculation done.")
        st.subheader("Step 5: Final Fund Table")
        st.dataframe(res_df, use_container_width=True)

        if tw>0:
            port_ter = twc/(tw/100)
            st.metric("📊 Weighted Average TER", f"{port_ter:.2%}")
        else:
            st.warning("Provide weights to compute TER.")

        if errors:
            st.subheader("⚠️ Issues Detected")
            for fam, msg in errors:
                st.error(f"{fam}: {msg}")

        # Step 6: Save for Comparison
        if tw>0 and not errors:
            st.subheader("Step 6: Compare Portfolios")
            if st.button("Save for Comparison"):
                label = "Portfolio " + ("I" if len(st.session_state.saved_portfolios)==0 else "II")
                st.session_state.saved_portfolios.append({
                    "label": label,
                    "table": res_df.copy(),
                    "ter": port_ter
                })
                st.success(f"{label} saved.")

        # Display comparisons
        if st.session_state.saved_portfolios:
            st.markdown("---")
            for p in st.session_state.saved_portfolios:
                st.subheader(p["label"])
                st.metric("Weighted Average TER", f"{p['ter']:.2%}")
                st.dataframe(p["table"], use_container_width=True)

            if len(st.session_state.saved_portfolios)==2:
                ter1 = st.session_state.saved_portfolios[0]["ter"]
                ter2 = st.session_state.saved_portfolios[1]["ter"]
                diff = ter2 - ter1
                st.markdown("---")
                st.subheader("TER Difference (II − I)")
                st.metric("Difference", f"{diff:.2%}")

