import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.markdown("# 📊 Portfolio TER Calculator")

# Step 1: Upload and Validate Excel
uploaded_file = st.file_uploader("Upload Excel file with share classes", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=2)

    required_columns = [
        "Family Name", "Type of Share", "Currency", "Hedged",
        "Min. Initial", "MiFID FH", "Ongoing Charge", "ISIN"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
    else:
        st.success("File uploaded successfully. All required columns are present.")

        # ─── Cleaning ───
        df["Ongoing Charge"] = (
            df["Ongoing Charge"]
              .astype(str)
              .str.replace("%", "")
              .str.replace(",", ".")
              .astype(float)
        )

        filter_cols = ["Type of Share", "Currency", "Hedged", "Min. Initial", "MiFID FH"]
        group_key = "Family Name"

        # ─── STEP 2: Global Filters ───
        st.subheader("Step 2: Global Share Class Filters")
        global_options = {col: sorted(df[col].dropna().unique()) for col in filter_cols}

        c1, c2, c3, c4, c5 = st.columns(5)
        global_filters = {}
        with c1:
            global_filters["Type of Share"] = st.selectbox("Type of Share", global_options["Type of Share"])
        with c2:
            global_filters["Currency"] = st.selectbox("Currency", global_options["Currency"])
        with c3:
            global_filters["Hedged"] = st.selectbox("Hedged", global_options["Hedged"])
        with c4:
            global_filters["Min. Initial"] = st.selectbox("Min. Initial", global_options["Min. Initial"])
        with c5:
            global_filters["MiFID FH"] = st.selectbox("MiFID FH", global_options["MiFID FH"])

        # ─── STEP 3: Per-Fund Cascading Dropdowns ───
        st.subheader("Step 3: Customize Share Class per Fund")
        st.write("🔽 Customize each fund’s settings. Dropdowns only show valid combinations per fund.")

        unique_families = df[group_key].dropna().unique()
        edited_rows = []

        for idx, family in enumerate(unique_families):
            fund_df = df[df[group_key] == family].copy()

            st.markdown(f"---\n#### {family}")
            cols = st.columns([1.7, 1.2, 1.2, 1.7, 1.5, 1.3])

            row = {
                "Family Name": family,
                "Type of Share": None,
                "Currency": None,
                "Hedged": None,
                "Min. Initial": None,
                "MiFID FH": None,
                "Weight %": 0.0
            }

            context = fund_df

            def cascade(col_idx, label, key):
                opts = sorted(context[key].dropna().unique().tolist())
                init = global_filters[key] if global_filters[key] in opts else "NOT FOUND"
                if init == "NOT FOUND":
                    opts = ["NOT FOUND"] + opts
                sel = cols[col_idx].selectbox(label, opts, index=opts.index(init), key=f"{key}_{idx}")
                return sel, context[context[key] == sel] if sel != "NOT FOUND" else context

            row["Type of Share"], context = cascade(0, "Type of Share", "Type of Share")
            row["Currency"],     context = cascade(1, "Currency",     "Currency")
            row["Hedged"],       context = cascade(2, "Hedged",       "Hedged")
            row["Min. Initial"], context = cascade(3, "Min. Initial", "Min. Initial")
            row["MiFID FH"],     context = cascade(4, "MiFID FH",     "MiFID FH")

            row["Weight %"] = cols[5].number_input(
                "Weight %",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"weight_{idx}"
            )

            edited_rows.append(row)

        # ─── Show a single Total Weight at end of Step 3 ───
        total_weight = sum(r["Weight %"] for r in edited_rows)
        st.markdown("---")

        # Place on the left
        col_left, _ = st.columns([1, 3])
        with col_left:
            st.subheader("Total Weight")
            st.write(f"{total_weight:.2f}%")

            # Notification if not 100%
            if abs(total_weight - 100.0) > 1e-6:
                st.warning(f"The total weight is {total_weight:.2f}%. It should sum to 100% before calculating TER.")

        st.divider()

        # ─── STEP 4: Calculate ISIN and TER ───
        st.subheader("Step 4: Calculate ISIN, Ongoing Charge, and TER")

        if st.button("Calculate TER"):
            results = []
            errors = []
            total_weighted_charge = 0.0
            total_w = 0.0

            for row in edited_rows:
                # validate selections
                if "NOT FOUND" in [row[k] for k in ["Type of Share","Currency","Hedged","Min. Initial","MiFID FH"]]:
                    errors.append((row["Family Name"], "One or more selections are 'NOT FOUND'"))
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
                    errors.append((row["Family Name"], "No matching share class found"))
                    continue

                best = match.loc[match["Ongoing Charge"].idxmin()]
                charge = best["Ongoing Charge"]
                weight = row["Weight %"]
                total_weighted_charge += charge * (weight / 100)
                total_w += weight

                results.append({
                    **row,
                    "ISIN": best["ISIN"],
                    "Ongoing Charge": charge
                })

            result_df = pd.DataFrame(results)
            st.success("TER calculation completed.")
            st.subheader("Step 5: Final Fund Table with ISINs and Charges")
            st.dataframe(result_df, use_container_width=True)

            if total_w > 0:
                portfolio_ter = total_weighted_charge / (total_w / 100)
                st.metric(label="📊 Weighted Average TER", value=f"{portfolio_ter:.2%}")
            else:
                st.warning("No weights provided to compute average TER.")

            if errors:
                st.subheader("⚠️ Issues Detected")
                for fam, reason in errors:
                    st.error(f"{fam}: {reason}")
