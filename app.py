import streamlit as st
import pandas as pd
import numpy as np

# Step 1: Upload and Validate Excel
uploaded_file = st.file_uploader("Upload Excel File with Share Classes", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_columns = [
        "Family Name", "Type of Share", "Currency", "Hedged",
        "Min. Initial", "MiFID FH", "Ongoing Charge", "ISIN"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing from the file: {missing_columns}")
    else:
        st.success("File uploaded successfully. All required columns are present.")

        # ✅ CLEANING STEP
        df["Ongoing Charge"] = df["Ongoing Charge"].astype(str).str.replace("%","").str.replace(",",".").astype(float)

        filter_cols = ["Type of Share", "Currency", "Hedged", "Min. Initial", "MiFID FH"]
        group_key = "Family Name"

        # ---------- STEP 2: Global Share Class Filters ----------
        st.subheader("Step 2: Global Share Class Filters")

        global_filter_options = {col: sorted(df[col].dropna().unique()) for col in filter_cols}

        c1, c2, c3, c4, c5 = st.columns(5)
        global_filters = {}
        with c1:
            global_filters["Type of Share"] = st.selectbox("Type of Share", global_filter_options["Type of Share"])
        with c2:
            global_filters["Currency"] = st.selectbox("Currency", global_filter_options["Currency"])
        with c3:
            global_filters["Hedged"] = st.selectbox("Hedged", global_filter_options["Hedged"])
        with c4:
            global_filters["Min. Initial"] = st.selectbox("Min. Initial", global_filter_options["Min. Initial"])
        with c5:
            global_filters["MiFID FH"] = st.selectbox("MiFID FH", global_filter_options["MiFID FH"])

        # ---------- STEP 3: Per-Fund Cascading Dropdowns ----------
        st.subheader("Step 3: Customize Share Class per Fund")
        st.write("🔽 Customize each fund’s settings. Dropdowns only show valid combinations for that fund based on previous selections.")

        unique_families = df[group_key].dropna().unique()
        edited_rows = []

        for idx, family in enumerate(unique_families):
            fund_data = df[df[group_key] == family].copy()

            st.markdown(f"---\n#### {family}")
            cols = st.columns([1.7, 1.2, 1.2, 1.7, 1.5, 1.3, 1.0, 1.2])  # 8 columns

            row_data = {
                "Family Name": family,
                "Weight %": 0.0,
                "ISIN": "",
                "Ongoing Charge": None
            }

            # Start with full fund subset and shrink as we pick values
            context = fund_data

            # Helper to build each dropdown
            def cascade_select(label, col_key, global_value, context_df, key_suffix):
                opts = sorted(context_df[col_key].dropna().unique().tolist())
                initial = global_value if global_value in opts else "NOT FOUND"
                if initial == "NOT FOUND":
                    opts = ["NOT FOUND"] + opts
                choice = cols[key_suffix].selectbox(
                    label,
                    options=opts,
                    index=opts.index(initial),
                    key=f"{col_key}_{idx}"
                )
                if choice != "NOT FOUND":
                    return choice, context_df[context_df[col_key] == choice]
                return choice, context_df

            # 0–4: dropdown filters
            row_data["Type of Share"], context = cascade_select("Type of Share", "Type of Share", global_filters["Type of Share"], context, 0)
            row_data["Currency"], context = cascade_select("Currency", "Currency", global_filters["Currency"], context, 1)
            row_data["Hedged"], context = cascade_select("Hedged", "Hedged", global_filters["Hedged"], context, 2)
            row_data["Min. Initial"], context = cascade_select("Min. Initial", "Min. Initial", global_filters["Min. Initial"], context, 3)
            row_data["MiFID FH"], context = cascade_select("MiFID FH", "MiFID FH", global_filters["MiFID FH"], context, 4)

            # 5: Weight input
            row_data["Weight %"] = cols[5].number_input(
                "Weight %",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"weight_{idx}"
            )

            edited_rows.append(row_data)

        edited_df = pd.DataFrame(edited_rows)
        st.divider()

        # ---------- STEP 4: Calculate ISIN and TER ----------
        st.subheader("Step 4: Calculate ISIN, Ongoing Charge, and TER")

        if st.button("Calculate TER"):
            results = []
            errors = []
            total_weighted_charge = 0.0
            total_weight = 0.0

            for i, row in edited_df.iterrows():
                if "NOT FOUND" in [
                    row["Type of Share"],
                    row["Currency"],
                    row["Hedged"],
                    row["Min. Initial"],
                    row["MiFID FH"]
                ]:
                    errors.append((row["Family Name"], "One or more selections are 'NOT FOUND'"))
                    continue

                match_df = df[
                    (df["Family Name"] == row["Family Name"]) &
                    (df["Type of Share"] == row["Type of Share"]) &
                    (df["Currency"] == row["Currency"]) &
                    (df["Hedged"] == row["Hedged"]) &
                    (df["Min. Initial"] == row["Min. Initial"]) &
                    (df["MiFID FH"] == row["MiFID FH"])
                ]

                if match_df.empty:
                    errors.append((row["Family Name"], "No matching share class found"))
                    continue

                best_row = match_df.loc[match_df["Ongoing Charge"].idxmin()]
                isin = best_row["ISIN"]
                charge = best_row["Ongoing Charge"]

                weight = row["Weight %"]
                weighted = charge * (weight / 100.0) if weight else 0.0

                total_weighted_charge += weighted
                total_weight += weight if weight else 0.0

                results.append({
                    "Family Name": row["Family Name"],
                    "Type of Share": row["Type of Share"],
                    "Currency": row["Currency"],
                    "Hedged": row["Hedged"],
                    "Min. Initial": row["Min. Initial"],
                    "MiFID FH": row["MiFID FH"],
                    "Weight %": weight,
                    "ISIN": isin,
                    "Ongoing Charge": charge
                })

            result_df = pd.DataFrame(results)

            st.success("TER calculation completed successfully.")
            st.subheader("Step 5: Final Fund Table with ISINs and Charges")
            st.dataframe(result_df, use_container_width=True)

            if total_weight > 0:
                portfolio_ter = total_weighted_charge / (total_weight / 100.0)
                st.metric(label="📊 Weighted Average TER", value=f"{portfolio_ter:.2%}")
            else:
                st.warning("No weights provided to compute average TER.")

            if errors:
                st.subheader("⚠️ Issues Detected")
                for fam, reason in errors:
                    st.error(f"{fam}: {reason}")
