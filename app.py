import streamlit as st
import pandas as pd
import numpy as np
import io  # ← add this

st.set_page_config(layout="wide")
st.markdown("# 📊 Portfolio TER Calculator")

# … all your existing session‐state and upload/cleaning/filter code …

# ─── Step 5: Show current portfolio ───
if st.session_state.current_portfolio:
    cp = st.session_state.current_portfolio
    st.subheader("Step 5: Final Fund Table with ISINs and Charges")
    st.dataframe(cp["table"], use_container_width=True)
    if cp["ter"] is not None:
        st.metric("📊 Weighted Average TER", f"{cp['ter']:.2%}")

    # ←── Add CSV and Excel download buttons here
    csv = cp["table"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current portfolio as CSV",
        data=csv,
        file_name="current_portfolio.csv",
        mime="text/csv",
    )

    # Excel export
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        cp["table"].to_excel(writer, index=False, sheet_name="Portfolio")
    towrite.seek(0)
    st.download_button(
        label="Download current portfolio as Excel",
        data=towrite,
        file_name="current_portfolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

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

    for p in st.session_state.saved_portfolios:
        st.markdown(f"#### {p['label']}")
        st.metric("Weighted Average TER", f"{p['ter']:.2%}")
        st.dataframe(p["table"], use_container_width=True)

        # ←── And here too, add an Excel download per saved portfolio
        csv2 = p["table"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {p['label']} as CSV",
            data=csv2,
            file_name=f"{p['label'].lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"csv_{p['label']}"
        )
        towrite2 = io.BytesIO()
        with pd.ExcelWriter(towrite2, engine="xlsxwriter") as writer:
            p["table"].to_excel(writer, index=False, sheet_name=p['label'])
        towrite2.seek(0)
        st.download_button(
            label=f"Download {p['label']} as Excel",
            data=towrite2,
            file_name=f"{p['label'].lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{p['label']}"
        )

    if len(st.session_state.saved_portfolios) == 2:
        p1, p2 = st.session_state.saved_portfolios
        diff = p2["ter"] - p1["ter"]
        st.markdown("---")
        st.subheader("TER Difference (II − I)")
        st.metric("Difference", f"{diff:.2%}")

