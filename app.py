import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.markdown("# 📊 Calculadora de TER de Cartera")

# ─── Estado de sesión ───
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = []
if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = None
if "current_errors" not in st.session_state:
    st.session_state.current_errors = []

# ─── Callbacks Guardar/Comparar ───
def save_as_I():
    st.session_state.saved_portfolios.append({
        "label": "Cartera I",
        **st.session_state.current_portfolio
    })

def save_as_II():
    st.session_state.saved_portfolios.append({
        "label": "Cartera II",
        **st.session_state.current_portfolio
    })

# ─── Paso 1: Cargar fichero maestro ───
master_file = st.file_uploader("Sube el Excel con TODAS las clases de participación", type=["xlsx"])
if not master_file:
    st.stop()

df = pd.read_excel(master_file, skiprows=2)
required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "Min. Initial","MiFID FH","Ongoing Charge","ISIN","Prospectus AF","Transferable"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el fichero maestro: {missing}")
    st.stop()
st.success("Fichero maestro cargado correctamente.")

# Limpieza de Ongoing Charge
df["Ongoing Charge"] = (
    df["Ongoing Charge"].astype(str)
       .str.replace("%","")
       .str.replace(",",".")
       .astype(float)
)

# ─── Modo de entrada ───
mode = st.radio(
    "¿Cómo quieres construir la cartera?",
    ("Elegir manualmente pesos y clases de participación",
     "Importar Excel con ISINs y pesos existentes")
)

edited = []
filter_cols = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]  # orden: MiFID FH antes que Min. Initial

if mode == "Importar Excel con ISINs y pesos existentes":
    weights_file = st.file_uploader(
        "Sube un Excel con las columnas 'ISIN' y 'Peso %'", type=["xlsx"], key="weights"
    )
    if weights_file:
        wdf = pd.read_excel(weights_file)
        req2 = ["ISIN","Peso %"]
        miss2 = [c for c in req2 if c not in wdf.columns]
        if miss2:
            st.error(f"Faltan columnas en el fichero de cartera: {miss2}")
        else:
            st.success("Cartera existente cargada.")
            merged = pd.merge(wdf, df, on="ISIN", how="left", validate="one_to_many")
            if merged["Family Name"].isnull().any():
                bad = merged[merged["Family Name"].isnull()]["ISIN"].tolist()
                st.error(f"No hay datos de clase de participación para los ISIN(s): {bad}")
            else:
                for _, row in merged.iterrows():
                    edited.append({
                        "Family Name":   row["Family Name"],
                        "Type of Share": row["Type of Share"],
                        "Currency":      row["Currency"],
                        "Hedged":        row["Hedged"],
                        "MiFID FH":      row["MiFID FH"],
                        "Min. Initial":  row["Min. Initial"],
                        "Weight %":      float(row["Peso %"])
                    })
                st.markdown("**Cartera precargada desde la importación.**")
else:
    # ─── Paso 2: Filtros globales ───
    st.subheader("Paso 2: Filtros globales de clases de participación")
    opts = {col: sorted(df[col].dropna().unique()) for col in filter_cols}
    c1,c2,c3,c4,c5 = st.columns(5)
    global_filters = {}
    with c1:
        global_filters["Type of Share"] = st.selectbox("Tipo de clase", opts["Type of Share"])
    with c2:
        global_filters["Currency"] = st.selectbox("Divisa", opts["Currency"])
    with c3:
        global_filters["Hedged"] = st.selectbox("Cobertura", opts["Hedged"])
    with c4:
        global_filters["MiFID FH"] = st.selectbox("MiFID FH", opts["MiFID FH"])
    with c5:
        global_filters["Min. Initial"] = st.selectbox("Mín. Inversión", opts["Min. Initial"])

    # ─── Paso 3: Selección manual ───
    st.subheader("Paso 3: Personaliza la clase por fondo")
    st.write("ℹ️ *Prospectus AF y Traspasable se calculan automáticamente (no editables).*")

    for idx, fam in enumerate(df["Family Name"].dropna().unique()):
        fund_df = df[df["Family Name"] == fam].copy()
        st.markdown(f"---\n#### {fam}")
        cols = st.columns([1.5,1.1,1.1,1.2,1.2,1.0,1.3,1.3])
        row = {"Family Name": fam}
        context = fund_df

        def cascade(i, label, key, ctx):
            options = sorted(ctx[key].dropna().unique().tolist())
            init = global_filters[key] if key in global_filters and global_filters[key] in options else "NO ENCONTRADO"
            if init == "NO ENCONTRADO":
                options = ["NO ENCONTRADO"] + options
            sel = cols[i].selectbox(label, options, index=options.index(init), key=f"{key}_{idx}")
            new_ctx = ctx[ctx[key] == sel] if sel != "NO ENCONTRADO" else ctx
            return sel, new_ctx

        row["Type of Share"], context = cascade(0, "Tipo de clase", "Type of Share", context)
        row["Currency"],     context = cascade(1, "Divisa",       "Currency",     context)
        row["Hedged"],       context = cascade(2, "Cobertura",    "Hedged",       context)
        row["MiFID FH"],     context = cascade(3, "MiFID FH",     "MiFID FH",     context)
        row["Min. Initial"], context = cascade(4, "Mín. Inversión","Min. Initial", context)

        row["Weight %"] = cols[5].number_input("Peso %", min_value=0.0, max_value=100.0, step=0.1, key=f"weight_{idx}")

        # Prospectus AF y Traspasable automáticos
        prospectus_info = "—"
        transferable_info = "—"
        valid = all(row.get(k) != "NO ENCONTRADO" for k in ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"])
        if valid:
            m = fund_df[
                (fund_df["Type of Share"] == row["Type of Share"]) &
                (fund_df["Currency"]      == row["Currency"]) &
                (fund_df["Hedged"]        == row["Hedged"]) &
                (fund_df["MiFID FH"]      == row["MiFID FH"]) &
                (fund_df["Min. Initial"]  == row["Min. Initial"])
            ]
            if not m.empty:
                best = m.loc[m["Ongoing Charge"].idxmin()]
                prospectus_info = str(best.get("Prospectus AF", "—"))
                transferable_info = str(best.get("Transferable", "—"))

        cols[6].markdown(f"**Prospectus AF:** {prospectus_info}")
        cols[7].markdown(f"**Traspasable:** {transferable_info}")

        edited.append(row)

# ─── Resumen de pesos ───
total_weight = sum(r["Weight %"] for r in edited)
n_funds = len(edited)

def equalize_weights():
    if n_funds > 0:
        w = 100.0 / n_funds
        for i in range(n_funds):
            st.session_state[f"weight_{i}"] = w

col_sum, col_eq = st.columns([3,1])
with col_sum:
    st.subheader("Peso total")
    st.write(f"{total_weight:.2f}%")
    if abs(total_weight - 100.0) > 1e-6:
        st.warning("El peso total debe sumar 100% antes de calcular el TER.")
with col_eq:
    st.button("Equiponderar cartera", on_click=equalize_weights)

st.divider()

# ─── Paso 4: Calcular TER ───
st.subheader("Paso 4: Calcular TER de Cartera")
if st.button("Calcular TER"):
    results, errors = [], []
    total_weighted = 0.0
    total_w = 0.0
    for row in edited:
        keys = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]
        if any(row.get(k) == "NO ENCONTRADO" for k in keys):
            errors.append((row["Family Name"], "Selección inválida"))
            continue

        match = df[
            (df["Family Name"] == row["Family Name"]) &
            (df["Type of Share"] == row["Type of Share"]) &
            (df["Currency"]     == row["Currency"]) &
            (df["Hedged"]       == row["Hedged"]) &
            (df["MiFID FH"]     == row["MiFID FH"]) &
            (df["Min. Initial"] == row["Min. Initial"])
        ]
        if match.empty:
            errors.append((row["Family Name"], "No se encontró una clase que coincida"))
            continue

        best = match.loc[match["Ongoing Charge"].idxmin()]
        charge = best["Ongoing Charge"]
        w = row["Weight %"]
        total_weighted += charge * (w/100)
        total_w += w

        results.append({
            **row,
            "ISIN": best["ISIN"],
            "Prospectus AF": best.get("Prospectus AF", "—"),
            "Traspasable": best.get("Transferable", "—"),
            "Ongoing Charge": charge
        })

    df_res = pd.DataFrame(results)
    if total_w > 0:
        ter = total_weighted / (total_w/100)
        st.session_state.current_portfolio = {"table": df_res, "ter": ter}
    else:
        st.session_state.current_portfolio = {"table": df_res, "ter": None}
    st.session_state.current_errors = errors

# ─── Paso 5: Mostrar cartera actual ───
if st.session_state.current_portfolio:
    cp = st.session_state.current_portfolio
    st.subheader("Paso 5: Tabla final con ISIN, Prospectus AF, Traspasable y comisiones")
    st.dataframe(cp["table"], use_container_width=True)
    if cp["ter"] is not None:
        st.metric("📊 TER medio ponderado", f"{cp['ter']:.2%}")
    if st.session_state.current_errors:
        st.subheader("⚠️ Incidencias detectadas")
        for fam, msg in st.session_state.current_errors:
            st.error(f"{fam}: {msg}")

# ─── Paso 6: Comparar carteras ───
if (
    st.session_state.current_portfolio and
    st.session_state.current_portfolio["ter"] is not None and
    not st.session_state.current_errors
):
    st.subheader("Paso 6: Comparar carteras")

    if len(st.session_state.saved_portfolios) == 0:
        st.button("Guardar para comparativa", on_click=save_as_I, key="save_I")
    elif len(st.session_state.saved_portfolios) == 1:
        st.button("Comparar con Cartera I", on_click=save_as_II, key="compare_with_I_top")
        for p in st.session_state.saved_portfolios:
            st.markdown(f"#### {p['label']}")
            st.metric("TER medio ponderado", f"{p['ter']:.2%}")
            st.dataframe(p["table"], use_container_width=True)
    else:
        for p in st.session_state.saved_portfolios:
            st.markdown(f"#### {p['label']}")
            st.metric("TER medio ponderado", f"{p['ter']:.2%}")
            st.dataframe(p["table"], use_container_width=True)

        p1, p2 = st.session_state.saved_portfolios
        diff = p2["ter"] - p1["ter"]
        st.markdown("---")
        st.subheader("Diferencia de TER (II − I)")
        st.metric("Diferencia", f"{diff:.2%}")
