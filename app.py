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
if "preview_ii" not in st.session_state:
    st.session_state.preview_ii = False

# NUEVO: flags/estado para poder editar una cartera importada
if "edit_import_to_manual" not in st.session_state:
    st.session_state.edit_import_to_manual = False
if "edited_rows" not in st.session_state:
    st.session_state.edited_rows = []

# ─── Callbacks Guardar/Comparar ───
def save_as_I():
    st.session_state.saved_portfolios.append({
        "label": "Cartera I",
        **st.session_state.current_portfolio
    })

def save_as_II():
    # Guarda la cartera actual como Cartera II
    st.session_state.saved_portfolios.append({
        "label": "Cartera II",
        **st.session_state.current_portfolio
    })
    # Deja de mostrarla como "previsualización"
    st.session_state.preview_ii = False
    # (opcional) cierra el modo editor si estaba abierto
    st.session_state.edit_import_to_manual = False

# ─── Paso 1: Cargar fichero maestro ───
master_file = st.file_uploader("Sube el Excel de All Funds con TODAS las clases de participación", type=["xlsx"])
if not master_file:
    st.stop()

df = pd.read_excel(master_file, skiprows=2)

required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el fichero importado: {missing}")
    st.stop()

# Indicador de si existe la columna Transferable en el Excel
has_transferable = "Transferable" in df.columns
st.success("Fichero importado correctamente.")

# Limpieza Ongoing Charge
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
# Orden: MiFID FH -> Min. Initial (como pediste)
filter_cols = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]

# Definir siempre para evitar NameError en funciones que lo referencian
global_filters = {}

if mode == "Importar Excel con ISINs y pesos existentes":
    weights_file = st.file_uploader(
        "Sube un Excel con columnas 'ISIN' y 'Peso %'", type=["xlsx"], key="weights"
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
                st.error(f"No hay datos para ISIN(s): {bad}")
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
                # NUEVO: guardar la cartera precargada para poder editarla luego
                st.session_state.edited_rows = edited.copy()

else:
    st.subheader("Paso 2: Filtros globales de clases de participación")
    opts = {col: sorted(df[col].dropna().unique()) for col in filter_cols}
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        global_filters["Type of Share"] = st.selectbox("Tipo de participación", opts["Type of Share"])
    with c2:
        global_filters["Currency"] = st.selectbox("Divisa", opts["Currency"])
    with c3:
        global_filters["Hedged"] = st.selectbox("Hedged", opts["Hedged"])
    with c4:
        global_filters["MiFID FH"] = st.selectbox("MiFID FH", opts["MiFID FH"])
    with c5:
        global_filters["Min. Initial"] = st.selectbox("Mín. Inversión", opts["Min. Initial"])

    # ─── Paso 3: Personaliza la clase por fondo ───
    st.subheader("Paso 3: Personaliza la clase por fondo")
    st.write("ℹ️ *Prospectus AF y Traspasable se calculan automáticamente con la clase seleccionada.*")

    for idx, fam in enumerate(df["Family Name"].dropna().unique()):
        fund_df = df[df["Family Name"] == fam].copy()
        st.markdown(f"---\n#### {fam}")

        cols = st.columns([1.5,1.1,1.1,1.2,1.2,1.0,1.5])
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

        row["Type of Share"], context = cascade(0, "Tipo de participación", "Type of Share", context)
        row["Currency"],     context = cascade(1, "Divisa", "Currency", context)
        row["Hedged"],       context = cascade(2, "Hedged", "Hedged", context)
        row["MiFID FH"],     context = cascade(3, "MiFID FH", "MiFID FH", context)
        row["Min. Initial"], context = cascade(4, "Mín. Inversión", "Min. Initial", context)

        row["Weight %"] = cols[5].number_input(
            "Peso %",
            min_value=0.0, max_value=100.0, step=0.1,
            key=f"weight_{idx}"
        )

        # Prospectus AF + Transferable automáticos (apilados)
        prospectus_info = "—"
        transferable_info = "—"
        valid = all(row.get(k) != "NO ENCONTRADO" for k in filter_cols)
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
                prospectus_info  = str(best.get("Prospectus AF", "—"))
                if has_transferable:
                    transferable_info = str(best.get("Transferable", "—"))

        with cols[6]:
            st.markdown(f"**Prospectus AF:** {prospectus_info}")
            st.markdown(f"**Traspasable:** {transferable_info}")

        edited.append(row)

    # NUEVO: guardar también la edición manual “corriente”
    st.session_state.edited_rows = edited.copy()

# ─── Total Weight Summary & Equal Weight Button ───
total_weight = sum(r["Weight %"] for r in edited)
n_funds = len(edited)

col_sum, col_eq = st.columns([3,1])
with col_sum:
    st.subheader("Total Weight")
    st.write(f"{total_weight:.2f}%")
    if abs(total_weight - 100.0) > 1e-6:
        st.warning("Total must sum to 100% before calculating TER")

# Solo mostrar el botón de reparto igual cuando el modo es MANUAL
if mode == "Elegir manualmente pesos y clases de participación":
    def equalize_weights():
        if n_funds > 0:
            w = 100.0 / n_funds
            for i in range(n_funds):
                st.session_state[f"weight_{i}"] = w

    with col_eq:
        st.button("Equal Weight", on_click=equalize_weights)


# ─── Paso 4: Calcular TER ───
st.subheader("Paso 4: Calcular TER")
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
            (df["Currency"] == row["Currency"]) &
            (df["Hedged"] == row["Hedged"]) &
            (df["MiFID FH"] == row["MiFID FH"]) &
            (df["Min. Initial"] == row["Min. Initial"])
        ]
        if match.empty:
            errors.append((row["Family Name"], "No se encontró clase que coincida"))
            continue

        best = match.loc[match["Ongoing Charge"].idxmin()]
        charge = best["Ongoing Charge"]
        w = row["Weight %"]
        total_weighted += charge * (w/100)
        total_w += w

        result_row = {
            **row,
            "ISIN": best["ISIN"],
            "Prospectus AF": best.get("Prospectus AF", "—"),
            "Ongoing Charge": charge
        }
        if has_transferable:
            result_row["Transferable"] = best.get("Transferable", "—")
        results.append(result_row)

    df_res = pd.DataFrame(results)
    if total_w > 0:
        ter = total_weighted / (total_w/100)
        st.session_state.current_portfolio = {"table": df_res, "ter": ter}
    else:
        st.session_state.current_portfolio = {"table": df_res, "ter": None}
    st.session_state.current_errors = errors

# ─── Utilidad: preparar tabla para mostrar con orden y etiquetas visuales ───
def pretty_table(df_in: pd.DataFrame) -> pd.DataFrame:
    tbl = df_in.copy()
    if "Traspasable" in tbl.columns:
        pass
    elif "Transferable" in tbl.columns:
        tbl.rename(columns={"Transferable": "Traspasable"}, inplace=True)
    desired = [
        "Family Name",
        "Type of Share", "Currency", "Hedged", "MiFID FH", "Min. Initial",
        "ISIN", "Prospectus AF", "Traspasable",
        "Ongoing Charge", "Weight %"
    ]
    existing = [c for c in desired if c in tbl.columns]
    rest = [c for c in tbl.columns if c not in existing]
    return tbl[existing + rest]

# ─── Paso 5: Mostrar cartera ───
if st.session_state.current_portfolio:
    cp = st.session_state.current_portfolio
    if st.session_state.get("preview_ii", False):
        st.subheader("Paso 5: Cartera II (previsualización)")
        if len(st.session_state.saved_portfolios) == 1:
            st.info("Revisa esta **Cartera II (previsualización)**. Si está OK, en el Paso 6 pulsa **Comparar con Cartera I** para fijarla y comparar.")
    else:
        st.subheader("Paso 5: Tabla final")

    st.dataframe(pretty_table(cp["table"]), use_container_width=True)
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

    num_saved = len(st.session_state.saved_portfolios)

    if num_saved == 0:
        st.button("Guardar para comparar", on_click=save_as_I, key="save_as_I_btn")

    elif num_saved == 1:
        if st.button("Editar esta cartera (cargar Paso 3)", key="edit_import_btn"):
            st.session_state.edit_import_to_manual = True
    
        IS_MANUAL = mode.startswith("Elegir manualmente")
        if IS_MANUAL:
            st.button("Comparar con Cartera I", on_click=save_as_II, key="compare_with_I_btn")
    
        p1 = st.session_state.saved_portfolios[0]
        st.markdown(f"#### {p1['label']}")
        st.metric("TER medio ponderado", f"{p1['ter']:.2%}")
        st.dataframe(pretty_table(p1["table"]), use_container_width=True)


    else:
        p1, p2 = st.session_state.saved_portfolios[0], st.session_state.saved_portfolios[1]

        st.markdown(f"#### {p1['label']}")
        st.metric("TER medio ponderado", f"{p1['ter']:.2%}")
        st.dataframe(pretty_table(p1["table"]), use_container_width=True)

        st.markdown(f"#### {p2['label']}")
        st.metric("TER medio ponderado", f"{p2['ter']:.2%}")
        st.dataframe(pretty_table(p2["table"]), use_container_width=True)

        diff = p2["ter"] - p1["ter"]
        st.markdown("---")
        st.subheader("Diferencia de TER (II − I)")
        st.metric("Diferencia", f"{diff:.2%}")

# ─── Editor tras importación: Paso 3 (prefill) ───
if st.session_state.edit_import_to_manual and st.session_state.edited_rows:
    st.markdown("---")
    st.subheader("Paso 3 (edición): Personaliza la clase por fondo (a partir de la cartera importada)")
    st.write("ℹ️ Los selectores y pesos se han precargado desde el Excel importado. Puedes cambiar la clase y recalcular el TER.")

    families = [r["Family Name"] for r in st.session_state.edited_rows]
    edited_from_import = []

    for idx, fam in enumerate(families):
        base_row = st.session_state.edited_rows[idx]
        fund_df = df[df["Family Name"] == fam].copy()

        st.markdown(f"---\n#### {fam}")
        cols = st.columns([1.5,1.1,1.1,1.2,1.2,1.0,1.5])
        context = fund_df
        row = {"Family Name": fam}

        def cascade_prefill(i, label, key, ctx, prefill_value):
            options = sorted(ctx[key].dropna().unique().tolist())
            if prefill_value in options:
                init = prefill_value
            else:
                init = "NO ENCONTRADO"
                options = ["NO ENCONTRADO"] + options
            sel = cols[i].selectbox(label, options, index=options.index(init), key=f"edit_{key}_{idx}")
            new_ctx = ctx[ctx[key] == sel] if sel != "NO ENCONTRADO" else ctx
            return sel, new_ctx

        row["Type of Share"], context = cascade_prefill(0, "Tipo de participación", "Type of Share", context, base_row.get("Type of Share"))
        row["Currency"],     context = cascade_prefill(1, "Divisa", "Currency", context, base_row.get("Currency"))
        row["Hedged"],       context = cascade_prefill(2, "Hedged", "Hedged", context, base_row.get("Hedged"))
        row["MiFID FH"],     context = cascade_prefill(3, "MiFID FH", "MiFID FH", context, base_row.get("MiFID FH"))
        row["Min. Initial"], context
