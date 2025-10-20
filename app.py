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
    st.session_state.saved_portfolios.append({
        "label": "Cartera II",
        **st.session_state.current_portfolio
    })
    st.session_state.preview_ii = False
    st.session_state.edit_import_to_manual = False

# ─── Paso 1: Cargar fichero maestro ───
master_file = st.file_uploader("Sube el Excel de All Funds con TODAS las clases de participación", type=["xlsx"])
if not master_file:
    st.stop()

df = pd.read_excel(master_file, skiprows=2)

required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF","Name"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el fichero importado: {missing}")
    st.stop()

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
filter_cols = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]
global_filters = {}

if mode == "Importar Excel con ISINs y pesos existentes":
    weights_file = st.file_uploader(
        "Sube un Excel con columnas 'ISIN' y 'Peso %'", type=["xlsx"], key="weights"
    )
    if weights_file:
        wdf = pd.read_excel(weights_file)
        req2 = ["ISIN", "Peso %"]
        miss2 = [c for c in req2 if c not in wdf.columns]
        if miss2:
            st.error(f"Faltan columnas en el fichero de cartera: {miss2}")
        else:
            st.success("Cartera existente cargada.")
    
            # Agrupar ISINs duplicados sumando los pesos
            wdf = wdf.groupby("ISIN", as_index=False)["Peso %"].sum()
    
            # Merge con el fichero maestro
            merged = pd.merge(wdf, df, on="ISIN", how="left", validate="one_to_many")
    
            if merged["Family Name"].isnull().any():
                bad = merged[merged["Family Name"].isnull()]["ISIN"].tolist()
                st.error(f"No hay datos para ISIN(s): {bad}")
            else:
                for _, row in merged.iterrows():
                    # Use "Name" if all criteria are found, "Family Name" otherwise (but for import, always Name)
                    edited.append({
                        "Name":   row["Name"],
                        "Type of Share": row["Type of Share"],
                        "Currency":      row["Currency"],
                        "Hedged":        row["Hedged"],
                        "MiFID FH":      row["MiFID FH"],
                        "Min. Initial":  row["Min. Initial"],
                        "Weight %":      float(row["Peso %"])
                    })
                st.markdown("**Cartera precargada desde la importación.**")
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

    # ─── Paso 3: Personaliza la clase por fondo (cascada) ───
    st.subheader("Paso 3: Personaliza la clase por fondo")
    st.write("ℹ️ *Prospectus AF y Traspasable se calculan automáticamente con la clase seleccionada.*")
    
    prev_rows = st.session_state.get("edited_rows", [])
    edited = []
    
    for idx, fam in enumerate(df["Family Name"].dropna().unique()):
        fund_df = df[df["Family Name"] == fam].copy()
        cols = st.columns([1.5,1.1,1.1,1.2,1.2,1.0,1.5])
        row = {}
        context = fund_df.copy()
    
        # Compute the current_name BEFORE the cascade (try to use previous selections if exist)
        temp_context = context.copy()
        sel_keys = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]
        selected = []
        # Try to use global filters first (same logic que antes)
        for k in sel_keys:
            v = global_filters.get(k)
            if v and v in temp_context[k].dropna().unique():
                temp_context = temp_context[temp_context[k] == v]
                selected.append(v)
            else:
                break
        valid_pre = len(selected) == len(sel_keys) and not temp_context.empty
        if valid_pre:
            best_pre = temp_context.loc[temp_context["Ongoing Charge"].idxmin()]
            current_name = best_pre.get("Name", fam)
        else:
            # fallback: if we have an edited row, try to compute name from those values
            prev = prev_rows[idx] if idx < len(prev_rows) else {}
            temp_context2 = context.copy()
            ok2 = True
            for k in sel_keys:
                v = prev.get(k)
                if v and v in temp_context2[k].dropna().unique():
                    temp_context2 = temp_context2[temp_context2[k] == v]
                else:
                    ok2 = False
                    break
            if ok2 and not temp_context2.empty:
                best2 = temp_context2.loc[temp_context2["Ongoing Charge"].idxmin()]
                current_name = best2.get("Name", fam)
            else:
                current_name = fam
    
        # Show the name in the first column so it appears BEFORE the selects
        with cols[0]:
            st.markdown(f"### {current_name}")
    
        def cascade(i, label, key, ctx, prefill=None):
            options = sorted(ctx[key].dropna().unique().tolist())
            # ensure there's at least one option to avoid empty selectboxes
            if not options:
                options = ["—"]
            # try to pick a sensible initial value:
            init = None
            if prefill and prefill in options:
                init = prefill
            elif global_filters.get(key) and global_filters.get(key) in options:
                init = global_filters.get(key)
            else:
                # keep existing behavior for "NO ENCONTRADO"
                init = options[0]
            try:
                index = options.index(init)
            except ValueError:
                index = 0
            sel = cols[i].selectbox(label, options, index=index, key=f"{key}_{idx}")
            new_ctx = ctx if sel in ["—", "NO ENCONTRADO"] else ctx[ctx[key] == sel]
            return sel, new_ctx
    
        # get prefill from previous edited_rows if present
        prev = prev_rows[idx] if idx < len(prev_rows) else {}
        sel_type, context = cascade(0, "Tipo de participación", "Type of Share", context, prev.get("Type of Share"))
        sel_cur,  context = cascade(1, "Divisa",                 "Currency",     context, prev.get("Currency"))
        sel_hed,  context = cascade(2, "Hedged",                 "Hedged",       context, prev.get("Hedged"))
        sel_mif,  context = cascade(3, "MiFID FH",               "MiFID FH",     context, prev.get("MiFID FH"))
        sel_min,  context = cascade(4, "Mín. Inversión",         "Min. Initial", context, prev.get("Min. Initial"))
    
        row["Type of Share"] = sel_type
        row["Currency"] = sel_cur
        row["Hedged"] = sel_hed
        row["MiFID FH"] = sel_mif
        row["Min. Initial"] = sel_min
    
        valid = all(row.get(k) not in (None, "NO ENCONTRADO", "—") for k in sel_keys) and not context.empty
        if valid:
            best = context.loc[context["Ongoing Charge"].idxmin()]
            row["Name"] = best.get("Name", fam)
            row["_show_name"] = True
        else:
            row["Name"] = fam
            row["_show_name"] = False
    
        # preserve weight in session_state so it remains when we re-render
        weight_key = f"weight_{idx}"
        if weight_key not in st.session_state:
            # try to restore from prev_rows
            st.session_state[weight_key] = float(prev.get("Weight %", 0.0))
        row["Weight %"] = cols[5].number_input(
            "Peso %",
            min_value=0.0, max_value=100.0, step=0.1,
            key=weight_key
        )
    
        # Info visual (misma lógica)
        prospectus_info = "—"
        transferable_info = "—"
        if valid and not context.empty:
            best = context.loc[context["Ongoing Charge"].idxmin()]
            prospectus_info  = str(best.get("Prospectus AF", "—"))
            if has_transferable:
                transferable_info = str(best.get("Transferable", "—"))
    
        with cols[6]:
            st.markdown(f"**Prospectus AF:** {prospectus_info}")
            st.markdown(f"**Traspasable:** {transferable_info}")
    
        edited.append(row)
    
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

# Sólo en modo MANUAL mostramos Equal Weight
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
            errors.append((row.get("Name", row.get("Family Name", "")), "Selección inválida"))
            continue

        match = df[
            (df["Family Name"] == row["Name"]) | (df["Name"] == row["Name"])
        ]
        for k in keys:
            match = match[match[k] == row[k]]
        if match.empty:
            errors.append((row.get("Name", row.get("Family Name", "")), "No se encontró clase que coincida"))
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
        # Always keep "Name" as main identifier for the table
        if not result_row.get("Name"):
            result_row["Name"] = best.get("Name", best.get("Family Name", ""))
        results.append(result_row)

    df_res = pd.DataFrame(results)
    if total_w > 0:
        ter = total_weighted / (total_w/100)
        st.session_state.current_portfolio = {"table": df_res, "ter": ter}
    else:
        st.session_state.current_portfolio = {"table": df_res, "ter": None}
    st.session_state.current_errors = errors

# ─── Utilidad: preparar tabla bonita ───
def pretty_table(df_in: pd.DataFrame) -> pd.DataFrame:
    tbl = df_in.copy()
    if "Traspasable" in tbl.columns:
        pass
    elif "Transferable" in tbl.columns:
        tbl.rename(columns={"Transferable": "Traspasable"}, inplace=True)
    # Replace "Family Name" by "Name" everywhere
    desired = [
        "Name",
        "Type of Share", "Currency", "Hedged", "MiFID FH", "Min. Initial",
        "ISIN", "Prospectus AF", "Traspasable",
        "Ongoing Charge", "Weight %"
    ]
    existing = [c for c in desired if c in tbl.columns]
    rest = [c for c in tbl.columns if c not in existing and not c.startswith("_")]
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

    # Copia de la tabla bonita
    df_show = pretty_table(cp["table"]).copy()

    # Formatear columnas numéricas en estilo europeo
    for col in ["Ongoing Charge", "Weight %"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
                if pd.notnull(x) else x
            )

    # Mostrar tabla con separador decimal europeo
    st.dataframe(df_show, use_container_width=True)

    if cp["ter"] is not None:
        st.metric("📊 TER medio ponderado", f"{cp['ter']:.2%}".replace(".", ","))

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

        # Sólo mostramos el botón directo de comparar si el modo es MANUAL
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

# ─── Editor tras importación: Paso 3 (prefill, EN CASCADA) ───
if st.session_state.edit_import_to_manual and st.session_state.edited_rows:
    st.markdown("---")
    st.subheader("Paso 3 (edición): Personaliza la clase por fondo (a partir de la cartera importada)")
    st.write("ℹ️ Los selectores y pesos se han precargado desde el Excel importado. Puedes cambiar la clase y guardar la Cartera II.")

    families = [r.get("Name", r.get("Family Name")) for r in st.session_state.edited_rows]
    edited_from_import = []

    for idx, fam in enumerate(families):
        base_row = st.session_state.edited_rows[idx]
        # Try both Family Name and Name for backward compatibility
        fund_df = df[(df["Family Name"] == fam) | (df["Name"] == fam)].copy()
        cols = st.columns([1.5,1.1,1.1,1.2,1.2,1.0,1.5])
        row = {}
        context = fund_df.copy()

        # Compute the current_name BEFORE the cascade (use base_row prefill)
        temp_context = fund_df.copy()
        sel_keys = ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]
        valid_pre = True
        for k in sel_keys:
            v = base_row.get(k)
            if v and v in temp_context[k].dropna().unique():
                temp_context = temp_context[temp_context[k] == v]
            else:
                valid_pre = False
                break
        if valid_pre and not temp_context.empty:
            best_pre = temp_context.loc[temp_context["Ongoing Charge"].idxmin()]
            current_name = best_pre.get("Name", fam)
        else:
            current_name = fam

        # Show the name in first column
        with cols[0]:
            st.markdown(f"### {current_name}")

        def cascade_prefill(i, label, key, ctx, prefill_value):
            options = sorted(ctx[key].dropna().unique().tolist())
            if not options:
                options = ["—"]
            init = prefill_value if (prefill_value in options) else options[0]
            try:
                index = options.index(init)
            except ValueError:
                index = 0
            sel = cols[i].selectbox(label, options, index=index, key=f"edit_{key}_{idx}")
            new_ctx = ctx if sel in ["—", "NO ENCONTRADO"] else ctx[ctx[key] == sel]
            return sel, new_ctx

        sel_type, context = cascade_prefill(0, "Tipo de participación", "Type of Share", context, base_row.get("Type of Share"))
        sel_cur,  context = cascade_prefill(1, "Divisa",                 "Currency",     context, base_row.get("Currency"))
        sel_hed,  context = cascade_prefill(2, "Hedged",                 "Hedged",       context, base_row.get("Hedged"))
        sel_mif,  context = cascade_prefill(3, "MiFID FH",               "MiFID FH",     context, base_row.get("MiFID FH"))
        sel_min,  context = cascade_prefill(4, "Mín. Inversión",         "Min. Initial", context, base_row.get("Min. Initial"))

        row["Type of Share"] = sel_type
        row["Currency"] = sel_cur
        row["Hedged"] = sel_hed
        row["MiFID FH"] = sel_mif
        row["Min. Initial"] = sel_min

        valid = all(row.get(k) not in (None, "NO ENCONTRADO", "—") for k in sel_keys) and not context.empty
        if valid and not context.empty:
            best = context.loc[context["Ongoing Charge"].idxmin()]
            row["Name"] = best.get("Name", fam)
            row["_show_name"] = True
        else:
            row["Name"] = fam
            row["_show_name"] = False

        weight_key = f"edit_weight_{idx}"
        if weight_key not in st.session_state:
            st.session_state[weight_key] = float(base_row.get("Weight %", 0.0))
        row["Weight %"] = cols[5].number_input("Peso %", min_value=0.0, max_value=100.0, step=0.1, key=weight_key)

        prospectus_info = "—"
        transferable_info = "—"
        if valid and not context.empty:
            best = context.loc[context["Ongoing Charge"].idxmin()]
            prospectus_info  = str(best.get("Prospectus AF", "—"))
            if has_transferable:
                transferable_info = str(best.get("Transferable", "—"))

        with cols[6]:
            st.markdown(f"**Prospectus AF:** {prospectus_info}")
            st.markdown(f"**Traspasable:** {transferable_info}")

        edited_from_import.append(row)

    total_weight2 = sum(r["Weight %"] for r in edited_from_import)
    st.subheader("Peso total (edición)")
    st.write(f"{total_weight2:.2f}%")
    if abs(total_weight2 - 100.0) > 1e-6:
        st.warning("El peso total debe sumar 100% antes de calcular el TER.")

    st.divider()

    # Botón único: calcular con la edición y guardar como Cartera II
    if st.button("Comparar con Cartera I (guardar edición como Cartera II)", key="save_edit_as_ii_btn"):
        # Usamos la edición RECIENTE del usuario
        st.session_state.edited_rows = edited_from_import.copy()

        results, errors = [], []
        twc, tw = 0.0, 0.0

        for row in edited_from_import:
            if any(row.get(k) == "NO ENCONTRADO" for k in ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]):
                errors.append((row.get("Name", row.get("Family Name", "")), "Selección inválida"))
                continue

            match = df[(df["Family Name"] == row["Name"]) | (df["Name"] == row["Name"])]
            for k in ["Type of Share","Currency","Hedged","MiFID FH","Min. Initial"]:
                match = match[match[k] == row[k]]
            if match.empty:
                errors.append((row.get("Name", row.get("Family Name", "")), "No se encontró clase que coincida"))
                continue

            best = match.loc[match["Ongoing Charge"].idxmin()]
            charge = best["Ongoing Charge"]
            w = row["Weight %"]

            twc += charge * (w/100)
            tw  += w

            out = {
                **row,
                "ISIN": best["ISIN"],
                "Prospectus AF": best.get("Prospectus AF","—"),
                "Ongoing Charge": charge
            }
            if has_transferable:
                out["Transferable"] = best.get("Transferable","—")
            if not out.get("Name"):
                out["Name"] = best.get("Name", best.get("Family Name", ""))

            results.append(out)

        df_res = pd.DataFrame(results)
        ter = (twc / (tw/100)) if tw > 0 else None

        st.session_state.current_portfolio = {"table": df_res, "ter": ter}
        st.session_state.current_errors = errors

        if len(st.session_state.saved_portfolios) >= 1:
            cartera_ii = {"label": "Cartera II", **st.session_state.current_portfolio}
            ii_idx = next((i for i,p in enumerate(st.session_state.saved_portfolios) if p.get("label")=="Cartera II"), None)
            if ii_idx is None:
                st.session_state.saved_portfolios.append(cartera_ii)
            else:
                st.session_state.saved_portfolios[ii_idx] = cartera_ii

        st.session_state.edit_import_to_manual = False
        st.toast("Cartera II guardada. Abriendo comparación…", icon="✅")
        st.rerun()
