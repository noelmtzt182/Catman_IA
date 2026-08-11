
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
st.set_page_config(
    page_title="Category Management AI Analyzer",
    page_icon="📊",
    layout="wide",
)
 
# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
 
REQUIRED_FIELDS = {
    "sku": "SKU / código de producto",
    "producto": "Nombre del producto",
    "marca": "Marca / fabricante",
    "ventas": "Ventas ($)",
}
OPTIONAL_FIELDS = {
    "unidades": "Unidades vendidas",
    "subcategoria": "Subcategoría",
    "fecha": "Fecha",
    "canal": "Canal / cadena / tienda",
}
 
 
def generar_datos_ejemplo(n_skus: int = 60, meses: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    marcas = ["Marca A", "Marca B", "Marca C", "Marca D", "Marca E", "Marca Propia"]
    subcats = ["Premium", "Estándar", "Económico"]
    fechas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=meses, freq="MS")
 
    filas = []
    for i in range(n_skus):
        marca = rng.choice(marcas, p=[0.28, 0.22, 0.18, 0.12, 0.10, 0.10])
        subcat = rng.choice(subcats)
        base_venta = rng.lognormal(mean=8.5, sigma=1.1)
        tendencia = rng.normal(0, 0.03)
        for j, fecha in enumerate(fechas):
            ruido = rng.normal(1, 0.15)
            venta = max(base_venta * (1 + tendencia) ** j * ruido, 0)
            unidades = max(int(venta / rng.uniform(15, 45)), 0)
            filas.append(
                {
                    "sku": f"SKU-{i+1:04d}",
                    "producto": f"Producto {i+1}",
                    "marca": marca,
                    "subcategoria": subcat,
                    "ventas": round(venta, 2),
                    "unidades": unidades,
                    "fecha": fecha,
                    "canal": rng.choice(["Supermercado", "Autoservicio", "Mayorista"]),
                }
            )
    return pd.DataFrame(filas)
 
 
def leer_archivo(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)
 
 
def clasificar_abc(df: pd.DataFrame, columna_valor: str) -> pd.DataFrame:
    d = df.sort_values(columna_valor, ascending=False).copy()
    total = d[columna_valor].sum()
    d["pct"] = d[columna_valor] / total if total else 0
    d["pct_acum"] = d["pct"].cumsum()
 
    def clase(p):
        if p <= 0.80:
            return "A"
        elif p <= 0.95:
            return "B"
        return "C"
 
    d["clase_abc"] = d["pct_acum"].apply(clase)
    return d
 
 
def kpi_card(col, label, value, delta=None, help_text=None):
    col.metric(label, value, delta=delta, help=help_text)
 
 
# ----------------------------------------------------------------------
# Sidebar: carga de datos
# ----------------------------------------------------------------------
 
st.sidebar.title("📊 Category Management AI")
st.sidebar.caption("Analizador de categorías para trade marketing")
 
modo = st.sidebar.radio(
    "Fuente de datos",
    ["Usar datos de ejemplo", "Subir mi archivo (CSV / Excel)"],
)
 
df_raw = None
if modo == "Usar datos de ejemplo":
    df_raw = generar_datos_ejemplo()
    st.sidebar.success("Usando dataset de ejemplo (12 meses, 60 SKUs).")
else:
    uploaded = st.sidebar.file_uploader("Cargar archivo", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df_raw = leer_archivo(uploaded)
        except Exception as e:
            st.sidebar.error(f"No se pudo leer el archivo: {e}")
 
if df_raw is None:
    st.title("📊 Category Management AI Analyzer")
    st.info(
        "Subí un archivo CSV o Excel con tus datos de categoría (ventas por SKU, "
        "marca, fecha, etc.) desde el panel izquierdo, o probá con el dataset de "
        "ejemplo para ver cómo funciona la herramienta."
    )
    st.stop()
 
# ----------------------------------------------------------------------
# Mapeo de columnas
# ----------------------------------------------------------------------
 
st.sidebar.markdown("---")
st.sidebar.subheader("Mapeo de columnas")
 
columnas = ["(ninguna)"] + list(df_raw.columns)
 
 
def selector_columna(campo, etiqueta, opcional=False):
    default_idx = 0
    for i, c in enumerate(columnas):
        if c.lower() == campo.lower():
            default_idx = i
            break
    return st.sidebar.selectbox(
        f"{etiqueta}{' (opcional)' if opcional else ''}", columnas, index=default_idx, key=f"map_{campo}"
    )
 
 
mapeo = {}
for campo, etiqueta in REQUIRED_FIELDS.items():
    mapeo[campo] = selector_columna(campo, etiqueta)
for campo, etiqueta in OPTIONAL_FIELDS.items():
    mapeo[campo] = selector_columna(campo, etiqueta, opcional=True)
 
faltantes = [c for c in REQUIRED_FIELDS if mapeo[c] == "(ninguna)"]
if faltantes:
    st.warning(
        "Faltan mapear columnas obligatorias: "
        + ", ".join(REQUIRED_FIELDS[c] for c in faltantes)
    )
    st.stop()
 
# Construir dataframe normalizado
df = pd.DataFrame()
for campo, col in mapeo.items():
    if col != "(ninguna)":
        df[campo] = df_raw[col]
 
if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
 
df["ventas"] = pd.to_numeric(df["ventas"], errors="coerce").fillna(0)
if "unidades" in df.columns:
    df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce").fillna(0)
 
# ----------------------------------------------------------------------
# Filtros
# ----------------------------------------------------------------------
 
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
 
if "canal" in df.columns:
    canales = sorted(df["canal"].dropna().unique().tolist())
    sel_canales = st.sidebar.multiselect("Canal", canales, default=canales)
    df = df[df["canal"].isin(sel_canales)]
 
if "subcategoria" in df.columns:
    subcats = sorted(df["subcategoria"].dropna().unique().tolist())
    sel_subcats = st.sidebar.multiselect("Subcategoría", subcats, default=subcats)
    df = df[df["subcategoria"].isin(sel_subcats)]
 
if "fecha" in df.columns and df["fecha"].notna().any():
    fmin, fmax = df["fecha"].min(), df["fecha"].max()
    rango = st.sidebar.date_input("Rango de fechas", value=(fmin, fmax))
    if isinstance(rango, tuple) and len(rango) == 2:
        df = df[(df["fecha"] >= pd.Timestamp(rango[0])) & (df["fecha"] <= pd.Timestamp(rango[1]))]
 
if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()
 
# ----------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------
 
st.title("📊 Category Management AI Analyzer")
 
total_ventas = df["ventas"].sum()
total_unidades = df["unidades"].sum() if "unidades" in df.columns else None
n_skus = df["sku"].nunique()
n_marcas = df["marca"].nunique()
 
crecimiento = None
if "fecha" in df.columns and df["fecha"].notna().any():
    serie_mensual = df.groupby(df["fecha"].dt.to_period("M"))["ventas"].sum().sort_index()
    if len(serie_mensual) >= 2:
        crecimiento = (serie_mensual.iloc[-1] / serie_mensual.iloc[-2] - 1) * 100 if serie_mensual.iloc[-2] else None
 
c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Ventas totales", f"${total_ventas:,.0f}")
if total_unidades is not None:
    kpi_card(c2, "Unidades totales", f"{total_unidades:,.0f}")
kpi_card(c3, "SKUs activos", f"{n_skus:,}")
kpi_card(c4, "Marcas", f"{n_marcas:,}", delta=f"{crecimiento:+.1f}% vs mes anterior" if crecimiento is not None else None)
 
tabs = st.tabs(["📈 Tendencias", "🅰️ ABC / Pareto", "🥧 Market Share", "🧩 Surtido", "💡 Recomendaciones"])
 
# ----------------------------------------------------------------------
# TAB: Tendencias
# ----------------------------------------------------------------------
with tabs[0]:
    if "fecha" in df.columns and df["fecha"].notna().any():
        serie = df.groupby(df["fecha"].dt.to_period("M").dt.to_timestamp())["ventas"].sum().reset_index()
        fig = px.line(serie, x="fecha", y="ventas", markers=True, title="Evolución de ventas por mes")
        st.plotly_chart(fig, use_container_width=True)
 
        top_var = (
            df.groupby(["sku", "producto"])
            .apply(lambda g: g.groupby(g["fecha"].dt.to_period("M"))["ventas"].sum())
            .unstack(fill_value=0)
        )
        if top_var.shape[1] >= 2:
            var_pct = (top_var.iloc[:, -1] - top_var.iloc[:, -2]) / top_var.iloc[:, -2].replace(0, np.nan) * 100
            var_pct = var_pct.dropna().sort_values(ascending=False)
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**🚀 SKUs en mayor crecimiento (últ. mes)**")
                st.dataframe(var_pct.head(10).rename("Δ% vs mes anterior").reset_index(), use_container_width=True)
            with colB:
                st.markdown("**📉 SKUs en mayor caída (últ. mes)**")
                st.dataframe(var_pct.tail(10).sort_values().rename("Δ% vs mes anterior").reset_index(), use_container_width=True)
    else:
        st.info("Mapeá una columna de fecha para ver tendencias temporales.")
 
# ----------------------------------------------------------------------
# TAB: ABC / Pareto
# ----------------------------------------------------------------------
with tabs[1]:
    st.markdown("Clasificación **ABC** de SKUs según su aporte acumulado a las ventas (regla 80/15/5).")
    ventas_sku = df.groupby(["sku", "producto"], as_index=False)["ventas"].sum()
    abc = clasificar_abc(ventas_sku, "ventas")
 
    resumen_abc = abc.groupby("clase_abc").agg(
        skus=("sku", "count"), ventas=("ventas", "sum")
    ).reset_index()
    resumen_abc["% ventas"] = (resumen_abc["ventas"] / resumen_abc["ventas"].sum() * 100).round(1)
    resumen_abc["% skus"] = (resumen_abc["skus"] / resumen_abc["skus"].sum() * 100).round(1)
 
    colA, colB = st.columns([1, 2])
    with colA:
        st.dataframe(resumen_abc, use_container_width=True, hide_index=True)
    with colB:
        fig = px.bar(
            resumen_abc, x="clase_abc", y="% ventas", color="clase_abc",
            title="% de ventas por clase ABC",
            color_discrete_map={"A": "#2E7D32", "B": "#F9A825", "C": "#C62828"},
        )
        st.plotly_chart(fig, use_container_width=True)
 
    fig2 = go.Figure()
    fig2.add_bar(x=abc["producto"], y=abc["ventas"], name="Ventas")
    fig2.add_scatter(x=abc["producto"], y=abc["pct_acum"] * 100, name="% acumulado", yaxis="y2")
    fig2.update_layout(
        title="Curva de Pareto por SKU",
        yaxis=dict(title="Ventas"),
        yaxis2=dict(title="% acumulado", overlaying="y", side="right", range=[0, 100]),
        xaxis=dict(showticklabels=False),
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)
 
    with st.expander("Ver detalle completo por SKU"):
        st.dataframe(
            abc[["sku", "producto", "ventas", "pct", "pct_acum", "clase_abc"]]
            .rename(columns={"pct": "% ventas", "pct_acum": "% acumulado"}),
            use_container_width=True,
        )
 
# ----------------------------------------------------------------------
# TAB: Market Share
# ----------------------------------------------------------------------
with tabs[2]:
    share = df.groupby("marca", as_index=False)["ventas"].sum().sort_values("ventas", ascending=False)
    share["share_%"] = (share["ventas"] / share["ventas"].sum() * 100).round(1)
 
    colA, colB = st.columns(2)
    with colA:
        fig = px.pie(share, names="marca", values="ventas", title="Market share por marca", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.dataframe(share, use_container_width=True, hide_index=True)
 
    if "fecha" in df.columns and df["fecha"].notna().any():
        share_tiempo = (
            df.groupby([df["fecha"].dt.to_period("M").dt.to_timestamp(), "marca"])["ventas"]
            .sum()
            .reset_index()
        )
        share_tiempo["share_%"] = share_tiempo.groupby("fecha")["ventas"].transform(lambda x: x / x.sum() * 100)
        fig3 = px.area(share_tiempo, x="fecha", y="share_%", color="marca", title="Evolución del share por marca")
        st.plotly_chart(fig3, use_container_width=True)
 
# ----------------------------------------------------------------------
# TAB: Surtido
# ----------------------------------------------------------------------
with tabs[3]:
    st.markdown("Productividad del surtido: compara el **% de SKUs** que aporta cada marca vs su **% de ventas**.")
    prod = df.groupby("marca").agg(skus=("sku", "nunique"), ventas=("ventas", "sum")).reset_index()
    prod["% skus"] = (prod["skus"] / prod["skus"].sum() * 100).round(1)
    prod["% ventas"] = (prod["ventas"] / prod["ventas"].sum() * 100).round(1)
    prod["indice_productividad"] = (prod["% ventas"] / prod["% skus"]).round(2)
 
    fig = px.scatter(
        prod, x="% skus", y="% ventas", size="ventas", color="marca", text="marca",
        title="Productividad de surtido por marca (arriba de la diagonal = eficiente)",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=max(prod["% skus"].max(), prod["% ventas"].max()),
                  y1=max(prod["% skus"].max(), prod["% ventas"].max()), line=dict(dash="dash", color="gray"))
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(prod.sort_values("indice_productividad", ascending=False), use_container_width=True, hide_index=True)
 
    st.markdown("**Cola larga (candidatos a revisión de surtido):** SKUs clase C con menor aporte individual.")
    ventas_sku_full = df.groupby(["sku", "producto", "marca"], as_index=False)["ventas"].sum()
    abc_full = clasificar_abc(ventas_sku_full, "ventas")
    cola_larga = abc_full[abc_full["clase_abc"] == "C"].sort_values("ventas")
    st.dataframe(cola_larga[["sku", "producto", "marca", "ventas", "pct"]].rename(columns={"pct": "% ventas"}),
                 use_container_width=True, hide_index=True)
 
# ----------------------------------------------------------------------
# TAB: Recomendaciones automáticas
# ----------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 💡 Insights automáticos")
    insights = []
 
    n_a = (abc_full["clase_abc"] == "A").sum() if "abc_full" in dir() else (abc["clase_abc"] == "A").sum()
    total_sku_count = ventas_sku["sku"].nunique()
    insights.append(
        f"**{n_a} SKUs ({n_a/total_sku_count*100:.0f}% del surtido)** generan el **80% de las ventas** "
        "de la categoría (clase A). Priorizá su disponibilidad y visibilidad en punto de venta."
    )
 
    n_c = (abc["clase_abc"] == "C").sum()
    insights.append(
        f"**{n_c} SKUs ({n_c/total_sku_count*100:.0f}% del surtido)** son clase C y aportan menos del 5% "
        "de las ventas en conjunto. Son candidatos a evaluar racionalización de surtido."
    )
 
    top_marca = share.iloc[0]
    insights.append(
        f"**{top_marca['marca']}** lidera la categoría con **{top_marca['share_%']}%** de share. "
        f"La marca #2 es **{share.iloc[1]['marca']}** con **{share.iloc[1]['share_%']}%**."
    )
 
    baja_productividad = prod[prod["indice_productividad"] < 0.7]
    if not baja_productividad.empty:
        marcas_bp = ", ".join(baja_productividad["marca"].tolist())
        insights.append(
            f"Las marcas **{marcas_bp}** tienen un índice de productividad de surtido bajo "
            "(muchos SKUs para poco aporte de ventas) — revisar racionalización o negociar mejor exhibición."
        )
 
    if "fecha" in df.columns and df["fecha"].notna().any() and crecimiento is not None:
        direccion = "creció" if crecimiento >= 0 else "cayó"
        insights.append(f"Las ventas de la categoría **{direccion} {abs(crecimiento):.1f}%** respecto al mes anterior.")
 
    for i, txt in enumerate(insights, 1):
        st.markdown(f"{i}. {txt}")
 
    st.markdown("---")
    st.markdown("### 📥 Exportar reporte")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        abc.to_excel(writer, sheet_name="ABC_Pareto", index=False)
        share.to_excel(writer, sheet_name="Market_Share", index=False)
        prod.to_excel(writer, sheet_name="Productividad_Surtido", index=False)
        cola_larga.to_excel(writer, sheet_name="Cola_Larga", index=False)
    st.download_button(
        "Descargar reporte Excel",
        data=buffer.getvalue(),
        file_name="reporte_category_management.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
 
