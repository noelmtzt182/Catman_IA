import io
import os
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
try:
    import anthropic
except ImportError:
    anthropic = None
 
st.set_page_config(
    page_title="Category Management AI Analyzer",
    page_icon="📊",
    layout="wide",
)
 
MODELOS_DISPONIBLES = [
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
]
 
# ----------------------------------------------------------------------
# Utilidades: datos
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
 
 
def md_tabla(df_in: pd.DataFrame, max_filas: int = None) -> str:
    """Convierte un dataframe a una tabla markdown compacta para pasarle a la IA."""
    d = df_in.copy()
    if max_filas:
        d = d.head(max_filas)
    # Evitar notación científica en columnas numéricas grandes (ventas, etc.)
    for col in d.select_dtypes(include=[np.number]).columns:
        if (d[col].abs() >= 1).any() and (d[col] % 1 == 0).all():
            d[col] = d[col].map(lambda v: f"{v:,.0f}")
        else:
            d[col] = d[col].map(lambda v: f"{v:,.2f}")
    try:
        return d.to_markdown(index=False, disable_numparse=True)
    except ImportError:
        return d.to_csv(index=False)
 
 
# ----------------------------------------------------------------------
# Utilidades: IA (Claude API)
# ----------------------------------------------------------------------
 
 
def get_client():
    api_key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or anthropic is None:
        return None
    return anthropic.Anthropic(api_key=api_key)
 
 
def generar_insights_ia(contexto: str, client, modelo: str) -> str:
    system = (
        "Sos un consultor senior de Category Management y Trade Marketing. "
        "A continuación te paso datos agregados YA CALCULADOS de una categoría "
        "(KPIs, clasificación ABC, market share, productividad de surtido, "
        "tendencias). Con esa información redactá un análisis ejecutivo en "
        "español, en formato markdown, con estas secciones: "
        "1) Diagnóstico general (2-3 líneas), "
        "2) Hallazgos clave por marca/subcategoría, "
        "3) Riesgos u oportunidades, "
        "4) Recomendaciones accionables priorizadas (máximo 5, concretas). "
        "Citá los números que te paso. No inventes datos que no estén en el "
        "contexto ni asumas causas que no puedas justificar con los datos."
    )
    resp = client.messages.create(
        model=modelo,
        max_tokens=1600,
        temperature=0.4,
        system=system,
        messages=[{"role": "user", "content": f"Datos de la categoría:\n\n{contexto}\n\nGenerá el análisis."}],
    )
    return resp.content[0].text
 
 
def responder_chat(pregunta_historial, contexto: str, client, modelo: str) -> str:
    system = (
        "Sos un analista de Category Management. Respondé preguntas sobre la "
        "categoría usando EXCLUSIVAMENTE los datos agregados que te paso a "
        "continuación. Si la pregunta no se puede responder con estos datos, "
        "decilo explícitamente en vez de inventar cifras. Respondé en español, "
        "de forma concisa y directa, citando números concretos cuando aplique.\n\n"
        f"DATOS DE LA CATEGORÍA:\n{contexto}"
    )
    resp = client.messages.create(
        model=modelo,
        max_tokens=900,
        temperature=0.3,
        system=system,
        messages=pregunta_historial,
    )
    return resp.content[0].text
 
 
# ----------------------------------------------------------------------
# Sidebar: configuración de IA + carga de datos
# ----------------------------------------------------------------------
 
st.sidebar.title("📊 Category Management AI")
st.sidebar.caption("Analizador de categorías para trade marketing")
 
with st.sidebar.expander("🤖 Configuración de IA", expanded=False):
    if anthropic is None:
        st.error("Falta instalar el paquete `anthropic` (pip install anthropic).")
    st.text_input(
        "API key de Anthropic",
        type="password",
        key="api_key",
        help="Se usa solo en esta sesión, no se guarda en ningún lado. "
        "También podés setearla como variable de entorno ANTHROPIC_API_KEY.",
    )
    st.selectbox("Modelo", MODELOS_DISPONIBLES, index=0, key="modelo_ia")
    st.caption(
        "Necesaria para los tabs '🤖 Insights IA' y '💬 Chat'. El resto de la "
        "app funciona sin API key."
    )
 
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
 
tiene_fecha = "fecha" in df.columns and df["fecha"].notna().any()
 
crecimiento = None
serie_mensual_df = None
if tiene_fecha:
    serie_mensual = df.groupby(df["fecha"].dt.to_period("M"))["ventas"].sum().sort_index()
    if len(serie_mensual) >= 2:
        crecimiento = (serie_mensual.iloc[-1] / serie_mensual.iloc[-2] - 1) * 100 if serie_mensual.iloc[-2] else None
    serie_mensual_df = serie_mensual.reset_index()
    serie_mensual_df["fecha"] = serie_mensual_df["fecha"].astype(str)
    serie_mensual_df.columns = ["mes", "ventas"]
 
c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Ventas totales", f"${total_ventas:,.0f}")
if total_unidades is not None:
    kpi_card(c2, "Unidades totales", f"{total_unidades:,.0f}")
kpi_card(c3, "SKUs activos", f"{n_skus:,}")
kpi_card(c4, "Marcas", f"{n_marcas:,}", delta=f"{crecimiento:+.1f}% vs mes anterior" if crecimiento is not None else None)
 
# ----------------------------------------------------------------------
# Cálculos centrales (se reutilizan en varios tabs, incluida la IA)
# ----------------------------------------------------------------------
 
ventas_sku = df.groupby(["sku", "producto", "marca"], as_index=False)["ventas"].sum()
abc = clasificar_abc(ventas_sku, "ventas")
 
resumen_abc = abc.groupby("clase_abc").agg(
    skus=("sku", "count"), ventas=("ventas", "sum")
).reset_index()
resumen_abc["% ventas"] = (resumen_abc["ventas"] / resumen_abc["ventas"].sum() * 100).round(1)
resumen_abc["% skus"] = (resumen_abc["skus"] / resumen_abc["skus"].sum() * 100).round(1)
 
share = df.groupby("marca", as_index=False)["ventas"].sum().sort_values("ventas", ascending=False)
share["share_%"] = (share["ventas"] / share["ventas"].sum() * 100).round(1)
 
prod = df.groupby("marca").agg(skus=("sku", "nunique"), ventas=("ventas", "sum")).reset_index()
prod["% skus"] = (prod["skus"] / prod["skus"].sum() * 100).round(1)
prod["% ventas"] = (prod["ventas"] / prod["ventas"].sum() * 100).round(1)
prod["indice_productividad"] = (prod["% ventas"] / prod["% skus"]).round(2)
 
cola_larga = abc[abc["clase_abc"] == "C"].sort_values("ventas")
 
share_tiempo = None
var_pct = None
if tiene_fecha:
    share_tiempo = (
        df.groupby([df["fecha"].dt.to_period("M").dt.to_timestamp(), "marca"])["ventas"]
        .sum()
        .reset_index()
    )
    share_tiempo["share_%"] = share_tiempo.groupby("fecha")["ventas"].transform(lambda x: x / x.sum() * 100)
 
    top_var = df.pivot_table(
        index=["sku", "producto"],
        columns=df["fecha"].dt.to_period("M"),
        values="ventas",
        aggfunc="sum",
        fill_value=0,
    )
    if top_var.shape[1] >= 2:
        var_pct = (top_var.iloc[:, -1] - top_var.iloc[:, -2]) / top_var.iloc[:, -2].replace(0, np.nan) * 100
        var_pct = var_pct.dropna().sort_values(ascending=False)
 
pivot_canal = None
if "canal" in df.columns:
    pivot_canal = df.groupby("canal", as_index=False)["ventas"].sum().sort_values("ventas", ascending=False)
 
pivot_subcat = None
if "subcategoria" in df.columns:
    pivot_subcat = df.groupby("subcategoria", as_index=False)["ventas"].sum().sort_values("ventas", ascending=False)
 
 
def construir_contexto_datos() -> str:
    partes = [
        "## KPIs generales",
        f"- Ventas totales: ${total_ventas:,.0f}",
        f"- SKUs activos: {n_skus}",
        f"- Marcas: {n_marcas}",
    ]
    if total_unidades is not None:
        partes.append(f"- Unidades totales: {total_unidades:,.0f}")
    if crecimiento is not None:
        partes.append(f"- Crecimiento último mes vs anterior: {crecimiento:+.1f}%")
 
    partes.append("\n## Clasificación ABC (resumen)")
    partes.append(md_tabla(resumen_abc))
 
    partes.append("\n## Market share por marca")
    partes.append(md_tabla(share[["marca", "ventas", "share_%"]]))
 
    partes.append("\n## Productividad de surtido por marca (skus, % skus, % ventas, índice)")
    partes.append(md_tabla(prod))
 
    partes.append("\n## SKUs cola larga (clase C, ordenados de menor a mayor venta) — top 20")
    partes.append(md_tabla(cola_larga[["sku", "producto", "marca", "ventas", "pct"]], max_filas=20))
 
    if serie_mensual_df is not None:
        partes.append("\n## Ventas totales por mes")
        partes.append(md_tabla(serie_mensual_df))
 
    if var_pct is not None:
        partes.append("\n## Variación % de ventas último mes vs anterior, por SKU (top 15 subas y bajas)")
        vp = var_pct.reset_index()
        vp.columns = ["sku", "producto", "variacion_%"]
        partes.append("Mayores subas:\n" + md_tabla(vp.head(15)))
        partes.append("Mayores bajas:\n" + md_tabla(vp.tail(15).sort_values("variacion_%")))
 
    if pivot_canal is not None:
        partes.append("\n## Ventas por canal")
        partes.append(md_tabla(pivot_canal))
 
    if pivot_subcat is not None:
        partes.append("\n## Ventas por subcategoría")
        partes.append(md_tabla(pivot_subcat))
 
    return "\n".join(partes)
 
 
contexto_datos = construir_contexto_datos()
 
tabs = st.tabs(
    [
        "📈 Tendencias",
        "🅰️ ABC / Pareto",
        "🥧 Market Share",
        "🧩 Surtido",
        "💡 Recomendaciones",
        "🤖 Insights IA",
        "💬 Chat con tus datos",
    ]
)
 
# ----------------------------------------------------------------------
# TAB: Tendencias
# ----------------------------------------------------------------------
with tabs[0]:
    if tiene_fecha:
        serie = df.groupby(df["fecha"].dt.to_period("M").dt.to_timestamp())["ventas"].sum().reset_index()
        fig = px.line(serie, x="fecha", y="ventas", markers=True, title="Evolución de ventas por mes")
        st.plotly_chart(fig, use_container_width=True)
 
        if var_pct is not None:
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
            abc[["sku", "producto", "marca", "ventas", "pct", "pct_acum", "clase_abc"]]
            .rename(columns={"pct": "% ventas", "pct_acum": "% acumulado"}),
            use_container_width=True,
        )
 
# ----------------------------------------------------------------------
# TAB: Market Share
# ----------------------------------------------------------------------
with tabs[2]:
    colA, colB = st.columns(2)
    with colA:
        fig = px.pie(share, names="marca", values="ventas", title="Market share por marca", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.dataframe(share, use_container_width=True, hide_index=True)
 
    if share_tiempo is not None:
        fig3 = px.area(share_tiempo, x="fecha", y="share_%", color="marca", title="Evolución del share por marca")
        st.plotly_chart(fig3, use_container_width=True)
 
# ----------------------------------------------------------------------
# TAB: Surtido
# ----------------------------------------------------------------------
with tabs[3]:
    st.markdown("Productividad del surtido: compara el **% de SKUs** que aporta cada marca vs su **% de ventas**.")
 
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
    st.dataframe(cola_larga[["sku", "producto", "marca", "ventas", "pct"]].rename(columns={"pct": "% ventas"}),
                 use_container_width=True, hide_index=True)
 
# ----------------------------------------------------------------------
# TAB: Recomendaciones automáticas (basadas en reglas)
# ----------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 💡 Insights automáticos (basados en reglas fijas)")
    st.caption(
        "Estos insights se generan con reglas de negocio simples. Para un análisis "
        "más matizado, redactado por un modelo de lenguaje, mirá el tab '🤖 Insights IA'."
    )
    insights = []
 
    n_a = (abc["clase_abc"] == "A").sum()
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
 
    if crecimiento is not None:
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
 
# ----------------------------------------------------------------------
# TAB: Insights IA (LLM)
# ----------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 🤖 Análisis ejecutivo generado con IA")
    st.caption(
        "Toma las tablas ya calculadas (ABC, share, productividad, tendencias) y le pide "
        "a Claude que redacte un diagnóstico y recomendaciones en lenguaje natural."
    )
 
    client = get_client()
    if client is None:
        st.info(
            "Configurá tu API key de Anthropic en '🤖 Configuración de IA' (panel izquierdo) "
            "para habilitar esta función."
        )
    else:
        if st.button("Generar análisis con IA", type="primary"):
            with st.spinner("Analizando la categoría..."):
                try:
                    resultado = generar_insights_ia(contexto_datos, client, st.session_state["modelo_ia"])
                    st.session_state["insights_ia"] = resultado
                except Exception as e:
                    st.error(f"No se pudo generar el análisis: {e}")
 
        if st.session_state.get("insights_ia"):
            st.markdown("---")
            st.markdown(st.session_state["insights_ia"])
 
        with st.expander("Ver contexto de datos enviado al modelo"):
            st.text(contexto_datos)
 
# ----------------------------------------------------------------------
# TAB: Chat con tus datos
# ----------------------------------------------------------------------
with tabs[6]:
    st.markdown("### 💬 Preguntale a tus datos")
    st.caption(
        "El modelo responde usando solo las tablas agregadas de esta categoría "
        "(no inventa datos que no estén en el contexto)."
    )
 
    client = get_client()
    if client is None:
        st.info(
            "Configurá tu API key de Anthropic en '🤖 Configuración de IA' (panel izquierdo) "
            "para habilitar el chat."
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
 
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
 
        pregunta = st.chat_input("Ej: ¿qué marca perdió más share este trimestre?")
        if pregunta:
            st.session_state.chat_history.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)
 
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        historial_api = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history
                        ]
                        respuesta = responder_chat(
                            historial_api, contexto_datos, client, st.session_state["modelo_ia"]
                        )
                    except Exception as e:
                        respuesta = f"Ocurrió un error al consultar el modelo: {e}"
                    st.markdown(respuesta)
            st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
 
        if st.session_state.chat_history:
            if st.button("Borrar conversación"):
                st.session_state.chat_history = []
                st.rerun()
