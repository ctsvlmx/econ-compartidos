# -*- coding: utf-8 -*-
"""
inegi_distribucion_historica.py

Descarga indicadores quincenales del Banco de Información Económica (BIE) del INEGI
y genera:
  - distribucion_historica_inegi.pdf      (una gráfica de distribución por indicador)
  - datos_distribucion_historica_inegi.csv (observaciones procesadas)
  - resumen_distribucion_historica.csv     (estadísticas por indicador)

Todos los archivos se guardan en la misma carpeta que este script.
"""

import os
import sys
import math
import datetime as dt

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
API_URL = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/"
    "910422,910423,910424,910426,910425,910420,910421/es/0/false/BIE-BISE/2.0/"
    "84e9582e-c6b4-4f0b-90d2-9ea2df019a0a?type=json"
)
TIMEOUT_SEG = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "distribucion_historica_inegi.pdf")
PDF_QUINC_PATH = os.path.join(BASE_DIR, "distribucion_historica_inegi_quincenal.pdf")
PDF_MENSUAL_PATH = os.path.join(BASE_DIR, "distribucion_historica_inegi_mensual.pdf")
PDF_QUINC_MENS_PATH = os.path.join(BASE_DIR, "distribucion_historica_inegi_quincenal_mensual.pdf")
DATOS_CSV = os.path.join(BASE_DIR, "datos_distribucion_historica_inegi.csv")
RESUMEN_CSV = os.path.join(BASE_DIR, "resumen_distribucion_historica.csv")

# La respuesta JSON del INEGI no trae el nombre del indicador (solo el ID).
# Si se conoce el nombre oficial, agréguelo aquí; si no, se usa un texto genérico.
NOMBRES_OFICIALES = {
    "910420": "Inflación general",
    "910421": "Inflación subyacente",
    "910422": "Mercancías",
    "910423": "Servicios",
    "910424": "Inflación no subyacente",
    "910425": "Agropecuarios",
    "910426": "Energéticos y tarifas autorizadas por el gobierno",
}

# Mínimo de observaciones históricas para una distribución útil.
MIN_HIST = 30

MESES_ABR = ["ene.", "feb.", "mar.", "abr.", "may.", "jun.",
             "jul.", "ago.", "sep.", "oct.", "nov.", "dic."]

# Estilo visual común
COLOR_TITULO = "#00205C"
COLOR_TEXTO = "#404040"
COLOR_HIST = "#A6A6A6"
COLOR_ACTUAL = "#00205C"


# ---------------------------------------------------------------------------
# 1. Descarga
# ---------------------------------------------------------------------------
def descargar_json():
    """Consulta la API. No imprime la URL (contiene la llave) ni str(excepción)."""
    try:
        resp = requests.get(
            API_URL,
            timeout=TIMEOUT_SEG,
            headers={"User-Agent": "inegi-distribucion-historica/1.0"},
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        sys.exit("ERROR: tiempo de espera agotado al consultar la API del INEGI.")
    except requests.exceptions.ConnectionError:
        sys.exit("ERROR: no fue posible establecer conexión con la API del INEGI.")
    except requests.exceptions.HTTPError as exc:
        sys.exit(f"ERROR HTTP: la API respondió con código {exc.response.status_code}.")
    except requests.exceptions.RequestException as exc:
        sys.exit(f"ERROR en la solicitud HTTP: {type(exc).__name__}.")

    if not resp.content or not resp.text.strip():
        sys.exit("ERROR: la API devolvió una respuesta vacía.")
    try:
        return resp.json()
    except ValueError:
        sys.exit("ERROR: la respuesta de la API no es JSON válido.")


def extraer_series(data):
    """Valida la estructura real del JSON: se espera un dict con la lista 'Series'."""
    if not isinstance(data, dict) or "Series" not in data:
        sys.exit("ERROR: estructura inesperada del JSON (no existe la llave 'Series').")
    series = data["Series"]
    if not isinstance(series, list) or len(series) == 0:
        sys.exit("ERROR: la lista 'Series' está vacía.")
    return series


# ---------------------------------------------------------------------------
# 2. Limpieza y validación
# ---------------------------------------------------------------------------
def parsear_fecha(texto):
    """TIME_PERIOD viene como 'YYYY/MM/DD'. Día 01 = 1ª quincena, día 02 = 2ª quincena."""
    try:
        return dt.datetime.strptime(str(texto).strip(), "%Y/%m/%d")
    except (TypeError, ValueError):
        return None


def procesar_serie(serie):
    """Convierte una serie del JSON en un DataFrame limpio y ordenado."""
    ind_id = str(serie.get("INDICADOR", "")).strip()
    freq = str(serie.get("FREQ", "")).strip()
    unidad_cod = str(serie.get("UNIT", "")).strip()
    nombre = NOMBRES_OFICIALES.get(ind_id, f"Indicador INEGI {ind_id}")

    if freq != "15":
        print(f"[{ind_id}] Omitido: FREQ={freq!r} no corresponde a quincenal (15).")
        return None

    periodicidad = "Quincenal"
    unidad = f"Índice (código de unidad INEGI {unidad_cod})"
    avisos = []

    registros = []
    n_nulos = 0
    n_fecha_invalida = 0
    for obs in serie.get("OBSERVATIONS") or []:
        raw = obs.get("OBS_VALUE")
        try:
            valor = float(str(raw).strip()) if raw not in (None, "") else math.nan
        except ValueError:
            valor = math.nan
        # Solo se eliminan valores nulos, vacíos o no numéricos (nunca se rellenan con cero).
        if not np.isfinite(valor):
            n_nulos += 1
            continue

        fecha = parsear_fecha(obs.get("TIME_PERIOD"))
        if fecha is None or fecha.day not in (1, 2):
            n_fecha_invalida += 1
            continue

        registros.append({
            "indicator_id": ind_id,
            "indicator_name": nombre,
            "unidad": unidad,
            "periodicidad": periodicidad,
            "anio": fecha.year,
            "mes": fecha.month,
            "quincena": 1 if fecha.day == 1 else 2,
            "fecha": fecha.date().isoformat(),
            "valor": valor,
            "obs_status": obs.get("OBS_STATUS"),
        })

    if n_nulos:
        avisos.append(f"{n_nulos} observaciones con valor nulo/vacío/no numérico descartadas")
    if n_fecha_invalida:
        avisos.append(f"{n_fecha_invalida} observaciones con fecha no reconocible descartadas")
    if not registros:
        print(f"[{ind_id}] Sin observaciones válidas.")
        return None

    df = pd.DataFrame(registros)

    # Duplicados por indicador y periodo. Criterio documentado: los metadatos de la API
    # no distinguen versiones; se conserva la primera aparición en el JSON
    # (todas las observaciones tienen OBS_STATUS y sin excepción, así que no hay preferencia).
    dup_mask = df.duplicated(subset=["anio", "mes", "quincena"], keep="first")
    if dup_mask.any():
        avisos.append(f"{int(dup_mask.sum())} duplicados por periodo; se conservó la primera aparición")
        df = df[~dup_mask]

    # Orden cronológico real (no depende del orden de las filas)
    df = df.sort_values(["anio", "mes", "quincena"]).reset_index(drop=True)
    df["_idx"] = df["anio"] * 24 + (df["mes"] - 1) * 2 + (df["quincena"] - 1)

    # Verificación de periodicidad quincenal: consecutivos = índice +1
    huecos = int((df["_idx"].diff().iloc[1:] != 1).sum())
    if huecos:
        avisos.append(f"{huecos} huecos en la secuencia quincenal (no se interpolan)")

    # 3A. Variación quincenal: contra la quincena inmediatamente anterior (solo si es consecutiva)
    prev_val = df["valor"].shift(1)
    consecutiva = df["_idx"].diff() == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        var_q = (df["valor"] / prev_val - 1) * 100
    var_q = var_q.where(consecutiva & (prev_val != 0), np.nan)

    # 3B. Variación anual: misma quincena (mes + número de quincena) del año anterior
    lookup = {(a, m, q): v for a, m, q, v in zip(df["anio"], df["mes"], df["quincena"], df["valor"])}
    var_a = []
    for a, m, q, v in zip(df["anio"], df["mes"], df["quincena"], df["valor"]):
        pv = lookup.get((a - 1, m, q))
        var_a.append((v / pv - 1) * 100 if (pv is not None and pv != 0) else np.nan)
    df["variacion_quincenal"] = var_q.values
    df["variacion_anual"] = var_a

    # Tratamiento: los valores son niveles de índice (no variaciones porcentuales),
    # por lo que las variaciones se calculan directamente sobre el nivel.
    df["tratamiento_aplicado"] = ("Nivel de índice: variación calculada directamente sobre el nivel; "
                                  "no se aplicó variación de una variación")
    df["es_ultima_observacion"] = False
    df.loc[df.index[-1], "es_ultima_observacion"] = True
    df["periodo"] = [f"{q}Q {MESES_ABR[m - 1]} {a}" for a, m, q in zip(df["anio"], df["mes"], df["quincena"])]

    return {
        "id": ind_id,
        "nombre": nombre,
        "unidad": unidad,
        "periodicidad": periodicidad,
        "df": df.drop(columns=["_idx"]),
        "avisos": avisos,
    }


# ---------------------------------------------------------------------------
# 4-5. Distribución histórica, dato actual y estadísticas
# ---------------------------------------------------------------------------
def clasificar(x, st):
    if x < st["min"]:
        return "Por debajo del rango histórico"
    if x < st["p10"]:
        return "Extremo inferior"
    if x < st["p25"]:
        return "Rango inferior"
    if x <= st["p75"]:
        return "Rango habitual"
    if x <= st["p90"]:
        return "Rango superior"
    if x <= st["max"]:
        return "Extremo superior"
    return "Por encima del rango histórico"


def analizar(ind, col="variacion_anual", texto=None):
    """Distribución histórica de la variable `col`."""
    df = ind["df"]
    ind["variable_texto"] = texto or ("variación anual" if col == "variacion_anual" else "variación quincenal")
    # Histórico: todas las variaciones válidas EXCEPTO la última observación
    hist = df.iloc[:-1][col].dropna().to_numpy(dtype=float)
    dato = df.iloc[-1][col]
    ind["hist"] = hist
    ind["actual"] = float(dato) if pd.notna(dato) else math.nan
    ind["ultimo_periodo"] = df.iloc[-1]["periodo"]
    ind["ultimo_valor"] = float(df.iloc[-1]["valor"])
    ind["ultima_q"] = df.iloc[-1]["variacion_quincenal"]
    ind["ultima_a"] = df.iloc[-1]["variacion_anual"]

    n = len(hist)
    if n == 0 or math.isnan(ind["actual"]):
        ind["stats"] = None
        ind["clasificacion"] = "Sin datos históricos suficientes"
        ind["pct"] = math.nan
        ind["avisos"].append("Sin variaciones anuales históricas válidas")
        return ind

    st = {
        "n": n,
        "min": float(hist.min()),
        "p10": float(np.percentile(hist, 10)),
        "p25": float(np.percentile(hist, 25)),
        "median": float(np.median(hist)),
        "p75": float(np.percentile(hist, 75)),
        "p90": float(np.percentile(hist, 90)),
        "max": float(hist.max()),
        "mean": float(hist.mean()),
    }
    pct = 100.0 * np.sum(hist <= ind["actual"]) / n
    st["pct_actual"] = float(pct)
    ind["stats"] = st
    ind["pct"] = float(pct)
    ind["clasificacion"] = clasificar(ind["actual"], st)
    if n < MIN_HIST:
        ind["avisos"].append(f"Historia insuficiente: solo {n} observaciones históricas (mínimo {MIN_HIST})")
    return ind


def validar_manual(ind):
    """Recalcula por otra vía (búsqueda directa por periodo) y compara con el programa."""
    df = ind["df"]
    v = df["valor"].tolist()
    last = df.iloc[-1]
    tol = 1e-9

    man_q = (v[-1] / v[-2] - 1) * 100 if len(v) >= 2 else math.nan

    prev = df[(df["anio"] == last["anio"] - 1) & (df["mes"] == last["mes"]) & (df["quincena"] == last["quincena"])]
    man_a = (v[-1] / float(prev["valor"].iloc[0]) - 1) * 100 if len(prev) else math.nan

    vals = []
    for _, r in df.iloc[:-1].iterrows():
        f = df[(df["anio"] == r["anio"] - 1) & (df["mes"] == r["mes"]) & (df["quincena"] == r["quincena"])]
        if len(f):
            vals.append((r["valor"] / float(f["valor"].iloc[0]) - 1) * 100)
    man_pct = 100.0 * sum(1 for x in vals if x <= man_a) / len(vals) if vals and not math.isnan(man_a) else math.nan

    def igual(a, b):
        if (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b))):
            return True
        return abs(float(a) - float(b)) < tol

    ok_q = igual(man_q, ind["ultima_q"])
    ok_a = igual(man_a, ind["ultima_a"])
    ok_p = (ind["stats"] is None) or igual(man_pct, ind["pct"])
    ok = ok_q and ok_a and ok_p
    ind["validacion"] = {
        "quincenal_manual": man_q, "quincenal_programa": ind["ultima_q"], "ok_q": ok_q,
        "anual_manual": man_a, "anual_programa": ind["ultima_a"], "ok_a": ok_a,
        "pct_manual": man_pct, "pct_programa": ind["pct"], "ok_p": ok_p,
        "ok": ok,
    }
    return ok


# ---------------------------------------------------------------------------
# 6-9. Gráfica
# ---------------------------------------------------------------------------
def dibujar(ax, ind):
    hist = ind["hist"]
    actual = ind["actual"]
    st = ind["stats"]

    ax.set_facecolor("white")
    ax.set_yticks([])
    ax.set_ylabel("")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(False)
    ax.set_xlabel(ind["variable_texto"].capitalize() + " (%)", fontsize=8, color=COLOR_TEXTO, labelpad=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
    ax.tick_params(axis="x", labelsize=7.5, colors=COLOR_TEXTO, length=3, width=0.6)

    # Encabezado
    if st is None:
        info = f"Último dato: {ind['actual']:.2f}% | Percentil histórico: n.d. | {ind['clasificacion']}" \
            if not math.isnan(ind["actual"]) else "Último dato: n.d. | " + ind["clasificacion"]
    else:
        info = f"Último dato: {ind['actual']:.2f}% | Percentil histórico: {ind['pct']:.0f} | {ind['clasificacion']}"
    ax.text(0, 1.31, ind["nombre"], transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=COLOR_TITULO, va="bottom", ha="left")
    ax.text(0, 1.19, f"Distribución histórica de la {ind['variable_texto']}", transform=ax.transAxes,
            fontsize=8, color=COLOR_TEXTO, va="bottom", ha="left")
    ax.text(0, 1.09, info, transform=ax.transAxes, fontsize=8, color=COLOR_TEXTO, va="bottom", ha="left")

    if st is None or math.isnan(ind["actual"]):
        ax.text(0.5, 0.5, "Sin datos suficientes para la distribución", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#7F7F7F")
        ax.set_xticks([])
        return

    # Escala: incluye histórico y dato actual, con margen horizontal
    lo = min(hist.min(), actual)
    hi = max(hist.max(), actual)
    span = hi - lo if hi > lo else 1.0
    x0, x1 = lo - 0.2 * span, hi + 0.2 * span
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.6, 0.6)

    # Referencias (bajo los puntos): IQR, mediana, mínimo, máximo, cero
    ax.axvspan(st["p25"], st["p75"], color="#E6E6E6", alpha=0.55, lw=0, zorder=0)
    ax.axvline(st["median"], color="#595959", lw=1.0, zorder=1)
    ax.axvline(st["min"], color="#8C8C8C", lw=0.6, ls=":", zorder=1)
    ax.axvline(st["max"], color="#8C8C8C", lw=0.6, ls=":", zorder=1)
    if x0 < 0 < x1:
        ax.axvline(0.0, color="#7F7F7F", lw=0.6, zorder=1)

    # Observaciones históricas: jitter vertical simétrico con semilla fija
    np.random.seed(42)
    jitter = np.random.uniform(-0.12, 0.12, size=len(hist))
    ax.scatter(hist, jitter, s=25, c=COLOR_HIST, alpha=0.60, edgecolors="none", zorder=2)

    # Dato actual: único punto azul, y = 0
    ax.scatter([actual], [0.0], s=25, c=COLOR_ACTUAL, alpha=1.0,
               edgecolors="none", linewidths=0, zorder=4)

    # Etiqueta del dato actual: a la derecha si el punto está en la mitad izquierda, y viceversa
    texto_q = f"{ind['ultima_q']:.2f}%" if not pd.isna(ind["ultima_q"]) else "n.d."
    texto_a = f"{ind['ultima_a']:.2f}%" if not pd.isna(ind["ultima_a"]) else "n.d."
    texto = f"{ind['ultimo_periodo']}\nAnual: {texto_a}\n{ind.get('etiqueta_sec', 'Quincenal')}: {texto_q}"
    a_derecha = (actual - x0) / (x1 - x0) < 0.5
    ax.annotate(
        texto,
        xy=(actual, 0.0),
        xytext=(26 if a_derecha else -26, 16),
        textcoords="offset points",
        ha="left" if a_derecha else "right",
        va="bottom",
        fontsize=7.5,
        color=COLOR_TEXTO,
        arrowprops=dict(arrowstyle="-", color="#7F7F7F", lw=0.6, shrinkA=0, shrinkB=5),
        zorder=5,
    )

    if ind["avisos"] and st["n"] < MIN_HIST:
        ax.text(0.5, 0.06, f"ADVERTENCIA: historia insuficiente (n={st['n']})",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#7F7F7F")


# ---------------------------------------------------------------------------
# Variaciones mensuales (índice mensual = promedio de las dos quincenas del mes)
# ---------------------------------------------------------------------------
def construir_mensual(ind):
    """
    Construye la serie mensual a partir de las quincenas:
      - valor mensual = promedio de la 1ª y 2ª quincena del mes;
      - solo se usan meses con AMBAS quincenas (el mes incompleto más reciente se excluye);
      - 'variacion_quincenal' contiene aquí la variación mensual (vs. mes inmediato anterior);
      - 'variacion_anual' compara el mes contra el mismo mes del año anterior.
    """
    df = ind["df"]
    filas = []
    incompletos = 0
    for (a, m), sub in df.groupby(["anio", "mes"]):
        if sorted(sub["quincena"].tolist()) != [1, 2]:
            incompletos += 1
            continue
        filas.append({"anio": int(a), "mes": int(m),
                      "valor": float(sub["valor"].mean()),
                      "periodo": f"{MESES_ABR[m - 1]} {a}"})
    mdf = pd.DataFrame(filas).sort_values(["anio", "mes"]).reset_index(drop=True)

    idx = mdf["anio"] * 12 + (mdf["mes"] - 1)
    consecutivo = idx.diff() == 1
    prev = mdf["valor"].shift(1)
    mdf["variacion_quincenal"] = ((mdf["valor"] / prev - 1) * 100).where(consecutivo & (prev != 0), np.nan)

    lookup = {(a, m): v for a, m, v in zip(mdf["anio"], mdf["mes"], mdf["valor"])}
    var_a = []
    for a, m in zip(mdf["anio"], mdf["mes"]):
        pv = lookup.get((a - 1, m))
        var_a.append((lookup[(a, m)] / pv - 1) * 100 if (pv is not None and pv != 0) else np.nan)
    mdf["variacion_anual"] = var_a

    avisos = []
    if incompletos:
        avisos.append(f"{incompletos} meses con una sola quincena excluidos del análisis mensual")
    if len(mdf) > 1 and not consecutivo.iloc[1:].all():
        avisos.append("hay meses faltantes en la serie mensual; esas variaciones quedan NaN")

    out = dict(ind)
    out["df"] = mdf
    out["avisos"] = list(ind["avisos"]) + avisos
    out["etiqueta_sec"] = "Mensual"
    return out


def validar_mensual(ind_q, ind_m):
    """Recalcula el último mes completo desde las quincenas originales y compara con el programa."""
    df = ind_q["df"]
    qv = {(a, m, q): v for a, m, q, v in zip(df["anio"], df["mes"], df["quincena"], df["valor"])}
    completos = sorted({(a, m) for (a, m, q) in qv if (a, m, 1) in qv and (a, m, 2) in qv})
    conjunto = set(completos)
    ua, um = completos[-1]

    def val(a, m):
        return (qv[(a, m, 1)] + qv[(a, m, 2)]) / 2

    ultimo = val(ua, um)
    pa, pm = (ua, um - 1) if um > 1 else (ua - 1, 12)
    man_mom = (ultimo / val(pa, pm) - 1) * 100 if (pa, pm) in conjunto else math.nan
    man_yoy = (ultimo / val(ua - 1, um) - 1) * 100 if (ua - 1, um) in conjunto else math.nan
    periodo_esperado = f"{MESES_ABR[um - 1]} {ua}"

    def igual(x, y):
        if math.isnan(x) and (y is None or (isinstance(y, float) and math.isnan(y))):
            return True
        return y is not None and not math.isnan(x) and not pd.isna(y) and abs(x - y) < 1e-9

    ok_p = ind_m["df"].iloc[-1]["periodo"] == periodo_esperado
    ok_mom = igual(man_mom, ind_m["ultima_q"])
    ok_yoy = igual(man_yoy, ind_m["ultima_a"])
    return {"periodo_esperado": periodo_esperado, "mom_manual": man_mom,
            "mom_programa": ind_m["ultima_q"], "yoy_manual": man_yoy,
            "yoy_programa": ind_m["ultima_a"],
            "ok": bool(ok_p and ok_mom and ok_yoy)}


def generar_pdf(indicadores, ruta_pdf, titulo_pdf):
    """Landscape carta, 2 columnas x 3 filas; 6 gráficas por página."""
    por_pagina = 6
    with PdfPages(ruta_pdf) as pdf:
        for ini in range(0, len(indicadores), por_pagina):
            bloque = indicadores[ini:ini + por_pagina]
            fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
            fig.subplots_adjust(left=0.06, right=0.97, bottom=0.16, top=0.885, hspace=0.9, wspace=0.18)
            for k in range(6):
                ax = axes[k // 2][k % 2]
                if k < len(bloque):
                    dibujar(ax, bloque[k])
                else:
                    ax.axis("off")
            fig.text(0.06, 0.075, "Fuente: INEGI. Elaboración propia con datos de la API de Indicadores.",
                     fontsize=7.5, color=COLOR_TEXTO)
            fig.text(0.06, 0.052, "La distribución histórica excluye la última observación.",
                     fontsize=7.5, color=COLOR_TEXTO)
            fig.text(0.06, 0.029, "La dispersión vertical (jitter) es aleatoria y no representa ninguna variable.",
                     fontsize=7.5, color=COLOR_TEXTO)
            fig.text(0.97, 0.029, f"Página {ini // por_pagina + 1} de {math.ceil(len(indicadores) / por_pagina)}",
                     fontsize=7.5, color=COLOR_TEXTO, ha="right")
            pdf.savefig(fig)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"Distribución histórica de la {titulo_pdf.lower()} - INEGI"
        info["Author"] = "Elaboración propia"


# ---------------------------------------------------------------------------
# Salidas CSV
# ---------------------------------------------------------------------------
def guardar_csv(indicadores):
    columnas = ["indicator_id", "indicator_name", "unidad", "periodicidad", "periodo", "fecha",
                "valor", "variacion_quincenal", "variacion_anual", "es_ultima_observacion",
                "tratamiento_aplicado", "anio", "mes", "quincena"]
    partes = []
    for ind in indicadores:
        partes.append(ind["df"][columnas])
    datos = pd.concat(partes, ignore_index=True)
    datos["variacion_quincenal"] = datos["variacion_quincenal"].round(6)
    datos["variacion_anual"] = datos["variacion_anual"].round(6)
    datos.to_csv(DATOS_CSV, index=False, encoding="utf-8-sig")

    filas = []
    for ind in indicadores:
        st = ind["stats"]
        def g(k):
            return round(st[k], 4) if st else math.nan
        filas.append({
            "indicator_id": ind["id"],
            "indicator_name": ind["nombre"],
            "ultimo_periodo": ind["ultimo_periodo"],
            "ultimo_valor": round(ind["ultimo_valor"], 6),
            "ultima_variacion_quincenal": round(ind["ultima_q"], 4) if pd.notna(ind["ultima_q"]) else math.nan,
            "ultima_variacion_anual": round(ind["ultima_a"], 4) if pd.notna(ind["ultima_a"]) else math.nan,
            "numero_observaciones_historicas": st["n"] if st else 0,
            "minimo_historico": g("min"),
            "percentil_10": g("p10"),
            "percentil_25": g("p25"),
            "mediana": g("median"),
            "percentil_75": g("p75"),
            "percentil_90": g("p90"),
            "maximo_historico": g("max"),
            "percentil_dato_actual": round(ind["pct"], 2) if st else math.nan,
            "clasificacion": ind["clasificacion"],
        })
    pd.DataFrame(filas).to_csv(RESUMEN_CSV, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # evita acentos corruptos en la consola de Windows
    except AttributeError:
        pass
    data = descargar_json()
    series = extraer_series(data)
    print(f"Series recibidas de la API: {len(series)}")

    indicadores = []
    for s in series:
        ind = procesar_serie(s)
        if ind is None:
            continue
        analizar(ind, "variacion_anual")
        validar_manual(ind)
        indicadores.append(ind)

    if not indicadores:
        sys.exit("ERROR: ningún indicador produjo datos válidos.")

    print(f"Indicadores procesados: {len(indicadores)}")
    for ind in indicadores:
        st = ind["stats"]
        v = ind["validacion"]
        print("-" * 78)
        print(f"ID: {ind['id']} | {ind['nombre']}")
        print(f"  Unidad: {ind['unidad']} | Periodicidad: {ind['periodicidad']}")
        print(f"  Observaciones válidas: {len(ind['df'])} | Último periodo: {ind['ultimo_periodo']}")
        print(f"  Último valor: {ind['ultimo_valor']:.4f}")
        q = ind["ultima_q"]
        a = ind["ultima_a"]
        print(f"  Variación quincenal: {q:.2f}%" if pd.notna(q) else "  Variación quincenal: n.d.")
        print(f"  Variación anual: {a:.2f}%" if pd.notna(a) else "  Variación anual: n.d.")
        if st:
            print(f"  Histórico n={st['n']} | min={st['min']:.2f} p25={st['p25']:.2f} "
                  f"mediana={st['median']:.2f} p75={st['p75']:.2f} max={st['max']:.2f}")
            print(f"  Percentil histórico del dato actual: {ind['pct']:.2f}")
        print(f"  Clasificación: {ind['clasificacion']}")
        print(f"  Validación manual: {'OK' if v['ok'] else 'DIFERENCIA'}")
        for aviso in ind["avisos"]:
            print(f"  AVISO: {aviso}")

    generar_pdf(indicadores, PDF_PATH, "Variación anual")
    guardar_csv(indicadores)

    # PDF adicional: la misma distribución, pero con la variación quincenal
    quincenales = []
    for ind in indicadores:
        q = dict(ind)
        q["avisos"] = list(ind["avisos"])  # copia para no mezclar avisos entre variantes
        analizar(q, "variacion_quincenal")
        quincenales.append(q)
    generar_pdf(quincenales, PDF_QUINC_PATH, "Variación quincenal")
    print(f"PDF quincenal: {PDF_QUINC_PATH}")

    # Variaciones mensuales: último mes completo (con ambas quincenas)
    mens_anual, mens_mensual = [], []
    for ind in indicadores:
        base = construir_mensual(ind)
        a = dict(base)
        a["avisos"] = list(base["avisos"])
        analizar(a, "variacion_anual", texto="variación anual")
        b = dict(base)
        b["avisos"] = list(base["avisos"])
        analizar(b, "variacion_quincenal", texto="variación mensual")
        a["validacion"] = validar_mensual(ind, a)
        b["validacion"] = a["validacion"]
        mens_anual.append(a)
        mens_mensual.append(b)
        print(f"[{ind['id']}] Último mes completo: {a['ultimo_periodo']} | "
              f"Var. mensual: {a['ultima_q']:.2f}% | Var. anual mensual: {a['ultima_a']:.2f}% | "
              f"Validación: {'OK' if a['validacion']['ok'] else 'DIFERENCIA'}")

    generar_pdf(mens_anual, PDF_MENSUAL_PATH, "Variación anual (mensual)")
    print(f"PDF mensual: {PDF_MENSUAL_PATH}")
    generar_pdf(mens_mensual, PDF_QUINC_MENS_PATH, "Variación mensual")
    print(f"PDF quincenal_mensual: {PDF_QUINC_MENS_PATH}")
    print("-" * 78)
    print(f"PDF:          {PDF_PATH}")
    print(f"CSV datos:    {DATOS_CSV}")
    print(f"CSV resumen:  {RESUMEN_CSV}")

    if not all(ind["validacion"]["ok"] for ind in indicadores) or not all(m["validacion"]["ok"] for m in mens_anual):
        sys.exit("ERROR: la validación manual no coincide con el programa en algún indicador.")


if __name__ == "__main__":
    main()
