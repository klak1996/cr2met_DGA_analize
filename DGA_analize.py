# %% [Celda 1: Instalar y cargar dependencias]
import sys
import subprocess
import pkg_resources
required = {'pandas', 'openpyxl', 'xlrd', 'rasterio'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
import os
import glob
import pandas as pd
import rasterio
from rasterio.crs import CRS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xlrd")
print("✅ Dependencias listas.")

# %% [Celda 2: Configurar rutas y CRS]
INPUT_EXCEL_DIR = r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA"
MASK_PATH = r"C:\GIS\klak1996\SIGHD\Tetis\Dem_fill.asc"
OUTPUT_EXCEL_PATH = os.path.join(INPUT_EXCEL_DIR, "Andelien_stations.xlsx")
TARGET_CRS = CRS.from_epsg(32718)
print(f"✅ Rutas configuradas. CRS: UTM 18S")

# %% [Celda 3: Función para extraer datos de hoja Excel]
def extract_station_data_from_sheet(df_sheet, sheet_name):
    station_name = None
    utm_e = None
    utm_n = None
    precip_data = []
    df_str = df_sheet.astype(str)
    for i in range(len(df_str)):
        row = df_str.iloc[i]
        if any("Estación:" in str(cell) for cell in row):
            for cell in row:
                if "Estación:" in str(cell):
                    station_name = str(cell).replace("Estación:", "").strip()
                    break
            if station_name: break
    if not station_name:
        station_name = sheet_name.strip()
    for i in range(len(df_str)):
        row = df_str.iloc[i]
        row_values = [str(cell).strip() for cell in row]
        if "UTM Este (mts):" in row_values:
            idx = row_values.index("UTM Este (mts):")
            if idx + 1 < len(row_values):
                try: utm_e = float(row_values[idx + 1])
                except ValueError: pass
        if "UTM Norte (mts):" in row_values:
            idx = row_values.index("UTM Norte (mts):")
            if idx + 1 < len(row_values):
                try: utm_n = float(row_values[idx + 1])
                except ValueError: pass
    data_start_row = None
    year_col_idx = None
    precip_col_idx = None
    for i in range(len(df_str)):
        row = df_str.iloc[i]
        row_values = [str(cell).strip() for cell in row]
        if "AÑO" in row_values and "MAXIMA EN 24 HS. PRECIPITACION (mm)" in row_values:
            year_col_idx = row_values.index("AÑO")
            precip_col_idx = row_values.index("MAXIMA EN 24 HS. PRECIPITACION (mm)")
            data_start_row = i + 1
            break
    if data_start_row is not None and year_col_idx is not None and precip_col_idx is not None:
        for j in range(data_start_row, len(df_sheet)):
            year_val = df_sheet.iloc[j, year_col_idx]
            precip_val = df_sheet.iloc[j, precip_col_idx]
            try:
                year = int(float(year_val))
                precip = float(precip_val)
                if not pd.isna(year) and not pd.isna(precip):
                    precip_data.append({"AÑO": year, "PRECIPITACION (mm)": precip})
            except (ValueError, TypeError):
                continue
    return station_name, utm_e, utm_n, precip_data
print("✅ Función de extracción definida.")

# %% [Celda 4: Función para verificar punto en máscara]
def point_in_raster_bounds(x, y, raster_transform, raster_width, raster_height):
    try:
        col, row = ~raster_transform * (x, y)
        return 0 <= row < raster_height and 0 <= col < raster_width
    except Exception:
        return False
print("✅ Función de verificación geográfica definida.")

# %% [Celda 5: Proceso principal — ejecutar filtrado y DEVOLVER valid_stations]

def main():
    print("🚀 Iniciando...")
    try:
        with rasterio.open(MASK_PATH) as src:
            mask_transform = src.transform
            mask_width = src.width
            mask_height = src.height
    except Exception as e:
        print(f"❌ Error máscara: {e}")
        return {}

    excel_files = glob.glob(os.path.join(INPUT_EXCEL_DIR, "*.xls")) + glob.glob(os.path.join(INPUT_EXCEL_DIR, "*.xlsx"))
    if not excel_files:
        print("❌ Sin archivos Excel.")
        return {}

    valid_stations = {}

    for file_path in excel_files:
        print(f"\n📄 Archivo: {os.path.basename(file_path)}")
        try:
            xls = pd.read_excel(file_path, sheet_name=None, header=None)
        except Exception as e:
            print(f"  ⚠️ Error lectura: {e}")
            continue

        for sheet_name, df_sheet in xls.items():
            print(f"  📄 Hoja: '{sheet_name}'")
            station_name, utm_e, utm_n, precip_data = extract_station_data_from_sheet(df_sheet, sheet_name)
            if utm_e is None or utm_n is None:
                print(f"    ⚠️ Sin UTM. Saltando.")
                continue
            if len(precip_data) == 0:
                print(f"    ⚠️ Sin datos. Saltando.")
                continue
            if point_in_raster_bounds(utm_e, utm_n, mask_transform, mask_width, mask_height):
                print(f"    ✅ DENTRO de máscara.")
                valid_stations[station_name] = pd.DataFrame(precip_data)
            else:
                print(f"    ❌ FUERA de máscara.")

    if valid_stations:
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
                for station_name, df_data in valid_stations.items():
                    safe_sheet_name = station_name[:31]
                    for char in ['\\', '/', '*', '[', ']', ':', '?']:
                        safe_sheet_name = safe_sheet_name.replace(char, '_')
                    df_data.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                    print(f"  📊 Hoja '{safe_sheet_name}' guardada.")
            print(f"\n🎉 ¡COMPLETADO! Archivo: {OUTPUT_EXCEL_PATH}")
        except Exception as e:
            print(f"❌ Error guardado: {e}")
    else:
        print("\n❌ Ninguna estación dentro de la máscara.")

    return valid_stations  # <-- ¡ESTA LÍNEA ES CLAVE!

# Ejecutar y capturar el resultado
valid_stations = main()  # <-- ¡ASÍ QUEDA DISPONIBLE PARA CELDAS POSTERIORES!

# %% [Celda 6: Verificar resultado (opcional)]
if os.path.exists(OUTPUT_EXCEL_PATH):
    print(f"🔍 Verificando: {OUTPUT_EXCEL_PATH}")
    try:
        output_xls = pd.read_excel(OUTPUT_EXCEL_PATH, sheet_name=None)
        for sheet_name, df in output_xls.items():
            print(f"\n📊 Hoja: '{sheet_name}'")
            print(f"   Filas: {len(df)}, Columnas: {list(df.columns)}")
            print(df.head(3))
    except Exception as e:
        print(f"❌ Error lectura: {e}")
else:
    print("❌ Archivo de salida no encontrado.")

# %% [Celda 7: Preparar todas las estaciones para IDW (incluyendo las fuera de la máscara)]

# Recopilar todas las estaciones de todos los archivos Excel
all_stations_full_data = {}

# Encontrar todos los archivos Excel nuevamente
excel_files_all = glob.glob(os.path.join(INPUT_EXCEL_DIR, "*.xls")) + glob.glob(os.path.join(INPUT_EXCEL_DIR, "*.xlsx"))

for file_path in excel_files_all:
    try:
        xls = pd.read_excel(file_path, sheet_name=None, header=None)
    except Exception as e:
        continue

    for sheet_name, df_sheet in xls.items():
        station_name, utm_e, utm_n, precip_data = extract_station_data_from_sheet(df_sheet, sheet_name)
        if utm_e is not None and utm_n is not None and len(precip_data) > 0:
            df_precip = pd.DataFrame(precip_data)
            all_stations_full_data[station_name] = {
                'utm_e': utm_e,
                'utm_n': utm_n,
                'data': df_precip.set_index('AÑO')['PRECIPITACION (mm)']
            }

print(f"✅ Total de estaciones cargadas para IDW: {len(all_stations_full_data)}")

# %% [Celda 8: Función de interpolación IDW (p=3, 5 vecinos)]

def idw_impute(target_year, target_station_name, all_stations_dict, p=3, max_neighbors=5):
    """
    Imputa valor para un año específico en una estación usando IDW con vecinos más cercanos.
    p=3 da más peso al más cercano.
    """
    target_coords = (all_stations_dict[target_station_name]['utm_e'], all_stations_dict[target_station_name]['utm_n'])
    neighbors = []

    for station_name, station_info in all_stations_dict.items():
        if station_name == target_station_name:
            continue  # No usar la misma estación
        if target_year in station_info['data'] and not pd.isna(station_info['data'][target_year]):
            # Calcular distancia euclidiana
            dist = ((target_coords[0] - station_info['utm_e'])**2 + (target_coords[1] - station_info['utm_n'])**2)**0.5
            if dist > 0:  # Evitar división por cero
                neighbors.append({
                    'station': station_name,
                    'value': station_info['data'][target_year],
                    'distance': dist
                })

    if len(neighbors) == 0:
        return None  # No hay vecinos con dato

    # Ordenar por distancia
    neighbors.sort(key=lambda x: x['distance'])
    # Tomar los 'max_neighbors' más cercanos
    neighbors = neighbors[:max_neighbors]

    # Calcular IDW
    weighted_sum = 0.0
    weight_sum = 0.0

    for n in neighbors:
        weight = 1 / (n['distance'] ** p)
        weighted_sum += n['value'] * weight
        weight_sum += weight

    if weight_sum == 0:
        return None

    return weighted_sum / weight_sum

# %% [Celda 9: Completar series 1990-2020 con IDW para estaciones dentro de la máscara]

# Crear rango completo de años
full_year_range = list(range(1990, 2021))

# Diccionario para estaciones completadas
completed_stations = {}

for station_name in valid_stations.keys():
    if station_name not in all_stations_full_data:
        continue  # Por si acaso, aunque no debería pasar

    original_series = valid_stations[station_name].set_index('AÑO')['PRECIPITACION (mm)']
    completed_series = []

    for year in full_year_range:
        if year in original_series.index and not pd.isna(original_series[year]):
            # Ya tiene dato observado
            completed_series.append({'AÑO': year, 'PRECIPITACION (mm)': original_series[year]})
        else:
            # Imputar con IDW
            imputed_value = idw_impute(year, station_name, all_stations_full_data, p=3, max_neighbors=5)
            completed_series.append({'AÑO': year, 'PRECIPITACION (mm)': imputed_value})

    completed_stations[station_name] = pd.DataFrame(completed_series)

print(f"✅ Series completadas para {len(completed_stations)} estaciones dentro de la máscara.")

# %% [Celda 10: Guardar Excel final con series completas (1990-2020)]

OUTPUT_EXCEL_PATH_COMPLETED = os.path.join(INPUT_EXCEL_DIR, "Andelien_stations_COMPLETED.xlsx")

try:
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH_COMPLETED, engine='openpyxl') as writer:
        for station_name, df_data in completed_stations.items():
            safe_sheet_name = station_name[:31]
            for char in ['\\', '/', '*', '[', ']', ':', '?']:
                safe_sheet_name = safe_sheet_name.replace(char, '_')
            df_data.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            print(f"  📊 Hoja '{safe_sheet_name}' guardada (serie completa 1990-2020).")
    print(f"\n🎉 ¡PROCESO DE COMPLETACIÓN FINALIZADO!")
    print(f"   Archivo generado: {OUTPUT_EXCEL_PATH_COMPLETED}")
except Exception as e:
    print(f"❌ Error al guardar el archivo completado: {e}")

# %% [Celda 11: Verificar resultado final (opcional)]

if os.path.exists(OUTPUT_EXCEL_PATH_COMPLETED):
    print(f"\n🔍 Verificando archivo completado: {OUTPUT_EXCEL_PATH_COMPLETED}")
    try:
        output_xls = pd.read_excel(OUTPUT_EXCEL_PATH_COMPLETED, sheet_name=None)
        for sheet_name, df in output_xls.items():
            print(f"\n📊 Hoja: '{sheet_name}'")
            print(f"   Total filas: {len(df)} (debe ser 31)")
            print(f"   Años cubiertos: {df['AÑO'].min()} - {df['AÑO'].max()}")
            # Mostrar primeros 3 y últimos 3 para ver imputados vs observados
            print("   Primeros 3:")
            print(df.head(3))
            print("   Últimos 3:")
            print(df.tail(3))
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
else:
    print("❌ Archivo de salida completado no encontrado.")

# %% [Celda 12: Importar librerías adicionales para análisis de frecuencia]

import scipy.stats as stats
from scipy.stats import genextreme
from scipy.spatial.distance import cdist
import pymannkendall as mk
from statsmodels.stats.stattools import durbin_watson
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

print("✅ Librerías para análisis de frecuencia cargadas.")

# %% [Celda 13: Configurar parámetros de análisis]

OUT_DIR_STATIONS = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA")
OUT_DIR_STATIONS.mkdir(parents=True, exist_ok=True)

PERIODOS_RETORNO = [2, 5, 10, 20, 25, 50, 100]
ALPHA = 0.05
IDW_POWER = 3  # p=3 como pediste
START_YEAR = 1990
END_YEAR = 2020

print("✅ Parámetros configurados.")

# %% [Celda 14: Preparar estructura de datos para análisis]

# Convertir completed_stations a un formato más analítico
station_series = {}
station_coords = {}

for station_name, df in completed_stations.items():
    # Asegurar que tiene 31 años
    if len(df) != 31:
        print(f"⚠️ {station_name} no tiene 31 años. Saltando.")
        continue
    station_series[station_name] = df.set_index('AÑO')['PRECIPITACION (mm)'].values
    # Obtener coordenadas de all_stations_full_data
    if station_name in all_stations_full_data:
        station_coords[station_name] = (
            all_stations_full_data[station_name]['utm_e'],
            all_stations_full_data[station_name]['utm_n']
        )
    else:
        print(f"❌ Coordenadas no encontradas para {station_name}. Saltando.")
        continue

station_names = list(station_series.keys())
n_stations = len(station_names)

print(f"✅ {n_stations} estaciones listas para análisis.")

# %% [Celda 15 MODIFICADA: Detectar outliers por umbral físico + Grubbs]

import numpy as np

print("=== DETECTANDO Y CORRIGIENDO OUTLIERS ===")

def grubbs_test(data, alpha=0.05):
    if len(data) < 3: return None
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0: return None
    deviations = np.abs(data - mean)
    max_dev_index = np.argmax(deviations)
    G = deviations[max_dev_index] / std
    N = len(data)
    t_value = stats.t.ppf(1 - alpha / (2 * N), N - 2)
    G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(t_value**2 / (N - 2 + t_value**2))
    return max_dev_index if G > G_critical else None

outliers_corrected = 0

# Primero: marcar outliers
for station_name in station_names:
    series = station_series[station_name].copy()

    # 1. Aplicar Grubbs (como antes)
    outlier_index_grubbs = grubbs_test(series, alpha=ALPHA)

    # 2. Aplicar filtro físico: cualquier valor < 30 mm es outlier
    outlier_indices_physical = np.where(series < 30)[0]

    # Combinar ambos métodos
    all_outlier_indices = set()
    if outlier_index_grubbs is not None:
        all_outlier_indices.add(outlier_index_grubbs)
    all_outlier_indices.update(outlier_indices_physical)

    # Marcar como NaN
    for idx in all_outlier_indices:
        station_series[station_name][idx] = np.nan
        outliers_corrected += 1

print(f"✅ Outliers marcados como NaN: {outliers_corrected}")

# Segundo: rellenar NaN con IDW espacial (usando otras estaciones) — SIN CAMBIOS
coords_array = np.array([station_coords[name] for name in station_names])
filled_count = 0

for i, station_name in enumerate(station_names):
    series = station_series[station_name]
    nan_mask = np.isnan(series)

    if not np.any(nan_mask):
        continue

    for t in range(len(series)):
        if not np.isnan(series[t]):
            continue

        # Encontrar vecinos válidos en ese año 't'
        neighbor_values = []
        neighbor_distances = []

        for j, other_name in enumerate(station_names):
            if i == j: continue
            other_series = station_series[other_name]
            if not np.isnan(other_series[t]):
                dist = np.sqrt((coords_array[i,0] - coords_array[j,0])**2 + (coords_array[i,1] - coords_array[j,1])**2)
                if dist > 0:
                    neighbor_values.append(other_series[t])
                    neighbor_distances.append(dist)

        if len(neighbor_values) == 0:
            median_val = np.nanmedian(series)
            if not np.isnan(median_val):
                station_series[station_name][t] = median_val
            else:
                station_series[station_name][t] = np.nanmean(series)
        else:
            neighbors = sorted(zip(neighbor_values, neighbor_distances), key=lambda x: x[1])[:5]
            values, distances = zip(*neighbors)
            weights = 1.0 / (np.array(distances) ** IDW_POWER)
            interpolated = np.sum(np.array(values) * weights) / np.sum(weights)
            station_series[station_name][t] = interpolated

        filled_count += 1

print(f"✅ Valores NaN rellenados con IDW: {filled_count}")
# %% [Celda 16: Análisis de calidad (4 tests) para cada estación]

print("=== ANÁLISIS DE CALIDAD: 4 TESTS FUNDAMENTALES ===")

quality_report = []

for station_name in tqdm(station_names, desc="Analizando estaciones"):
    series = station_series[station_name]
    if np.any(np.isnan(series)) or len(series) < 10:
        continue

    record = {"station": station_name}

    # 1. Aleatoriedad (Rachas)
    try:
        median_val = np.median(series)
        binary_seq = (series > median_val).astype(int)
        runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1])
        n1 = np.sum(binary_seq)
        n0 = len(binary_seq) - n1
        if n1 == 0 or n0 == 0:
            raise ValueError
        mean_runs = 1 + (2 * n1 * n0) / (n1 + n0)
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
        std_runs = np.sqrt(var_runs)
        z_stat = (runs - mean_runs) / std_runs
        p_value_runs = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        reject_random = p_value_runs < ALPHA
        record.update({
            "runs_p_value": p_value_runs,
            "reject_H0_randomness": reject_random
        })
    except:
        record.update({
            "runs_p_value": np.nan,
            "reject_H0_randomness": True
        })

    # 2. Independencia (Durbin-Watson)
    try:
        residuals = series - np.mean(series)
        dw_stat = durbin_watson(residuals)
        DL, DU = 1.36, 1.49
        reject_indep = (dw_stat < DL) or (dw_stat > (4 - DL))
        record.update({
            "dw_statistic": dw_stat,
            "reject_H0_independence": reject_indep
        })
    except:
        record.update({
            "dw_statistic": np.nan,
            "reject_H0_independence": True
        })

    # 3. Homogeneidad (SNHT)
    try:
        n = len(series)
        std_series = (series - np.mean(series)) / np.std(series, ddof=1)
        max_snht = 0
        for k in range(1, n):
            mean1 = np.mean(std_series[:k])
            mean2 = np.mean(std_series[k:])
            snht = (k * mean1**2 + (n - k) * mean2**2) / n
            if snht > max_snht:
                max_snht = snht
        critical_snht = 20.0
        p_value_snht = stats.chi2.sf(max_snht, df=1)
        reject_homog = max_snht > critical_snht
        record.update({
            "snht_statistic": max_snht,
            "snht_p_value": p_value_snht,
            "reject_H0_homogeneity": reject_homog
        })
    except:
        record.update({
            "snht_statistic": np.nan,
            "snht_p_value": np.nan,
            "reject_H0_homogeneity": True
        })

    # 4. Estacionariedad (Mann-Kendall)
    try:
        mk_result = mk.original_test(series)
        reject_station = mk_result.p < ALPHA
        record.update({
            "mk_p_value": mk_result.p,
            "reject_H0_stationarity": reject_station
        })
    except:
        record.update({
            "mk_p_value": np.nan,
            "reject_H0_stationarity": True
        })

    # Decisión final
    is_unreliable = (
        record["reject_H0_randomness"] or
        record["reject_H0_independence"] or
        record["reject_H0_homogeneity"] or
        record["reject_H0_stationarity"]
    )
    record["unreliable"] = is_unreliable
    quality_report.append(record)

df_quality = pd.DataFrame(quality_report)
unreliable_count = df_quality["unreliable"].sum()
print(f"\n📊 Estaciones no confiables: {unreliable_count}/{len(df_quality)} ({unreliable_count/len(df_quality)*100:.1f}%)")

df_quality.to_csv(OUT_DIR_STATIONS / "stations_quality_report.csv", index=False)
print(f"📄 Reporte de calidad guardado: stations_quality_report.csv")

# %% [Celda 17: Ajustar distribución GEV y tests de bondad de ajuste]

print("=== AJUSTANDO DISTRIBUCIÓN GEV ===")

gev_results = []
goodness_results = []

for record in quality_report:
    station_name = record["station"]
    if record["unreliable"]:
        continue

    series = station_series[station_name]
    if np.any(np.isnan(series)) or len(series) < 10:
        continue

    # Ajustar GEV
    c, l, s = np.nan, np.nan, np.nan
    success = False

    try:
        c, l, s = genextreme.fit(series)
        if np.isfinite(c) and np.isfinite(l) and np.isfinite(s) and s > 0:
            success = True
    except:
        pass

    if not success:
        try:
            from lmoments3 import distr
            lmom = distr.gev.lmom_fit(series)
            c, l, s = lmom['c'], lmom['loc'], lmom['scale']
            if np.isfinite(c) and np.isfinite(l) and np.isfinite(s) and s > 0:
                success = True
        except:
            pass

    if not success:
        continue

    gev_results.append({
        "station": station_name,
        "shape": c,
        "loc": l,
        "scale": s
    })

    # Tests de bondad de ajuste
    try:
        ks_stat, ks_p = kstest(series, lambda x: genextreme.cdf(x, c, loc=l, scale=s))
    except:
        ks_p = np.nan

    try:
        ad_result = anderson(series, dist='genextreme')
        ad_p = 0.01
        for idx, cv in enumerate(ad_result.critical_values):
            if ad_result.statistic > cv:
                ad_p = [0.15, 0.10, 0.05, 0.025, 0.01][idx]
                break
    except:
        ad_p = np.nan

    goodness_results.append({
        "station": station_name,
        "ks_pvalue": ks_p,
        "ad_pvalue": ad_p,
        "ks_rejected": ks_p < ALPHA if not np.isnan(ks_p) else False,
        "ad_rejected": ad_p < ALPHA if not np.isnan(ad_p) else False
    })

df_gev = pd.DataFrame(gev_results)
df_goodness = pd.DataFrame(goodness_results)

df_gev.to_csv(OUT_DIR_STATIONS / "stations_gev_parameters.csv", index=False)
df_goodness.to_csv(OUT_DIR_STATIONS / "stations_goodness_of_fit.csv", index=False)

print(f"✅ Parámetros GEV guardados: stations_gev_parameters.csv")
print(f"✅ Bondad de ajuste guardada: stations_goodness_of_fit.csv")

# %% [Celda 18: Calcular precipitaciones para periodos de retorno]

print("=== CALCULANDO PRECIPITACIONES PARA PERIODOS DE RETORNO ===")

results_T = []

for idx, row in df_gev.iterrows():
    station_name = row["station"]
    c, l, s = row["shape"], row["loc"], row["scale"]

    for T in PERIODOS_RETORNO:
        p = 1 - 1/T
        try:
            precip_T = genextreme.ppf(p, c, loc=l, scale=s)
            if np.isfinite(precip_T):
                results_T.append({
                    "station": station_name,
                    "return_period": T,
                    "precipitation_mm": precip_T
                })
        except:
            continue

df_return_periods = pd.DataFrame(results_T)
df_return_periods.to_csv(OUT_DIR_STATIONS / "stations_return_periods.csv", index=False)

print(f"🎉 ¡ANÁLISIS COMPLETADO!")
print(f"📊 Resultados finales guardados en: {OUT_DIR_STATIONS}")
# %% [Celda 19: Instalar dependencias para Kriging]

%pip install --quiet pykrige rasterio numpy pandas tqdm

print("✅ Dependencias para Kriging instaladas.")

# %% [Celda 20: Importar librerías para interpolación espacial]

import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm
from pathlib import Path

print("✅ Librerías para interpolación cargadas.")

# %% [Celda 21: Configurar rutas y parámetros]

# Rutas
INPUT_EXCEL_COMPLETED = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA\Andelien_stations_COMPLETED.xlsx")
RETURN_PERIODS_FILE = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA\stations_return_periods.csv")
MASK_PATH = Path(r"C:\GIS\klak1996\SIGHD\Tetis\Dem_fill.asc")
OUTPUT_DIR = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA\KRIGING_MAPS")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODOS_RETORNO = [2, 5, 10, 20, 25, 50, 100]

print("✅ Rutas y parámetros configurados.")

# %% [Celda 22: Cargar datos de retorno y coordenadas]

# Cargar el archivo de períodos de retorno
df_return = pd.read_csv(RETURN_PERIODS_FILE)

# Cargar coordenadas desde all_stations_full_data (ya lo tenemos del script anterior)
station_coords_dict = {}
for name in df_return['station'].unique():
    if name in all_stations_full_data:
        station_coords_dict[name] = {
            'x': all_stations_full_data[name]['utm_e'],
            'y': all_stations_full_data[name]['utm_n']
        }

# Crear DataFrame final con coordenadas
data_list = []
for _, row in df_return.iterrows():
    station = row['station']
    if station in station_coords_dict:
        data_list.append({
            'station': station,
            'x': station_coords_dict[station]['x'],
            'y': station_coords_dict[station]['y'],
            'return_period': row['return_period'],
            'precipitation_mm': row['precipitation_mm']
        })

df_points = pd.DataFrame(data_list)
print(f"✅ Datos cargados: {len(df_points)} registros de {df_points['station'].nunique()} estaciones.")

# %% [Celda 23: Cargar máscara de referencia y crear grilla]

with rasterio.open(MASK_PATH) as src:
    ref_transform = src.transform
    ref_width = src.width
    ref_height = src.height
    ref_crs = src.crs if src.crs else CRS.from_epsg(32718)

    # Crear grilla de coordenadas
    rows, cols = np.meshgrid(np.arange(ref_height), np.arange(ref_width), indexing='ij')
    xs, ys = rasterio.transform.xy(ref_transform, rows.flatten(), cols.flatten(), offset='center')
    grid_x = np.array(xs)
    grid_y = np.array(ys)

print(f"✅ Grilla de referencia creada con {len(grid_x)} puntos.")

# %% [Celda 24: Generar mapas TIFF con Kriging Ordinario]

print("=== GENERANDO MAPAS CON KRIGING ORDINARIO ===")

for T in tqdm(PERIODOS_RETORNO, desc="Periodos de Retorno"):
    # Filtrar datos para este período de retorno
    df_T = df_points[df_points['return_period'] == T]
    
    if len(df_T) < 3:
        print(f"⚠️  Insuficientes puntos para T={T}. Saltando.")
        continue
    
    x_obs = df_T['x'].values
    y_obs = df_T['y'].values
    z_obs = df_T['precipitation_mm'].values
    
    try:
        # Crear modelo de Kriging Ordinario
        OK = OrdinaryKriging(
            x_obs, y_obs, z_obs,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        
        # Interpolar en la grilla completa
        z_interp, ss = OK.execute('points', grid_x, grid_y)
        z_interp_2d = z_interp.reshape((ref_height, ref_width))
        
        # Guardar como TIFF
        output_meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'crs': ref_crs,
            'transform': ref_transform,
            'compress': 'LZW'
        }
        
        output_path = OUTPUT_DIR / f"Pmax_T{T:02d}_Kriging.tif"
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(z_interp_2d, 1)
        
        print(f"✅ Mapa guardado: {output_path}")
        
    except Exception as e:
        print(f"❌ Error al generar mapa para T={T}: {e}")

print(f"\n🎉 ¡PROCESO DE INTERPOLACIÓN COMPLETADO!")
print(f"Los mapas se guardaron en: {OUTPUT_DIR}")

# %% [Celda 25: Evaluación POR ESTACIÓN — GEV vs CR2MET (curvas de crecimiento)]
import os
import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# --- CONFIGURACIÓN ---
GEV_FILE = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA\stations_return_periods.csv")
CR2MET_RASTERS_DIR = Path(r"C:\Python\klak1996\cr2met_analize\INTERPOLATED_MAPS")
OUTPUT_DIR = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- CARGAR VALORES GEV ---
print("📊 Cargando valores GEV por estación y período de retorno...")
df_gev = pd.read_csv(GEV_FILE)
df_gev['return_period'] = df_gev['return_period'].astype(int)

# --- OBTENER COORDENADAS ---
print("📍 Asignando coordenadas UTM a las estaciones...")
station_coords = {}
for station in df_gev['station'].unique():
    if station in all_stations_full_data:
        station_coords[station] = (
            all_stations_full_data[station]['utm_e'],
            all_stations_full_data[station]['utm_n']
        )

df_gev['x'] = df_gev['station'].map(lambda s: station_coords.get(s, (None, None))[0])
df_gev['y'] = df_gev['station'].map(lambda s: station_coords.get(s, (None, None))[1])
df_gev = df_gev.dropna(subset=['x', 'y']).reset_index(drop=True)

# --- IDENTIFICAR RASTERS ---
print("\n🗺️ Identificando rasters de CR2MET...")
raster_files = list(CR2MET_RASTERS_DIR.glob("Pmax_T*_Interpolado_Kriging_Ordinario.tif"))
T_to_raster = {}
for f in raster_files:
    match = re.search(r'Pmax_T(\d+)_Interpolado_Kriging_Ordinario\.tif', f.name)
    if match:
        T = int(match.group(1))
        T_to_raster[T] = f

if not T_to_raster:
    raise FileNotFoundError("No se encontraron rasters con el patrón esperado.")

# --- EXTRAER VALORES DE CR2MET POR ESTACIÓN ---
print("\n🔍 Extrayendo valores de CR2MET en ubicaciones de estaciones...")
all_data = []

# Crear un diccionario para almacenar valores por estación
station_data = {}

# Inicializar
for station in df_gev['station'].unique():
    station_data[station] = {
        'x': station_coords[station][0],
        'y': station_coords[station][1],
        'gev_values': {},
        'cr2met_values': {}
    }

# Llenar valores GEV
for _, row in df_gev.iterrows():
    station_data[row['station']]['gev_values'][row['return_period']] = row['precipitation_mm']

# Extraer valores CR2MET
for T, raster_path in T_to_raster.items():
    print(f"  Extrayendo T={T} desde {raster_path.name}...")
    stations_for_T = [s for s, d in station_data.items() if T in d['gev_values']]
    if not stations_for_T:
        continue
    points = [(station_data[s]['x'], station_data[s]['y']) for s in stations_for_T]
    try:
        with rasterio.open(raster_path) as src:
            for i, val in enumerate(src.sample(points)):
                v = val[0]
                station = stations_for_T[i]
                if np.isnan(v) or (src.nodata is not None and v == src.nodata):
                    station_data[station]['cr2met_values'][T] = np.nan
                else:
                    station_data[station]['cr2met_values'][T] = float(v)
    except Exception as e:
        print(f"    ❌ Error en T={T}: {e}")
        for s in stations_for_T:
            station_data[s]['cr2met_values'][T] = np.nan

# --- CALCULAR MÉTRICAS POR ESTACIÓN ---
print("\n📈 Calculando métricas POR ESTACIÓN (comparando curvas de crecimiento)...")
metrics_per_station = []

for station, data in station_data.items():
    # Obtener períodos comunes con datos válidos
    gev_vals = []
    cr2met_vals = []
    common_Ts = []

    for T in sorted(T_to_raster.keys()):
        if T in data['gev_values'] and T in data['cr2met_values']:
            gev = data['gev_values'][T]
            cr2met = data['cr2met_values'][T]
            if not (np.isnan(gev) or np.isnan(cr2met)):
                gev_vals.append(gev)
                cr2met_vals.append(cr2met)
                common_Ts.append(T)

    if len(gev_vals) < 2:
        print(f"  ⚠️ {station}: Insuficientes datos ({len(gev_vals)})")
        continue

    y_true = np.array(gev_vals)   # GEV
    y_pred = np.array(cr2met_vals)  # CR2MET

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    bias = np.mean(y_pred - y_true)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    mape = np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6)) * 100

    metrics_per_station.append({
        'station': station,
        'n_periods': len(gev_vals),
        'periods_used': ','.join(map(str, common_Ts)),
        'MAE_mm': mae,
        'RMSE_mm': rmse,
        'Bias_mm': bias,
        'R2': r2,
        'MAPE_%': mape
    })

    print(f"  📊 {station} | n={len(gev_vals)} | MAE={mae:6.2f} | RMSE={rmse:6.2f} | Bias={bias:6.2f} | R²={r2:6.3f}")

# --- GUARDAR RESULTADOS ---
if metrics_per_station:
    df_metrics = pd.DataFrame(metrics_per_station)
    df_metrics.to_csv(OUTPUT_DIR / "station_level_validation_metrics.csv", index=False)
    print(f"\n✅ Métricas por estación guardadas: {OUTPUT_DIR / 'station_level_validation_metrics.csv'}")
else:
    print("\n❌ No se calcularon métricas para ninguna estación.")

# Opcional: guardar datos detallados por estación y período
detail_rows = []
for station, data in station_data.items():
    for T in sorted(T_to_raster.keys()):
        if T in data['gev_values'] and T in data['cr2met_values']:
            detail_rows.append({
                'station': station,
                'return_period': T,
                'gev_value': data['gev_values'][T],
                'cr2met_value': data['cr2met_values'][T],
                'error': data['cr2met_values'][T] - data['gev_values'][T] if not (np.isnan(data['cr2met_values'][T]) or np.isnan(data['gev_values'][T])) else np.nan
            })

df_detail = pd.DataFrame(detail_rows)
df_detail.to_csv(OUTPUT_DIR / "station_level_validation_details.csv", index=False)
print(f"✅ Detalles por estación y período guardados: {OUTPUT_DIR / 'station_level_validation_details.csv'}")

print("\n🎉 Evaluación completada: Métricas POR ESTACIÓN (curvas de crecimiento GEV vs CR2MET).")
# %% [Celda 26: Visualización de curvas de crecimiento por estación]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
OUTPUT_DIR = Path(r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA")
DETAILS_FILE = OUTPUT_DIR / "station_level_validation_details.csv"
METRICS_FILE = OUTPUT_DIR / "station_level_validation_metrics.csv"

# --- CARGAR DATOS ---
print("📊 Cargando datos para visualización...")
df_detail = pd.read_csv(DETAILS_FILE)
df_metrics = pd.read_csv(METRICS_FILE)

# Asegurar tipo de dato
df_detail['return_period'] = df_detail['return_period'].astype(int)

# --- CREAR GRÁFICOS ---
print("🖼️ Generando gráficos de curvas de crecimiento...")

# Configurar estilo
plt.rcParams.update({'font.size': 12})

# Obtener lista de estaciones
stations = df_metrics['station'].tolist()
n_stations = len(stations)

# Crear subgráficos (una fila por estación)
fig, axes = plt.subplots(n_stations, 1, figsize=(10, 5 * n_stations), sharex=True)
if n_stations == 1:
    axes = [axes]

for ax, station in zip(axes, stations):
    # Filtrar datos de la estación
    df_sta = df_detail[df_detail['station'] == station].copy()
    df_sta = df_sta.sort_values('return_period')
    
    T_vals = df_sta['return_period'].values
    gev_vals = df_sta['gev_value'].values
    cr2met_vals = df_sta['cr2met_value'].values
    
    # Graficar
    ax.plot(T_vals, gev_vals, 'o-', color='tab:blue', linewidth=2, markersize=8, label='GEV (Estación)')
    ax.plot(T_vals, cr2met_vals, 's--', color='tab:red', linewidth=2, markersize=8, label='CR2MET (Raster)')
    
    # Escala logarítmica en X
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Etiquetas y título
    ax.set_ylabel('Precipitación (mm)', fontsize=13)
    ax.set_title(f'Curva de Crecimiento — Estación: {station}', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=12)
    
    # Añadir métricas como texto en el gráfico
    metrics_row = df_metrics[df_metrics['station'] == station].iloc[0]
    metrics_text = (
        f"MAE: {metrics_row['MAE_mm']:.1f} mm\n"
        f"Bias: {metrics_row['Bias_mm']:.1f} mm\n"
        f"R²: {metrics_row['R2']:.3f}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# Etiqueta común en X
axes[-1].set_xlabel('Período de Retorno (años)', fontsize=13)

# Ajustar diseño
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Guardar figura
output_fig = OUTPUT_DIR / "curvas_crecimiento_GEV_vs_CR2MET.png"
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"✅ Gráfico guardado: {output_fig}")

# Mostrar en notebook
plt.show()

print("\n🎉 Visualización completada.")

# %%
