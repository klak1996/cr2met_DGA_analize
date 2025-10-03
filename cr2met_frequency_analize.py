#%% 0) Instalar Dependencias
# Ejecuta esta celda primero.

%pip install --quiet xarray rioxarray rasterio pyproj numpy scipy tqdm statsmodels netCDF4 pymannkendall dask lmoments3

#%% 1) Importar Librerías Necesarias

import sys
import pathlib
import xarray as xr
import rioxarray as riox
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genextreme
from scipy.spatial.distance import cdist
import pymannkendall as mk
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
print("Python:", sys.executable)
print("Working dir:", pathlib.Path().resolve())

#%% 2) Configuración de Rutas y Parámetros

IN_DIR = Path(r"C:\Python\klak1996\cr2met_analize\pr_utm18S")
OUT_DIR = Path(r"C:\Python\klak1996\cr2met_analize\FREQUENCY_ANALYSIS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== CONFIGURACIÓN ===")
print(f"Entrada: {IN_DIR}")
print(f"Salida:  {OUT_DIR}")

PERIODOS_RETORNO = [2, 5, 10, 20, 25, 50, 100]
ALPHA = 0.05  # ✅ Valor estándar en hidrología
IDW_POWER = 2
START_DATE = "1990-01-01"
END_DATE = "2020-12-31"

nc_files = sorted(IN_DIR.glob("CR2MET_pr_*.nc"))
if len(nc_files) == 0:
    raise FileNotFoundError(f"No se encontraron archivos .nc en {IN_DIR}")
print(f"✅ Encontrados {len(nc_files)} archivos de precipitación.")

#%% 3) Cargar, Filtrar y Calcular Máximos Anuales

print("=== CARGANDO Y FILTRANDO DATOS ===")

ds = xr.open_mfdataset(nc_files, combine='by_coords', engine='netcdf4')

if {"lat", "lon"}.issubset(ds.dims):
    ds = ds.rename({'lat': 'y', 'lon': 'x'})
if ds.rio.crs is None:
    ds = ds.rio.write_crs("EPSG:32718", inplace=True)

var_name = [v for v in ds.data_vars if v not in ['spatial_ref', 'crs']][0]
da_daily = ds[var_name]

if {"x", "y"}.issubset(da_daily.dims):
    da_daily = da_daily.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

# Filtrar por fechas
da_daily = da_daily.sel(time=slice(START_DATE, END_DATE))

# Calcular máximos anuales
da_annual_max = da_daily.resample(time='A').max(dim='time')
annual_max = da_annual_max.values  # (n_years, n_y, n_x)
n_years, n_y, n_x = annual_max.shape

print(f"✅ Máximos anuales: {n_years} años ({START_DATE[:4]}-{END_DATE[:4]}), {n_y}x{n_x} celdas.")
ds.close()

#%% 4) Rellenar Valores Faltantes (IDW Espacial)

print("=== RELLENANDO NaN CON IDW ===")

yy, xx = np.meshgrid(da_annual_max.y.values, da_annual_max.x.values, indexing='ij')
coords = np.column_stack([xx.ravel(), yy.ravel()])
filled_count = 0

for t in tqdm(range(n_years), desc="Rellenando por año"):
    data = annual_max[t, :, :].copy()
    nan_mask = np.isnan(data)
    if not np.any(nan_mask): continue

    valid_mask = ~nan_mask
    valid_points = coords[valid_mask.ravel()]
    valid_values = data[valid_mask]

    if len(valid_values) == 0:
        mean_val = np.nanmean(annual_max[t, :, :])
        annual_max[t, :, :] = np.where(nan_mask, mean_val, data)
        filled_count += np.sum(nan_mask)
        continue

    nan_points = coords[nan_mask.ravel()]
    distances = cdist(nan_points, valid_points, 'euclidean')
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (distances ** IDW_POWER)
    weighted_sum = np.dot(weights, valid_values)
    sum_weights = np.sum(weights, axis=1)
    interpolated_values = weighted_sum / sum_weights

    data[nan_mask] = interpolated_values
    annual_max[t, :, :] = data
    filled_count += len(interpolated_values)

print(f"✅ Valores rellenados: {filled_count}")

#%% 5) Detectar y Corregir Outliers (Test de Grubbs)

print("=== CORRIGIENDO OUTLIERS ===")

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
for i in tqdm(range(n_y), desc="Filas"):
    for j in range(n_x):
        series = annual_max[:, i, j]
        if np.any(np.isnan(series)) or len(series) < 3: continue

        sorted_series = np.sort(series)[::-1]
        second_max = sorted_series[1] if len(sorted_series) > 1 else sorted_series[0]

        outlier_index = grubbs_test(series, alpha=0.05)
        if outlier_index is not None:
            annual_max[outlier_index, i, j] = second_max
            outliers_corrected += 1

print(f"✅ Outliers corregidos: {outliers_corrected}")

#%% 6) ANÁLISIS DE CALIDAD DE DATOS: Aleatoriedad, Independencia, Homogeneidad, Estacionariedad

print("=== ANÁLISIS DE CALIDAD: 4 TESTS FUNDAMENTALES (α=0.05) ===")
print("Tests aplicados: Aleatoriedad (Rachas), Independencia (Durbin-Watson), Homogeneidad (SNHT), Estacionariedad (Mann-Kendall)")

from statsmodels.stats.stattools import durbin_watson

detailed_report = []

for i in tqdm(range(n_y), desc="Celdas"):
    for j in range(n_x):
        series = annual_max[:, i, j]
        if np.any(np.isnan(series)) or len(series) < 10:
            continue

        record = {"row": i, "col": j, "n_years": len(series)}

        # --- 1. TEST DE ALEATORIEDAD (Rachas - Wald-Wolfowitz) ---
        try:
            median_val = np.median(series)
            binary_seq = (series > median_val).astype(int)
            runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1])
            n1 = np.sum(binary_seq)
            n0 = len(binary_seq) - n1

            if n1 == 0 or n0 == 0:
                raise ValueError("Serie constante respecto a la mediana.")

            mean_runs = 1 + (2 * n1 * n0) / (n1 + n0)
            var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
            std_runs = np.sqrt(var_runs)
            if std_runs == 0: raise ValueError("Varianza cero.")

            z_stat = (runs - mean_runs) / std_runs
            p_value_runs = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            reject_H0_random = p_value_runs < ALPHA

            record.update({
                "runs_statistic": z_stat,
                "runs_p_value": p_value_runs,
                "runs_critical_value": stats.norm.ppf(1 - ALPHA/2),
                "reject_H0_randomness": reject_H0_random
            })
        except Exception:
            record.update({
                "runs_statistic": np.nan,
                "runs_p_value": np.nan,
                "runs_critical_value": np.nan,
                "reject_H0_randomness": True
            })

        # --- 2. TEST DE INDEPENDENCIA (Durbin-Watson) ---
        try:
            residuals = series - np.mean(series)
            dw_stat = durbin_watson(residuals)
            DL, DU = 1.36, 1.49  # Valores críticos para n=31, k=1, α=0.05
            reject_H0_indep = (dw_stat < DL) or (dw_stat > (4 - DL))

            record.update({
                "dw_statistic": dw_stat,
                "dw_p_value": np.nan,
                "dw_critical_low": DL,
                "dw_critical_high": 4 - DL,
                "reject_H0_independence": reject_H0_indep
            })
        except Exception:
            record.update({
                "dw_statistic": np.nan,
                "dw_p_value": np.nan,
                "dw_critical_low": np.nan,
                "dw_critical_high": np.nan,
                "reject_H0_independence": True
            })

        # --- 3. TEST DE HOMOGENEIDAD (SNHT) ---
        try:
            n = len(series)
            standardized_series = (series - np.mean(series)) / np.std(series, ddof=1)
            max_snht_stat = 0
            for k in range(1, n):
                mean1 = np.mean(standardized_series[:k])
                mean2 = np.mean(standardized_series[k:])
                snht_stat = (k * mean1**2 + (n - k) * mean2**2) / n
                if snht_stat > max_snht_stat:
                    max_snht_stat = snht_stat

            snht_statistic = max_snht_stat
            critical_value_snht = 20.0  # Valor crítico para n≈30, α=0.05
            p_value_snht = stats.chi2.sf(snht_statistic, df=1)
            reject_H0_homogeneity = snht_statistic > critical_value_snht

            record.update({
                "snht_statistic": snht_statistic,
                "snht_p_value": p_value_snht,
                "snht_critical_value": critical_value_snht,
                "reject_H0_homogeneity": reject_H0_homogeneity
            })
        except Exception:
            record.update({
                "snht_statistic": np.nan,
                "snht_p_value": np.nan,
                "snht_critical_value": np.nan,
                "reject_H0_homogeneity": True
            })

        # --- 4. TEST DE ESTACIONARIEDAD (Mann-Kendall) ---
        try:
            mk_result = mk.original_test(series)
            reject_H0_stationarity = mk_result.p < ALPHA

            record.update({
                "mk_statistic": mk_result.Tau,
                "mk_p_value": mk_result.p,
                "mk_critical_value": np.nan,
                "reject_H0_stationarity": reject_H0_stationarity
            })
        except Exception:
            record.update({
                "mk_statistic": np.nan,
                "mk_p_value": np.nan,
                "mk_critical_value": np.nan,
                "reject_H0_stationarity": True
            })

        # --- DECISIÓN FINAL: ¿Es la serie CONFIABLE? ---
        is_unreliable = (
            record["reject_H0_randomness"] or
            record["reject_H0_independence"] or
            record["reject_H0_homogeneity"] or
            record["reject_H0_stationarity"]
        )
        record["unreliable"] = is_unreliable
        detailed_report.append(record)

# Convertir a DataFrame
df_detailed_quality = pd.DataFrame(detailed_report)
total_valid = len(detailed_report)
unreliable_count = df_detailed_quality["unreliable"].sum()

print(f"\n📊 RESULTADO DEL ANÁLISIS DE CALIDAD (4 TESTS, α=0.05):")
print(f"Total de celdas válidas: {total_valid}")
print(f"Celdas no confiables: {unreliable_count} ({unreliable_count/total_valid*100:.1f}%)")
print(f" → Debido a fallar en al menos una prueba: Aleatoriedad, Independencia, Homogeneidad o Estacionariedad")

# Guardar reporte detallado
df_detailed_quality.to_csv(OUT_DIR / "detailed_quality_report_4tests.csv", index=False)
print(f"📄 Reporte detallado de calidad (4 tests) guardado: {OUT_DIR / 'detailed_quality_report_4tests.csv'}")

# Mostrar estadísticas resumidas por prueba
print(f"\n📈 RESUMEN POR PRUEBA:")
for test_name, col_name in [
    ("Aleatoriedad (Rachas)", "reject_H0_randomness"),
    ("Independencia (Durbin-Watson)", "reject_H0_independence"),
    ("Homogeneidad (SNHT)", "reject_H0_homogeneity"),
    ("Estacionariedad (Mann-Kendall)", "reject_H0_stationarity")
]:
    failed_count = df_detailed_quality[col_name].sum()
    print(f"   {test_name}: {failed_count} celdas fallidas ({failed_count/total_valid*100:.1f}%)")

#%% 7) Ajustar Distribución GEV por Celda (SOLO series confiables)

print("=== AJUSTANDO DISTRIBUCIÓN GEV (SOLO SERIES CONFIABLES) ===")

shape = np.full((n_y, n_x), np.nan)
loc = np.full((n_y, n_x), np.nan)
scale = np.full((n_y, n_x), np.nan)

# Crear máscara de confiabilidad
reliable_mask = np.full((n_y, n_x), False)
for idx, row in df_detailed_quality.iterrows():
    i, j = int(row['row']), int(row['col'])
    reliable_mask[i, j] = not row['unreliable']

gev_success_count = 0

for i in tqdm(range(n_y), desc="Ajustando GEV"):
    for j in range(n_x):
        if not reliable_mask[i, j]:
            continue

        series = annual_max[:, i, j]
        if np.any(np.isnan(series)) or len(series) < 10:
            continue

        success = False
        try:
            c, l, s = genextreme.fit(series)
            if np.isfinite(c) and np.isfinite(l) and np.isfinite(s) and s > 0:
                shape[i, j], loc[i, j], scale[i, j] = c, l, s
                success = True
        except:
            pass

        if not success:
            try:
                from lmoments3 import distr
                lmom = distr.gev.lmom_fit(series)
                c, l, s = lmom['c'], lmom['loc'], lmom['scale']
                if np.isfinite(c) and np.isfinite(l) and np.isfinite(s) and s > 0:
                    shape[i, j], loc[i, j], scale[i, j] = c, l, s
                    success = True
            except:
                pass

        if success:
            gev_success_count += 1

print(f"✅ Ajuste GEV completado. Éxito en {gev_success_count} celdas confiables.")

#%% 7.1) TEST DE BONDAD DE AJUSTE (Kolmogorov-Smirnov y Anderson-Darling)

print("=== APLICANDO TESTS DE BONDAD DE AJUSTE ===")

from scipy.stats import kstest, anderson

ks_pvalues = np.full((n_y, n_x), np.nan)
ad_pvalues = np.full((n_y, n_x), np.nan)
ks_rejected = 0
ad_rejected = 0
total_valid_gof = 0

for i in tqdm(range(n_y), desc="Aplicando tests de bondad de ajuste"):
    for j in range(n_x):
        if not reliable_mask[i, j]:
            continue

        series = annual_max[:, i, j]
        if np.any(np.isnan(series)) or len(series) < 10:
            continue

        c = shape[i, j]
        loc_param = loc[i, j]
        scale_param = scale[i, j]

        if not (np.isfinite(c) and np.isfinite(loc_param) and np.isfinite(scale_param) and scale_param > 0):
            continue

        total_valid_gof += 1

        try:
            ks_stat, ks_p = kstest(series, lambda x: genextreme.cdf(x, c, loc=loc_param, scale=scale_param))
            ks_pvalues[i, j] = ks_p
            if ks_p < ALPHA:
                ks_rejected += 1
        except:
            ks_pvalues[i, j] = np.nan

        try:
            ad_result = anderson(series, dist='genextreme')
            critical_values = ad_result.critical_values
            significance_levels = [15, 10, 5, 2.5, 1]
            ad_stat = ad_result.statistic
            ad_p = 0.01
            for idx, cv in enumerate(critical_values):
                if ad_stat > cv:
                    ad_p = significance_levels[idx] / 100.0
                    break
            ad_pvalues[i, j] = ad_p
            if ad_p < ALPHA:
                ad_rejected += 1
        except:
            ad_pvalues[i, j] = np.nan

print(f"\n📊 RESULTADOS DE BONDAD DE AJUSTE (α={ALPHA}):")
print(f"Total de celdas válidas (confiables y ajustadas): {total_valid_gof}")
print(f"Celdas rechazadas por Kolmogorov-Smirnov (p < {ALPHA}): {ks_rejected} ({ks_rejected/total_valid_gof*100:.1f}%)")
print(f"Celdas rechazadas por Anderson-Darling (p < {ALPHA}): {ad_rejected} ({ad_rejected/total_valid_gof*100:.1f}%)")

# Guardar informe de bondad de ajuste.
goodness_report = []
for i in range(n_y):
    for j in range(n_x):
        if not np.isnan(ks_pvalues[i, j]) or not np.isnan(ad_pvalues[i, j]):
            goodness_report.append({
                "row": i,
                "col": j,
                "ks_pvalue": ks_pvalues[i, j],
                "ad_pvalue": ad_pvalues[i, j],
                "ks_rejected": ks_pvalues[i, j] < ALPHA if not np.isnan(ks_pvalues[i, j]) else False,
                "ad_rejected": ad_pvalues[i, j] < ALPHA if not np.isnan(ad_pvalues[i, j]) else False
            })

df_goodness = pd.DataFrame(goodness_report)
df_goodness.to_csv(OUT_DIR / "goodness_of_fit_report.csv", index=False)
print(f"📄 Informe de bondad de ajuste guardado: {OUT_DIR / 'goodness_of_fit_report.csv'}")

# Generar mapas de p-valores.
da_template = da_annual_max.isel(time=0).copy()

da_ks = da_template.copy()
da_ks.values = ks_pvalues
da_ks.rio.to_raster(OUT_DIR / "KS_pvalues.tif", driver="GTiff", compress="LZW", dtype="float32", nodata=np.nan)

da_ad = da_template.copy()
da_ad.values = ad_pvalues
da_ad.rio.to_raster(OUT_DIR / "AD_pvalues.tif", driver="GTiff", compress="LZW", dtype="float32", nodata=np.nan)

print(f"✅ Mapas de p-valores generados: KS_pvalues.tif, AD_pvalues.tif")

#%% 8) Generar Mapas TIFF para Periodos de Retorno

print("=== GENERANDO MAPAS DE FRECUENCIA ===")

da_template = da_annual_max.isel(time=0).copy()

for T in PERIODOS_RETORNO:
    print(f"Generando T={T} años...")
    p = 1 - 1/T
    precip_T = genextreme.ppf(p, c=shape, loc=loc, scale=scale)

    da_T = da_template.copy()
    da_T.values = precip_T

    out_path = OUT_DIR / f"Pmax_T{T:02d}.tif"
    da_T.rio.to_raster(
        out_path,
        driver="GTiff",
        compress="LZW",
        dtype="float32",
        nodata=np.nan
    )

print(f"\n🎉 ¡PROCESO COMPLETADO!")
print(f"Mapas generados en: {OUT_DIR}")

# %%
