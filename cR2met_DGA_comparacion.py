#%% [markdown]
# 🌍 **ANÁLISIS COMPARATIVO ROBUSTO: CR2MET vs DGA (3 ESTACIONES)**
## 📊 Comparación pixel a pixel para 7 períodos de retorno (T=2,5,10,20,25,50,100 años)
# ✅ Totalmente automático — Instala dependencias, carga mapas, calcula métricas, genera gráficos y reportes.
# 🧭 Diseñado para Jupyter con celdas `#%%` — Ejecuta bloque por bloque.
# 📁 Rutas predefinidas — Solo asegúrate que tus mapas estén donde deben.

#%% 
# 📦 Celda 1: Instalar dependencias automáticamente (solo si es necesario)
import sys
import subprocess
import pkg_resources

def install_packages():
    required = {'rasterio', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'scikit-image', 'tqdm'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        print(f"📦 Instalando paquetes faltantes: {missing}")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        print("✅ Todos los paquetes instalados correctamente.")
    else:
        print("✅ Todos los paquetes ya están instalados.")

install_packages()

#%% 
# 🧰 Celda 2: Importar librerías y configurar entorno (¡CORREGIDA!)
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error  # ¡CORREGIDO AQUÍ!
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuración visual global
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

print("✅ Librerías cargadas y entorno configurado.")

#%% 
# 📁 Celda 3: Definir rutas de trabajo y crear directorios de salida
CR2MET_MAPS = r"C:\Python\klak1996\cr2met_analize\INTERPOLATED_MAPS"
DGA_MAPS = r"C:\Python\klak1996\cr2met_analize\ESTACIONES_DGA\KRIGING_MAPS"
OUTPUT_DIR = r"C:\Python\klak1996\cr2met_analize\COMPARATION"

# Crear estructura de carpetas
subfolders = ["METRICS", "DIFF_MAPS", "SCATTER_PLOTS", "HISTOGRAMS", "ERROR_MAPS", "REPORTES"]
for folder in subfolders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

RETURN_PERIODS = [2, 5, 10, 20, 25, 50, 100]

print(f"📂 Carpeta de entrada CR2MET: {CR2MET_MAPS}")
print(f"📂 Carpeta de entrada DGA: {DGA_MAPS}")
print(f"📂 Carpeta de salida: {OUTPUT_DIR}")
print(f"🔢 Períodos de retorno: {RETURN_PERIODS}")

#%% 
# 🔍 Celda 4 (CORREGIDA): Funciones robustas para encontrar archivos por período de retorno
import re

def find_cr2met_file(period):
    pattern = f"Pmax_T{period}.*Interpolado_Kriging_Ordinario.*\\.tif$"
    for f in os.listdir(CR2MET_MAPS):
        if re.search(pattern, f, re.IGNORECASE):
            return os.path.join(CR2MET_MAPS, f)
    # Si no lo encuentra, listar todos los archivos para diagnóstico
    available_files = [f for f in os.listdir(CR2MET_MAPS) if f.endswith(".tif")]
    print(f"⚠️ Archivos disponibles en CR2MET_MAPS: {available_files}")
    raise FileNotFoundError(f"❌ No se encontró mapa CR2MET para T={period} con patrón '{pattern}' en {CR2MET_MAPS}")

def find_dga_file(period):
    pattern = f"Pmax_T{period}.*Kriging.*\\.tif$"
    for f in os.listdir(DGA_MAPS):
        if re.search(pattern, f, re.IGNORECASE):
            return os.path.join(DGA_MAPS, f)
    # Si no lo encuentra, listar todos los archivos para diagnóstico
    available_files = [f for f in os.listdir(DGA_MAPS) if f.endswith(".tif")]
    print(f"⚠️ Archivos disponibles en DGA_MAPS: {available_files}")
    raise FileNotFoundError(f"❌ No se encontró mapa DGA para T={period} con patrón '{pattern}' en {DGA_MAPS}")

# Verificación rápida de disponibilidad de archivos
print("🔍 Verificando disponibilidad de archivos...")
for period in RETURN_PERIODS:
    try:
        cr2met_file = find_cr2met_file(period)
        dga_file = find_dga_file(period)
        print(f"✅ T={period}: CR2MET='{os.path.basename(cr2met_file)}', DGA='{os.path.basename(dga_file)}'")
    except FileNotFoundError as e:
        print(f"❌ ERROR en T={period}: {e}")
        continue

print("✅ Verificación completada.")

#%% 
# 🧩 Celda 5: Función para cargar y validar mapas (pixel a pixel)
def load_and_validate_rasters(path1, path2):
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        # Validar alineación espacial
        if src1.shape != src2.shape:
            raise ValueError(f"Formas no coinciden: {src1.shape} vs {src2.shape}")
        if src1.transform != src2.transform:
            raise ValueError("Transformaciones espaciales no coinciden")
        if src1.crs != src2.crs:
            raise ValueError("SRC no coincide")

        # Leer banda 1 como float32
        arr1 = src1.read(1).astype(np.float32)
        arr2 = src2.read(1).astype(np.float32)

        # Enmascarar valores nulos (asumiendo -9999 o NaN como nulos)
        arr1 = np.where((arr1 == -9999) | np.isnan(arr1), np.nan, arr1)
        arr2 = np.where((arr2 == -9999) | np.isnan(arr2), np.nan, arr2)

        # Máscara de píxeles válidos
        valid_mask = ~(np.isnan(arr1) | np.isnan(arr2))
        if not np.any(valid_mask):
            raise ValueError("No hay píxeles válidos para comparar")

        # Metadatos para guardar mapas de salida
        meta = src1.meta.copy()
        meta.update(dtype=rasterio.float32, nodata=np.nan)

        return arr1, arr2, valid_mask, meta

print("✅ Función de carga y validación lista.")

#%% 
# 📐 Celda 6: Funciones de métricas matemáticas robustas
def calculate_metrics(y_true, y_pred, mask):
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Sesgo (MBE)
    mbe = np.mean(y_pred - y_true)

    # Pearson r y R²
    r, _ = pearsonr(y_true, y_pred)
    r2 = r**2

    # Willmott d (índice de acuerdo)
    d = 1 - (np.sum((y_pred - y_true)**2) /
             np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2))

    # MAPE (evitando división por cero)
    nonzero_mask = y_true > 1.0  # umbral para evitar errores
    mape = np.mean(np.abs((y_pred[nonzero_mask] - y_true[nonzero_mask]) / y_true[nonzero_mask])) * 100 if np.sum(nonzero_mask) > 0 else np.nan

    return {
        "Periodo": None,  # se asignará después
        "MAE": mae,
        "RMSE": rmse,
        "MBE": mbe,
        "Pearson_r": r,
        "R2": r2,
        "Willmott_d": d,
        "MAPE (%)": mape,
        "N_pixels": n
    }

print("✅ Métricas robustas definidas: MAE, RMSE, MBE, Pearson_r, R², Willmott_d, MAPE.")

#%% 
# 🎨 Celda 7: Funciones para generar gráficos y mapas
def plot_comparison_scatter(y_true, y_pred, period, output_path):
    plt.figure(figsize=(8, 8))
    max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
    min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
    plt.scatter(y_true, y_pred, alpha=0.65, edgecolors='k', linewidth=0.5, color='#2E8B57', s=30)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label="Línea 1:1 (Perfecto acuerdo)")
    plt.xlabel("DGA (3 estaciones más cercanas) [mm]", fontsize=13, weight='bold')
    plt.ylabel("CR2MET [mm]", fontsize=13, weight='bold')
    plt.title(f"Dispersión Pixel a Pixel: CR2MET vs DGA (T={period} años)", fontsize=14, weight='bold')
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.4)
    plt.axis('equal')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_histograms(y_true, y_pred, period, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=60, alpha=0.75, label="DGA", color='#1F77B4', edgecolor='black', linewidth=0.5)
    plt.hist(y_pred, bins=60, alpha=0.75, label="CR2MET", color='#D62728', edgecolor='black', linewidth=0.5)
    plt.xlabel("Precipitación Máxima (mm)", fontsize=13, weight='bold')
    plt.ylabel("Frecuencia (conteo de píxeles)", fontsize=13, weight='bold')
    plt.title(f"Histogramas de Valores: CR2MET vs DGA (T={period} años)", fontsize=14, weight='bold')
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.4)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_error_map(error_map, meta, title, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    vmin, vmax = np.nanpercentile(error_map, [2, 98])  # Evitar outliers extremos
    im = ax.imshow(error_map, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, label="Diferencia (CR2MET - DGA) [mm]", pad=0.02)
    ax.set_title(title, fontsize=15, weight='bold', pad=20)
    ax.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

print("✅ Funciones gráficas listas: dispersión, histogramas, mapas de error.")

#%% 
# 🔄 Celda 8: Procesar todos los períodos de retorno (¡NÚCLEO DEL ANÁLISIS!)
all_metrics = []

print("🔄 Iniciando procesamiento de los 7 períodos de retorno...")

for period in tqdm(RETURN_PERIODS, desc="📊 Procesando períodos de retorno", unit="periodo"):
    print(f"\n=== 📅 Procesando T = {period} años ===")

    # Encontrar y cargar archivos
    cr2met_path = find_cr2met_file(period)
    dga_path = find_dga_file(period)
    cr2met_arr, dga_arr, valid_mask, meta = load_and_validate_rasters(cr2met_path, dga_path)

    # Calcular métricas
    metrics = calculate_metrics(dga_arr, cr2met_arr, valid_mask)
    metrics["Periodo"] = period
    all_metrics.append(metrics)

    # Generar y guardar mapa de diferencias
    diff_map = cr2met_arr - dga_arr
    diff_path = os.path.join(OUTPUT_DIR, "DIFF_MAPS", f"Diff_Pmax_T{period}.tif")
    with rasterio.open(diff_path, 'w', **meta) as dst:
        dst.write(diff_map, 1)

    # Generar gráficos
    plot_comparison_scatter(dga_arr[valid_mask], cr2met_arr[valid_mask], period,
                            os.path.join(OUTPUT_DIR, "SCATTER_PLOTS", f"scatter_T{period}.png"))
    plot_histograms(dga_arr[valid_mask], cr2met_arr[valid_mask], period,
                    os.path.join(OUTPUT_DIR, "HISTOGRAMS", f"hist_T{period}.png"))
    plot_error_map(diff_map, meta, f"Diferencia Espacial: CR2MET - DGA (T={period} años)",
                   os.path.join(OUTPUT_DIR, "ERROR_MAPS", f"error_map_T{period}.png"))

    # Guardar métricas individuales
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "METRICS", f"metrics_T{period}.csv"), index=False)

    print(f"✅ Resultados guardados para T={period}.")

print("\n🎉 ¡Procesamiento de todos los períodos completado!")

#%% 
# 📈 Celda 9: Consolidar métricas y generar gráficos de evolución
df_summary = pd.DataFrame(all_metrics)
df_summary = df_summary[[
    "Periodo", "MAE", "RMSE", "MBE", "Pearson_r", "R2", "Willmott_d", "MAPE (%)", "N_pixels"
]]

# Guardar resumen global
df_summary.to_csv(os.path.join(OUTPUT_DIR, "METRICS", "metrics_summary_all_periods.csv"), index=False)

# Mostrar tabla en notebook
print("\n📊 TABLA RESUMEN DE MÉTRICAS PARA TODOS LOS PERÍODOS DE RETORNO:")
print("="*110)
print(df_summary.round(4).to_string(index=False))

# Gráfico de evolución de métricas clave
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("📈 EVOLUCIÓN DE MÉTRICAS DE COMPARACIÓN POR PERÍODO DE RETORNO", fontsize=18, weight='bold')

# MAE
axes[0,0].plot(df_summary["Periodo"], df_summary["MAE"], 'o-', color='#1F77B4', linewidth=3, markersize=10, markeredgecolor='k', markeredgewidth=1)
axes[0,0].set_title("Error Absoluto Medio (MAE)", fontsize=14, weight='bold')
axes[0,0].set_xlabel("Período de Retorno [años]", fontsize=12)
axes[0,0].set_ylabel("MAE [mm]", fontsize=12)
axes[0,0].grid(True, alpha=0.4)

# RMSE
axes[0,1].plot(df_summary["Periodo"], df_summary["RMSE"], 's-', color='#FF7F0E', linewidth=3, markersize=10, markeredgecolor='k', markeredgewidth=1)
axes[0,1].set_title("Raíz del Error Cuadrático Medio (RMSE)", fontsize=14, weight='bold')
axes[0,1].set_xlabel("Período de Retorno [años]", fontsize=12)
axes[0,1].set_ylabel("RMSE [mm]", fontsize=12)
axes[0,1].grid(True, alpha=0.4)

# Sesgo (MBE)
axes[1,0].plot(df_summary["Periodo"], df_summary["MBE"], '^-', color='#2CA02C', linewidth=3, markersize=10, markeredgecolor='k', markeredgewidth=1)
axes[1,0].set_title("Sesgo Medio (MBE)", fontsize=14, weight='bold')
axes[1,0].set_xlabel("Período de Retorno [años]", fontsize=12)
axes[1,0].set_ylabel("Sesgo [mm]", fontsize=12)
axes[1,0].axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Sesgo cero")
axes[1,0].legend(fontsize=11)
axes[1,0].grid(True, alpha=0.4)

# Willmott d
axes[1,1].plot(df_summary["Periodo"], df_summary["Willmott_d"], 'd-', color='#9467BD', linewidth=3, markersize=10, markeredgecolor='k', markeredgewidth=1)
axes[1,1].set_title("Índice de Acuerdo de Willmott (d)", fontsize=14, weight='bold')
axes[1,1].set_xlabel("Período de Retorno [años]", fontsize=12)
axes[1,1].set_ylabel("Willmott d [0-1]", fontsize=12)
axes[1,1].set_ylim(0, 1.05)
axes[1,1].grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "METRICS", "evolucion_metricas_comparacion.png"), dpi=200)
plt.show()

#%% 
# 📄 Celda 10: Generar reporte de texto profesional
report_path = os.path.join(OUTPUT_DIR, "REPORTES", "resumen_analisis_CR2MET_vs_DGA.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("📊 REPORTE FINAL: COMPARACIÓN ROBUSTA CR2MET vs DGA (3 ESTACIONES)\n")
    f.write("="*80 + "\n\n")
    f.write("📅 Períodos de retorno analizados: " + ", ".join(map(str, RETURN_PERIODS)) + "\n\n")

    # Hallazgos clave
    mejor_mae_row = df_summary.loc[df_summary['MAE'].idxmin()]
    peor_mae_row = df_summary.loc[df_summary['MAE'].idxmax()]
    f.write(f"🏆 MEJOR DESEMPEÑO (menor MAE): {mejor_mae_row['MAE']:.3f} mm (T={mejor_mae_row['Periodo']} años)\n")
    f.write(f"📉 PEOR DESEMPEÑO (mayor MAE): {peor_mae_row['MAE']:.3f} mm (T={peor_mae_row['Periodo']} años)\n\n")

    sesgo_promedio = df_summary['MBE'].mean()
    f.write(f"⚖️ SESGO PROMEDIO GENERAL: {sesgo_promedio:+.3f} mm\n")
    if sesgo_promedio > 0.5:
        f.write("   → CR2MET TIENDE A SOBREESTIMAR SISTEMÁTICAMENTE LA PRECIPITACIÓN.\n")
    elif sesgo_promedio < -0.5:
        f.write("   → CR2MET TIENDE A SUBESTIMAR SISTEMÁTICAMENTE LA PRECIPITACIÓN.\n")
    else:
        f.write("   → CR2MET NO PRESENTA SESGO SIGNIFICATIVO EN PROMEDIO.\n\n")

    f.write(f"🔗 CORRELACIÓN LINEAL PROMEDIO (Pearson r): {df_summary['Pearson_r'].mean():.3f}\n")
    f.write(f"🎯 ACUERDO PROMEDIO (Willmott d): {df_summary['Willmott_d'].mean():.3f}\n")
    f.write(f"🧮 ERROR RELATIVO PROMEDIO (MAPE): {df_summary['MAPE (%)'].mean():.2f}% (solo donde P>1mm)\n\n")

    f.write("📁 RESULTADOS GENERADOS:\n")
    f.write("   - Mapas de diferencia georreferenciados (.tif) en /DIFF_MAPS\n")
    f.write("   - Gráficos de dispersión en /SCATTER_PLOTS\n")
    f.write("   - Histogramas comparativos en /HISTOGRAMS\n")
    f.write("   - Mapas de error espacial en /ERROR_MAPS\n")
    f.write("   - Tablas de métricas detalladas en /METRICS\n\n")
    f.write("💡 RECOMENDACIÓN: Use los mapas de error y métricas para decidir si CR2MET es adecuado para su cuenca en cada período de retorno.\n")

print(f"\n✅ REPORTE FINAL GENERADO: {report_path}")

#%% 
# 🎉 Celda 11: Mensaje final y resumen ejecutivo
print("\n" + "🎉"*50)
print("🎉 ¡ANÁLISIS COMPARATIVO COMPLETADO CON ÉXITO!".center(100))
print("🎉"*50)
print(f"📁 TODOS LOS RESULTADOS GUARDADOS EN: {OUTPUT_DIR}")
print("\n📊 RESUMEN EJECUTIVO:")
print(f"   • Períodos procesados: {len(RETURN_PERIODS)}")
print(f"   • Métricas calculadas: 7 por período (MAE, RMSE, MBE, r, R², Willmott_d, MAPE)")
print(f"   • Gráficos generados: {len(RETURN_PERIODS)*3 + 1} (dispersión, histogramas, mapas de error + evolución)")
print(f"   • Mapas de diferencia georreferenciados: {len(RETURN_PERIODS)}")
print(f"   • Reporte profesional generado: {os.path.basename(report_path)}")
print("\n🔍 PARA PROFUNDIZAR: Abra los mapas de error y gráficos de dispersión para identificar patrones espaciales de discrepancia.")
print("💡 CR2MET puede ser más robusto en períodos largos si la correlación y Willmott_d aumentan con el período de retorno.")

#%% 
# 🖼️ Celda 12 (Opcional): Mostrar mapa de error de ejemplo (T=100 años)
print("\n🖼️ Mostrando mapa de error para T=100 años como ejemplo...")

period_ejemplo = 100
diff_path = os.path.join(OUTPUT_DIR, "DIFF_MAPS", f"Diff_Pmax_T{period_ejemplo}.tif")

if os.path.exists(diff_path):
    with rasterio.open(diff_path) as src:
        data = src.read(1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        vmin, vmax = np.nanpercentile(data, [2, 98])
        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, label='Diferencia (CR2MET - DGA) [mm]', pad=0.02)
        ax.set_title(f"Mapa de Diferencias Espaciales: CR2MET - DGA (T={period_ejemplo} años)", fontsize=15, weight='bold', pad=20)
        ax.axis('off')
        plt.show()
else:
    print(f"❌ No se encontró el mapa de diferencias para T={period_ejemplo}. Verifique la ruta.")

print("\n✅ ¡Todo listo! Este análisis es 100% reproducible y robusto. ¡Feliz interpretación!")