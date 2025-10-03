#%% 0) Instalar Dependencias (¡Ejecuta esta celda primero!)

# ⚠️ ADVERTENCIA: pykrige requiere Microsoft C++ Build Tools en Windows.
# Si no los tienes instalados, esta celda fallará.
# Descárgalos e instálalos desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/

%pip install --quiet rasterio numpy pandas scipy scikit-learn tqdm pykrige openpyxl

print("✅ Dependencias instaladas correctamente.")
print("⚠️  Si falló la instalación de 'pykrige', instala 'Microsoft C++ Build Tools' y reinicia el kernel.")

#%% 1) Importar Librerías Necesarias

import sys
import os
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf, griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

print("✅ Python:", sys.executable)
print("✅ Working dir:", pathlib.Path().resolve())

#%% 2) Configuración de Rutas y Parámetros (¡CONFIGURA ESTAS RUTAS!)

# --- 🎯 RUTAS QUE EL USUARIO DEBE CONFIGURAR ---
# ¡Cambia estas rutas según tu estructura de carpetas!

# Carpeta base del proyecto (donde están tus datos de CR2MET)
BASE_DIR = Path(r"C:\Python\klak1996\cr2met_analize")  # <-- ¡CONFIGURAR ESTA!

# Ruta al archivo DEM de referencia (formato .asc)
MASK_PATH = Path(r"C:\GIS\klak1996\SIGHD\Tetis\Dem_fill.asc")  # <-- ¡CONFIGURAR ESTA!

# --- Rutas Derivadas (No necesitas cambiar estas) ---
FREQUENCY_DIR = BASE_DIR / "FREQUENCY_ANALYSIS"          # Carpeta con mapas Pmax_T*.tif
INTERP_EVAL_DIR = BASE_DIR / "INTERPOLATION_EVALUATION"  # Carpeta para guardar resultados de evaluación
INTERP_FINAL_DIR = BASE_DIR / "INTERPOLATED_MAPS"        # Carpeta para guardar mapas finales

# Crear carpetas de salida si no existen
for folder in [INTERP_EVAL_DIR, INTERP_FINAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=== CONFIGURACIÓN DE RUTAS ===")
print(f"Base del Proyecto: {BASE_DIR}")
print(f"Entrada (Mapas de Frecuencia): {FREQUENCY_DIR}")
print(f"Máscara de Referencia: {MASK_PATH}")
print(f"Salida (Evaluación): {INTERP_EVAL_DIR}")
print(f"Salida (Mapas Finales): {INTERP_FINAL_DIR}")

# --- Parámetros (Puedes ajustarlos si lo deseas) ---
VALIDATION_SPLIT = 0.30  # 30% de los puntos para validación
RANDOM_SEED = 42          # Semilla para reproducibilidad
PERIODOS_RETORNO = [2, 5, 10, 20, 25, 50, 100]  # Períodos de retorno a procesar

METODOS = [
    "IDW",
    "Kriging_Ordinario",
    "Splines_RBF",
    "Natural_Neighbor",
    "Nearest_Neighbor"
]

print(f"\n=== PARÁMETROS ===")
print(f"Porcentaje de Validación: {VALIDATION_SPLIT*100}%")
print(f"Semilla Aleatoria: {RANDOM_SEED}")
print(f"Periodos de Retorno: {PERIODOS_RETORNO}")
print(f"Métodos a Evaluar: {METODOS}")

#%% 3) Validación Inicial de Archivos de Entrada

# Verificar que todos los archivos de frecuencia existan
missing_files = []
for T in PERIODOS_RETORNO:
    tiff_file = FREQUENCY_DIR / f"Pmax_T{T:02d}.tif"
    if not tiff_file.exists():
        missing_files.append(tiff_file.name)

if missing_files:
    raise FileNotFoundError(f"❌ Faltan los siguientes archivos de entrada: {missing_files}. Verifique la carpeta {FREQUENCY_DIR}.")

print("✅ Todos los archivos de frecuencia están presentes.")

#%% 4) Función: Convertir GeoTIFF a Puntos (Centro de Píxel)

def tiff_to_points(tiff_path):
    """
    Convierte un archivo GeoTIFF en un DataFrame de puntos (x_centro, y_centro, valor).
    """
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        height, width = data.shape

        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()
        values = data.flatten()

        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

        df = pd.DataFrame({
            'x_centro': xs,
            'y_centro': ys,
            'valor': values
        })

        df = df.dropna(subset=['valor']).reset_index(drop=True)

        metadata = {
            'transform': transform,
            'crs': crs,
            'width': width,
            'height': height
        }

        return df, metadata

#%% 5) Funciones de Interpolación para los 5 Métodos

def interpolate_idw(x_train, y_train, z_train, x_pred, y_pred, power=2):
    """Interpolación IDW."""
    points_train = np.column_stack((x_train, y_train))
    points_pred = np.column_stack((x_pred, y_pred))
    distances = np.sqrt(((points_pred[:, np.newaxis, :] - points_train[np.newaxis, :, :]) ** 2).sum(axis=2))
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    z_pred = np.sum(weights * z_train, axis=1) / np.sum(weights, axis=1)
    return z_pred

def interpolate_kriging(x_train, y_train, z_train, x_pred, y_pred):
    """Interpolación con Kriging Ordinario."""
    try:
        OK = OrdinaryKriging(x_train, y_train, z_train, variogram_model='spherical', verbose=False, enable_plotting=False)
        z_pred, ss = OK.execute('points', x_pred, y_pred)
        return z_pred.data
    except Exception as e:
        print(f"⚠️ Error en Kriging: {e}")
        return np.full(len(x_pred), np.nan)

def interpolate_rbf(x_train, y_train, z_train, x_pred, y_pred, function='thin_plate'):
    """Interpolación con Splines RBF."""
    try:
        rbf = Rbf(x_train, y_train, z_train, function=function)
        z_pred = rbf(x_pred, y_pred)
        return z_pred
    except Exception as e:
        print(f"⚠️ Error en RBF: {e}")
        return np.full(len(x_pred), np.nan)

def interpolate_natural_neighbor(x_train, y_train, z_train, x_pred, y_pred):
    """Interpolación Natural Neighbor."""
    points_train = np.column_stack((x_train, y_train))
    points_pred = np.column_stack((x_pred, y_pred))
    try:
        z_pred = griddata(points_train, z_train, points_pred, method='cubic')
        if np.any(np.isnan(z_pred)):
            z_pred = griddata(points_train, z_train, points_pred, method='linear')
        return z_pred
    except Exception as e:
        print(f"⚠️ Error en Natural Neighbor: {e}")
        return np.full(len(x_pred), np.nan)

def interpolate_nearest(x_train, y_train, z_train, x_pred, y_pred):
    """Interpolación Nearest Neighbor."""
    points_train = np.column_stack((x_train, y_train))
    points_pred = np.column_stack((x_pred, y_pred))
    try:
        z_pred = griddata(points_train, z_train, points_pred, method='nearest')
        return z_pred
    except Exception as e:
        print(f"⚠️ Error en Nearest Neighbor: {e}")
        return np.full(len(x_pred), np.nan)

INTERPOLATION_FUNCTIONS = {
    "IDW": interpolate_idw,
    "Kriging_Ordinario": interpolate_kriging,
    "Splines_RBF": interpolate_rbf,
    "Natural_Neighbor": interpolate_natural_neighbor,
    "Nearest_Neighbor": interpolate_nearest
}

#%% 6) Función: Calcular Métricas de Error

def calculate_metrics(y_true, y_pred):
    """Calcula RMSE, MAE, R² y NMAE."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'nmae': np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nmae = mae / (np.max(y_true) - np.min(y_true)) if (np.max(y_true) - np.min(y_true)) > 0 else np.nan
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nmae': nmae}

#%% 7) Fase 1: Evaluación de Métodos de Interpolación

all_results = []
print("=== INICIANDO FASE 1: EVALUACIÓN DE MÉTODOS ===")

for T in tqdm(PERIODOS_RETORNO, desc="Periodos de Retorno"):
    tiff_file = FREQUENCY_DIR / f"Pmax_T{T:02d}.tif"
    df_points, metadata = tiff_to_points(tiff_file)
    print(f"\n📊 T={T}: {len(df_points)} puntos extraídos.")
    
    df_train, df_val = train_test_split(df_points, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    val_file = INTERP_EVAL_DIR / f"validation_points_T{T:02d}.csv"
    df_val.to_csv(val_file, index=False)
    print(f"💾 Puntos de validación guardados: {val_file}")
    
    x_train = df_train['x_centro'].values
    y_train = df_train['y_centro'].values
    z_train = df_train['valor'].values
    x_val = df_val['x_centro'].values
    y_val = df_val['y_centro'].values
    z_val_true = df_val['valor'].values
    
    for metodo in METODOS:
        try:
            func = INTERPOLATION_FUNCTIONS[metodo]
            z_val_pred = func(x_train, y_train, z_train, x_val, y_val)
            metrics = calculate_metrics(z_val_true, z_val_pred)
            
            result = {
                'periodo_retorno': T,
                'metodo': metodo,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'nmae': metrics['nmae'],
                'n_puntos_entrenamiento': len(df_train),
                'n_puntos_validacion': len(df_val)
            }
            all_results.append(result)
            print(f"   ✅ {metodo}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error al evaluar {metodo} para T={T}: {e}")
            all_results.append({
                'periodo_retorno': T,
                'metodo': metodo,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'nmae': np.nan,
                'n_puntos_entrenamiento': len(df_train),
                'n_puntos_validacion': len(df_val)
            })

df_results = pd.DataFrame(all_results)
results_file = INTERP_EVAL_DIR / "interpolation_evaluation_results.csv"
df_results.to_csv(results_file, index=False)
print(f"\n✅ FASE 1 COMPLETADA. Resultados guardados en: {results_file}")

#%% 8) Fase 1: Análisis Multicriterio y Selección del Mejor Método

print("=== ANÁLISIS MULTICRITERIO PARA SELECCIÓN DEL MEJOR MÉTODO ===")

df_avg = df_results.groupby('metodo')[['rmse', 'mae', 'r2', 'nmae']].mean().reset_index()

def normalize_metric(values, direction='minimize'):
    if direction == 'minimize':
        return 1 / (1 + (values - values.min()) / (values.max() - values.min() + 1e-8))
    else:
        return (values - values.min()) / (values.max() - values.min() + 1e-8)

df_avg['rmse_norm'] = normalize_metric(df_avg['rmse'], direction='minimize')
df_avg['mae_norm'] = normalize_metric(df_avg['mae'], direction='minimize')
df_avg['r2_norm'] = normalize_metric(df_avg['r2'], direction='maximize')
df_avg['nmae_norm'] = normalize_metric(df_avg['nmae'], direction='minimize')

weights = {'rmse_norm': 0.25, 'mae_norm': 0.25, 'r2_norm': 0.25, 'nmae_norm': 0.25}
df_avg['indice_compuesto'] = (
    df_avg['rmse_norm'] * weights['rmse_norm'] +
    df_avg['mae_norm'] * weights['mae_norm'] +
    df_avg['r2_norm'] * weights['r2_norm'] +
    df_avg['nmae_norm'] * weights['nmae_norm']
)

df_avg = df_avg.sort_values('indice_compuesto', ascending=False).reset_index(drop=True)
best_method = df_avg.iloc[0]['metodo']
best_index = df_avg.iloc[0]['indice_compuesto']

print("\n📊 Tabla de Índices Compuestos (Ordenados):")
print(df_avg[['metodo', 'rmse', 'mae', 'r2', 'nmae', 'indice_compuesto']])
print(f"\n🏆 ¡MEJOR MÉTODO SELECCIONADO!: {best_method}")
print(f"   Índice Compuesto: {best_index:.4f}")

selection_report = INTERP_EVAL_DIR / "method_selection_report.xlsx"
df_avg.to_excel(selection_report, index=False)
print(f"📄 Reporte de selección guardado: {selection_report}")

best_method_file = INTERP_EVAL_DIR / "best_method.txt"
with open(best_method_file, 'w') as f:
    f.write(best_method)
print(f"📄 Nombre del mejor método guardado: {best_method_file}")

#%% 9) Fase 2: Cargar Máscara de Referencia (Dem_fill.asc)

if not MASK_PATH.exists():
    raise FileNotFoundError(f"❌ Máscara de referencia no encontrada: {MASK_PATH}")

with rasterio.open(MASK_PATH) as mask_src:
    ref_transform = mask_src.transform
    ref_width = mask_src.width
    ref_height = mask_src.height
    ref_bounds = mask_src.bounds
    
    if mask_src.crs is not None:
        print(f"⚠️  El archivo {MASK_PATH.name} tiene un CRS definido: {mask_src.crs}")
        ref_crs = mask_src.crs
    else:
        print(f"ℹ️  El archivo {MASK_PATH.name} no tiene CRS definido. Asignando EPSG:32718 (UTM 18S).")
        ref_crs = CRS.from_epsg(32718)

    print("=== MÁSCARA DE REFERENCIA (Dem_fill.asc) ===")
    print(f"Resolución: {ref_transform.a:.2f} x {abs(ref_transform.e):.2f} metros/píxel")
    print(f"Tamaño: {ref_width} x {ref_height} píxeles")
    print(f"Extensión: {ref_bounds}")
    print(f"CRS: {ref_crs}")

#%% 10) Fase 2: Generar Grilla de Coordenadas de Referencia

rows_ref, cols_ref = np.meshgrid(np.arange(ref_height), np.arange(ref_width), indexing='ij')
rows_flat = rows_ref.flatten()
cols_flat = cols_ref.flatten()
xs_ref, ys_ref = rasterio.transform.xy(ref_transform, rows_flat, cols_flat, offset='center')
x_coords_ref = np.array(xs_ref)
y_coords_ref = np.array(ys_ref)
print(f"✅ Grilla de referencia generada con {len(x_coords_ref)} puntos (centros de celda).")

#%% 11) Fase 2: Interpolación Final y Generación de Mapas TIFF

best_method_file = INTERP_EVAL_DIR / "best_method.txt"
if not best_method_file.exists():
    raise FileNotFoundError("❌ Archivo 'best_method.txt' no encontrado. Ejecuta primero la Fase 1.")

with open(best_method_file, 'r') as f:
    best_method = f.read().strip()

print(f"=== INICIANDO FASE 2: INTERPOLACIÓN FINAL CON '{best_method}' ===")

if best_method not in INTERPOLATION_FUNCTIONS:
    raise ValueError(f"❌ Método '{best_method}' no está definido.")

interpolate_func = INTERPOLATION_FUNCTIONS[best_method]

for T in tqdm(PERIODOS_RETORNO, desc="Interpolando Periodos de Retorno"):
    tiff_file = FREQUENCY_DIR / f"Pmax_T{T:02d}.tif"
    if not tiff_file.exists():
        print(f"❌ Archivo no encontrado: {tiff_file}")
        continue
    
    df_points_full, _ = tiff_to_points(tiff_file)
    print(f"\n📊 T={T}: {len(df_points_full)} puntos cargados para interpolación final.")
    
    x_train = df_points_full['x_centro'].values
    y_train = df_points_full['y_centro'].values
    z_train = df_points_full['valor'].values
    
    print(f"   ↳ Interpolando {len(x_coords_ref)} puntos con {best_method}...")
    try:
        z_interp = interpolate_func(x_train, y_train, z_train, x_coords_ref, y_coords_ref)
    except Exception as e:
        print(f"   ❌ Error al interpolar T={T}: {e}")
        continue
    
    z_interp_2d = z_interp.reshape((ref_height, ref_width))
    
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
    
    output_path = INTERP_FINAL_DIR / f"Pmax_T{T:02d}_Interpolado_{best_method}.tif"
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(z_interp_2d, 1)
    
    print(f"   ✅ Mapa guardado: {output_path}")

print(f"\n🎉 ¡PROCESO COMPLETADO!")
print(f"Todos los mapas interpolados se generaron con la resolución y extensión de: {MASK_PATH}")
print(f"Mapas guardados en: {INTERP_FINAL_DIR}")
# %%
