#%% 0) Instalar Dependencias
# Ejecuta esta celda primero para asegurar que todas las librerías necesarias están instaladas.
# Estas son las mismas librerías que usaba el código original, más 'tqdm' para barras de progreso.

%pip install --quiet xarray rioxarray rasterio pyproj numpy tqdm netCDF4
#%% 1) Importar Librerías Necesarias
# Importamos todas las librerías que usaremos en el script, tal como en el código original.

import sys
import pathlib
import xarray as xr
import rioxarray as riox
from pyproj import Transformer
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Ignoramos advertencias para mantener la salida limpia.
warnings.filterwarnings("ignore")

# Imprimimos información del entorno para depuración.
print("Python:", sys.executable)
print("Working dir:", pathlib.Path().resolve())

#%% 2) Configuración de Rutas (Manteniendo la carpeta base)
# Definimos las rutas de entrada y salida.
# Mantenemos la carpeta base C:\Python\klak1996\cr2met_analize.
# Creamos nuevas subcarpetas para los datos recortados y reproyectados.

# --- Rutas de Entrada ---
# Carpeta raíz donde están los datos originales de CR2MET (v2.5_R1).
IN_DIRS = {
    "et0": Path(r"C:\Python\klak1996\cr2met_download\data_v2.5_R1\et0\v2.5_R1_day"),
    "pr": Path(r"C:\Python\klak1996\cr2met_download\data_v2.5_R1\pr\v2.5_R1_day"),
    "tmean": Path(r"C:\Python\klak1996\cr2met_download\data_v2.5_R1\tmean\v2.5_R1_day"),
    "txn": Path(r"C:\Python\klak1996\cr2met_download\data_v2.5_R1\txn\v2.5_R1_day"),
}

# Ruta a la máscara ASC (DEM) que define el área de estudio. Solo se usará para obtener el bounding box.
ASC_PATH = Path(r"C:\GIS\klak1996\SIGHD\Tetis\Dem_fill.asc")
ASC_CRS = "EPSG:32718"  # El CRS del archivo ASC (UTM Zona 18S).

# --- Rutas de Salida ---
# Carpeta raíz para todos los resultados. ¡Mantenemos la carpeta base!
OUT_ROOT = Path(r"C:\Python\klak1996\cr2met_analize")

# Creamos subcarpetas NUEVAS para cada etapa del proceso: clip y reproyección UTM.
# Esto permite borrar las carpetas antiguas sin afectar esta ejecución.
OUT_CLIP_DIRS = {k: OUT_ROOT / f"{k}_clip" for k in IN_DIRS}
OUT_UTM_DIRS = {k: OUT_ROOT / f"{k}_utm18S" for k in IN_DIRS}

# Creamos todas las carpetas de salida.
for d in list(OUT_CLIP_DIRS.values()) + list(OUT_UTM_DIRS.values()):
    d.mkdir(parents=True, exist_ok=True)

# Imprimimos la configuración para verificación.
print("=== CONFIGURACIÓN DEL PROYECTO ===")
print(f"Máscara ASC (solo para BBox): {ASC_PATH}")
print(f"CRS de la máscara: {ASC_CRS}")
print(f"Carpeta Raíz de Salida: {OUT_ROOT}\n")

print("Carpetas de Salida (NUEVAS):")
for k, p in OUT_CLIP_DIRS.items():
    print(f" - {k} (Clip): {p}")
for k, p in OUT_UTM_DIRS.items():
    print(f" - {k} (UTM18S): {p}")

#%% 3) Calcular Bounding Box Geográfico (EPSG:4326) desde la Máscara ASC
# Usamos el archivo DEM (ASC) para obtener su extent en coordenadas UTM.
# Luego, transformamos este extent a coordenadas geográficas (EPSG:4326, lat/lon)
# para usarlo como caja de recorte en los datos CR2MET, que están en este sistema.

# Abrimos el archivo ASC para obtener sus límites (bounds).
with riox.open_rasterio(ASC_PATH, masked=False) as mask_da:
    # Aseguramos que el CRS esté asignado.
    if mask_da.rio.crs is None:
        mask_da.rio.write_crs(ASC_CRS, inplace=True)
    # Obtenemos los límites en UTM (xmin, ymin, xmax, ymax).
    xmin, ymin, xmax, ymax = mask_da.rio.bounds()

# Creamos un transformador de coordenadas de UTM18S a WGS84 (EPSG:4326).
transformer = Transformer.from_crs(ASC_CRS, "EPSG:4326", always_xy=True)

# Transformamos las esquinas del bounding box.
lon_min, lat_min = transformer.transform(xmin, ymin)
lon_max, lat_max = transformer.transform(xmax, ymax)

# Aseguramos el orden correcto (W, S, E, N) por si la proyección lo invierte.
lon_left = min(lon_min, lon_max)
lon_right = max(lon_min, lon_max)
lat_bottom = min(lat_min, lat_max)
lat_top = max(lat_min, lat_max)

print("=== BOUNDING BOX PARA RECORTAR DATOS CR2MET (EPSG:4326) ===")
print(f"Lon Izquierda (min):  {lon_left:.6f}")
print(f"Lon Derecha (max):    {lon_right:.6f}")
print(f"Lat Inferior (min):   {lat_bottom:.6f}")
print(f"Lat Superior (max):   {lat_top:.6f}")

#%% 4) Funciones de Utilidad para Manejo de Datos NetCDF
# Definimos funciones para normalizar coordenadas, seleccionar variables y
# preparar DataArrays para operaciones geoespaciales, mejorando el código original.

def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normaliza los nombres de las coordenadas del dataset.
    Cambia 'latitude' -> 'lat' y 'longitude' -> 'lon' si existen.
    """
    rename_dict = {}
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds

def pick_main_var(ds: xr.Dataset, expected_var: str) -> str:
    """
    Selecciona la variable principal del dataset.
    Primero intenta con el nombre esperado (et0, pr, tmean, txn).
    Si no existe, toma la primera variable que no sea 'spatial_ref' o similar.
    """
    if expected_var in ds.data_vars:
        return expected_var
    # Buscar la primera variable válida.
    for var_name in ds.data_vars:
        if var_name not in ["spatial_ref", "crs"]:
            return var_name
    raise ValueError(f"No se encontró una variable climática válida en el dataset: {list(ds.data_vars)}")

def make_geographic_da(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """
    Convierte una variable del dataset en un DataArray georreferenciado en EPSG:4326.
    Asegura que las dimensiones espaciales sean 'lat' y 'lon' y asigna el CRS si es necesario.
    """
    da = ds[var_name]

    # Asignar dimensiones espaciales.
    if {"lat", "lon"}.issubset(da.dims):
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    elif {"y", "x"}.issubset(da.dims):
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    else:
        raise ValueError(f"No se reconocen dimensiones espaciales en {var_name}: {da.dims}")

    # Asignar CRS si no tiene.
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da

#%% 5) Recortar Datos CR2MET por Bounding Box (Operación Ligera)
# Esta función recorta cada archivo NetCDF a la caja geográfica definida.
# Es una operación ligera porque no cambia la resolución ni el CRS, solo selecciona un subconjunto de datos.
# Esto reduce drásticamente el tamaño de los archivos antes de la costosa reproyección.

def list_nc_files(folder: Path) -> list[Path]:
    """Lista todos los archivos .nc en una carpeta."""
    return sorted([p for p in folder.glob("*.nc") if p.is_file()])

def clip_nc_by_bbox(nc_path: Path, out_dir: Path, var_group: str, bbox: tuple) -> tuple[bool, str]:
    """
    Recorta un archivo NetCDF por un bounding box geográfico y lo guarda.
    Args:
        nc_path: Ruta al archivo NetCDF de entrada.
        out_dir: Carpeta de salida.
        var_group: Grupo de variable (et0, pr, tmean, txn) para inferir nombre de variable.
        bbox: Tupla (lon_left, lat_bottom, lon_right, lat_top).
    Returns:
        Tupla (éxito: bool, mensaje: str).
    """
    try:
        # Abrir el dataset.
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        ds = normalize_coords(ds)
        var_name = pick_main_var(ds, var_group)

        # Crear DataArray geográfico.
        da_geo = make_geographic_da(ds, var_name)

        # Recortar por bounding box.
        # Usamos 'rio.clip_box' que es eficiente y maneja CRS automáticamente.
        da_clip = da_geo.rio.clip_box(
            minx=bbox[0], miny=bbox[1],
            maxx=bbox[2], maxy=bbox[3],
            crs="EPSG:4326"
        )

        # Definir ruta de salida.
        out_path = out_dir / f"{nc_path.stem}_clip.nc"

        # Guardar el dataset recortado.
        # Usamos compresión para ahorrar espacio.
        encoding = {var_name: {"zlib": True, "complevel": 4}}
        da_clip.to_dataset(name=var_name).to_netcdf(out_path, encoding=encoding, engine="netcdf4")

        ds.close()
        return True, f"OK -> {out_path.name}"

    except Exception as e:
        return False, f"ERROR {nc_path.name}: {str(e)}"

# --- Ejecutar Recorte para Todas las Variables ---
bbox = (lon_left, lat_bottom, lon_right, lat_top)
summary_clip = {}

for var_group, in_dir in IN_DIRS.items():
    out_dir = OUT_CLIP_DIRS[var_group]
    files = list_nc_files(in_dir)
    ok_count = 0
    fail_count = 0

    print(f"\n=== Recortando {var_group}: {len(files)} archivos ===")
    for f in tqdm(files, desc=f"Recortando {var_group}"):
        success, msg = clip_nc_by_bbox(f, out_dir, var_group, bbox)
        if success:
            ok_count += 1
        else:
            fail_count += 1
            print(msg) # Imprime el error para depuración.

    summary_clip[var_group] = (ok_count, fail_count)
    print(f"Resumen {var_group}: {ok_count} OK | {fail_count} Fallidos -> {out_dir}")

print("\n=== Resumen General de Recorte ===")
for var_group, (ok, fail) in summary_clip.items():
    print(f" - {var_group}: {ok} OK | {fail} Fallidos")


#%% 6) Preparar Plantilla de Referencia UTM desde la Máscara ASC (SOLO PARA EXTENT, NO PARA RESOLUCIÓN)
# Esta plantilla se usará SOLO para obtener el CRS y el bounding box en UTM.
# ¡NO se usará su resolución! La resolución de salida se derivará automáticamente de la resolución original de CR2MET.

# Abrimos la máscara ASC para usarla como referencia de CRS y extent.
template = riox.open_rasterio(ASC_PATH, masked=False).squeeze()

# Aseguramos que tiene el CRS correcto.
if template.rio.crs is None:
    template.rio.write_crs(ASC_CRS, inplace=True)

# Obtenemos la resolución original del CR2MET (aproximadamente 0.05 grados).
# Tomamos un archivo de ejemplo (por ejemplo, 'pr') para confirmar.
example_nc = list_nc_files(IN_DIRS["pr"])[0]
with xr.open_dataset(example_nc) as ds_ex:
    ds_ex = normalize_coords(ds_ex)
    da_ex = make_geographic_da(ds_ex, "pr")
    # La resolución en grados.
    orig_res_lon, orig_res_lat = da_ex.rio.resolution()
    print(f"Resolución Original CR2MET (grados): Lon={abs(orig_res_lon):.4f}, Lat={abs(orig_res_lat):.4f}")

print(f"\n=== Plantilla de Referencia UTM (SOLO PARA CRS Y EXTENT) ===")
print(f"CRS: {template.rio.crs}")
print(f"Bounding Box (UTM): {template.rio.bounds()}")
print(f"Resolución de Plantilla (m): {template.rio.resolution()} (¡ADVERTENCIA! Esta resolución NO se usará en la salida final.)")

# --- Guardamos el bounding box en UTM para usarlo en el recorte posterior ---
UTM_BOUNDS = template.rio.bounds()  # (xmin, ymin, xmax, ymax) en UTM18S
print(f"Bounding Box de recorte en UTM18S: {UTM_BOUNDS}")

#%% 7) Función CORREGIDA para Reproyectar a UTM18S (Manteniendo Resolución Original)
# Esta función reprojeta el DataArray a UTM18S y luego lo recorta al extent de la máscara.
# NO usa la resolución de la máscara, sino que mantiene la densidad de puntos original.

# Definimos el método de remuestreo por variable, como en el código original.
RESAMPLING_BY_VAR = {
    "et0": Resampling.bilinear,
    "pr": Resampling.average,  # Para precipitación acumulada, se usa 'average' para conservar el volumen.
    "tmean": Resampling.bilinear,
    "txn": Resampling.bilinear,
}

def reproject_to_utm_preserve_resolution(da_geo: xr.DataArray, var_group: str, target_crs: str, clip_bounds: tuple) -> xr.DataArray:
    """
    Reproyecta un DataArray geográfico (EPSG:4326) a un CRS de destino (e.g., EPSG:32718),
    y luego lo recorta a un bounding box en el CRS de destino.
    Mantiene la resolución "original" al no forzar un tamaño de píxel específico.
    """
    # Obtener el método de remuestreo.
    resampling_method = RESAMPLING_BY_VAR.get(var_group, Resampling.bilinear)

    # Paso 1: Reproyectar al CRS de destino SIN especificar una plantilla de resolución.
    # Esto crea una nueva cuadrícula con la resolución "natural" derivada de la original.
    da_utm = da_geo.rio.reproject(
        dst_crs=target_crs,
        resampling=resampling_method
    )

    # Paso 2: Recortar al bounding box de la máscara (que ya está en UTM).
    da_utm_clipped = da_utm.rio.clip_box(
        minx=clip_bounds[0], miny=clip_bounds[1],
        maxx=clip_bounds[2], maxy=clip_bounds[3],
        crs=target_crs
    )

    # Convertir a float32 para ahorrar espacio si es necesario.
    if da_utm_clipped.dtype.kind == "f" and da_utm_clipped.dtype.itemsize > 4:
        da_utm_clipped = da_utm_clipped.astype("float32")

    return da_utm_clipped

def save_nc(da: xr.DataArray, out_path: Path, var_name: str):
    """Guarda un DataArray como archivo NetCDF."""
    da.name = var_name
    out_ds = da.to_dataset(name=var_name)
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    try:
        out_ds.to_netcdf(out_path, encoding=encoding, engine="netcdf4")
    except Exception as e:
        print(f"Aviso al guardar {out_path.name}: {e}. Guardando sin compresión.")
        out_ds.to_netcdf(out_path, engine="netcdf4")
    out_ds.close()

#%% 8) Convertir Datos Recortados a UTM18S (Versión CORREGIDA)
# Aplicamos la reproyección y recorte usando la nueva función.

def list_clip_files(folder: Path) -> list[Path]:
    """Lista archivos _clip.nc en una carpeta."""
    return sorted([p for p in folder.glob("*_clip.nc") if p.is_file()])

def convert_clip_to_utm_CORRECTED(nc_path: Path, out_dir: Path, var_group: str) -> tuple[bool, str]:
    """
    Convierte un archivo _clip.nc a UTM18S y lo guarda, preservando la resolución original.
    """
    try:
        # Definir ruta de salida.
        out_path = out_dir / f"{nc_path.stem.replace('_clip', '')}_utm18S.nc"

        # Abrir y preparar datos.
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        ds = normalize_coords(ds)
        var_name = pick_main_var(ds, var_group)
        da_geo = make_geographic_da(ds, var_name)

        # Validar datos de entrada.
        valid_geo = int(np.isfinite(da_geo.values).sum())
        if valid_geo == 0:
            raise ValueError("El archivo recortado no contiene datos válidos.")

        # Reproyectar y recortar (¡CORREGIDO!).
        da_utm = reproject_to_utm_preserve_resolution(da_geo, var_group, ASC_CRS, UTM_BOUNDS)
        da_utm.name = var_name

        # Validar datos de salida.
        valid_utm = int(np.isfinite(da_utm.values).sum())
        if valid_utm == 0:
            raise ValueError("La reproyección resultó en un archivo sin datos válidos.")

        # Guardar.
        save_nc(da_utm, out_path, var_name)
        ds.close()

        return True, f"OK -> {out_path.name}"

    except Exception as e:
        return False, f"ERROR {nc_path.name}: {str(e)}"

# --- Ejecutar Conversión a UTM (CORREGIDA) ---
summary_utm = {}

for var_group, in_dir in OUT_CLIP_DIRS.items():
    out_dir = OUT_UTM_DIRS[var_group]
    files = list_clip_files(in_dir)
    ok_count = 0
    fail_count = 0

    print(f"\n=== Convirtiendo {var_group} a UTM18S (CORREGIDO): {len(files)} archivos ===")
    for f in tqdm(files, desc=f"{var_group} → UTM18S"):
        success, msg = convert_clip_to_utm_CORRECTED(f, out_dir, var_group)
        if success:
            ok_count += 1
        else:
            fail_count += 1
            print(msg)

    summary_utm[var_group] = (ok_count, fail_count)
    print(f"Resumen {var_group}: {ok_count} OK | {fail_count} Fallidos -> {out_dir}")

print("\n=== Resumen General de Conversión a UTM (CORREGIDO) ===")
for var_group, (ok, fail) in summary_utm.items():
    print(f" - {var_group}: {ok} OK | {fail} Fallidos")

#%% 9) Auditoría de los Archivos UTM Generados (VERIFICACIÓN DE RESOLUCIÓN)
# Verificamos que los archivos generados tienen el CRS y extent correctos,
# y que su resolución en metros es coherente con la resolución original de CR2MET (~5 km).

def audit_utm_files(var_group: str, out_dir: Path):
    """
    Auditoría básica de los archivos UTM generados.
    Comprueba CRS, bounds, forma y resolución.
    """
    issues = []
    files = list(out_dir.glob("*_utm18S.nc"))

    print(f"\nAuditando {len(files)} archivos en {out_dir}...")
    for nc_file in tqdm(files, desc="Auditando"):
        try:
            ds = xr.open_dataset(nc_file, engine="netcdf4")
            var_name = pick_main_var(ds, var_group)
            da = ds[var_name]

            # Asegurar dimensiones espaciales.
            if {"x", "y"}.issubset(da.dims):
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            else:
                raise ValueError(f"Dimensiones espaciales no reconocidas en {nc_file.name}.")

            # Asegurar CRS.
            if da.rio.crs is None:
                da = da.rio.write_crs(ASC_CRS, inplace=False)

            # Obtener bounds, resolución y forma.
            bounds = tuple(round(v, 6) for v in da.rio.bounds())
            res_x, res_y = da.rio.resolution()
            shape = da.shape[-2:]  # (height, width)

            # Verificar que la resolución es grande (aprox. 4000-6000 metros, no 100).
            if abs(res_x) < 1000 or abs(res_y) < 1000:
                issues.append((nc_file.name, f"Resolución sospechosamente pequeña: {res_x:.1f}m x {res_y:.1f}m", {
                    "bounds": bounds,
                    "resolution_m": (res_x, res_y),
                    "shape": shape
                }))
            elif abs(res_x) > 7000 or abs(res_y) > 7000:
                issues.append((nc_file.name, f"Resolución sospechosamente grande: {res_x:.1f}m x {res_y:.1f}m", {
                    "bounds": bounds,
                    "resolution_m": (res_x, res_y),
                    "shape": shape
                }))

            valid_count = int(np.isfinite(da.values).sum())
            if valid_count == 0:
                issues.append((nc_file.name, "Sin datos válidos", None))

            ds.close()

        except Exception as e:
            issues.append((nc_file.name, f"Error al abrir: {str(e)}", None))

    if issues:
        print(f"\n¡Se encontraron {len(issues)} problemas en {var_group}!")
        for fname, prob, details in issues[:5]:  # Mostrar solo los primeros 5.
            print(f" - {fname}: {prob}")
            if details:
                print(f"   Detalles: {details}")
        if len(issues) > 5:
            print(f"   ... y {len(issues)-5} más.")
    else:
        print(f"¡Todos los archivos de {var_group} pasaron la auditoría! Resolución en metros es coherente con CR2MET.")

# --- Ejecutar Auditoría ---
for var_group in OUT_UTM_DIRS.keys():
    audit_utm_files(var_group, OUT_UTM_DIRS[var_group])


#%% 10a) Configuración Inicial y Extracción de Coordenadas

import os
from datetime import datetime
from pyproj import Transformer, CRS
import calendar, math

# --- CONFIGURACIÓN ---
INCLUDE_TEMPERATURE_IN_TETIS = True   # Incluir Tmean en CEDEX si modelas nieve
SWAT_USE_LATLON = True                # SWAT requiere lat/lon
SWAT_MISSING = -99.0                  # Valor faltante estándar en SWAT

# --- Rutas de salida ---
TETIS_DIR = OUT_ROOT / "TETIS_INPUT"
SWAT_DIR = OUT_ROOT / "SWAT_INPUT"
TETIS_DIR.mkdir(parents=True, exist_ok=True)
SWAT_DIR.mkdir(parents=True, exist_ok=True)

print("=== CARPETAS DE SALIDA ===")
print(f"TETIS: {TETIS_DIR}")
print(f"SWAT:  {SWAT_DIR}")

# --- Obtener CRS y coordenadas de la grilla reproyectada ---
print("\n=== OBTENIENDO COORDENADAS DE LA GRILLA ===")
example_file = list(OUT_UTM_DIRS["pr"].glob("*_utm18S.nc"))[0]
with xr.open_dataset(example_file, engine="netcdf4") as ds_ex:
    ds_ex = normalize_coords(ds_ex)
    var_name_ex = pick_main_var(ds_ex, "pr")
    da_ex = make_geographic_da(ds_ex, var_name_ex)

    if {"x", "y"}.issubset(da_ex.dims):
        da_ex = da_ex.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    if da_ex.rio.crs is None:
        raise ValueError("CRS no asignado en archivo reproyectado.")

    GRID_CRS = da_ex.rio.crs
    print(f"CRS detectado: {GRID_CRS}")

    x_coords = da_ex.x.values
    y_coords = da_ex.y.values
    xx, yy = np.meshgrid(x_coords, y_coords)
    points_x = xx.flatten()
    points_y = yy.flatten()
    NUM_STATIONS = len(points_x)
    print(f"✅ Total de estaciones (celdas): {NUM_STATIONS}")

    if SWAT_USE_LATLON:
        transformer_utm_to_ll = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
        points_lon, points_lat = transformer_utm_to_ll.transform(points_x, points_y)
        print("✅ Coordenadas Lat/Lon generadas para SWAT.")
    else:
        points_lon, points_lat = [None] * NUM_STATIONS, [None] * NUM_STATIONS


#%% 10b) Extracción de Fechas y Series Temporales

print("\n=== EXTRAYENDO FECHAS DE LOS ARCHIVOS ===")
all_dates = set()

for var_group in ['pr', 'tmean', 'et0', 'txn']:
    in_dir = OUT_UTM_DIRS[var_group]
    files = sorted(in_dir.glob("*_utm18S.nc"))
    for f in files:
        parts = f.stem.split('_')
        year, month = None, None
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if i + 1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 2:
                    month = int(parts[i+1]); break
        if year is None or month is None: continue
        num_days = 29 if (month==2 and calendar.isleap(year)) else calendar.monthrange(year, month)[1]
        for day in range(1, num_days+1):
            all_dates.add(f"{year}{month:02d}{day:02d}")

all_dates = sorted(list(all_dates))
print(f"✅ Total de días: {len(all_dates)}")

START_DATE, END_DATE = "19900101", "20201231"
all_dates = [d for d in all_dates if START_DATE <= d <= END_DATE]
print(f"📅 Fechas filtradas: {len(all_dates)} días ({START_DATE}–{END_DATE})")

# --- Estructura de datos ---
station_data = {i: {'P': [], 'T': [], 'E': [], 'TMIN': []} for i in range(NUM_STATIONS)}

# --- Crear mapeo por variable ---
date_to_file_band = {var: {} for var in ['pr','tmean','et0','txn']}
for var_group in ['pr','tmean','et0','txn']:
    in_dir = OUT_UTM_DIRS[var_group]
    files = sorted(in_dir.glob("*_utm18S.nc"))
    for f in files:
        parts = f.stem.split('_')
        year, month = None, None
        for i, part in enumerate(parts):
            if part.isdigit() and len(part)==4:
                year = int(part)
                if i+1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1])==2:
                    month = int(parts[i+1]); break
        if year is None or month is None: continue
        num_days = 29 if (month==2 and calendar.isleap(year)) else calendar.monthrange(year, month)[1]
        for day in range(1, num_days+1):
            date_str = f"{year}{month:02d}{day:02d}"
            date_to_file_band[var_group][date_str] = (f, day-1)

# --- Extraer datos ---
print(f"\n=== EXTRAYENDO DATOS PARA {NUM_STATIONS} ESTACIONES ===")
for date_str in tqdm(all_dates, desc="Procesando días"):
    for var_group in ['pr','tmean','et0','txn']:
        if date_str not in date_to_file_band[var_group]:
            val = np.nan
            for i in range(NUM_STATIONS):
                if var_group=='pr': station_data[i]['P'].append(0.0)
                elif var_group=='tmean': station_data[i]['T'].append(20.0)
                elif var_group=='et0': station_data[i]['E'].append(3.0)
                elif var_group=='txn': station_data[i]['TMIN'].append(10.0)
            continue

        f, band_index = date_to_file_band[var_group][date_str]
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
            var_name = pick_main_var(ds, var_group)
            da = ds[var_name]
            if {"x","y"}.issubset(da.dims):
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            if da.rio.crs is None:
                da = da.rio.write_crs(GRID_CRS, inplace=False)
            da_day = da.isel(time=band_index) if "time" in da.dims else da
            for i in range(NUM_STATIONS):
                try:
                    value = da_day.sel(x=points_x[i], y=points_y[i], method="nearest").values.item()
                except: value = np.nan
                if var_group=='pr': station_data[i]['P'].append(value if not np.isnan(value) else 0.0)
                elif var_group=='tmean': station_data[i]['T'].append(value if not np.isnan(value) else 20.0)
                elif var_group=='et0': station_data[i]['E'].append(value if not np.isnan(value) else 3.0)
                elif var_group=='txn': station_data[i]['TMIN'].append(value if not np.isnan(value) else 10.0)
            ds.close()
        except Exception as e:
            print(f"Error {f.name}, {date_str}: {e}")


#%% 10c) Generar Archivo CEDEX para TETIS (P, E, T opcional)

from pyproj import CRS
import math

CED_EX_PATH = TETIS_DIR / "ENTRADA.TXT"
print(f"\n=== GENERANDO {CED_EX_PATH.name} PARA TETIS ===")

# Derivar zona UTM desde el CRS
try:
    zone_number = CRS.from_user_input(GRID_CRS).to_dict().get('zone', 18)
except Exception:
    zone_number = 18
    print("⚠️ No se pudo inferir zona UTM, usando 18 como fallback.")

def sanitize(vals, default):
    out=[]
    for v in vals:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))):
            out.append(default)
        else:
            out.append(v)
    return [f"{val:.1f}" for val in out]

with open(CED_EX_PATH, 'w', encoding='utf-8') as cedex_file:
    cedex_file.write("* Archivo de entrada para TETIS generado desde CR2MET\n")
    cedex_file.write(f"* Estaciones: {NUM_STATIONS}\n")
    cedex_file.write(f"* Período: {all_dates[0]} - {all_dates[-1]}\n")
    cedex_file.write(f"* Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    cedex_file.write("* Formato CEDEX\n*\n")

    for i in range(NUM_STATIONS):
        station_id = f"CR2MET_{i+1:04d}"
        px, py = points_x[i], points_y[i]

        # Precipitación
        p_vals = sanitize(station_data[i]['P'], 0.0)
        cedex_file.write(f'P "{station_id}_P"  {px:.3f}  {py:.3f}  {zone_number}  0.0  ')
        cedex_file.write("  ".join(p_vals) + "\n")

        # Temperatura media (opcional)
        if INCLUDE_TEMPERATURE_IN_TETIS:
            t_vals = sanitize(station_data[i]['T'], 20.0)
            cedex_file.write(f'T "{station_id}_T"  {px:.3f}  {py:.3f}  {zone_number}  0.0  ')
            cedex_file.write("  ".join(t_vals) + "\n")

        # Evapotranspiración
        e_vals = sanitize(station_data[i]['E'], 3.0)
        cedex_file.write(f'E "{station_id}_E"  {px:.3f}  {py:.3f}  {zone_number}  0.0  ')
        cedex_file.write("  ".join(e_vals) + "\n")

        if (i+1) % 10 == 0:
            cedex_file.write("* ---\n")

print(f"✅ Archivo CEDEX generado: {CED_EX_PATH}")


#%% 10d) Generar Archivos para SWAT (.pcp y .tmp)

print(f"\n=== GENERANDO {NUM_STATIONS} ESTACIONES PARA SWAT ===")

SWAT_MISSING = -99.0

def swat_val(v, default):
    if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))):
        return f"{SWAT_MISSING:.1f}"
    return f"{float(v):.1f}"

for i in range(NUM_STATIONS):
    station_id = f"CR2MET_{i+1:04d}"

    # --- Precipitación (.pcp) ---
    pcp_path = SWAT_DIR / f"{station_id}.pcp"
    with open(pcp_path, 'w', encoding='utf-8') as pcp_file:
        pcp_file.write(f"Location={station_id}\n")
        if SWAT_USE_LATLON:
            pcp_file.write(f"Latitude= {points_lat[i]:.6f}\n")
            pcp_file.write(f"Longitude= {points_lon[i]:.6f}\n")
        else:
            pcp_file.write("Latitude= Unknown\nLongitude= Unknown\n")
        pcp_file.write("Elevation= 0\n")
        pcp_file.write("Missing= -99\n.\n")
        for j, date in enumerate(all_dates):
            val = swat_val(station_data[i]['P'][j], 0.0)
            pcp_file.write(f"{date}\t{val}\n")

    # --- Temperaturas (.tmp) ---
    tmp_path = SWAT_DIR / f"{station_id}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        tmp_file.write(f"Location={station_id}\n")
        if SWAT_USE_LATLON:
            tmp_file.write(f"Latitude= {points_lat[i]:.6f}\n")
            tmp_file.write(f"Longitude= {points_lon[i]:.6f}\n")
        else:
            tmp_file.write("Latitude= Unknown\nLongitude= Unknown\n")
        tmp_file.write("Elevation= 0\n")
        tmp_file.write("Missing= -99\n.\n")
        for j, date in enumerate(all_dates):
            tmax = swat_val(station_data[i]['T'][j], 20.0)   # tmean como Tmax
            tmin = swat_val(station_data[i]['TMIN'][j], 10.0) # txn como Tmin
            tmp_file.write(f"{date}\t{tmax}\t{tmin}\n")

    if (i+1) % 20 == 0:
        print(f"  {i+1}/{NUM_STATIONS} estaciones generadas...")

print(f"✅ Archivos SWAT generados en: {SWAT_DIR}")
print("\n🎉 ¡EXPORTACIÓN COMPLETADA PARA TETIS Y SWAT!")


# %%
