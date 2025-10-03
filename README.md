# cr2met_DGA_analize
Set of 5 Python scripts for hydrological analysis: CR2MET data processing, quality control (randomness, independence, homogeneity, stationarity), GEV fitting, goodness-of-fit tests (KS, AD), and generation of TIFF maps of maximum precipitation for selected return periods.
## cr2met_cut_basin2.py
# Spanish version:
Qué hace el script
Recorte geográfico: Obtiene el bounding box de tu cuenca desde un DEM en UTM (ASC) y lo transforma a EPSG:4326 para recortar NetCDFs CR2MET sin cambiar resolución ni CRS.
Reproyección a UTM 18S: Convierte el recorte a EPSG:32718, manteniendo la “densidad de puntos” del grid original (no fuerza la resolución del DEM).
Conservación física: Usa remuestreo apropiado por variable (average para precipitación acumulada, bilinear para variables continuas) para evitar sesgos.
Auditoría de salida: Verifica CRS, bounds, forma y resolución (esperada ~4–6 km) y reporta problemas detectados.
Outputs organizados: Guarda versiones _clip.nc (recortado en EPSG:4326) y _utm18S.nc (reproyectado a UTM) por variable.
Requisitos
Python: 3.9+ recomendado.
Librerías: xarray, rioxarray, rasterio, pyproj, numpy, pandas, tqdm, netCDF4.
Datos de entrada:
Carpetas CR2MET v2.5_R1 por variable (et0, pr, tmean, txn).
Un DEM en formato .asc con CRS UTM 18S (EPSG:32718), que cubra tu cuenca (puedes cambiar a formato tiff si lo deseas)
Configuración clave
Rutas de entrada CR2MET:
IN_DIRS = {"et0": ..., "pr": ..., "tmean": ..., "txn": ...}
Máscara DEM (ASC) y CRS:
ASC_PATH (ruta al DEM .asc)
ASC_CRS = "EPSG:32718"
Carpeta raíz de salida:
OUT_ROOT (se crean subcarpetas *_clip y *_utm18S)
Remuestreo por variable:
RESAMPLING_BY_VAR = {"pr": Resampling.average, "et0": Resampling.bilinear, ...}
Si usas otras variables (p. ej. prcp diario acumulado, temperatura máxima), añade su clave a IN_DIRS y define el remuestreo adecuado en RESAMPLING_BY_VAR.
Cómo usarlo
Instala dependencias:
bash
pip install xarray rioxarray rasterio pyproj numpy pandas tqdm netCDF4
Actualiza rutas y CRS:
IN_DIRS: Ajusta las carpetas de tus NetCDF CR2MET.
ASC_PATH: Pon la ruta al DEM en UTM 18S.
OUT_ROOT: Define la carpeta destino del proyecto.
Ejecuta las secciones:
Bounding box: Lee el ASC y calcula límites UTM y su transformación a EPSG:4326.
Clip: Recorta cada archivo NetCDF a ese bounding box en EPSG:4326 y guarda *_clip.nc.
Reproyección: Convierte cada *_clip.nc a UTM 18S preservando resolución y guarda *_utm18S.nc.
Auditoría: Revisa los archivos UTM (CRS, bounds, resolución ~4–6 km, datos válidos).
Verifica la consola:
El script imprime resumen por variable (archivos OK/fallidos) y reporta potenciales problemas de resolución o datos nulos.
Metodología aplicada
Normalización del dataset: Renombra coordenadas latitude/longitude a lat/lon si es necesario, o maneja x/y según el dataset.
Georreferenciación robusta: Asigna CRS cuando falte y consolida dimensiones espaciales antes de cualquier recorte/reproyección.
Recorte “ligero”: Usa rio.clip_box en EPSG:4326 para reducir tamaño antes de reproyectar — más eficiente y reproducible.
Reproyección conservativa:
Evita imponer la resolución del DEM.
Usa un método de remuestreo por variable consistente con su naturaleza (p. ej., Resampling.average para acumulados como precipitación).
Auditoría cuantitativa: Chequea resolución en metros frente a umbrales (1000–7000 m), valores válidos, y consistencia del CRS.
Salidas y estructura
Carpetas en OUT_ROOT:
<var>_clip/ → archivos *_clip.nc (EPSG:4326, recortados).
<var>_utm18S/ → archivos *_utm18S.nc (EPSG:32718, reproyectados).
Logs por variable:
Totales OK/Fail al recortar y reproyectar.
Advertencias si no se reconocen dimensiones o faltan datos válidos.
Personalización para tu cuenca
Cambiar variables:
Añade/retira claves en IN_DIRS. Ajusta RESAMPLING_BY_VAR si incorporas nuevas.
CRS de la máscara:
Si tu DEM no está en UTM 18S, actualiza ASC_CRS y revisa el CRS de salida objetivo.
Rango espacial del recorte:
El recorte toma el bounding box del DEM; si necesitas “buffer”, amplía UTM_BOUNDS manualmente antes del clip.
Gestión de nombres de variables:
Si tu NetCDF usa nombres no estándar (p. ej., precip, rr, tasmean, tasmax), adapta pick_main_var para seleccionar la variable correcta.
Errores comunes y cómo resolverlos
“Dimensiones espaciales no reconocidas”:
Asegura que el dataset tiene lat/lon o x/y. Corrige en make_geographic_da.
“El archivo recortado no contiene datos válidos”:
Verifica que el bounding box en EPSG:4326 realmente intersecte el grid CR2MET (no te quedes fuera del dominio).
Resolución UTM sospechosamente baja/alta:
Revisa que no estés imponiendo la resolución del DEM; confirma que la reproyección se hace sin plantilla.
Valida que tu versión de CR2MET tenga resolución ~0.05°.
CRS faltante:
Algunos ASC no incluyen CRS. Asigna ASC_CRS cuidadosamente según tu fuente.
Ejemplo mínimo
python
# 1) Ajusta rutas
IN_DIRS = {
  "pr": Path(r"D:\data\CR2MET\pr\v2.5_R1_day"),
  "tmean": Path(r"D:\data\CR2MET\tmean\v2.5_R1_day"),
}
ASC_PATH = Path(r"D:\GIS\Andalien\Dem_fill.asc")
OUT_ROOT = Path(r"D:\projects\andalien\cr2met_analize")
# 2) Mantén 'ASC_CRS' en EPSG:32718 si tu DEM está en UTM 18S
ASC_CRS = "EPSG:32718"
# 3) Ajusta remuestreo si añades variables nuevas
RESAMPLING_BY_VAR = {"pr": Resampling.average, "tmean": Resampling.bilinear}
Buenas prácticas
Versiona tu entorno: Usa requirements.txt o conda env para reproducibilidad.
Verifica variable principal: Confirma que pick_main_var captura la variable correcta de tus NetCDFs.
Mantén trazabilidad: Guarda los logs de consola o redirígelos a archivo.
Revisa auditorías: No uses resultados “sin datos válidos” o con resolución fuera de rango.
Licencia y atribuciones
Datos CR2MET: Respeta la licencia y cita al CR2 y la versión del dataset utilizada.
Código: Puedes licenciar este script bajo MIT/Apache, según tu repositorio.

# English Version:
This script clips and reprojects CR2MET data to your basin using a DEM mask, preserves the dataset’s native resolution, and audits the outputs to ensure spatial consistency. Designed for reproducible distributed hydrology workflows.
What the script does
Geographic clip: Reads a UTM DEM extent and transforms it to EPSG:4326 to clip CR2MET NetCDFs without changing resolution or CRS.
Reproject to UTM 18S: Converts clipped data to EPSG:32718 while preserving the original grid density (no forced DEM resolution).
Physical consistency: Applies variable-specific resampling (average for accumulated precipitation, bilinear for continuous variables).
Output auditing: Checks CRS, bounds, shape, resolution (~4–6 km expected) and flags potential issues.
Organized outputs: Produces paired _clip.nc (EPSG:4326) and _utm18S.nc (EPSG:32718) per variable.
Requirements
Python: 3.9+ recommended.
Packages: xarray, rioxarray, rasterio, pyproj, numpy, pandas, tqdm, netCDF4.
Inputs:
CR2MET v2.5_R1 folders by variable (et0, pr, tmean, txn).
A DEM in .asc with UTM 18S CRS (EPSG:32718) covering your basin.
Key configuration
CR2MET input folders:
IN_DIRS = {"et0": ..., "pr": ..., "tmean": ..., "txn": ...}
DEM mask and CRS:
ASC_PATH (.asc file)
ASC_CRS = "EPSG:32718"
Output root folder:
OUT_ROOT (creates *_clip and *_utm18S subfolders)
Resampling per variable:
RESAMPLING_BY_VAR = {"pr": Resampling.average, "et0": Resampling.bilinear, ...}
If you add variables (e.g., daily precipitation, maximum temperature), include them in IN_DIRS and specify an appropriate resampling in RESAMPLING_BY_VAR.
Usage
Install dependencies:
bash
pip install xarray rioxarray rasterio pyproj numpy pandas tqdm netCDF4
Update paths and CRS:
IN_DIRS: Your CR2MET NetCDF directories.
ASC_PATH: Path to UTM DEM.
OUT_ROOT: Destination project folder.
Run the steps:
Bounding box: Read the ASC, compute UTM bounds and their EPSG:4326 transform.
Clip: Clip each NetCDF to the EPSG:4326 bbox and save *_clip.nc.
Reproject: Convert *_clip.nc to UTM 18S preserving resolution; save *_utm18S.nc.
Audit: Validate UTM files (CRS, bounds, resolution ~4–6 km, valid data).
Check the console:
The script summarizes per variable (OK/failed) and highlights resolution/data issues.
Methodology
Dataset normalization: Renames latitude/longitude to lat/lon if needed or handles x/y.
Robust georeferencing: Assigns CRS if missing and enforces spatial dims prior to clip/reproject.
Lightweight clip: Uses rio.clip_box in EPSG:4326 to reduce file size before reprojection — efficient and reproducible.
Conservative reprojection:
Avoids imposing DEM resolution.
Uses variable-appropriate resampling (e.g., Resampling.average for accumulations like precipitation).
Quantitative audit: Checks resolution in meters against thresholds (1000–7000 m), presence of valid data, and CRS consistency.
Outputs and structure
Folders in OUT_ROOT:
<var>_clip/ → *_clip.nc (EPSG:4326, clipped).
<var>_utm18S/ → *_utm18S.nc (EPSG:32718, reprojected).
Logs per variable:
Totals OK/Fail for clip and reproject.
Warnings for unrecognized spatial dims or missing valid data.
Customization
Add variables:
Extend IN_DIRS. Update RESAMPLING_BY_VAR for new variables.
Mask CRS:
If your DEM is not UTM 18S, change ASC_CRS and (optionally) your target CRS.
Clip extent:
Clip uses DEM bbox; if you need a buffer, expand UTM_BOUNDS before clipping.
Variable names:
If NetCDF uses non-standard names (e.g., precip, rr, tasmean, tasmax), adjust pick_main_var.
Common pitfalls and fixes
“Spatial dims not recognized”:
Ensure the dataset has lat/lon or x/y. Fix in make_geographic_da.
“Clipped file has no valid data”:
Confirm bbox in EPSG:4326 intersects the CR2MET grid domain.
Suspicious UTM resolution:
Verify no template resolution is being enforced; confirm CR2MET ~0.05° native resolution.
Missing CRS:
Some ASC files lack CRS. Set ASC_CRS carefully.
Minimal example
python
# 1) Set paths
IN_DIRS = {
  "pr": Path(r"D:\data\CR2MET\pr\v2.5_R1_day"),
  "tmean": Path(r"D:\data\CR2MET\tmean\v2.5_R1_day"),
}
ASC_PATH = Path(r"D:\GIS\Andalien\Dem_fill.asc")
OUT_ROOT = Path(r"D:\projects\andalien\cr2met_analize")
# 2) Keep ASC_CRS in EPSG:32718 if your DEM is UTM 18S
ASC_CRS = "EPSG:32718"
# 3) Adjust resampling if adding variables
RESAMPLING_BY_VAR = {"pr": Resampling.average, "tmean": Resampling.bilinear}
Best practices
Pin your environment: Use requirements.txt or conda env for reproducibility.
Check variable selection: Ensure pick_main_var captures the intended NetCDF variable.
Keep logs: Save console outputs or redirect to a file.
Review audits: Don’t use outputs with “no valid data” or out-of-range resolutions.
License and attribution
CR2MET data: Respect the license and cite CR2 and the dataset version.
Code: License this script under MIT/Apache as suitable for your repository.

## cr2met_frequency_analize
# Spanish version
Este script implementa el análisis de frecuencias de precipitaciones máximas diarias anuales a partir de datos CR2MET previamente recortados y reproyectados. El objetivo es ajustar la distribución de valores extremos GEV (Generalized Extreme Value) en cada celda de la grilla, evaluar la calidad de las series y generar mapas de precipitación máxima para distintos períodos de retorno.
Limitación metodológica: solo se evalúa la distribución GEV, dado que el interés está en las precipitaciones máximas anuales en un día.
Funcionalidades principales
Carga de archivos NetCDF de precipitación reproyectados a UTM 18S.
Cálculo de máximos anuales diarios en un rango temporal definido.
Relleno de valores faltantes mediante interpolación IDW.
Detección y corrección de outliers con el test de Grubbs.
Evaluación de calidad de series mediante cuatro pruebas estadísticas:
1. Aleatoriedad (Rachas de Wald-Wolfowitz)
2. Independencia (Durbin-Watson)
3. Homogeneidad (SNHT)
4. Estacionariedad (Mann-Kendall)
Ajuste de la distribución GEV en cada celda confiable.
Pruebas de bondad de ajuste (Kolmogorov-Smirnov y Anderson-Darling).
Generación de reportes CSV con resultados de calidad y ajuste.
Creación de mapas GeoTIFF de p-valores y de precipitación máxima para períodos de retorno definidos.
Configuración del usuario
En la sección de parámetros iniciales se deben ajustar:
IN_DIR: carpeta con archivos NetCDF de precipitación reproyectados (pr_utm18S).
OUT_DIR: carpeta de salida para resultados.
PERIODOS_RETORNO: lista de períodos de retorno a calcular (ej. [2, 5, 10, 20, 25, 50, 100]).
ALPHA: nivel de significancia para pruebas estadísticas (por defecto 0.05).
START_DATE, END_DATE: rango temporal de análisis (ej. 1990–2020).
IDW_POWER: potencia para interpolación IDW en el relleno de datos faltantes.
Flujo de trabajo
Carga y filtrado: lectura de NetCDF, selección temporal y cálculo de máximos anuales.
Preprocesamiento: relleno de valores faltantes y corrección de outliers.
Control de calidad: aplicación de 4 pruebas estadísticas y generación de reporte.
Ajuste GEV: estimación de parámetros por celda confiable.
Bondad de ajuste: pruebas KS y AD, mapas de p-valores.
Mapas de retorno: generación de GeoTIFF de precipitación máxima para cada período de retorno.
Resultados
detailed_quality_report_4tests.csv: reporte de calidad de series.
goodness_of_fit_report.csv: reporte de bondad de ajuste.
KS_pvalues.tif y AD_pvalues.tif: mapas de p-valores.
Pmax_Txx.tif: mapas de precipitación máxima para cada período de retorno.

# English version
Overview
This script performs frequency analysis of annual daily maximum precipitation from preprocessed CR2MET data. It fits the GEV (Generalized Extreme Value) distribution at each grid cell, evaluates series quality, and generates maximum precipitation maps for different return periods.
Methodological limitation: only the GEV distribution is evaluated, since the focus is on annual daily maxima.
Main features
Loads reprojected precipitation NetCDF files (pr_utm18S).
Computes annual daily maxima within a defined time range.
Fills missing values using IDW interpolation.
Detects and corrects outliers with Grubbs test.
Applies four quality tests:
Randomness (Wald-Wolfowitz runs)
Independence (Durbin-Watson)
Homogeneity (SNHT)
Stationarity (Mann-Kendall)
Fits GEV distribution at each reliable cell.
Performs goodness-of-fit tests (Kolmogorov-Smirnov and Anderson-Darling).
Exports CSV reports of quality and fit.
Produces GeoTIFF maps of p-values and maximum precipitation for selected return periods.
User configuration
IN_DIR: folder with reprojected precipitation NetCDF files (pr_utm18S).
OUT_DIR: output folder for results.
PERIODOS_RETORNO: list of return periods (e.g., [2, 5, 10, 20, 25, 50, 100]).
ALPHA: significance level (default 0.05).
START_DATE, END_DATE: analysis time range (e.g., 1990–2020).
IDW_POWER: power parameter for IDW interpolation.
Workflow
Load and filter: read NetCDF, select time range, compute annual maxima.
Preprocessing: fill missing values, correct outliers.
Quality control: apply 4 statistical tests, export report.
GEV fitting: estimate parameters per reliable cell.
Goodness-of-fit: KS and AD tests, generate p-value maps.
Return period maps: export GeoTIFFs of maximum precipitation.
Outputs
detailed_quality_report_4tests.csv: series quality report.
goodness_of_fit_report.csv: goodness-of-fit report.
KS_pvalues.tif, AD_pvalues.tif: p-value maps.
Pmax_Txx.tif: maximum precipitation maps for each return period.

## cr2met_interpolation2
# Spanish version
Descripción general
Este script ejecuta la evaluación y aplicación de cinco métodos de interpolación espacial sobre los mapas de precipitación máxima por período de retorno generados previamente por el análisis de frecuencias. Su salida son mapas GeoTIFF interpolados con la extensión y resolución de un DEM de referencia.
Dependencia del flujo: Requiere que previamente existan los archivos Pmax_Txx.tif en FREQUENCY_ANALYSIS y un DEM en .asc (UTM 18S) para definir la grilla de salida.
Métodos evaluados: IDW, Kriging Ordinario, Splines RBF, Natural Neighbor (aproximado con griddata cúbico/lineal), Nearest Neighbor.
Selección automática: Calcula métricas (RMSE, MAE, R², NMAE), normaliza, pondera y selecciona el mejor método con un índice compuesto, guardando best_method.txt.
Interpolación final: Aplica el método ganador sobre toda la grilla del DEM, generando Pmax_Txx_Interpolado_<método>.tif.
Dependencias e instalación
Python: 3.9+ recomendado.
Paquetes:
rasterio, numpy, pandas, scipy, scikit-learn, tqdm, openpyxl
pykrige (para Kriging Ordinario). En Windows requiere Microsoft C++ Build Tools.
Instalación recomendada:
Código
pip install rasterio numpy pandas scipy scikit-learn tqdm pykrige openpyxl
Nota crítica (Windows): Si falla pykrige, instala Microsoft C++ Build Tools y reinicia el entorno. Si no puedes disponer de pykrige, desactiva temporalmente Kriging en la lista de métodos.
Configuración del usuario
Rutas:
BASE_DIR: carpeta raíz del proyecto (contiene FREQUENCY_ANALYSIS).
MASK_PATH: ruta al DEM .asc (define CRS, extensión y resolución de salida).
Derivadas: FREQUENCY_ANALYSIS, INTERPOLATION_EVALUATION, INTERPOLATED_MAPS.
Parámetros:
PERIODOS_RETORNO: [2, 5, 10, 20, 25, 50, 100] (ajustables).
VALIDATION_SPLIT: proporción de validación (por defecto 0.30).
RANDOM_SEED: semilla para reproducibilidad (por defecto 42).
METODOS: lista de métodos a evaluar (puedes añadir/retirar).
Ajusta BASE_DIR y MASK_PATH obligatoriamente antes de ejecutar. Verifica que FREQUENCY_ANALYSIS contenga Pmax_Txx.tif con nombres acordes al patrón.
Flujo de trabajo
Validación de entradas:
Verifica: existencia de Pmax_Txx.tif para cada T en FREQUENCY_ANALYSIS.
Fase 1 — Evaluación de métodos:
Conversión a puntos: cada TIFF se transforma a un conjunto de puntos (centro de píxel, valor).
Partición: entrenamiento/validación según VALIDATION_SPLIT.
Interpolación: aplica los cinco métodos sobre el conjunto de validación.
Métricas: calcula RMSE, MAE, R², NMAE; guarda interpolation_evaluation_results.csv.
Selección del mejor método:
Normalización y ponderación: índice compuesto con pesos iguales (0.25 cada métrica).
Salida: method_selection_report.xlsx y best_method.txt con el método ganador.
Fase 2 — Interpolación final:
DEM de referencia: lee MASK_PATH para obtener transform, tamaño y CRS.
Grilla: genera coordenadas de centro de celda de la malla del DEM.
Interpolación: aplica el método ganador a todos los puntos de la grilla.
Exportación: mapas GeoTIFF Pmax_Txx_Interpolado_<método>.tif en INTERPOLATED_MAPS.
Métodos de interpolación
IDW: pondera por distancia con potencia configurable (por defecto p=2); sensible a clustering de puntos.
Kriging Ordinario: modelo de variograma esférico por defecto; requiere pykrige. Con pocos puntos por T puede fallar o ser inestable.
Splines RBF: núcleo “thin plate”; suaviza superficies, puede atenuar extremos.
Natural Neighbor (aprox.): se aproxima usando griddata cúbico/lineal; no es NN exacto, úsalo como alternativa de suavizado.
Nearest Neighbor: asigna el valor del punto más cercano; útil como base, suele tener mayor error.
Resultados y archivos generados
INTERPOLATION_EVALUATION:
interpolation_evaluation_results.csv: métricas por método y T.
method_selection_report.xlsx: ranking multicriterio.
best_method.txt: nombre del método seleccionado.
INTERPOLATED_MAPS:
Pmax_Txx_Interpolado_<método>.tif: mapas finales por período de retorno y mejor método.
Consola: resumen de métricas y progreso por período de retorno.
Buenas prácticas y precauciones
CRS y alineación: Asegúrate de que los Pmax_Txx.tif y el DEM usan el mismo CRS espacial (UTM 18S) y que la interpolación final se hace sobre la malla del DEM.
Cantidad de puntos por T: Kriging requiere ≥3 puntos; si hay menos, el script saltará ese T o fallará. Considera filtrar T con datos suficientes.
Natural Neighbor: La aproximación vía griddata no garantiza las propiedades exactas del método clásico; revísalo con cuidado si lo usas para decisiones.
Métricas sospechosamente bajas/altas: Revisa entrada, partición y escalas; confirma que evaluas sobre puntos de validación reales, no sobre el mismo conjunto de entrenamiento.
Rendimiento: El número de píxeles del DEM define el costo de la interpolación final; tamaños grandes aumentan tiempos y memoria.
Solución de problemas
Falta pykrige: Instala Microsoft C++ Build Tools (Windows), reinstala pykrige, o elimina temporalmente “Kriging_Ordinario” de METODOS.
No encuentra Pmax_Txx.tif: Verifica nombres exactos y la carpeta FREQUENCY_ANALYSIS.
Error por pocos puntos: Reduce la lista de T, agrega métodos robustos (IDW, Nearest), o usa una máscara/entrenamiento con más cobertura.
CRS inexistente en DEM: Asigna EPSG:32718 manualmente y confirma transform y resolución antes de interpolar.

# English version
Overview
This script evaluates and applies five spatial interpolation methods on maximum precipitation maps (return periods) previously produced by frequency analysis. It outputs GeoTIFF maps interpolated to the extent and resolution of a reference DEM.
Pipeline dependency: Requires Pmax_Txx.tif in FREQUENCY_ANALYSIS and a UTM 18S DEM (.asc) for the output grid.
Methods: IDW, Ordinary Kriging, RBF Splines, Natural Neighbor (approximated via cubic/linear griddata), Nearest Neighbor.
Automatic selection: Computes metrics (RMSE, MAE, R², NMAE), normalizes and weights to pick the best method, saving best_method.txt.
Final interpolation: Applies the best method on the DEM grid, creating Pmax_Txx_Interpolado_<method>.tif.
Dependencies and installation
Python: 3.9+ recommended.
Packages:
rasterio, numpy, pandas, scipy, scikit-learn, tqdm, openpyxl
pykrige (for Ordinary Kriging). On Windows, requires Microsoft C++ Build Tools.
Install:
Código
pip install rasterio numpy pandas scipy scikit-learn tqdm pykrige openpyxl
Critical note (Windows): If pykrige fails, install Microsoft C++ Build Tools and restart your environment. If unavailable, temporarily remove Kriging from the methods list.
User configuration
Paths:
BASE_DIR: project root folder (contains FREQUENCY_ANALYSIS).
MASK_PATH: path to DEM .asc (drives output CRS, extent, resolution).
Derived: FREQUENCY_ANALYSIS, INTERPOLATION_EVALUATION, INTERPOLATED_MAPS.
Parameters:
PERIODOS_RETORNO: [2, 5, 10, 20, 25, 50, 100] (adjustable).
VALIDATION_SPLIT: validation share (default 0.30).
RANDOM_SEED: reproducibility (default 42).
METODOS: interpolation methods to evaluate (add/remove as needed).
Update BASE_DIR and MASK_PATH before running. Ensure FREQUENCY_ANALYSIS contains Pmax_Txx.tif matching the naming pattern.
Workflow
Input validation:
Checks: existence of Pmax_Txx.tif for each T in FREQUENCY_ANALYSIS.
Phase 1 — Method evaluation:
Pixel centers: convert TIFF to point cloud (center coordinates, value).
Split: train/validation according to VALIDATION_SPLIT.
Interpolation: apply five methods on the validation set.
Metrics: compute RMSE, MAE, R², NMAE; save interpolation_evaluation_results.csv.
Best method selection:
Normalization and weighting: composite index with equal weights (0.25 each metric).
Outputs: method_selection_report.xlsx and best_method.txt.
Phase 2 — Final interpolation:
DEM reference: read MASK_PATH to obtain transform, size, CRS.
Grid: build the DEM center-of-cell coordinates.
Interpolate: apply the best method across the grid.
Export: GeoTIFF Pmax_Txx_Interpolado_<method>.tif in INTERPOLATED_MAPS.
Interpolation methods
IDW: distance-weighted with configurable power (default p=2); sensitive to point clustering.
Ordinary Kriging: spherical variogram by default; requires pykrige. With few points per T, may fail or be unstable.
RBF Splines: “thin plate” kernel; smooth surfaces, may attenuate extremes.
Natural Neighbor (approx.): approximated using griddata cubic/linear; not exact NN—treat as a smoothing alternative.
Nearest Neighbor: value of the nearest observation; baseline, typically higher errors.
Outputs
INTERPOLATION_EVALUATION:
interpolation_evaluation_results.csv
method_selection_report.xlsx
best_method.txt
INTERPOLATED_MAPS:
Pmax_Txx_Interpolado_<method>.tif
Console: metrics summary and progress per T.
Best practices and cautions
CRS and alignment: Ensure Pmax_Txx.tif and the DEM share spatial CRS (UTM 18S) and the final interpolation uses the DEM grid.
Points per T: Kriging needs ≥3 points; if fewer, the script will skip or error. Consider filtering T values with sufficient data.
Natural Neighbor: The griddata approximation lacks strict NN properties; validate its suitability for decision contexts.
Suspicious metrics: Re-check inputs, splits, and scales; confirm evaluation uses true validation points, not training data.
Performance: DEM pixel count drives runtime and memory; large grids increase cost.
Troubleshooting
Missing pykrige: Install Microsoft C++ Build Tools (Windows), reinstall pykrige, or remove “Kriging_Ordinario” from METODOS.
Missing Pmax_Txx.tif: Verify exact names and FREQUENCY_ANALYSIS path.
Too few points: Reduce the set of T values, prefer robust methods (IDW, Nearest), or enlarge training coverage.
DEM without CRS: Assign EPSG:32718 manually; confirm transform and resolution before interpolation.

## DGA_analize
# Spanish version
Descripción general
Este script procesa estaciones pluviométricas DGA para construir series anuales completas (1990–2020), ajustar la distribución GEV por estación, calcular precipitaciones para períodos de retorno, interpolar mapas con Kriging Ordinario y comparar esas curvas de crecimiento con los mapas CR2MET interpolados. Es un módulo de integración “punto→mapa→validación” orientado a control de calidad y contraste entre fuentes.
Depende de archivos Excel de estaciones DGA (descargables desde el sitio de la DGA) y de una máscara DEM (ASC) para filtrar estaciones dentro del área de estudio.
Genera parámetros GEV por estación, valores de retorno T, mapas interpolados por T y métricas comparativas CR2MET vs DGA a nivel de estación y pixel a pixel.
Funcionalidades
Extracción y filtrado de estaciones: Lee múltiples Excel DGA, detecta nombre, UTM Este/Norte, serie “máxima en 24 h” anual, y valida si cada estación cae dentro de la máscara DEM.
Completación de series 1990–2020: Imputa datos faltantes por IDW con p=3 y máximo 5 vecinos; añade filtro físico (<30 mm como outlier) y test de Grubbs antes de completar.
Control de calidad por estación: Aplica cuatro pruebas (Rachas, Durbin–Watson, SNHT, Mann–Kendall) y marca estaciones “no confiables”.
Ajuste GEV y periodos de retorno: Ajusta GEV por estación (MLE con respaldo L-moments), calcula precipitaciones para T=[2, 5, 10, 20, 25, 50, 100].
Interpolación espacial: Genera mapas por T usando Kriging Ordinario (variograma esférico) sobre la grilla del DEM.
Validación CR2MET vs DGA:
Por estación: compara valores CR2MET interpolados (en coordenadas UTM de estación) vs GEV-estación, computando MAE, RMSE, sesgo, R², MAPE.
Pixel a pixel: compara mapas CR2MET vs mapas DGA (alineación espacial y CRS obligatoria), guardando métricas, mapas de diferencias y gráficos.
Dependencias e instalación
Python: 3.9+ recomendado.
Paquetes:
pandas, openpyxl, xlrd (lectura Excel)
rasterio (DEM y muestreo de mapas)
numpy, scipy, statsmodels, pymannkendall (estadística y tests)
pykrige (Kriging Ordinario; en Windows requiere Microsoft C++ Build Tools)
matplotlib, seaborn (visualización opcional)
Instalación:
Código
pip install pandas openpyxl xlrd rasterio numpy scipy statsmodels pymannkendall pykrige matplotlib seaborn tqdm
Notas críticas:
Si pykrige falla en Windows, instala Microsoft C++ Build Tools y reinicia el entorno.
xlrd puede emitir advertencias en hojas modernas; openpyxl cubre .xlsx más robustamente.
Configuración del usuario
Rutas:
INPUT_EXCEL_DIR: carpeta con exceles DGA.
MASK_PATH: ruta al DEM .asc (UTM 18S).
OUTPUT_EXCEL_PATH: Excel consolidado de estaciones dentro de la máscara.
OUT_DIR_STATIONS: carpeta de salida para parámetros GEV y periodos por estación.
KRIGING_MAPS (OUTPUT_DIR): carpeta para mapas interpolados DGA.
CR2MET_RASTERS_DIR: carpeta con mapas CR2MET interpolados (nombres Pmax_Txx_Interpolado_<método>.tif).
Parámetros:
PERIODOS_RETORNO: [2, 5, 10, 20, 25, 50, 100].
ALPHA: 0.05 (tests estadísticos).
IDW_POWER: 3 (completación de series DGA).
START_YEAR/END_YEAR: 1990/2020.
Supuestos de columnas en Excel DGA: “AÑO” y “MAXIMA EN 24 HS. PRECIPITACION (mm)”; etiquetas “Estación:”, “UTM Este (mts):”, “UTM Norte (mts):”.
Flujo de trabajo
Cargar estaciones DGA y filtrar por máscara:
Extrae nombre, coordenadas UTM y serie anual.
Guarda Excel consolidado con estaciones dentro de la máscara.
Completar series 1990–2020:
Marca outliers (<30 mm, Grubbs).
Imputa con IDW espacial (p=3, 5 vecinos).
Exporta “Andelien_stations_COMPLETED.xlsx”.
Calidad y GEV por estación:
Aplica 4 tests; excluye “no confiables”.
Ajusta GEV (MLE; respaldo L-moments).
Exporta CSV de parámetros GEV y bondad de ajuste.
Calcula y exporta CSV de precipitaciones por T (stations_return_periods.csv).
Interpolación con Kriging Ordinario:
Lee DEM; genera grilla de centros de celda.
Interpola para cada T si hay ≥3 estaciones; exporta Pmax_Txx_Kriging.tif.
Validación CR2MET vs DGA:
Por estación: extrae CR2MET en coordenadas de estación; computa métricas; exporta CSV y figura de curvas de crecimiento.
Pixel a pixel: carga pares de mapas CR2MET–DGA por T; verifica alineación (forma, transform, CRS), computa métricas globales, guarda mapas de diferencia y gráficos.
Resultados
Extracción y completación:
“Andelien_stations.xlsx”
“Andelien_stations_COMPLETED.xlsx”
Estaciones:
“stations_quality_report.csv” (tests 4-calidad)
“stations_gev_parameters.csv”
“stations_goodness_of_fit.csv”
“stations_return_periods.csv”
Mapas DGA (Kriging):
“KRIGING_MAPS/Pmax_Txx_Kriging.tif”
Comparación CR2MET vs DGA:
“station_level_validation_metrics.csv”
“station_level_validation_details.csv”
“curvas_crecimiento_GEV_vs_CR2MET.png”
Comparación pixel a pixel (si se ejecuta el bloque adicional):
“DIFF_MAPS/Diff_Pmax_Txx.tif”
“SCATTER_PLOTS/scatter_Txx.png”
“HISTOGRAMS/hist_Txx.png”
“ERROR_MAPS/error_map_Txx.png”
“METRICS/metrics_Txx.csv” y resumen global
Limitaciones críticas y cuándo no interpolar
Pocas estaciones en el área de estudio: Con 3 estaciones para ~750 km², la interpolación espacial se vuelve altamente sesgada. Kriging con variograma esférico puede producir superficies excesivamente suaves o artefactos; IDW y Nearest Neighbor serán sensibles a la geometría de pocos puntos. Con densidad tan baja, los mapas interpolados reflejan más la geometría de los puntos que la estructura real de la precipitación extrema.
Mínimo para Kriging: El script exige ≥3 puntos por T; con menos, se omite el T. Incluso con 3, el ajuste del variograma no es robusto.
Imputación por IDW p=3: Si la red es escasa y/o las distancias son grandes, la imputación puede introducir sesgos sistemáticos en las series y, por ende, en los parámetros GEV.
Comparación CR2MET vs DGA: R² negativo o sesgos grandes no necesariamente invalidan CR2MET; indican que la variabilidad de los T o la intensidad local no se captura bien. Úsalo como diagnóstico, no como único criterio.
Recomendación: No utilizar mapas DGA interpolados como productos operativos cuando haya menos de ~5–7 estaciones bien distribuidas en la máscara o cuando la geometría resulte colineal/agrupada. Prioriza análisis por estación (curvas GEV), factores de corrección locales y validación independiente.
Recomendaciones prácticas
Aumentar densidad de estaciones: Ampliar el radio de búsqueda, integrar estaciones cercanas (aunque fuera de la máscara) y documentar su inclusión.
Usar co-variables: Considerar co-kriging o kriging con deriva externa (altitud, distancia al mar) si se dispone de suficientes estaciones para ajuste.
Reportar incertidumbre: Añadir intervalos de confianza para GEV o rangos por T, y mapas de error de interpolación.
Separar productos: Mantener por separado productos “diagnóstico” (interpolados con red escasa) de productos “operativos” para diseño.
Verificar alineación: Siempre validar forma, transform y CRS antes de comparaciones pixel a pixel; cualquier desalineación invalida métricas.
Solución de problemas
No se detectan estaciones dentro de la máscara: Revisar CRS del DEM, límites del bounding box, coordenadas UTM de Excel; transformar si fuese necesario.
Fallos de pykrige: Instalar Microsoft C++ Build Tools, usar entorno limpio; si persiste, desactivar Kriging y generar solo productos por estación.
Series incompletas: Si IDW p=3 no logra imputar (sin vecinos válidos ese año), usar mediana de la serie de la estación como fallback (ya implementado).
R² negativo en validación CR2MET vs DGA: Explicar en el informe que con pocos puntos y variabilidad de T limitada, R² puede volverse negativo por definición (SS_tot pequeño). Priorizar MAE, RMSE, sesgo y Willmott d.

# English version
Overview
This script processes DGA rain-gauge stations to build complete annual series (1990–2020), fit GEV per station, compute return-period precipitation, interpolate maps via Ordinary Kriging, and compare station growth curves against interpolated CR2MET maps. It’s a “point→map→validation” module built for quality control and cross-source contrast.
Requires DGA Excel files and a DEM mask (ASC) to filter stations within the study area.
Outputs station-level GEV parameters and T values, DGA kriging maps per T, and comparative metrics CR2MET vs DGA at station and pixel scales.
Features
Station extraction and filtering: Reads multiple Excel files, detects name, UTM East/North, annual “24 h maximum” series, validates location within a DEM mask.
Series completion (1990–2020): IDW imputation with p=3 and up to 5 neighbors; physical filter (<30 mm outlier) and Grubbs test beforehand.
Station quality control: Four tests (Runs, Durbin–Watson, SNHT, Mann–Kendall); marks “unreliable” stations.
GEV fitting and return periods: Fits GEV (MLE; fallback L-moments), computes precipitation for T=[2, 5, 10, 20, 25, 50, 100].
Spatial interpolation: Generates per-T maps using Ordinary Kriging (spherical variogram) on the DEM grid.
CR2MET vs DGA validation:
Station-level: samples CR2MET at station UTM coordinates, compares to station GEV values, computes MAE, RMSE, bias, R², MAPE.
Pixel-level: compares CR2MET vs DGA maps (requires exact spatial alignment), saves metrics, difference maps, scatter/hist/error plots.
Dependencies and installation
Python: 3.9+ recommended.
Packages:
pandas, openpyxl, xlrd
rasterio
numpy, scipy, statsmodels, pymannkendall
pykrige (Ordinary Kriging; Windows needs Microsoft C++ Build Tools)
matplotlib, seaborn
Install:
Código
pip install pandas openpyxl xlrd rasterio numpy scipy statsmodels pymannkendall pykrige matplotlib seaborn tqdm
Critical notes:
If pykrige fails on Windows, install Microsoft C++ Build Tools and restart.
Excel parsing relies on specific labels; adjust parsers if your sheets differ.
User configuration
Paths:
INPUT_EXCEL_DIR: DGA Excel folder.
MASK_PATH: DEM .asc (UTM 18S).
OUTPUT_EXCEL_PATH: consolidated Excel of in-mask stations.
OUT_DIR_STATIONS: outputs for station GEV and return periods.
KRIGING_MAPS (OUTPUT_DIR): DGA interpolated maps.
CR2MET_RASTERS_DIR: CR2MET interpolated maps (naming: Pmax_Txx_Interpolado_<method>.tif).
Parameters:
PERIODOS_RETORNO: [2, 5, 10, 20, 25, 50, 100].
ALPHA: 0.05.
IDW_POWER: 3 (series completion).
START_YEAR/END_YEAR: 1990/2020.
Workflow
Load and filter stations:
Extract station name, UTM coordinates, annual series; filter by DEM mask.
Complete series:
Mark outliers (<30 mm, Grubbs), impute with spatial IDW (p=3), export completed Excel.
Quality and GEV per station:
Apply 4 tests, fit GEV (MLE; fallback L-moments), export parameters and goodness-of-fit; compute return-period precipitation.
Spatial interpolation (DGA):
Build DEM grid, krige per T if ≥3 stations; export TIF maps.
CR2MET vs DGA validation:
Station-level curves; pixel-level comparison with strict alignment checks; export metrics, maps, and plots.
Outputs
Extraction & completion: “Andelien_stations.xlsx”, “Andelien_stations_COMPLETED.xlsx”.
Stations: “stations_quality_report.csv”, “stations_gev_parameters.csv”, “stations_goodness_of_fit.csv”, “stations_return_periods.csv”.
DGA maps: “KRIGING_MAPS/Pmax_Txx_Kriging.tif”.
CR2MET vs DGA: “station_level_validation_metrics.csv”, “station_level_validation_details.csv”, “curvas_crecimiento_GEV_vs_CR2MET.png”.
Pixel-level (optional): difference TIFs, scatter/hist/error images, metrics per T, and a summary.
Critical limitations and when not to interpolate
Sparse station network: With only 3 stations over ~750 km², spatial interpolation becomes strongly biased. Ordinary Kriging may over-smooth or produce artifacts; IDW and Nearest are highly sensitive to point geometry. In such cases, maps reflect station geometry more than actual extreme precipitation structure.
Minimum for Kriging: ≥3 points per T required; with only 3, variogram estimation is fragile.
IDW p=3 imputation: Large inter-station distances or poor coverage can introduce systematic biases into station series and hence GEV parameters.
CR2MET vs DGA R² negative: With few points and limited T variability, R² may be negative by construction; rely on MAE, RMSE, bias, Willmott d primarily.
Recommendation: Avoid using interpolated DGA maps as operational products when <5–7 well-distributed stations exist within the mask or station geometry is collinear/clustered. Prefer station-level GEV, local correction factors, and independent validation.
Practical recommendations
Increase station density: Expand search radius, include near-border stations with rationale.
Use covariates: Consider co-kriging or external drift (elevation, coastal distance) if enough stations permit robust fitting.
Quantify uncertainty: Provide confidence intervals for GEV parameters and T values; include interpolation error maps.
Separate diagnostics vs operations: Keep sparse-network maps as diagnostics; do not use for design without caution.
Alignment checks: Always verify map shape, transform, and CRS before pixel-wise comparisons.
Troubleshooting
No stations within mask: Check DEM CRS, mask bounds, and station UTM coordinates; reproject if needed.
pykrige errors: Install Microsoft C++ Build Tools; if still failing, disable Kriging temporarily.
Incomplete series: If IDW cannot impute (no neighbors), fallback to station median (implemented).
Negative R² in CR2MET vs DGA: Explain metric behavior; prioritize error magnitudes and agreement indexes.

## cr2met_DGA_comparacion
# Spanish Version
Descripción general
Este script implementa un análisis comparativo robusto entre mapas interpolados de CR2MET y mapas derivados de estaciones DGA cuando se dispone de una alta densidad de estaciones pluviométricas en la cuenca. A diferencia de escenarios con pocas estaciones (donde la interpolación se sesga), aquí la red densa permite generar superficies más representativas y realizar comparaciones espaciales confiables.
Funcionalidades
Carga automática de mapas CR2MET y DGA para períodos de retorno T = [2, 5, 10, 20, 25, 50, 100].
Validación espacial: asegura que ambos mapas tengan misma extensión, resolución y CRS.
Cálculo de métricas robustas:
MAE, RMSE, MBE (sesgo medio)
Correlación de Pearson (r) y R²
Índice de acuerdo de Willmott (d)
MAPE (%)
Generación de productos gráficos:
Mapas de diferencias (CR2MET – DGA)
Diagramas de dispersión pixel a pixel
Histogramas comparativos
Mapas de error espacial
Evolución de métricas por período de retorno
Reporte automático en texto: resume hallazgos clave, mejores y peores desempeños, sesgos y recomendaciones.
Dependencias
Python 3.9+
Paquetes: rasterio, numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, scikit-image, tqdm
Instalación automática incluida en el script.
Configuración del usuario
CR2MET_MAPS: carpeta con mapas interpolados de CR2MET.
DGA_MAPS: carpeta con mapas interpolados de estaciones DGA.
OUTPUT_DIR: carpeta de salida para métricas, gráficos y reportes.
RETURN_PERIODS: lista de períodos de retorno a analizar.
Los nombres de archivos deben seguir el patrón Pmax_Txx_...tif para detección automática.
Flujo de trabajo
Instalación de dependencias.
Importación de librerías y configuración visual.
Definición de rutas y períodos de retorno.
Búsqueda y validación de archivos CR2MET y DGA.
Comparación pixel a pixel:
Validación de alineación.
Cálculo de métricas.
Generación de mapas de diferencias y gráficos.
Consolidación de métricas en tablas y gráficos de evolución.
Generación de reporte final en texto con hallazgos y recomendaciones.
Resultados
/METRICS: métricas por período y resumen global.
/DIFF_MAPS: mapas de diferencias (GeoTIFF).
/SCATTER_PLOTS: diagramas de dispersión.
/HISTOGRAMS: histogramas comparativos.
/ERROR_MAPS: mapas de error espacial.
/REPORTES: reporte final en texto.
Limitaciones y recomendaciones
Alta densidad de estaciones: este script es recomendable solo cuando la red de estaciones es suficientemente densa y bien distribuida, lo que permite interpolaciones representativas y comparaciones espaciales confiables.
Sesgo sistemático: CR2MET puede subestimar o sobreestimar dependiendo del período de retorno; el reporte final indica la dirección del sesgo.
Interpretación cuidadosa: los mapas de error y las métricas deben usarse como insumo para calibrar o corregir CR2MET, no como producto final de diseño sin validación adicional.

# English version
Overview
This script performs a robust comparison between CR2MET interpolated maps and DGA station-based maps when a high density of stations is available in the basin. Unlike sparse networks (where interpolation is biased), a dense network allows more representative surfaces and reliable spatial comparisons.
Features
Loads CR2MET and DGA maps for T = [2, 5, 10, 20, 25, 50, 100].
Validates spatial alignment (extent, resolution, CRS).
Computes robust metrics: MAE, RMSE, MBE, Pearson r, R², Willmott d, MAPE.
Generates outputs:
Difference maps (CR2MET – DGA)
Scatter plots
Histograms
Error maps
Metric evolution plots
Produces a text report summarizing key findings, best/worst performance, biases, and recommendations.
Dependencies
Python 3.9+
Packages: rasterio, numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, scikit-image, tqdm
Automatic installation included in the script.
User configuration
CR2MET_MAPS: folder with CR2MET interpolated maps.
DGA_MAPS: folder with DGA interpolated maps.
OUTPUT_DIR: output folder for metrics, plots, and reports.
RETURN_PERIODS: list of return periods to analyze.
File names must follow the pattern Pmax_Txx_...tif for automatic detection.
Workflow
Install dependencies.
Import libraries and configure environment.
Define paths and return periods.
Locate and validate CR2MET and DGA files.
Pixel-to-pixel comparison:
Validate alignment.
Compute metrics.
Generate difference maps and plots.
Consolidate metrics into summary tables and plots.
Generate final text report with findings and recommendations.
Outputs
/METRICS: per-period metrics and summary.
/DIFF_MAPS: difference maps (GeoTIFF).
/SCATTER_PLOTS: scatter plots.
/HISTOGRAMS: histograms.
/ERROR_MAPS: error maps.
/REPORTES: final text report.
Limitations and recommendations
High station density required: This script is only recommended when the station network is dense and well distributed, enabling representative interpolation and reliable spatial comparisons.
Systematic bias: CR2MET may under- or overestimate depending on T; the report highlights this.
Careful interpretation: Error maps and metrics should guide calibration or correction of CR2MET, not be used directly for hydraulic design without further validation.
