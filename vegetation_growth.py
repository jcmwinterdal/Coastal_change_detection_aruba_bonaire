"""
Vegetation change between two vegetation GeoJSONs from vegetation_detection.py
(start and end year), per grid cell and per section, for styling in QGIS.
Set the paths and options below, then run:  python vegetation_growth.py
"""
import os
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

# ======================== SETTINGS ========================
START_FILES = ["output_vegetation/aruba_summer_2016_vegetation.geojson"]
END_FILES = ["output_vegetation/aruba_summer_2023_vegetation.geojson"]
AOI_FILE = "aruba_outline.geojson"
SECTIONS_FOLDER = "data_coastline/sections/aruba"
OUTPUT_FOLDER = "output_vegetation"

USE_SECTIONS = True          # False = only the grid, sections are skipped

CELL_SIZE_M = 500           # grid cell size in metres
TARGET_CRS = "EPSG:32619"    # metric CRS (UTM 19N works for Aruba and Bonaire)
SAVE_TIFFS = True
NODATA = -9999.0
# ==========================================================


def read_polygons(path):
    """Read a vector file into the metric CRS, fix geometries, keep polygons only."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # GEE exports are lon/lat
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]]


def read_year(files):
    """Read one year's vegetation file(s); several files are merged so overlaps aren't counted twice."""
    gdf = pd.concat([read_polygons(f) for f in files], ignore_index=True)
    if len(files) > 1:
        gdf = gpd.GeoDataFrame(geometry=[unary_union(gdf.geometry)], crs=TARGET_CRS).explode(index_parts=False)
    return gdf


def read_sections(folder):
    """One section per GeoJSON file, named after the file."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".geojson", ".json"))
    return gpd.GeoDataFrame({"name": [p.stem for p in files]},
                            geometry=[unary_union(read_polygons(p).geometry) for p in files],
                            crs=TARGET_CRS)


def make_grid(aoi):
    """Square grid snapped to CELL_SIZE_M, clipped to the AOI."""
    minx, miny, maxx, maxy = aoi.bounds
    xs = np.arange(minx // CELL_SIZE_M * CELL_SIZE_M, maxx, CELL_SIZE_M)
    ys = np.arange(miny // CELL_SIZE_M * CELL_SIZE_M, maxy, CELL_SIZE_M)
    grid = gpd.GeoDataFrame(
        {"x": [x for x in xs for y in ys], "y": [y for x in xs for y in ys]},
        geometry=[box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M) for x in xs for y in ys],
        crs=TARGET_CRS)
    grid = gpd.clip(grid, aoi, keep_geom_type=True)
    grid = grid[~grid.is_empty].reset_index(drop=True)
    grid["name"] = [f"cell_{int(x)}_{int(y)}" for x, y in zip(grid.x, grid.y)]
    return grid


def veg_area(veg, zones):
    """Vegetation area (m²) inside each zone."""
    parts = gpd.overlay(zones[["name", "geometry"]], veg, how="intersection", keep_geom_type=True)
    return zones["name"].map(parts.area.groupby(parts["name"]).sum()).fillna(0.0)


def add_metrics(zones, start, end, aoi):
    z = zones
    z["land_m2"] = z.intersection(aoi).area
    z["veg_m2_start"] = veg_area(start, z)
    z["veg_m2_end"] = veg_area(end, z)
    z["veg_change_m2"] = z.veg_m2_end - z.veg_m2_start
    z["veg_change_pct"] = (z.veg_change_m2 / z.veg_m2_start * 100).where(z.veg_m2_start > 0)  # NaN if it started at 0
    z["veg_cover_start_pct"] = (z.veg_m2_start / z.land_m2 * 100).where(z.land_m2 > 0)        # % of land that is vegetation
    z["veg_cover_end_pct"] = (z.veg_m2_end / z.land_m2 * 100).where(z.land_m2 > 0)
    z["veg_cover_change_pp"] = z.veg_cover_end_pct - z.veg_cover_start_pct                    # percentage points
    return z


def save_tiff(grid, column, out_path):
    """One pixel per grid cell."""
    minx = grid.x.min()
    maxy = grid.y.max() + CELL_SIZE_M
    cols = ((grid.x - minx) // CELL_SIZE_M).astype(int)
    rows = ((maxy - CELL_SIZE_M - grid.y) // CELL_SIZE_M).astype(int)
    data = np.full((rows.max() + 1, cols.max() + 1), NODATA, dtype="float32")
    data[rows, cols] = grid[column].fillna(NODATA)
    with rasterio.open(out_path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs=TARGET_CRS, nodata=NODATA, compress="lzw",
                       transform=from_origin(minx, maxy, CELL_SIZE_M, CELL_SIZE_M)) as dst:
        dst.write(data, 1)
    print(f"  {out_path.name} saved")


def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = read_year(START_FILES), read_year(END_FILES)
    print(f"Vegetation patches: {len(start):,} at start, {len(end):,} at end")

    aoi = unary_union(read_polygons(AOI_FILE).geometry)

    # grid over the whole island
    print(f"Grid ({CELL_SIZE_M} m)")
    grid = add_metrics(make_grid(aoi), start, end, aoi)
    grid.to_crs("EPSG:4326").to_file(out_dir / f"veg_grid_{CELL_SIZE_M}m.geojson", driver="GeoJSON")
    if SAVE_TIFFS:
        for column in ("veg_change_m2", "veg_change_pct", "veg_cover_change_pp"):
            save_tiff(grid, column, out_dir / f"veg_grid_{CELL_SIZE_M}m_{column}.tif")

    # your own sections
    if USE_SECTIONS:
        print("Sections")
        sections = add_metrics(read_sections(SECTIONS_FOLDER), start, end, aoi)
        sections.to_crs("EPSG:4326").to_file(out_dir / "veg_sections.geojson", driver="GeoJSON")
        sections.drop(columns="geometry").to_csv(out_dir / "veg_sections.csv", index=False)
        print(sections[["name", "veg_cover_start_pct", "veg_cover_end_pct", "veg_cover_change_pp", "veg_change_pct"]]
              .round(2).to_string(index=False))

    # whole island
    s, e = grid.veg_m2_start.sum(), grid.veg_m2_end.sum()
    land = grid.land_m2.sum()
    print(f"Whole island: {s / land * 100:.2f}% → {e / land * 100:.2f}% of land is vegetation "
          f"({(e - s) / s * 100:+.1f}% vegetation area)")


if __name__ == "__main__":
    main()