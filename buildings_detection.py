"""
Building growth between two Open Buildings GeoJSONs (start and end year),
per grid cell and per section, for styling in QGIS.
Set the paths and options below, then run:  python building_growth.py
"""
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

# ======================== SETTINGS ========================
START_FILE = "data_buildings/bonaire/bonaire_buildings_2016.geojson"
END_FILE = "data_buildings/bonaire/bonaire_buildings_2023.geojson"
AOI_FILE = "bonaire_outline.geojson"
SECTIONS_FOLDER = "data_coastline/sections/bonaire"
OUTPUT_FOLDER = "output_buildings"

USE_SECTIONS = True

CELL_SIZE_M = 500
TARGET_CRS = "EPSG:32619"
SAVE_TIFFS = False
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


def building_area(buildings, zones):
    """Building area (m²) inside each zone."""
    parts = gpd.overlay(zones[["name", "geometry"]], buildings, how="intersection", keep_geom_type=True)
    return zones["name"].map(parts.area.groupby(parts["name"]).sum()).fillna(0.0)


def add_metrics(zones, start, end, aoi):
    z = zones
    z["land_m2"] = z.intersection(aoi).area
    z["bldg_m2_start"] = building_area(start, z)
    z["bldg_m2_end"] = building_area(end, z)
    z["growth_m2"] = z.bldg_m2_end - z.bldg_m2_start
    z["growth_pct"] = (z.growth_m2 / z.bldg_m2_start * 100).where(z.bldg_m2_start > 0)  # NaN if it started at 0
    z["cover_start_pct"] = (z.bldg_m2_start / z.land_m2 * 100).where(z.land_m2 > 0)      # % of land covered
    z["cover_end_pct"] = (z.bldg_m2_end / z.land_m2 * 100).where(z.land_m2 > 0)
    z["cover_change_pp"] = z.cover_end_pct - z.cover_start_pct                            # percentage points
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

    start, end = read_polygons(START_FILE), read_polygons(END_FILE)
    print(f"Buildings: {len(start):,} at start, {len(end):,} at end")

    aoi = unary_union(read_polygons(AOI_FILE).geometry)

    # grid over the whole island
    print(f"Grid ({CELL_SIZE_M} m)")
    grid = add_metrics(make_grid(aoi), start, end, aoi)
    grid.to_crs("EPSG:4326").to_file(out_dir / f"grid_{CELL_SIZE_M}m.geojson", driver="GeoJSON")
    if SAVE_TIFFS:
        for column in ("growth_m2", "growth_pct", "cover_change_pp"):
            save_tiff(grid, column, out_dir / f"grid_{CELL_SIZE_M}m_{column}.tif")

    # your own sections
    if USE_SECTIONS:
        print("Sections")
        sections = add_metrics(read_sections(SECTIONS_FOLDER), start, end, aoi)
        sections.to_crs("EPSG:4326").to_file(out_dir / "sections.geojson", driver="GeoJSON")
        sections.drop(columns="geometry").to_csv(out_dir / "sections.csv", index=False)
        print(sections[["name", "cover_start_pct", "cover_end_pct", "cover_change_pp", "growth_pct"]]
              .round(2).to_string(index=False))

    # whole island
    s, e = grid.bldg_m2_start.sum(), grid.bldg_m2_end.sum()
    land = grid.land_m2.sum()
    print(f"Whole island: {s / land * 100:.2f}% → {e / land * 100:.2f}% of land built up "
          f"({(e - s) / s * 100:+.1f}% building area)")


if __name__ == "__main__":
    main()