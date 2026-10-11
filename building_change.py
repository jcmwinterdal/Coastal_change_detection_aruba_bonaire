"""
Building changes between two Open Buildings GeoJSONs (e.g. 2016 and 2023), per section.
Outputs polygons labelled new / removed / unchanged, for styling in QGIS.
"new" also covers extensions of existing buildings, "removed" also covers buildings that got smaller.
Small shifts in how the same building was detected count as unchanged.
Set the paths and options below, then run:  python building_change.py
"""
import os
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# ======================== SETTINGS ========================
START_FILE = "data_buildings/bonaire/bonaire_buildings_2016.geojson"
END_FILE = "data_buildings/bonaire/bonaire_buildings_2023.geojson"
SECTIONS_FOLDER = "data_coastline/sections/bonaire"
AOI_FILE = "bonaire_outline.geojson"
OUTPUT_FOLDER = "output_buildings"

USE_SECTIONS = True

TOLERANCE_M = 4           # differences closer than this to the other year's building are ignored
MIN_WIDTH_M = 4           # drop thin slivers narrower than this
MIN_AREA_M2 = 50          # drop change polygons smaller than this

TARGET_CRS = "EPSG:32619"
# ==========================================================


def read_polygons(path):
    """Read a vector file into the metric CRS, fix geometries, keep polygons only."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # GEE exports are lon/lat
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]]


def read_areas():
    """Sections (one per GeoJSON file, named after the file) or the whole island."""
    if USE_SECTIONS:
        files = sorted(p for p in Path(SECTIONS_FOLDER).iterdir()
                       if p.suffix.lower() in (".geojson", ".json"))
    else:
        files = [Path(AOI_FILE)]
    return {p.stem: unary_union(read_polygons(p).geometry) for p in files}


def parts(geom):
    """Remove slivers, split into single polygons, drop tiny ones."""
    half = MIN_WIDTH_M / 2
    geom = geom.buffer(-half, join_style="mitre").buffer(half, join_style="mitre")
    return [p for p in getattr(geom, "geoms", [geom]) if p.area >= MIN_AREA_M2]


def compare(start, end):
    """Label where buildings appeared, disappeared or stayed, ignoring detection wobble."""
    near_start = start.buffer(TOLERANCE_M, join_style="mitre")
    near_end = end.buffer(TOLERANCE_M, join_style="mitre")

    rows = []
    appeared = parts(end.difference(near_start))
    for p in appeared:  # new building, or an extension of one that already existed
        rows.append(("new", p))
    for p in parts(start.difference(near_end)):  # building gone, or part of it gone
        rows.append(("removed", p))
    for p in parts(end.difference(unary_union(appeared))):  # existed in both years
        rows.append(("unchanged", p))
    return rows


def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_all, end_all = read_polygons(START_FILE), read_polygons(END_FILE)
    print(f"Buildings: {len(start_all):,} at start, {len(end_all):,} at end")

    results = []
    for name, area in read_areas().items():
        start = unary_union(gpd.clip(start_all, area).geometry)
        end = unary_union(gpd.clip(end_all, area).geometry)
        rows = compare(start, end)
        if not rows:
            print(f"  {name}: no buildings")
            continue
        gdf = gpd.GeoDataFrame({"section": name, "change": [c for c, _ in rows]},
                               geometry=[g for _, g in rows], crs=TARGET_CRS)
        gdf["area_m2"] = gdf.area.round(1)
        results.append(gdf)
        counts = gdf["change"].value_counts()
        print(f"  {name}: " + ", ".join(f"{counts.get(c, 0)} {c}"
                                        for c in ("new", "removed", "unchanged")))

    changes = pd.concat(results, ignore_index=True)
    changes.to_crs("EPSG:4326").to_file(out_dir / "building_changes.geojson", driver="GeoJSON")

    summary = changes.pivot_table(index="section", columns="change", values="area_m2",
                                  aggfunc="sum", fill_value=0).round(0)
    summary.to_csv(out_dir / "building_changes_summary_m2.csv")
    print("\nArea per change type (m²):")
    print(summary.to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()