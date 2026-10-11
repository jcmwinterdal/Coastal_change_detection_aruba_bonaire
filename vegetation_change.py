"""
Vegetation change from two vegetation GeoJSON files (polygons from the vegetation detection).
Takes a "before" and an "after" file, crops both to the sections and shows per section
where vegetation was gained, lost or stayed.

Set the comparisons and options below, then run:  python vegetation_change.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

# ======================== SETTINGS ========================

COMPARISONS = [
    ("aruba", "output_vegetation/change/combined_winter_2016_aruba.geojson",
              "output_vegetation/change/combined_summer_2023_aruba.geojson",
              "data_coastline/sections/aruba"),
    ("bonaire", "output_vegetation/change/combined_winter_2016_bonaire.geojson",
              "output_vegetation/change/combined_summer_2023_bonaire.geojson",
              "data_coastline/sections/bonaire"),
]
OUTPUT_FOLDER = "output_vegetation_change"   # where the results are saved

MIN_PATCH_M2 = 20   # gain/loss patches smaller than this (m2) are treated as noise and left out,
                    # e.g. thin slivers where the two outlines differ by a pixel. 0 = keep everything
# ==========================================================


def polygons_only(geom, min_area=0):
    """The polygon parts of a geometry (drops lines/points left over by overlays and tiny patches)."""
    parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and g.area > min_area]
    return MultiPolygon(parts)


def vegetation_in(gdf, section):
    """All vegetation polygons of a file inside the section, merged into one geometry."""
    inside = gdf[gdf.intersects(section)]
    return polygons_only(make_valid(unary_union(list(inside.geometry))).intersection(section))


def load_polygons(path, crs=None):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.to_crs(crs or gdf.estimate_utm_crs())   # metres instead of degrees
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    return gdf


def analyse(name, before_path, after_path, sections_folder, out_dir):
    before = load_polygons(before_path)
    after = load_polygons(after_path, before.crs)
    sections = sorted(p for p in Path(sections_folder).iterdir() if p.suffix.lower() in (".geojson", ".json"))

    polygons, stats = [], []
    for s in sections:
        section = make_valid(unary_union(list(gpd.read_file(s).to_crs(before.crs).geometry)))
        veg_before, veg_after = vegetation_in(before, section), vegetation_in(after, section)
        change = {"stable": polygons_only(veg_before.intersection(veg_after)),
                  "gain": polygons_only(veg_after.difference(veg_before), MIN_PATCH_M2),
                  "loss": polygons_only(veg_before.difference(veg_after), MIN_PATCH_M2)}

        row = {"name": name, "section": s.stem, "section_area_m2": section.area,
               "veg_before_m2": veg_before.area, "veg_after_m2": veg_after.area,
               "gain_m2": change["gain"].area, "loss_m2": change["loss"].area, "stable_m2": change["stable"].area,
               "net_change_m2": veg_after.area - veg_before.area,
               "net_change_pct": 100 * (veg_after.area - veg_before.area) / veg_before.area if veg_before.area else np.nan,
               "veg_before_pct": 100 * veg_before.area / section.area,   # share of the section
               "veg_after_pct": 100 * veg_after.area / section.area}
        row = {k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()}
        stats.append({**row, "geometry": section})
        for label, geom in change.items():
            if not geom.is_empty:
                polygons.append({"name": name, "section": s.stem, "change": label,
                                 "area_m2": round(geom.area, 1), "geometry": geom})

        print(f"  {s.stem}: vegetation {row['veg_before_m2']:.0f} -> {row['veg_after_m2']:.0f} m2 "
              f"(gain {row['gain_m2']:.0f}, loss {row['loss_m2']:.0f}, net {row['net_change_m2']:+.0f} m2"
              + (f", {row['net_change_pct']:+.1f}%)" if veg_before.area else ")"))

    if not stats:
        print("  no sections found")
        return
    if polygons:
        gpd.GeoDataFrame(polygons, geometry="geometry", crs=before.crs).to_crs("EPSG:4326") \
            .to_file(out_dir / f"vegetation_change_{name}.geojson", driver="GeoJSON")
    summary = gpd.GeoDataFrame(stats, geometry="geometry", crs=before.crs).to_crs("EPSG:4326")
    summary.to_file(out_dir / f"vegetation_sections_{name}.geojson", driver="GeoJSON")
    summary.drop(columns="geometry").to_csv(out_dir / f"vegetation_summary_{name}.csv", index=False)


def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, before_path, after_path, sections_folder in COMPARISONS:
        print(f"{name}: {Path(before_path).name} -> {Path(after_path).name}")
        missing = [p for p in (before_path, after_path, sections_folder) if not Path(p).exists()]
        if missing:
            print(f"  {missing[0]} not found, skipped")
            continue
        analyse(name, before_path, after_path, sections_folder, out_dir)


if __name__ == "__main__":
    main()
