"""
Coastline extraction for PlanetScope 4-band images (NDWI + Otsu).
Set the paths and options below, then run:  python coastline_detection.py
"""
import re
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, label
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import unary_union
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== SETTINGS ========================
TIFF_FOLDER = "data_coastline/tiffs/aruba"
SECTIONS_FOLDER = "data_coastline/sections/aruba"
OUTPUT_FOLDER = "output_coastline"

CROP_TO_SECTIONS = True   # False = detect on each whole TIFF instead of per section

LIMIT_TIFFS = False
MAX_TIFFS = 2

USE_CUSTOM_THRESHOLD = False
CUSTOM_THRESHOLD = 0.0

DEEP_WATER_FIX = True  # True = pixels that are dark in NIR also count as water. Fixes coastlines that
                       # follow the edge between shallow and deep water instead of the shore.

KEEP_ONLY_MAIN_COASTLINE = True

COMBINED_FILE = "all_coastlines.geojson"

GREEN_BAND = 2
NIR_BAND = 4
RGB_BANDS = [3, 2, 1]
# ==========================================================

METHOD = "custom" if USE_CUSTOM_THRESHOLD else "Otsu"


def detect_coastline(green, nir, valid, transform):
    """Returns (threshold, list of LineStrings in the TIFF's CRS)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)
    valid = valid & np.isfinite(ndwi)
    if valid.sum() < 10:
        return None, []

    # Threshold (your own value or Otsu) -> water mask, cleaned of speckle
    thr = float(CUSTOM_THRESHOLD) if USE_CUSTOM_THRESHOLD else float(threshold_otsu(ndwi[valid]))
    is_water = ndwi > thr
    if DEEP_WATER_FIX:
        # Deep water is dark in both bands, so its NDWI is noisy and can end up on the land side
        # of the threshold. But all water, deep or shallow, is dark in NIR.
        is_water |= nir < threshold_otsu(nir[valid])
    water = np.pad(is_water & valid, 3, mode="edge")
    water = binary_closing(binary_opening(water, iterations=2), iterations=2)[3:-3, 3:-3] & valid

    if KEEP_ONLY_MAIN_COASTLINE:
        # keep only the biggest water body (the sea) and the biggest land body
        water = largest_region(water)
        land = largest_region(~water & valid)
        water = ~land & valid

    # Pixels next to nodata or the image border: coastline there is not real
    edge = np.pad(~valid, 1, constant_values=True)
    edge = binary_dilation(edge, iterations=2)[1:-1, 1:-1]

    lines = []
    for c in find_contours(water.astype(float), 0.5):
        r = np.clip(np.round(c[:, 0]).astype(int), 0, water.shape[0] - 1)
        k = np.clip(np.round(c[:, 1]).astype(int), 0, water.shape[1] - 1)
        good = ~edge[r, k]
        # split the contour wherever it touches an edge, keep the real parts
        start = None
        for i, g in enumerate(np.append(good, False)):
            if g and start is None:
                start = i
            elif not g and start is not None:
                if i - start >= 2:
                    xs, ys = rasterio.transform.xy(transform, c[start:i, 0], c[start:i, 1])
                    lines.append(LineString(zip(xs, ys)))
                start = None
    if KEEP_ONLY_MAIN_COASTLINE and lines:
        lines = [max(lines, key=lambda l: l.length)]  # the one continuous coastline
    return thr, lines


def largest_region(binary):
    """Keep only the largest connected group of True pixels."""
    labels, n = label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels == sizes.argmax()


SEASON_WORDS = ("dry", "wet", "rainy", "spring", "summer", "autumn", "fall", "winter")


def year_and_season(name):
    """Read (year, season) from a file name; None for whatever can't be found."""
    year = season = None
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name)
    if m:
        year = int(m.group(1))
    for w in SEASON_WORDS:
        if re.search(rf"(?<![a-z]){w}(?![a-z])", name.lower()):
            season = w
            break
    return year, season


def save_geojson(thr, lines, crs, out_path, tif_name, section):
    """Saves one coastline and returns it (EPSG:4326) for the combined file."""
    year, season = year_and_season(tif_name)
    gdf = gpd.GeoDataFrame({"year": [year], "season": [season], "section": [section], "tiff": [tif_name],
                            "threshold": [thr], "method": [METHOD]},
                           geometry=[MultiLineString(lines)], crs=crs).to_crs("EPSG:4326")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"  {out_path.name}: {METHOD} threshold = {thr:.3f}, {len(lines)} line(s), "
          f"year = {year}, season = {season}")
    return gdf


def save_png(src, section_geom, lines, thr, out_path):
    """True-colour crop of the section with the coastline drawn in red."""
    data, transform = mask(src, [mapping(section_geom)], crop=True, filled=False, indexes=RGB_BANDS)
    valid = ~np.ma.getmaskarray(data[0])

    # stretch each band to its 2-98 % range so the image isn't too dark
    rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype="float32")
    for b in range(3):
        band = data[b].filled(0).astype("float32")
        vals = band[valid & (band > 0)]
        if vals.size:
            lo, hi = np.percentile(vals, (2, 98))
            rgb[..., b] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb[~valid] = 1.0  # white outside the section

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=plotting_extent(data[0], transform))

    # only draw the part of the coastline inside this section
    if lines:
        coast = unary_union(lines).intersection(section_geom)
        for g in getattr(coast, "geoms", [coast]):
            if g.geom_type == "LineString" and not g.is_empty:
                ax.plot(*g.xy, color="red", linewidth=1.5)

    title = out_path.stem + (f"  ({METHOD} threshold = {thr:.3f})" if thr is not None else "  (no coastline)")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name} saved")


def main():
    tiffs = sorted(p for p in Path(TIFF_FOLDER).iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    sections = sorted(p for p in Path(SECTIONS_FOLDER).iterdir()
                      if p.suffix.lower() in (".geojson", ".json"))
    if LIMIT_TIFFS:
        print(f"Testing mode: only the first {MAX_TIFFS} of {len(tiffs)} TIFF(s)")
        tiffs = tiffs[:MAX_TIFFS]
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_coasts = []  # every coastline, for the combined file

    for tif in tiffs:
        print(tif.name)
        with rasterio.open(tif) as src:
            section_geoms = {s.stem: unary_union(gpd.read_file(s).to_crs(src.crs).geometry)
                             for s in sections}

            if not CROP_TO_SECTIONS:
                # detect once on the whole TIFF
                green = src.read(GREEN_BAND).astype("float32")
                nir = src.read(NIR_BAND).astype("float32")
                valid = src.read_masks(GREEN_BAND) > 0
                thr, lines = detect_coastline(green, nir, valid, src.transform)
                if lines:
                    all_coasts.append(save_geojson(thr, lines, src.crs,
                                                   out_dir / f"{tif.stem}_coastline.geojson", tif.stem, "whole"))
                else:
                    print("  no coastline found in whole TIFF")

                # PNG of every section, using the whole-TIFF coastline
                for name, geom in section_geoms.items():
                    try:
                        save_png(src, geom, lines, thr, out_dir / f"{tif.stem}_{name}.png")
                    except ValueError:
                        print(f"  {name}: doesn't overlap this TIFF, skipped")
                continue

            # crop to each section and detect there
            for name, geom in section_geoms.items():
                try:
                    data, transform = mask(src, [mapping(geom)], crop=True, filled=False,
                                           indexes=[GREEN_BAND, NIR_BAND])
                except ValueError:
                    print(f"  {name}: doesn't overlap this TIFF, skipped")
                    continue
                valid = ~np.ma.getmaskarray(data[0])
                thr, lines = detect_coastline(data[0].filled(0).astype("float32"),
                                              data[1].filled(0).astype("float32"), valid, transform)
                if lines:
                    all_coasts.append(save_geojson(thr, lines, src.crs,
                                                   out_dir / f"{tif.stem}_{name}_coastline.geojson", tif.stem, name))
                else:
                    print(f"  {name}: no coastline found (all land or all water?)")
                save_png(src, geom, lines, thr, out_dir / f"{tif.stem}_{name}.png")

    if all_coasts:
        combined = gpd.GeoDataFrame(pd.concat(all_coasts, ignore_index=True), crs="EPSG:4326")
        combined = combined.sort_values(["year", "season", "section"], na_position="last")
        combined.to_file(out_dir / COMBINED_FILE, driver="GeoJSON")
        print(f"{COMBINED_FILE}: {len(combined)} coastline(s) combined")

        # one extra file per season, e.g. all_coastlines_winter.geojson
        for season, group in combined.dropna(subset=["season"]).groupby("season"):
            name = f"{Path(COMBINED_FILE).stem}_{season}.geojson"
            group.to_file(out_dir / name, driver="GeoJSON")
            print(f"{name}: {len(group)} coastline(s) combined")


if __name__ == "__main__":
    main()