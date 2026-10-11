"""
Vegetation detection for PlanetScope 4-band images (NDVI + Otsu).
Set the paths and options below, then run:  python vegetation_detection.py
"""
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes, geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.errors import WindowError
from rasterio.plot import plotting_extent
from scipy.ndimage import binary_opening, binary_closing, label
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ======================== SETTINGS ========================
TIFF_FOLDER = "data_coastline/tiffs/bonaire"
SECTIONS_FOLDER = "data_coastline/sections/bonaire"
OUTPUT_FOLDER = "output_vegetation"

CROP_TO_SECTIONS = True   # False = detect on each whole TIFF instead of per section

# ==== troubleshooting ====
LIMIT_TIFFS = False
MAX_TIFFS = 1

USE_CUSTOM_THRESHOLD = False
CUSTOM_THRESHOLD = 0.3

EXCLUDE_WATER = True
MIN_PATCH_PIXELS = 20    # remove vegetation patches smaller than this (1 pixel is about 3x3 m)

BLUE_BAND = 1
GREEN_BAND = 2
RED_BAND = 3
NIR_BAND = 4
# ==========================================================

METHOD = "custom" if USE_CUSTOM_THRESHOLD else "Otsu"
BANDS = [BLUE_BAND, GREEN_BAND, RED_BAND, NIR_BAND]


def detect_vegetation(bands, valid):
    """bands = [blue, green, red, nir] as float arrays.
    Returns (ndvi, vegetation mask, threshold, land mask)."""
    blue, green, red, nir = bands
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)
    valid = valid & np.isfinite(ndvi)
    land = valid & ~(ndwi > 0) if EXCLUDE_WATER else valid

    if land.sum() < 10:
        return ndvi, np.zeros_like(valid), None, land

    thr = float(CUSTOM_THRESHOLD) if USE_CUSTOM_THRESHOLD else float(threshold_otsu(ndvi[land]))
    veg = (ndvi > thr) & land

    # clean up speckle (padded so the image border isn't eroded)
    veg = np.pad(veg, 2, mode="edge")
    veg = binary_closing(binary_opening(veg, iterations=1), iterations=1)[2:-2, 2:-2] & land

    # remove patches that are too small
    if MIN_PATCH_PIXELS > 1:
        labels, n = label(veg)
        if n:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= MIN_PATCH_PIXELS
            keep[0] = False
            veg = keep[labels]
    return ndvi, veg, thr, land


def save_geojson(veg, transform, crs, thr, out_path):
    """Vegetation as polygons, one feature per patch, with its area in m²."""
    polys = [shape(g) for g, _ in shapes(veg.astype("uint8"), mask=veg, transform=transform)]
    if not polys:
        print(f"  no vegetation found: {out_path.name}")
        return
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    metric = gdf if not gdf.crs.is_geographic else gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = metric.area.round(1)
    gdf["threshold"] = thr
    gdf["method"] = METHOD
    gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"  {out_path.name}: {METHOD} threshold = {thr:.3f}, {len(gdf)} patch(es), "
          f"{gdf['area_m2'].sum():,.0f} m² vegetation")


def stretch_rgb(bands, valid):
    """True-colour image, each band stretched to its 2-98 % range."""
    blue, green, red = bands[0], bands[1], bands[2]
    rgb = np.zeros(red.shape + (3,), dtype="float32")
    for i, band in enumerate((red, green, blue)):
        vals = band[valid & (band > 0)]
        if vals.size:
            lo, hi = np.percentile(vals, (2, 98))
            rgb[..., i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb[~valid] = 1.0  # white outside the section
    return rgb


def save_png(bands, ndvi, veg, land, valid, transform, thr, out_path):
    ext = plotting_extent(ndvi, transform)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # left: true colour with vegetation in green
    axes[0].imshow(stretch_rgb(bands, valid), extent=ext)
    axes[0].imshow(np.ma.masked_where(~veg, veg), cmap=ListedColormap(["#00ff40"]),
                   alpha=0.5, extent=ext, interpolation="nearest")
    axes[0].set_title("True colour + detected vegetation")

    # right: NDVI map
    im = axes[1].imshow(np.where(valid, ndvi, np.nan), cmap="RdYlGn", vmin=-1, vmax=1, extent=ext)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="NDVI")
    axes[1].set_title("NDVI")

    for ax in axes:
        ax.set_axis_off()
    land_px = (land & valid).sum()
    pct = 100 * (veg & valid).sum() / land_px if land_px else 0
    thr_txt = f"{METHOD} threshold = {thr:.3f}" if thr is not None else "no land pixels"
    fig.suptitle(f"{out_path.stem}  |  {thr_txt}  |  vegetation = {pct:.1f}% of land")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name} saved")


def section_window(geom, transform, height, width):
    """Pixel window of a section inside the full TIFF + mask of pixels inside the section."""
    win = from_bounds(*geom.bounds, transform=transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, width, height))  # WindowError if no overlap
    rows, cols = win.toslices()
    win_transform = rasterio.windows.transform(win, transform)
    inside = geometry_mask([mapping(geom)], out_shape=(int(win.height), int(win.width)),
                           transform=win_transform, invert=True)
    return rows, cols, win_transform, inside


def main():
    tiffs = sorted(p for p in Path(TIFF_FOLDER).iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    sections = sorted(p for p in Path(SECTIONS_FOLDER).iterdir()
                      if p.suffix.lower() in (".geojson", ".json"))
    if LIMIT_TIFFS:
        print(f"Testing mode: only the first {MAX_TIFFS} of {len(tiffs)} TIFF(s)")
        tiffs = tiffs[:MAX_TIFFS]
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tif in tiffs:
        print(tif.name)
        with rasterio.open(tif) as src:
            section_geoms = {s.stem: unary_union(gpd.read_file(s).to_crs(src.crs).geometry)
                             for s in sections}

            if not CROP_TO_SECTIONS:
                # detect once on the whole TIFF
                bands = src.read(BANDS).astype("float32")
                valid = src.read_masks(1) > 0
                ndvi, veg, thr, land = detect_vegetation(bands, valid)
                if thr is not None:
                    save_geojson(veg, src.transform, src.crs, thr,
                                 out_dir / f"{tif.stem}_vegetation.geojson")

                # PNG of every section, cut out of the whole-TIFF result
                for name, geom in section_geoms.items():
                    try:
                        r, c, t, inside = section_window(geom, src.transform, src.height, src.width)
                    except WindowError:
                        print(f"  {name}: doesn't overlap this TIFF, skipped")
                        continue
                    v = valid[r, c] & inside
                    save_png(bands[:, r, c], ndvi[r, c], veg[r, c] & v, land[r, c] & v, v, t, thr,
                             out_dir / f"{tif.stem}_{name}.png")
                continue

            # crop to each section and detect there
            for name, geom in section_geoms.items():
                try:
                    data, transform = mask(src, [mapping(geom)], crop=True, filled=False, indexes=BANDS)
                except ValueError:
                    print(f"  {name}: doesn't overlap this TIFF, skipped")
                    continue
                valid = ~np.ma.getmaskarray(data[0])
                bands = data.filled(0).astype("float32")
                ndvi, veg, thr, land = detect_vegetation(bands, valid)
                if thr is not None:
                    save_geojson(veg, transform, src.crs, thr,
                                 out_dir / f"{tif.stem}_{name}_vegetation.geojson")
                save_png(bands, ndvi, veg, land, valid, transform, thr, out_dir / f"{tif.stem}_{name}.png")


if __name__ == "__main__":
    main()