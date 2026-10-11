"""
Coastline validation: compares ground-truth points with the coastlines made by
coastline_detection.py. For every ground-truth point it measures the distance in metres
to the detected coastline, then reports the error statistics.

Set the pairs and options below, then run:  python validation_coastlines.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== SETTINGS ========================
VALIDATIONS = [
    ("aruba_section_1", "data_validation/aruba/ground_truth_aruba_s1_winter_2023.geojson",
                  "output_coastline/aruba_winter_2023_section_1_coastline.geojson"),
    ("aruba_section_2", "data_validation/aruba/ground_truth_aruba_s2_winter_2023.geojson",
                  "output_coastline/aruba_winter_2023_section_2_coastline.geojson"),
    ("aruba_section_3", "data_validation/aruba/ground_truth_aruba_s3_winter_2023.geojson",
                  "output_coastline/aruba_winter_2023_section_3_coastline.geojson"),
    ("bonaire_section_1", "data_validation/bonaire/ground_truth_bonaire_s1_winter_2023.geojson",
                  "output_coastline/bonaire_winter_2023_section_1_coastline.geojson"),
    ("bonaire_section_2", "data_validation/bonaire/ground_truth_bonaire_s2_winter_2023.geojson",
                  "output_coastline/bonaire_winter_2023_section_2_coastline.geojson"),
    ("bonaire_section_3", "data_validation/bonaire/ground_truth_bonaire_s3_winter_2023.geojson",
                  "output_coastline/bonaire_winter_2023_section_3_coastline.geojson"),
]
OUTPUT_FOLDER = "output_validation"   # where the results are saved

PIXEL_SIZE = 3   # metres (PlanetScope)

ACCURACY_THRESHOLDS = [1.5, 3, 6, 9, 15]


ISLAND_CENTER = (-69.97, 12.51)
# ==========================================================

def measure(points_path, line_path):
    """Returns (points with error columns, detected line), both in metres (UTM)."""
    pts = gpd.read_file(points_path).explode(index_parts=False).reset_index(drop=True)
    if not (pts.geom_type == "Point").all():
        raise ValueError(f"{points_path} must contain only points")
    lines = gpd.read_file(line_path)

    utm = lines.estimate_utm_crs()  # metres instead of degrees
    pts, lines = pts.to_crs(utm), lines.to_crs(utm)
    line = lines.geometry.union_all() if hasattr(lines.geometry, "union_all") else lines.geometry.unary_union
    centre = gpd.GeoSeries([Point(ISLAND_CENTER)], crs="EPSG:4326").to_crs(utm).iloc[0]

    # error = shortest distance from the point to the detected coastline
    pts["error_m"] = pts.geometry.distance(line)
    # offset = the same distance with a direction:
    # + = detected coastline lies landward of the point, - = seaward of it
    nearest = [nearest_points(line, p)[0] for p in pts.geometry]
    seaward = [p.distance(centre) >= q.distance(centre) for p, q in zip(pts.geometry, nearest)]
    pts["offset_m"] = np.where(seaward, pts["error_m"], -pts["error_m"])
    return pts, line


def summarise(name, pts):
    """Returns (dict of statistics, table of % of points within each threshold)."""
    err = pts["error_m"].values
    stats = {
        "name": name, "n_points": len(err),
        "mean_error_m": err.mean(), "median_error_m": np.median(err),
        "rmse_m": np.sqrt((err ** 2).mean()), "std_m": err.std(),
        "min_m": err.min(), "max_m": err.max(),
        "p90_m": np.percentile(err, 90), "p95_m": np.percentile(err, 95),
        "mean_offset_m": pts["offset_m"].mean(),  # systematic shift: + landward, - seaward
    }
    stats = {k: round(float(v), 2) if isinstance(v, float) else v for k, v in stats.items()}
    matrix = pd.DataFrame({"name": name, "threshold_m": ACCURACY_THRESHOLDS,
                           "pct_points_within": [round((err <= t).mean() * 100, 2) for t in ACCURACY_THRESHOLDS]})
    for t, pct in zip(matrix["threshold_m"], matrix["pct_points_within"]):
        stats[f"pct_within_{t:g}m"] = pct
    return stats, matrix


def plot_cumulative(pts, name, out_path):
    """% of points within X metres, for every X (the continuous version of the threshold table)."""
    err = np.sort(pts["error_m"].values)
    pct = np.arange(1, len(err) + 1) / len(err) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(err, pct, linewidth=2, color="steelblue")
    for t in ACCURACY_THRESHOLDS:  # mark the thresholds
        if t <= err.max():
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.text(t, 2, f"{t / PIXEL_SIZE:g}px\n({t:g}m)", fontsize=8, ha="center", color="gray")
    ax.set_xlabel("distance from ground-truth point to detected coastline (m)")
    ax.set_ylabel("cumulative % of points within distance")
    ax.set_title(f"Cumulative accuracy - {name}")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(err.max(), 1e-6) * 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar(matrix, name, out_path):
    """Bar chart of the threshold table."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{t:g}m\n({t / PIXEL_SIZE:g}px)" for t in matrix["threshold_m"]]
    bars = ax.bar(labels, matrix["pct_points_within"], color="steelblue", edgecolor="black")
    for bar, pct in zip(bars, matrix["pct_points_within"]):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1, f"{pct:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("distance threshold")
    ax.set_ylabel("% of points within threshold")
    ax.set_title(f"Accuracy by threshold - {name}")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_map(pts, line, name, out_path):
    """The detected coastline with the ground-truth points coloured by their error."""
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([line], crs=pts.crs).plot(ax=ax, color="black", linewidth=1)
    pts.plot(ax=ax, column="error_m", cmap="RdYlGn_r", markersize=25,
             legend=True, legend_kwds={"label": "error (m)", "shrink": 0.6})
    ax.set_title(f"Validation - {name}")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_maps(maps, out_path):
    """One image with every validated coastline side by side (same colour scale for all)."""
    top = max(max(pts["error_m"].max() for _, pts, _ in maps), 1e-6)
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 7), squeeze=False, layout="constrained")
    for ax, (name, pts, line) in zip(axes[0], maps):
        gpd.GeoSeries([line], crs=pts.crs).plot(ax=ax, color="black", linewidth=1)
        pts.plot(ax=ax, column="error_m", cmap="RdYlGn_r", vmin=0, vmax=top, markersize=20)
        # zoom to the part of the coastline that has ground-truth points
        x0, y0, x1, y1 = pts.total_bounds
        margin = 0.1 * max(x1 - x0, y1 - y0, 10)
        ax.set_xlim(x0 - margin, x1 + margin)
        ax.set_ylim(y0 - margin, y1 + margin)
        # same panel size for every section (the view widens instead), so all titles are on one line
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(
            f"{name.replace('_', ' ').title()}\nmean {pts['error_m'].mean():.1f} m, RMSE {(pts['error_m'] ** 2).mean() ** 0.5:.1f} m",
            fontsize=10)
    scale = plt.cm.ScalarMappable(norm=plt.Normalize(0, top), cmap="RdYlGn_r")
    # legend bar: shrink = its height (1 = full height), aspect = lower is thicker
    bar = fig.colorbar(scale, ax=axes[0].tolist(), shrink=0.9, aspect=15)
    bar.set_label("error (m)", fontsize=14)
    bar.ax.tick_params(labelsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats, all_matrices, all_points, all_maps = [], [], [], []
    for name, points_path, line_path in VALIDATIONS:
        print(name)
        missing = [p for p in (points_path, line_path) if not Path(p).exists()]
        if missing:
            print(f"  {missing[0]} not found, skipped")
            continue
        pts, line = measure(points_path, line_path)
        stats, matrix = summarise(name, pts)
        pts.insert(0, "name", name)
        pts[["error_m", "offset_m"]] = pts[["error_m", "offset_m"]].round(2)
        all_stats.append(stats)
        all_matrices.append(matrix)
        all_points.append(pts.to_crs("EPSG:4326"))
        all_maps.append((name, pts, line))

        plot_cumulative(pts, name, out_dir / f"cumulative_accuracy_{name}.png")
        plot_bar(matrix, name, out_dir / f"accuracy_bar_{name}.png")
        plot_error_map(pts, line, name, out_dir / f"error_map_{name}.png")

        print(f"  {stats['n_points']} points: mean {stats['mean_error_m']} m, median {stats['median_error_m']} m, "
              f"RMSE {stats['rmse_m']} m, 95% within {stats['p95_m']} m, max {stats['max_m']} m")
        print(f"  mean offset {stats['mean_offset_m']:+} m (+ = detected coastline landward of the points)")
        print("  " + ", ".join(f"{pct:.0f}% within {t:g} m"
                               for t, pct in zip(matrix["threshold_m"], matrix["pct_points_within"])))

    if all_stats:
        pd.DataFrame(all_stats).to_csv(out_dir / "validation_summary.csv", index=False)
        pd.concat(all_matrices, ignore_index=True).to_csv(out_dir / "validation_matrix.csv", index=False)
        gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True), crs="EPSG:4326") \
            .to_file(out_dir / "validation_points.geojson", driver="GeoJSON")
        # one combined image per island: the island is the first word of the name (aruba_section_1)
        islands = {}
        for entry in all_maps:
            islands.setdefault(entry[0].split("_")[0].lower(), []).append(entry)
        for island, maps in islands.items():
            plot_all_maps(maps, out_dir / f"error_map_{island}.png")
        print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
