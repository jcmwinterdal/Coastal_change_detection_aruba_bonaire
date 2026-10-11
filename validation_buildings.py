"""
Building detection validation: compares ground-truth points (one point on every building you
see in the high-resolution image) with the detected building polygons (Open Buildings)
inside a validation section, and makes a confusion matrix.

  detected (TP)      = a ground-truth building that has a detected polygon
  missed (FN)        = a ground-truth building without a detected polygon
  false alarm (FP)   = a detected polygon with no ground-truth building in it
  TN does not exist here: "places without a building that were correctly not detected"
  can't be counted, so that cell stays empty.

Set the validations and options below, then run:  python validation_buildings.py
"""
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== SETTINGS ========================
VALIDATIONS = [
    ("bonaire_2023", "data_validation/buildings/ground_truth.geojson",
                     "data_buildings/bonaire/bonaire_buildings_2023.geojson",
                     "data_validation/buildings/validation_section.geojson"),
    ("aruba_2018", "data_validation/buildings/ground_truth_aruba.geojson",
                     "data_buildings/aruba/aruba_buildings_2018.geojson",
                     "data_validation/buildings/validation_section_aruba.geojson"),
]
OUTPUT_FOLDER = "output_validation/buildings"   # where the results are saved

MATCH_DISTANCE = 3     # metres: a point this close to a polygon still counts as that building

MIN_CONFIDENCE = None  # e.g. 0.7 = only use detections with confidence >= 0.7

# ==========================================================


def validate(name, points_path, buildings_path, section_path):
    """Returns (ground-truth points with result, detected polygons with result, counts)."""
    pts = gpd.read_file(points_path).explode(index_parts=False)
    pts = pts[pts.geom_type == "Point"].reset_index(drop=True)
    utm = pts.estimate_utm_crs()   # metres instead of degrees
    pts = pts.to_crs(utm)
    section = unary_union(list(gpd.read_file(section_path).to_crs(utm).geometry))

    buildings = gpd.read_file(buildings_path, bbox=gpd.GeoSeries([section.buffer(100)], crs=utm))
    buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(utm).reset_index(drop=True)
    if MIN_CONFIDENCE is not None and "confidence" in buildings.columns:
        buildings = buildings[buildings["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)

    outside = ~pts.within(section)
    if outside.any():
        print(f"  {outside.sum()} ground-truth point(s) outside the section ignored")
        pts = pts[~outside].reset_index(drop=True)

    # match every point to the polygon it lies in (or the closest one within MATCH_DISTANCE)
    match = gpd.sjoin_nearest(pts[["geometry"]], buildings[["geometry"]], max_distance=MATCH_DISTANCE)
    match = match[~match.index.duplicated()]["index_right"]
    pts["result"] = ["detected" if i in match.index else "missed" for i in pts.index]

    # detections that count: polygons in the section, plus any polygon matched to a point
    points_per_building = match.value_counts()
    in_section = buildings.representative_point().within(section) | buildings.index.isin(points_per_building.index)
    buildings = buildings[in_section].copy()
    buildings["n_points"] = points_per_building.reindex(buildings.index).fillna(0).astype(int).values
    buildings["result"] = ["detected" if n else "false alarm" for n in buildings["n_points"]]

    tp, fn = int((pts["result"] == "detected").sum()), int((pts["result"] == "missed").sum())
    tp_polygons, fp = int((buildings["n_points"] > 0).sum()), int((buildings["n_points"] == 0).sum())
    precision = tp_polygons / (tp_polygons + fp) if tp_polygons + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    counts = {
        "name": name, "ground_truth_buildings": tp + fn, "detected_polygons": tp_polygons + fp,
        "TP": tp, "FN": fn, "FP": fp,
        "recall_pct": round(100 * recall, 1),         # share of the real buildings that were detected
        "precision_pct": round(100 * precision, 1),   # share of the detections that are real buildings
        "f1_pct": round(100 * 2 * precision * recall / (precision + recall), 1) if precision + recall else float("nan"),
        # one polygon covering several ground-truth buildings (e.g. row houses detected as one)
        "merged_polygons": int((buildings["n_points"] > 1).sum()),
    }
    return pts, buildings, counts


def plot_confusion_matrix(c, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cells = [[(c["TP"], "detected (TP)", "#b7e1a1"), (c["FN"], "missed (FN)", "#f4b6a6")],
             [(c["FP"], "false alarm (FP)", "#f4b6a6"), ("-", "not countable (TN)", "#e6e6e6")]]
    for r, row in enumerate(cells):
        for k, (value, label, colour) in enumerate(row):
            ax.add_patch(plt.Rectangle((k, 1 - r), 1, 1, facecolor=colour, edgecolor="black"))
            ax.text(k + 0.5, 1.58 - r, str(value), ha="center", va="center", fontsize=22, fontweight="bold")
            ax.text(k + 0.5, 1.25 - r, label, ha="center", va="center", fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["detected", "not detected"])
    ax.set_yticks([1.5, 0.5], ["building", "no building"], rotation=90, va="center")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Detection (Open Buildings)")
    ax.set_ylabel("Ground truth")
    ax.tick_params(length=0)
    ax.set_title(f"{c['name']}:  recall {c['recall_pct']}%   precision {c['precision_pct']}%   F1 {c['f1_pct']}%",
                 fontsize=10, y=-0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_counts = []
    for name, points_path, buildings_path, section_path in VALIDATIONS:
        print(name)
        missing = [p for p in (points_path, buildings_path, section_path) if not Path(p).exists()]
        if missing:
            print(f"  {missing[0]} not found, skipped")
            continue
        pts, buildings, c = validate(name, points_path, buildings_path, section_path)
        all_counts.append(c)

        pd.DataFrame([[c["TP"], c["FN"]], [c["FP"], ""]],
                     index=["truth: building", "truth: no building"],
                     columns=["detected", "not detected"]).to_csv(out_dir / f"confusion_matrix_{name}.csv")
        plot_confusion_matrix(c, out_dir / f"confusion_matrix_{name}.png")
        # map layers: colour by the "result" field in QGIS
        pts.to_crs("EPSG:4326").to_file(out_dir / f"validation_points_{name}.geojson", driver="GeoJSON")
        buildings.to_crs("EPSG:4326").to_file(out_dir / f"validation_buildings_{name}.geojson", driver="GeoJSON")

        print(f"  {c['ground_truth_buildings']} ground-truth buildings, {c['detected_polygons']} detected polygons")
        print(f"  detected (TP) {c['TP']}, missed (FN) {c['FN']}, false alarms (FP) {c['FP']}")
        print(f"  recall {c['recall_pct']}%, precision {c['precision_pct']}%, F1 {c['f1_pct']}%")
        if c["merged_polygons"]:
            print(f"  {c['merged_polygons']} polygon(s) cover more than one ground-truth building")

    if all_counts:
        pd.DataFrame(all_counts).to_csv(out_dir / "validation_buildings_summary.csv", index=False)
        print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
