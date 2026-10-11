"""
Coastline change analysis: compares the yearly coastlines made by coastline_detection.py.
Winter and summer are analysed separately.

Method (transects):
  1. The oldest coastline of a section is the baseline.
  2. Every TRANSECT_SPACING metres a line is drawn straight across the smoothed baseline (a transect).
  3. On each transect, the position of every year's coastline is measured in metres.
  4. Position differences between years = how far the coast moved at that spot.
     Positive = coast moved seaward (land gained), negative = moved landward (erosion).

Summer vs winter: the same transects are also used to compare the two seasons of the same
year (seasonal_* files), see SEASON_COMPARISON.

Set the options below, then run:  python coastline_change.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== SETTINGS ========================
INPUT_FOLDER = "output_coastline"
OUTPUT_FOLDER = "output_change"

ISLANDS = {
    "aruba": (-69.97, 12.51),
    "bonaire": (-68.27, 12.18),
}
SEASONS = ["winter", "summer"]

FIRST_YEAR = 2016
LAST_YEAR = 2023

FIRST_VS_LAST_ONLY = False

SEASON_COMPARISON = ("summer", "winter")

TRANSECT_SPACING = 50    # metres between transects along the coast
TRANSECT_LENGTH = 200    # metres a transect reaches to each side of the baseline
                         # (coast movements bigger than this are not measured)
BASELINE_SMOOTHING = 300 # metres; the baseline is smoothed over this distance so transects
                         # follow the general coast direction instead of every pixel step,
                         # small bump or sharp corner (too low = gaps and spikes in the chart)

# Which crossing to use when a transect crosses the coastline more than once
# (hooks, spits, small bays, inlets):
#   "landward" = the most inland one, so a loop or spit sticking out into the sea is ignored
#   "seaward"  = the outermost one
#   "closest"  = the one closest to the baseline (can jump between shores from year to year)
CROSSING = "landward"

# ==========================================================


def longest_part(geom):
    """The longest single line of a coastline. Parts that touch end to end are joined first,
    so a coastline saved in several pieces still gives one long baseline."""
    parts = list(getattr(geom, "geoms", [geom]))
    joined = unary_union(parts)
    if joined.geom_type == "MultiLineString":
        joined = linemerge(joined)
    return max(parts + list(getattr(joined, "geoms", [joined])), key=lambda g: g.length)


def smooth_line(line, window, step=5):
    """Moving average of the line over `window` metres (removes pixel steps and small bumps)."""
    n = max(int(line.length / step), 2)
    pts = np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, n + 1)])
    k = max(int(window / step), 1) | 1   # odd number of points in the window
    if k >= len(pts):
        return line
    padded = np.pad(pts, ((k // 2, k // 2), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return LineString(np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)]))


def make_transects(baseline, centre):
    """List of (distance along baseline, centre point, seaward unit vector)."""
    baseline = smooth_line(baseline, BASELINE_SMOOTHING)
    out = []
    for d in np.arange(TRANSECT_SPACING / 2, baseline.length, TRANSECT_SPACING):
        p = baseline.interpolate(d)
        a = baseline.interpolate(max(d - BASELINE_SMOOTHING / 2, 0))
        b = baseline.interpolate(min(d + BASELINE_SMOOTHING / 2, baseline.length))
        tx, ty = b.x - a.x, b.y - a.y
        size = np.hypot(tx, ty)
        if size == 0:
            continue
        nx, ny = -ty / size, tx / size                       # perpendicular to the coast
        if nx * (p.x - centre.x) + ny * (p.y - centre.y) < 0:  # make it point away from the island
            nx, ny = -nx, -ny
        out.append((d, p, (nx, ny)))
    return out


def transect_line(p, n):
    return LineString([(p.x - n[0] * TRANSECT_LENGTH, p.y - n[1] * TRANSECT_LENGTH),
                       (p.x + n[0] * TRANSECT_LENGTH, p.y + n[1] * TRANSECT_LENGTH)])


def coast_position(coast, p, n):
    """Where the coastline crosses the transect, in metres from the baseline (+ = seaward)."""
    hit = transect_line(p, n).intersection(coast)
    if hit.is_empty:
        return np.nan
    dists = [(x - p.x) * n[0] + (y - p.y) * n[1]
             for g in getattr(hit, "geoms", [hit]) for x, y in g.coords]
    if CROSSING == "landward":
        return min(dists)
    return max(dists) if CROSSING == "seaward" else min(dists, key=abs)


def year_coasts(gdf):
    """{year: coastline}. All line parts of a year, kept whole (a union would cut a line
    that touches itself into pieces)."""
    return {int(y): MultiLineString([part for geom in g.geometry for part in getattr(geom, "geoms", [geom])])
            for y, g in gdf.groupby("year")}


def analyse_section(season, section, gdf, centre, to_wgs):
    """Returns (transect table, periods for the summary, periods for the chart, baseline)."""
    coasts = year_coasts(gdf)
    years = sorted(coasts)
    if len(years) < 2:
        print(f"  {section}: only {len(years)} year(s), nothing to compare")
        return None, [], [], None
    all_years = years   # the chart always shows every year
    if FIRST_VS_LAST_ONLY:
        years = [years[0], years[-1]]

    rows = []
    baseline = longest_part(coasts[years[0]])
    for i, (d, p, n) in enumerate(make_transects(baseline, centre)):
        pos = {y: coast_position(coasts[y], p, n) for y in all_years}
        row = {"season": season, "section": section, "transect_id": i, "distance_along_m": round(d, 1),
               "first_year": years[0], "last_year": years[-1]}
        row.update({f"pos_{y}": pos[y] for y in years})
        for y0, y1 in zip(all_years[:-1], all_years[1:]):
            row[f"change_{y0}_{y1}"] = pos[y1] - pos[y0]

        ok = [y for y in years if np.isfinite(pos[y])]
        vals = [pos[y] for y in ok]
        row["net_change_m"] = vals[-1] - vals[0] if len(ok) >= 2 else np.nan         # last - first year
        row["rate_m_per_year"] = np.polyfit(ok, vals, 1)[0] if len(ok) >= 2 else np.nan  # trend line
        row["range_m"] = max(vals) - min(vals) if len(ok) >= 2 else np.nan           # most seaward - most landward
        row["geometry"] = transect_line(p, n)
        row["_point"] = p
        rows.append(row)
    if not rows:
        return None, [], [], None

    table = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    centres = gpd.GeoSeries(table.pop("_point"), crs=gdf.crs).to_crs(to_wgs)
    table["lon"], table["lat"] = centres.x.round(6).values, centres.y.round(6).values
    plot_periods = [f"change_{y0}_{y1}" for y0, y1 in zip(all_years[:-1], all_years[1:])]
    periods = [] if FIRST_VS_LAST_ONLY else plot_periods
    return table, periods, plot_periods, baseline


def summarise(table, periods):
    """One row per period: average change and where the biggest changes were."""
    rows = []
    for col in periods + ["net_change_m"]:
        t = table.dropna(subset=[col])
        if t.empty:
            continue
        sea, land = t.loc[t[col].idxmax()], t.loc[t[col].idxmin()]
        rows.append({
            "season": t["season"].iloc[0], "section": t["section"].iloc[0],
            "period": col.replace("change_", "").replace("_", "-") if col != "net_change_m" else "first-last",
            "first_year": t["first_year"].iloc[0], "last_year": t["last_year"].iloc[0],
            "n_transects": len(t),
            "mean_change_m": t[col].mean(), "median_change_m": t[col].median(),
            "mean_abs_change_m": t[col].abs().mean(),   # how much the coast moved, either direction
            "pct_landward": 100 * (t[col] < 0).mean(),  # share of transects that eroded
            "max_seaward_m": sea[col], "max_seaward_lon": sea["lon"], "max_seaward_lat": sea["lat"],
            "max_landward_m": land[col], "max_landward_lon": land["lon"], "max_landward_lat": land["lat"],
        })
    return round_metres(pd.DataFrame(rows))


def round_metres(df):
    """Round the metre values to cm, keep lon/lat precise."""
    cols = [c for c in df.select_dtypes("float").columns if not c.endswith(("lon", "lat"))]
    df[cols] = df[cols].round(2)
    return df


def section_overview(summary, baseline):
    """One row for the section: overall change + its most and least changed period."""
    row = {"season": summary["season"].iloc[0], "section": summary["section"].iloc[0]}
    yearly = summary[summary["period"] != "first-last"]
    if not yearly.empty:
        most = yearly.loc[yearly["mean_abs_change_m"].idxmax()]
        least = yearly.loc[yearly["mean_abs_change_m"].idxmin()]
        row.update({"most_changed_period": most["period"], "most_changed_m": most["mean_abs_change_m"],
                    "least_changed_period": least["period"], "least_changed_m": least["mean_abs_change_m"]})
    net = summary[summary["period"] == "first-last"]
    if not net.empty:
        net = net.iloc[0]
        row.update({"first_year": net["first_year"], "last_year": net["last_year"],
                    "net_mean_change_m": net["mean_change_m"], "net_mean_abs_change_m": net["mean_abs_change_m"],
                    "net_pct_landward": net["pct_landward"],
                    "net_max_seaward_m": net["max_seaward_m"], "net_max_landward_m": net["max_landward_m"]})
        row["label"] = f"{row['section']}: {net['mean_change_m']:+.1f} m"
    row["geometry"] = baseline
    return row


def hotspots(summary):
    """Points where the biggest seaward and landward change of each period were."""
    rows = []
    for _, r in summary.iterrows():
        for kind in ("seaward", "landward"):
            rows.append({"season": r["season"], "section": r["section"], "period": r["period"],
                         "type": f"biggest {kind}", "change_m": r[f"max_{kind}_m"],
                         "label": f"{r[f'max_{kind}_m']:+.1f} m ({r['first_year']}-{r['last_year']})"
                         if r["period"] == "first-last" else f"{r[f'max_{kind}_m']:+.1f} m ({r['period']})",
                         "geometry": Point(r[f"max_{kind}_lon"], r[f"max_{kind}_lat"])})
    return rows


def save_plot(table, periods, out_path):
    """Change along the coast, one line per period."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in periods:
        ax.plot(table["distance_along_m"], table[col], linewidth=1, label=col.replace("change_", "").replace("_", "-"))
    ax.plot(table["distance_along_m"], table["net_change_m"], color="black", linewidth=2, label=f"{table['first_year'].iloc[0]}-{table['last_year'].iloc[0]}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("distance along the coast (m)")
    ax.set_ylabel("change (m)   + seaward / - landward")
    ax.set_title(f"{out_path.parent.name}: {out_path.stem}")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_coastlines():
    """All coastlines in INPUT_FOLDER, read from the per-image *_coastline.geojson files,
    with an 'island' column taken from the file name."""
    frames, skipped = [], 0
    for f in sorted(Path(INPUT_FOLDER).glob("*_coastline.geojson")):
        gdf = gpd.read_file(f)
        island = next((i for i in ISLANDS if i in f.name.lower()), None)
        if island is None or not {"year", "season", "section"} <= set(gdf.columns):
            skipped += 1   # no island in the name, or made by an old version without year/season
            continue
        gdf["island"] = island
        frames.append(gdf.to_crs("EPSG:4326"))
    if skipped:
        print(f"{skipped} file(s) skipped: no island name in the file name, or no year/season/section attributes")
    if not frames:
        return None
    data = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    data = data.dropna(subset=["year", "season", "section"])
    in_range = data["year"].astype(int).between(FIRST_YEAR, LAST_YEAR)
    if not in_range.all():
        ignored = sorted(set(data.loc[~in_range, "year"].astype(int)))
        print(f"years outside {FIRST_YEAR}-{LAST_YEAR} ignored: {', '.join(map(str, ignored))}")
    return data[in_range]


def compare_seasons(data, island_centre, out_dir):
    """Summer vs winter of the same year, per section, on one shared set of transects."""
    a, b = SEASON_COMPARISON
    gdf_a, gdf_b = data[data["season"] == a], data[data["season"] == b]
    if gdf_a.empty or gdf_b.empty:
        print(f"{a} vs {b}: needs coastlines of both seasons, skipped")
        return
    print(f"{a} vs {b}")
    utm = gdf_a.estimate_utm_crs()
    gdf_a, gdf_b = gdf_a.to_crs(utm), gdf_b.to_crs(utm)
    centre = gpd.GeoSeries([Point(island_centre)], crs="EPSG:4326").to_crs(utm).iloc[0]

    tables, summaries = [], []
    for section in sorted(set(gdf_a["section"]) & set(gdf_b["section"])):
        coasts_a = year_coasts(gdf_a[gdf_a["section"] == section])
        coasts_b = year_coasts(gdf_b[gdf_b["section"] == section])
        years = sorted(set(coasts_a) & set(coasts_b))   # years that have both seasons
        if not years:
            print(f"  {section}: no year with both {a} and {b}, skipped")
            continue

        rows = []
        for i, (d, p, n) in enumerate(make_transects(longest_part(coasts_b[years[0]]), centre)):
            diffs = [coast_position(coasts_a[y], p, n) - coast_position(coasts_b[y], p, n) for y in years]
            ok = [v for v in diffs if np.isfinite(v)]
            row = {"section": section, "transect_id": i, "distance_along_m": round(d, 1)}
            row.update({f"diff_{y}": v for y, v in zip(years, diffs)})
            row["mean_diff_m"] = np.mean(ok) if ok else np.nan       # the usual seasonal difference here
            row["std_diff_m"] = np.std(ok) if ok else np.nan         # how much it varies between years
            row["n_years"] = len(ok)
            row["geometry"] = transect_line(p, n)
            row["_point"] = p
            rows.append(row)
        if not rows:
            continue
        table = gpd.GeoDataFrame(rows, geometry="geometry", crs=utm)
        centres = gpd.GeoSeries(table.pop("_point"), crs=utm).to_crs("EPSG:4326")
        table["lon"], table["lat"] = centres.x.round(6).values, centres.y.round(6).values
        tables.append(table)
        save_season_plot(table, years, a, b, out_dir / f"seasonal_{section}.png")

        # report for this section: one line per year
        print(f"  {section}")
        for y in years:
            v = table[f"diff_{y}"].dropna()
            if v.empty:
                continue
            summaries.append({"section": section, "year": y, "n_transects": len(v),
                              "mean_diff_m": v.mean(), "median_diff_m": v.median(),
                              "mean_abs_diff_m": v.abs().mean(),
                              f"pct_{a}_seaward": 100 * (v > 0).mean(),
                              "max_diff_m": v.max(), "min_diff_m": v.min()})
            print(f"    {y}: {a} coast on average {v.mean():+.1f} m vs {b} "
                  f"({v.abs().mean():.1f} m apart, {a} further seaward on {100 * (v > 0).mean():.0f}% of the coast)")
        overall = table["mean_diff_m"].mean()
        print(f"    all years: {a} coast on average {overall:+.1f} m vs {b}")
    if not tables:
        return

    transects = round_metres(gpd.GeoDataFrame(pd.concat(tables, ignore_index=True), crs=utm))
    transects.to_crs("EPSG:4326").to_file(out_dir / "seasonal_transects.geojson", driver="GeoJSON")
    transects.drop(columns="geometry").to_csv(out_dir / "seasonal_transects.csv", index=False)
    round_metres(pd.DataFrame(summaries)).to_csv(out_dir / "seasonal_summary.csv", index=False)


def save_season_plot(table, years, a, b, out_path):
    """Top: seasonal difference for every year and place (colours). Bottom: the average over the years."""
    x = table["distance_along_m"].values
    data = np.array([table[f"diff_{y}"].values for y in years])
    x_edges = np.append(x - TRANSECT_SPACING / 2, x[-1] + TRANSECT_SPACING / 2)
    finite = np.abs(data[np.isfinite(data)])
    limit = max(np.percentile(finite, 98), 1) if finite.size else 1   # colour range, ignoring extreme spikes

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 3 + 0.4 * len(years) + 3), sharex=True, layout="constrained",
                                      gridspec_kw={"height_ratios": [max(0.4 * len(years), 1.5), 3]})
    mesh = top.pcolormesh(x_edges, np.arange(len(years) + 1), data, cmap="RdBu", vmin=-limit, vmax=limit)
    top.set_yticks(np.arange(len(years)) + 0.5, [str(y) for y in years])
    top.invert_yaxis()
    top.set_facecolor("lightgrey")   # grey = not measured
    top.set_title(f"{out_path.parent.name} {out_path.stem}: {a} minus {b} coastline position, same year\n"
                  f"blue = {a} coast further seaward, red = {a} coast further landward")
    fig.colorbar(mesh, ax=[top, bottom], label=f"{a} - {b} (m)", extend="both", shrink=0.8)

    mean, std = table["mean_diff_m"].values, table["std_diff_m"].values
    bottom.fill_between(x, mean - std, mean + std, color="grey", alpha=0.3, label="variation between years (± 1 std)")
    bottom.plot(x, mean, color="black", linewidth=2, label="average of all years")
    bottom.axhline(0, color="grey", linewidth=0.8)
    bottom.set_xlabel("distance along the coast (m)")
    bottom.set_ylabel(f"{a} - {b} (m)")
    bottom.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyse_island(island, island_centre, data):
    out_dir = Path(OUTPUT_FOLDER) / island
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        gdf = data[data["season"] == season]
        if gdf.empty:
            print(f"no {season} coastlines, skipped")
            continue
        print(season)
        utm = gdf.estimate_utm_crs()   # metres instead of degrees
        gdf = gdf.to_crs(utm)
        centre = gpd.GeoSeries([Point(island_centre)], crs="EPSG:4326").to_crs(utm).iloc[0]

        tables, summaries, overviews = [], [], []
        for section, part in gdf.groupby("section"):
            table, periods, plot_periods, baseline = analyse_section(season, section, part, centre, "EPSG:4326")
            if table is None:
                continue
            summary = summarise(table, periods)
            overview = section_overview(summary, baseline)
            # the year-to-year columns are only for the chart when comparing first vs last
            tables.append(table.drop(columns=plot_periods) if FIRST_VS_LAST_ONLY else table)
            summaries.append(summary)
            overviews.append(overview)
            save_plot(table, plot_periods, out_dir / f"change_{season}_{section}.png")

            # report for this section
            print(f"  {section}")
            for _, r in summary.iterrows():
                period = f"{r['first_year']}-{r['last_year']}" if r["period"] == "first-last" else r["period"]
                print(f"    {period}: mean {r['mean_change_m']:+.1f} m, "
                      f"moved {r['mean_abs_change_m']:.1f} m on average, {r['pct_landward']:.0f}% landward | "
                      f"biggest seaward {r['max_seaward_m']:+.1f} m at ({r['max_seaward_lon']}, {r['max_seaward_lat']}) | "
                      f"biggest landward {r['max_landward_m']:+.1f} m at ({r['max_landward_lon']}, {r['max_landward_lat']})")
            if "most_changed_period" in overview:
                print(f"    most change:  {overview['most_changed_period']} ({overview['most_changed_m']:.1f} m on average)")
                print(f"    least change: {overview['least_changed_period']} ({overview['least_changed_m']:.1f} m on average)")
        if not tables:
            continue

        transects = round_metres(gpd.GeoDataFrame(pd.concat(tables, ignore_index=True), crs=utm))
        summary = pd.concat(summaries, ignore_index=True)
        transects.to_crs("EPSG:4326").to_file(out_dir / f"transects_{season}.geojson", driver="GeoJSON")
        transects.drop(columns="geometry").to_csv(out_dir / f"transects_{season}.csv", index=False)
        summary.to_csv(out_dir / f"summary_{season}.csv", index=False)

        # map layers: one line per section, and points at the biggest changes
        sections = gpd.GeoDataFrame(overviews, geometry="geometry", crs=utm).to_crs("EPSG:4326")
        sections.to_file(out_dir / f"map_sections_{season}.geojson", driver="GeoJSON")
        sections.drop(columns=["geometry", "label"], errors="ignore").to_csv(out_dir / f"sections_{season}.csv", index=False)
        gpd.GeoDataFrame(hotspots(summary), geometry="geometry", crs="EPSG:4326") \
            .to_file(out_dir / f"map_hotspots_{season}.geojson", driver="GeoJSON")

    if SEASON_COMPARISON:
        compare_seasons(data, island_centre, out_dir)


def main():
    data = load_coastlines()
    if data is None:
        print(f"No usable *_coastline.geojson files found in {INPUT_FOLDER}")
        return
    for island, island_centre in ISLANDS.items():
        print(f"=== {island} ===")
        island_data = data[data["island"] == island]
        if island_data.empty:
            print("no coastlines found, skipped")
            continue
        analyse_island(island, island_centre, island_data)


if __name__ == "__main__":
    main()