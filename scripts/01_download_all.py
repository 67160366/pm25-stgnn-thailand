"""CLI entry point for downloading all raw data sources.

Commands:
    discover      -- Discover Northern Thailand stations + write metadata.
    backfill      -- Backfill OpenAQ historical for curated stations.
    firms         -- Fetch FIRMS hotspots for a date range (single source, chunked).
    firms-hybrid  -- Fetch FIRMS hotspots using SP for history + NRT for recent days.
    era5          -- Download ERA5 reanalysis via CDS API and save to data/processed/.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from src.data.loader import build_station_metadata
from src.data.scrapers.firms import (
    BBOX_NORTHERN_THAILAND,
    fetch_hotspots_historical,
    fetch_hotspots_hybrid,
)
from src.data.scrapers.openaq import backfill_station

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_ERA5_RAW_DIR = Path("data/raw/era5")
_ERA5_PARQUET = Path("data/processed/era5.parquet")


@app.command()
def discover() -> None:
    """Discover Northern Thailand stations + write metadata."""
    df = build_station_metadata()
    typer.echo(f"Discovered {len(df)} curated stations.")
    typer.echo(df.head(10).to_string())


@app.command()
def backfill(
    start_year: int = typer.Option(2022, help="First year to backfill (inclusive)."),
    end_year: int = typer.Option(2025, help="Last year to backfill (inclusive)."),
    stations: str = typer.Option("all", help="Comma-separated location_ids to backfill, or 'all'."),
) -> None:
    """Backfill OpenAQ historical for curated stations."""
    df = build_station_metadata(use_cache=True)

    if stations != "all":
        requested_ids = [int(loc_id.strip()) for loc_id in stations.split(",")]
        df = df[df["location_id"].isin(requested_ids)].reset_index(drop=True)
        typer.echo(f"Filtered to {len(df)} station(s) matching location_ids: {requested_ids}")
    else:
        typer.echo(f"Backfilling all {len(df)} curated station(s).")

    output_dir = Path("data/raw/openaq")

    for _, row in df.iterrows():
        sensor_id = int(row["sensor_id_pm25"])
        typer.echo(
            f"  Backfilling sensor_id={sensor_id} ({row['name']}) " f"{start_year}-{end_year} ..."
        )
        paths = backfill_station(
            sensor_id=sensor_id,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
        )
        typer.echo(f"    -> {len(paths)} file(s) written/confirmed.")

    typer.echo("Backfill complete.")


@app.command()
def firms(
    start_date: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD), inclusive."),
    end_date: str = typer.Option("2025-12-31", help="End date (YYYY-MM-DD), inclusive."),
    source: str = typer.Option(
        "VIIRS_NOAA20_SP",
        help=(
            "FIRMS data source key.  Use VIIRS_NOAA20_SP for historical backfill "
            "(2022-01-01 through ~today-60d) or VIIRS_NOAA20_NRT for the last 10 days."
        ),
    ),
) -> None:
    """Fetch FIRMS hotspots for a date range using a single source, batched in 10-day chunks.

    For a combined SP + NRT approach that maximises coverage, use the
    firms-hybrid command instead.
    """
    try:
        df = fetch_hotspots_historical(
            bbox=BBOX_NORTHERN_THAILAND,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"FIRMS done: {len(df)} hotspot rows, source={source} {start_date}->{end_date}.")


@app.command(name="firms-hybrid")
def firms_hybrid(
    start_date: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD), inclusive."),
    end_date: str = typer.Option(
        None, help="End date (YYYY-MM-DD), inclusive.  Defaults to today."
    ),
    sp_source: str = typer.Option(
        "VIIRS_NOAA20_SP",
        help="Standard Product source key for historical data.",
    ),
    nrt_source: str = typer.Option(
        "VIIRS_NOAA20_NRT",
        help="Near Real-Time source key for recent data (last ~10 days).",
    ),
) -> None:
    """Fetch FIRMS hotspots using SP for history and NRT for recent days.

    Observed SP lag as of 2026-05-18: ~48 days (conservative cutoff: 60 days).
    A gap of roughly 38-48 days between SP end and NRT start cannot be filled
    from the FIRMS area CSV API — this is a NASA pipeline constraint.
    """
    resolved_end = end_date if end_date else date.today().isoformat()

    try:
        df = fetch_hotspots_hybrid(
            bbox=BBOX_NORTHERN_THAILAND,
            start_date=start_date,
            end_date=resolved_end,
            sp_source=sp_source,
            nrt_source=nrt_source,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"FIRMS hybrid done: {len(df)} hotspot rows, "
        f"SP={sp_source} + NRT={nrt_source}, {start_date}->{resolved_end}."
    )


@app.command()
def era5(
    start_year: int = typer.Option(2022, help="First year to download (inclusive)."),
    end_year: int = typer.Option(2025, help="Last year to download (inclusive)."),
    output_dir: Annotated[
        Path, typer.Option(help="Directory for raw NetCDF files.")
    ] = _ERA5_RAW_DIR,
    processed_path: Annotated[
        Path,
        typer.Option(help="Destination parquet for the tidy station-interpolated output."),
    ] = _ERA5_PARQUET,
) -> None:
    """Download ERA5 reanalysis (u10, v10, t2m, d2m, blh) via CDS API.

    Requires a valid ~/.cdsapirc with your CDS API key.
    See https://cds.climate.copernicus.eu/how-to-api for setup instructions.

    Downloads one NetCDF per year into output_dir (skips existing files),
    then bilinearly interpolates to the 18 station locations and writes a
    tidy parquet to processed_path.
    """
    from datetime import date as _date

    from src.data.scrapers.era5 import fetch_era5

    date_from = _date(start_year, 1, 1)
    date_to = _date(end_year, 12, 31)

    typer.echo(f"Downloading ERA5 {start_year}-{end_year} -> {output_dir} ...")
    typer.echo("(This may take 10-30 min per year. Existing files are skipped.)")

    try:
        df = fetch_era5(
            date_from=date_from,
            date_to=date_to,
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        typer.echo(f"ERA5 download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)

    n_rows = len(df)
    n_stations = df["station_id"].nunique() if n_rows > 0 else 0
    typer.echo(f"ERA5 done: {n_rows:,} rows, {n_stations} stations -> {processed_path}")


if __name__ == "__main__":
    app()
