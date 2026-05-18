"""CLI entry point for downloading all raw data sources.

Commands:
    discover      -- Discover Northern Thailand stations + write metadata.
    backfill      -- Backfill OpenAQ historical for curated stations.
    firms         -- Fetch FIRMS hotspots for a date range (single source, chunked).
    firms-hybrid  -- Fetch FIRMS hotspots using SP for history + NRT for recent days.
    era5          -- STUB - pending CDS profile completion.
"""

import logging
from datetime import date
from pathlib import Path

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
def era5() -> None:
    """STUB - pending CDS profile completion."""
    typer.echo("ERA5 download not yet implemented. See docs/SESSION1_NOTES.md.")


if __name__ == "__main__":
    app()
