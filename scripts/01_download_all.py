"""CLI entry point for downloading all raw data sources.

Commands:
    discover  -- Discover Northern Thailand stations + write metadata.
    backfill  -- Backfill OpenAQ historical for curated stations.
    firms     -- Fetch FIRMS hotspots batched in 10-day chunks.
    era5      -- STUB - pending CDS profile completion.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import typer

from src.data.loader import build_station_metadata
from src.data.scrapers.firms import fetch_hotspots
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
    start_date: str = typer.Option(..., help="Start date in ISO 8601 format (YYYY-MM-DD)."),
    end_date: str = typer.Option(..., help="End date in ISO 8601 format (YYYY-MM-DD)."),
) -> None:
    """Fetch FIRMS hotspots batched in 10-day chunks."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if end < start:
        typer.echo("Error: end_date must be >= start_date.", err=True)
        raise typer.Exit(code=1)

    total_rows = 0
    n_chunks = 0
    chunk_start = start

    while chunk_start <= end:
        days_remaining = (end - chunk_start).days + 1
        days_in_chunk = min(10, days_remaining)

        typer.echo(
            f"  Fetching FIRMS chunk {chunk_start.isoformat()} " f"(day_range={days_in_chunk}) ..."
        )

        df_chunk = fetch_hotspots(date_=chunk_start, day_range=days_in_chunk)
        total_rows += len(df_chunk)
        n_chunks += 1

        chunk_start += timedelta(days=days_in_chunk)

    typer.echo(f"FIRMS done: {total_rows} hotspot rows across {n_chunks} chunks.")


@app.command()
def era5() -> None:
    """STUB - pending CDS profile completion."""
    typer.echo("ERA5 download not yet implemented. See docs/SESSION1_NOTES.md.")


if __name__ == "__main__":
    app()
