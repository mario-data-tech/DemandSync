"""
cli.py
======

Interfaz de línea de comandos de DemandSync.ai.

Uso:
    python -m src.cli analyze data/sample/sales.csv
    python -m src.cli analyze data/sample/sales.csv --window 5 --lead-time 10
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.forecaster import ForecastResult, StockStatus, WeightedMovingAverageForecaster

app = typer.Typer(
    name="demandsync",
    help="DemandSync.ai — Predicción de demanda y recomendaciones de inventario por SKU.",
    add_completion=False,
)
console = Console()

_STATUS_STYLE = {
    StockStatus.REPONER: "bold red",
    StockStatus.OK: "bold green",
    StockStatus.SOBRE_STOCK: "bold yellow",
}


def _load_sales_csv(csv_path: Path) -> pd.DataFrame:
    """Carga y valida el CSV de ventas desde disco."""
    if not csv_path.exists():
        console.print(f"[bold red]Error:[/bold red] no se encontró el archivo '{csv_path}'.")
        raise typer.Exit(code=1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 - se reporta al usuario y se aborta
        console.print(f"[bold red]Error leyendo el CSV:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    return df


def _render_results_table(results: list[ForecastResult]) -> Table:
    """Construye la tabla Rich con los resultados del pronóstico."""
    table = Table(title="DemandSync.ai — Recomendaciones de inventario por SKU")

    table.add_column("SKU", style="cyan", no_wrap=True)
    table.add_column("Demanda diaria (WMA)", justify="right")
    table.add_column("Demanda proyectada (lead time)", justify="right")
    table.add_column("Stock actual", justify="right")
    table.add_column("Días de cobertura", justify="right")
    table.add_column("Estado", justify="center")
    table.add_column("Cantidad sugerida a reponer", justify="right")

    for result in results:
        style = _STATUS_STYLE[result.status]
        coverage = (
            "∞" if result.days_of_coverage == float("inf") else f"{result.days_of_coverage}"
        )
        table.add_row(
            result.sku,
            f"{result.avg_daily_demand:.2f}",
            f"{result.forecast_horizon_demand:.2f}",
            f"{result.current_stock:.0f}",
            coverage,
            f"[{style}]{result.status.value}[/{style}]",
            f"{result.recommended_reorder_qty:.0f}" if result.recommended_reorder_qty else "-",
        )

    return table


@app.command()
def analyze(
    csv_path: Path = typer.Argument(
        ..., help="Ruta al CSV de ventas (columnas: sku, date, units_sold, current_stock)."
    ),
    window: int = typer.Option(
        7, "--window", "-w", help="Tamaño de la ventana móvil ponderada (en días)."
    ),
    lead_time: int = typer.Option(
        7, "--lead-time", "-l", help="Días de lead time de reabastecimiento."
    ),
    safety_margin: float = typer.Option(
        0.2, "--safety-margin", "-s", help="Margen de seguridad sobre la demanda proyectada."
    ),
) -> None:
    """Analiza un CSV de ventas y muestra recomendaciones de inventario por SKU."""
    df = _load_sales_csv(csv_path)

    forecaster = WeightedMovingAverageForecaster(window=window)

    try:
        results = forecaster.forecast_dataframe(
            df, lead_time_days=lead_time, safety_margin=safety_margin
        )
    except KeyError as exc:
        console.print(f"[bold red]Error de formato en el CSV:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]No se encontraron SKUs para analizar en el archivo.[/yellow]")
        raise typer.Exit(code=0)

    table = _render_results_table(results)
    console.print(table)

    reponer_count = sum(1 for r in results if r.status == StockStatus.REPONER)
    if reponer_count:
        console.print(
            f"\n[bold red]⚠ {reponer_count} SKU(s) requieren reposición inmediata.[/bold red]"
        )


@app.command()
def version() -> None:
    """Muestra la versión de DemandSync.ai."""
    console.print("DemandSync.ai — v0.1.0")


if __name__ == "__main__":
    app()
