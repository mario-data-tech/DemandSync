"""
forecaster.py
=============

Motor de predicción de demanda de DemandSync.ai.

Implementa un pronóstico de Media Móvil Ponderada (Weighted Moving Average,
WMA) sobre series de ventas diarias por SKU, junto con la lógica de
recomendación de inventario (reposición / stock óptimo / sobre-stock).

El WMA da más peso a las observaciones recientes que a las antiguas, lo
que lo hace más reactivo que una media móvil simple ante cambios de
tendencia recientes en la demanda, manteniendo una implementación simple,
determinística y fácilmente auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class StockStatus(str, Enum):
    """Estado de inventario recomendado para un SKU."""

    REPONER = "REPONER"
    OK = "OK"
    SOBRE_STOCK = "SOBRE-STOCK"


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Resultado del pronóstico y recomendación de inventario para un SKU."""

    sku: str
    avg_daily_demand: float
    forecast_horizon_demand: float
    current_stock: float
    days_of_coverage: float
    status: StockStatus
    recommended_reorder_qty: float


class InsufficientDataError(ValueError):
    """Se lanza cuando no hay suficientes observaciones para pronosticar."""


class WeightedMovingAverageForecaster:
    """
    Pronosticador de demanda basado en Media Móvil Ponderada (WMA).

    La ponderación es lineal: la observación más reciente recibe el peso
    más alto y decrece linealmente hacia el pasado dentro de la ventana.

    Ejemplo con window=3 y pesos [1, 2, 3] (más reciente = mayor peso):
        WMA = (v_t-2 * 1 + v_t-1 * 2 + v_t * 3) / (1 + 2 + 3)

    Attributes:
        window: Cantidad de observaciones (días) a considerar en la ventana
            móvil. Debe ser >= 1.
    """

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError("window debe ser un entero >= 1")
        self.window = window

    def _weights(self, n: int) -> np.ndarray:
        """Genera pesos lineales crecientes de tamaño n, normalizados a suma 1."""
        raw_weights = np.arange(1, n + 1, dtype=float)
        return raw_weights / raw_weights.sum()

    def compute_wma(self, sales: pd.Series) -> float:
        """
        Calcula la Media Móvil Ponderada sobre las últimas `window`
        observaciones de una serie de ventas.

        Args:
            sales: Serie de ventas diarias, ordenada cronológicamente
                (el último valor es el más reciente).

        Returns:
            El valor de demanda diaria promedio ponderada.

        Raises:
            InsufficientDataError: si la serie tiene menos de 1 observación.
        """
        if sales.empty:
            raise InsufficientDataError(
                "No hay datos de ventas para calcular el pronóstico."
            )

        # Si hay menos observaciones que la ventana configurada, se usa
        # toda la serie disponible en lugar de fallar.
        effective_window = min(self.window, len(sales))
        recent_sales = sales.tail(effective_window).to_numpy(dtype=float)
        weights = self._weights(effective_window)

        return float(np.dot(recent_sales, weights))

    def forecast_sku(
        self,
        sku: str,
        sales_history: pd.Series,
        current_stock: float,
        lead_time_days: int = 7,
        safety_margin: float = 0.2,
    ) -> ForecastResult:
        """
        Genera el pronóstico y la recomendación de inventario para un SKU.

        Args:
            sku: Identificador del producto.
            sales_history: Serie histórica de ventas diarias del SKU,
                ordenada cronológicamente.
            current_stock: Unidades actualmente en stock.
            lead_time_days: Días que tarda en llegar un reabastecimiento;
                define el horizonte del pronóstico.
            safety_margin: Margen de seguridad adicional sobre la demanda
                proyectada del lead time (ej. 0.2 = 20% extra de colchón).

        Returns:
            Un ForecastResult con el diagnóstico y la recomendación.
        """
        avg_daily_demand = self.compute_wma(sales_history)
        forecast_horizon_demand = avg_daily_demand * lead_time_days

        days_of_coverage = (
            current_stock / avg_daily_demand if avg_daily_demand > 0 else float("inf")
        )

        reorder_point = forecast_horizon_demand * (1 + safety_margin)

        if current_stock < reorder_point:
            status = StockStatus.REPONER
            recommended_reorder_qty = max(reorder_point - current_stock, 0.0)
        elif current_stock > reorder_point * 2.5:
            status = StockStatus.SOBRE_STOCK
            recommended_reorder_qty = 0.0
        else:
            status = StockStatus.OK
            recommended_reorder_qty = 0.0

        return ForecastResult(
            sku=sku,
            avg_daily_demand=round(avg_daily_demand, 2),
            forecast_horizon_demand=round(forecast_horizon_demand, 2),
            current_stock=current_stock,
            days_of_coverage=round(days_of_coverage, 1)
            if days_of_coverage != float("inf")
            else days_of_coverage,
            status=status,
            recommended_reorder_qty=round(recommended_reorder_qty, 2),
        )

    def forecast_dataframe(
        self,
        df: pd.DataFrame,
        lead_time_days: int = 7,
        safety_margin: float = 0.2,
    ) -> list[ForecastResult]:
        """
        Genera pronósticos para todos los SKUs presentes en un DataFrame.

        Args:
            df: DataFrame con columnas obligatorias:
                - 'sku' (str)
                - 'date' (parseable a datetime)
                - 'units_sold' (numérico)
                - 'current_stock' (numérico, constante por SKU)
            lead_time_days: Ver `forecast_sku`.
            safety_margin: Ver `forecast_sku`.

        Returns:
            Lista de ForecastResult, uno por cada SKU único en el DataFrame.

        Raises:
            KeyError: si faltan columnas requeridas en el DataFrame.
        """
        required_columns = {"sku", "date", "units_sold", "current_stock"}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["sku", "date"])

        results: list[ForecastResult] = []
        for sku, group in df.groupby("sku", sort=False):
            sales_history = group["units_sold"]
            current_stock = float(group["current_stock"].iloc[-1])
            results.append(
                self.forecast_sku(
                    sku=str(sku),
                    sales_history=sales_history,
                    current_stock=current_stock,
                    lead_time_days=lead_time_days,
                    safety_margin=safety_margin,
                )
            )
        return results
