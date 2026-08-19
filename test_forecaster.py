"""
test_forecaster.py
===================

Suite de pruebas unitarias para el motor de forecasting de DemandSync.ai.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecaster import (
    ForecastResult,
    InsufficientDataError,
    StockStatus,
    WeightedMovingAverageForecaster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def forecaster() -> WeightedMovingAverageForecaster:
    return WeightedMovingAverageForecaster(window=3)


@pytest.fixture
def constant_sales() -> pd.Series:
    """Serie con demanda constante: el WMA debe coincidir con ese valor."""
    return pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])


@pytest.fixture
def increasing_sales() -> pd.Series:
    """Serie con demanda creciente: el WMA debe ser mayor que la media simple."""
    return pd.Series([5.0, 10.0, 15.0, 20.0, 25.0])


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame multi-SKU válido, análogo al CSV de ejemplo del proyecto."""
    rows = []
    for day in range(1, 8):
        rows.append(
            {
                "sku": "SKU-A",
                "date": f"2026-08-{day:02d}",
                "units_sold": 10 + day,
                "current_stock": 50,
            }
        )
        rows.append(
            {
                "sku": "SKU-B",
                "date": f"2026-08-{day:02d}",
                "units_sold": 2,
                "current_stock": 200,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------


def test_init_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        WeightedMovingAverageForecaster(window=0)


def test_init_accepts_valid_window() -> None:
    fc = WeightedMovingAverageForecaster(window=5)
    assert fc.window == 5


# ---------------------------------------------------------------------------
# compute_wma
# ---------------------------------------------------------------------------


def test_wma_constant_series_equals_constant_value(
    forecaster: WeightedMovingAverageForecaster, constant_sales: pd.Series
) -> None:
    result = forecaster.compute_wma(constant_sales)
    assert result == pytest.approx(10.0)


def test_wma_weights_recent_observations_more(
    forecaster: WeightedMovingAverageForecaster, increasing_sales: pd.Series
) -> None:
    wma = forecaster.compute_wma(increasing_sales)
    simple_mean = increasing_sales.tail(3).mean()
    # Con una tendencia creciente, el WMA (que pondera más lo reciente)
    # debe ser mayor que la media simple de la misma ventana.
    assert wma > simple_mean


def test_wma_manual_calculation() -> None:
    fc = WeightedMovingAverageForecaster(window=3)
    sales = pd.Series([10.0, 20.0, 30.0])
    # Pesos normalizados para n=3: [1/6, 2/6, 3/6]
    expected = (10.0 * 1 + 20.0 * 2 + 30.0 * 3) / 6
    assert fc.compute_wma(sales) == pytest.approx(expected)


def test_wma_uses_effective_window_when_series_shorter_than_window() -> None:
    fc = WeightedMovingAverageForecaster(window=10)
    sales = pd.Series([4.0, 6.0])
    # Con menos observaciones que la ventana, se debe usar toda la serie.
    expected = (4.0 * 1 + 6.0 * 2) / 3
    assert fc.compute_wma(sales) == pytest.approx(expected)


def test_wma_raises_on_empty_series(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    with pytest.raises(InsufficientDataError):
        forecaster.compute_wma(pd.Series([], dtype=float))


# ---------------------------------------------------------------------------
# forecast_sku
# ---------------------------------------------------------------------------


def test_forecast_sku_returns_forecast_result(
    forecaster: WeightedMovingAverageForecaster, constant_sales: pd.Series
) -> None:
    result = forecaster.forecast_sku(
        sku="SKU-TEST",
        sales_history=constant_sales,
        current_stock=100.0,
        lead_time_days=7,
    )
    assert isinstance(result, ForecastResult)
    assert result.sku == "SKU-TEST"
    assert result.avg_daily_demand == pytest.approx(10.0)
    assert result.forecast_horizon_demand == pytest.approx(70.0)


def test_forecast_sku_status_reponer_when_stock_low(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    sales = pd.Series([20.0, 20.0, 20.0])
    result = forecaster.forecast_sku(
        sku="SKU-LOW", sales_history=sales, current_stock=10.0, lead_time_days=7
    )
    assert result.status == StockStatus.REPONER
    assert result.recommended_reorder_qty > 0


def test_forecast_sku_status_sobre_stock_when_stock_excessive(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    sales = pd.Series([1.0, 1.0, 1.0])
    result = forecaster.forecast_sku(
        sku="SKU-EXCESS", sales_history=sales, current_stock=1000.0, lead_time_days=7
    )
    assert result.status == StockStatus.SOBRE_STOCK
    assert result.recommended_reorder_qty == 0.0


def test_forecast_sku_status_ok_in_healthy_range(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    sales = pd.Series([10.0, 10.0, 10.0])
    # Demanda proyectada (lead_time=7) * (1+margin=0.2) ≈ 84 -> "OK" entre
    # el punto de reorden y 2.5x ese punto.
    result = forecaster.forecast_sku(
        sku="SKU-OK", sales_history=sales, current_stock=120.0, lead_time_days=7
    )
    assert result.status == StockStatus.OK
    assert result.recommended_reorder_qty == 0.0


def test_forecast_sku_days_of_coverage_is_infinite_when_no_demand(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    sales = pd.Series([0.0, 0.0, 0.0])
    result = forecaster.forecast_sku(
        sku="SKU-ZERO", sales_history=sales, current_stock=50.0, lead_time_days=7
    )
    assert result.days_of_coverage == float("inf")
    assert result.status == StockStatus.SOBRE_STOCK


# ---------------------------------------------------------------------------
# forecast_dataframe
# ---------------------------------------------------------------------------


def test_forecast_dataframe_returns_one_result_per_sku(
    forecaster: WeightedMovingAverageForecaster, sample_dataframe: pd.DataFrame
) -> None:
    results = forecaster.forecast_dataframe(sample_dataframe)
    skus = {r.sku for r in results}
    assert skus == {"SKU-A", "SKU-B"}
    assert len(results) == 2


def test_forecast_dataframe_raises_on_missing_columns(
    forecaster: WeightedMovingAverageForecaster,
) -> None:
    bad_df = pd.DataFrame({"sku": ["SKU-X"], "units_sold": [10]})
    with pytest.raises(KeyError):
        forecaster.forecast_dataframe(bad_df)


def test_forecast_dataframe_sorts_by_date_before_forecasting() -> None:
    """El orden de las filas en el CSV no debería afectar el resultado."""
    fc = WeightedMovingAverageForecaster(window=3)
    shuffled_rows = [
        {"sku": "SKU-Z", "date": "2026-08-03", "units_sold": 30, "current_stock": 100},
        {"sku": "SKU-Z", "date": "2026-08-01", "units_sold": 10, "current_stock": 100},
        {"sku": "SKU-Z", "date": "2026-08-02", "units_sold": 20, "current_stock": 100},
    ]
    df = pd.DataFrame(shuffled_rows)
    result = fc.forecast_dataframe(df)[0]

    expected = (10.0 * 1 + 20.0 * 2 + 30.0 * 3) / 6
    assert result.avg_daily_demand == pytest.approx(expected, rel=1e-2)
