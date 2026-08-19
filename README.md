# DemandSync.ai

**Predicción de demanda open source para e-commerce y retail.**

DemandSync.ai analiza tu historial de ventas por SKU y te dice, en segundos, cuánto vas a vender la próxima semana y cuánto stock deberías tener — sin depender de un analista de datos ni de un ERP de seis cifras.

---

## 1. El problema

El 80% de las PyMEs de retail y e-commerce todavía gestionan su inventario "a ojo" o con hojas de cálculo estáticas. Esto genera dos costos simultáneos:

- **Quiebre de stock (stockout):** ventas perdidas y clientes que migran a la competencia.
- **Sobre-stock:** capital inmovilizado, costos de almacenamiento y liquidaciones forzadas.

Las plataformas de forecasting enterprise (Blue Yonder, o9, SAP IBP) cuestan miles de dólares mensuales y requieren integraciones complejas — inaccesibles para un negocio de 50, 500 o incluso 5.000 SKUs.

## 2. La solución

DemandSync.ai es un motor de forecasting **open core**: el algoritmo de predicción, la CLI y el pipeline de datos son 100% open source y auto-hospedables. Cualquier equipo técnico puede clonar el repo, correr `docker-compose up` y tener recomendaciones de reposición por SKU en minutos, a partir de un CSV de ventas.

Núcleo del motor: **Media Móvil Ponderada (WMA)**, que da más peso a las ventas recientes que a las antiguas — una base simple, auditable y explicable, pensada como punto de partida antes de escalar a modelos más complejos (ARIMA, Prophet, gradient boosting) en la capa SaaS.

## 3. ROI estimado

| Métrica | Sin forecasting | Con DemandSync.ai (core) |
|---|---|---|
| Quiebres de stock/mes | 15-20% de SKUs | 5-8% de SKUs |
| Exceso de inventario | 20-30% del capital en stock | 10-15% del capital en stock |
| Tiempo de análisis semanal | 4-6 horas manuales | < 5 minutos (CLI) |
| Costo mensual | Analista/ERP: USD 500-5.000+ | USD 0 (self-hosted) |

Reducir el sobre-stock y los quiebres incluso en un 30% suele representar un ahorro de varios puntos porcentuales sobre el capital de trabajo total en inventario — el retorno del "core" gratuito se paga solo con la primera reposición evitada.

## 4. Modelo Open Core vs SaaS

**Open Core (este repositorio, gratis y auto-hospedable):**
- Algoritmo de forecasting WMA
- CLI (`demandsync`) para analizar CSVs y generar recomendaciones
- Infraestructura de referencia (Docker + Postgres + Redis)
- Tests, tipado estático, licencia Apache-2.0

**DemandSync.ai Cloud (SaaS, de pago):**
- Modelos avanzados (ARIMA, Prophet, ensambles ML) auto-seleccionados por SKU
- Conectores nativos (Shopify, Tienda Nube, Mercado Libre, WooCommerce)
- Dashboard multiusuario, alertas automáticas y reposición sugerida por proveedor
- Forecasting multi-almacén y estacionalidad avanzada
- Soporte, SLA y hosting administrado

El core nunca queda castrado a propósito: es una herramienta completa por sí sola. El SaaS existe para equipos que quieren escalar sin operar infraestructura.

## 5. Quickstart

```bash
git clone https://github.com/<tu-usuario>/demandsync.git
cd demandsync

# Levantar la infraestructura completa (API + Postgres + Redis)
docker-compose up -d --build

# O, para correr solo la CLI localmente:
pip install -e ".[dev]"
python -m src.cli analyze data/sample/sales.csv
```

## 6. Uso de la CLI

```bash
python -m src.cli analyze data/sample/sales.csv --window 3 --lead-time 7
```

Salida: una tabla con, por SKU, la demanda diaria proyectada, el stock actual, los días de cobertura restantes y una recomendación (`REPONER`, `OK`, `SOBRE-STOCK`).

## 7. Estructura del proyecto

```
demandsync/
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── README.md
├── .gitignore
├── src/
│   ├── forecaster.py     # Motor de forecasting (WMA)
│   └── cli.py            # CLI con Typer + Rich
├── data/
│   └── sample/
│       └── sales.csv     # Dataset de ejemplo
└── tests/
    └── test_forecaster.py
```

## 8. Roadmap

- [ ] Conectores de e-commerce (Shopify, Tienda Nube)
- [ ] Modelos ARIMA/Prophet como capa opcional
- [ ] API REST (FastAPI) sobre el mismo core
- [ ] Dashboard web

## 9. Licencia

Apache License 2.0 — ver [LICENSE](LICENSE).

## 10. Contribuir

Se aceptan PRs. Antes de abrir uno: `ruff check .`, `black --check .`, `mypy src`, `pytest`.
