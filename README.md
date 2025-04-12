# hpiPy

**hpiPy** simplifies and standardizes the creation of house price indices in Python.

The package provides tools to evaluate index quality through predictive accuracy, volatility, and revision metrics—enabling meaningful comparisons across different methods and estimators. It focuses on the most widely used approaches: repeat sales and hedonic pricing models, with support for base, robust, and weighted estimators where applicable. The package also includes a random forest–based method with partial dependence plots for post-model interpretability, as well as a neural network approach that separates property-specific and market-level effects to jointly estimate quality and index components from property-level data. The package is based on [hpiR](https://github.com/andykrause/hpiR).

## Running the Tests

1. Create a virtual environment (we recommend [`uv`](https://github.com/astral-sh/uv)):

```bash
uv venv
```

2. Install base and development dependencies:

```bash
uv pip install --requirements pyproject.toml --extra dev
```

3. Run the test suite:

```bash
pytest
```

## Example

A basic example of creating a house price index:

```python
import pandas as pd
from hpipy.price_index import RepeatTransactionIndex

# Load prepared data.
df = pd.read_csv("ex_sales.csv", parse_dates=["sale_date"])

# Create an index.
hpi = RepeatTransactionIndex.create_index(
    trans_data=df,
    date="sale_date",
    price="sale_price",
    prop_id="pinx",
    trans_id="sale_id",
    estimator="robust",
    log_dep=True,
    smooth=True,
)
```

## Acknowledgements

Based on the [hpiR package](https://github.com/andykrause/hpiR).
