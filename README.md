# hpiPy
House price indices in Python.


## Running the tests

```
poetry install
```

```
poetry run pytest
```

## Example

This is a basic example of creating a house price index:

```python
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
