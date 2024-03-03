"""Error utilities."""


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if model is used before fitting."""
