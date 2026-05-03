from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ansitable import ANSITable, Column

from ..base.types import NumpyArray


__all__ = ["array_to_table"]


FormatSpecifier = str | Callable[[Any], str]


def array_to_table(
    array: NumpyArray,
    headers: Sequence[str] | None = None,
    *,
    fmt: FormatSpecifier = "{:.6g}",
    border: str | None = "ascii",
    index: bool = False,
    index_header: str = "",
    colalign: str = ">",
    headalign: str = ">",
    **table_kwargs: Any,
) -> str:
    """
    Convert a 1D or 2D NumPy array to a text table.

    Parameters
    ----------
    array : NumpyArray
        Array with one or two dimensions.
    headers : Sequence[str] | None, default=None
        Optional column headers. If omitted, generated headers are used.
    fmt : str | Callable[[Any], str], default="{:.6g}"
        Format string or callable used for each array value.
    border : str | None, default="ascii"
        ANSITable border style. Examples are "ascii", "thin", "thick",
        "double", or None.
    index : bool, default=False
        If True, add a leading zero-based row index column.
    index_header : str, default=""
        Header for the optional row index column.
    colalign : str, default=">"
        ANSITable column alignment for array values.
    headalign : str, default=">"
        ANSITable header alignment.
    **table_kwargs : Any
        Extra keyword arguments passed to ``ANSITable``.

    Returns
    -------
    str
        Text representation of the array.

    Raises
    ------
    ValueError
        If the array is not 1D or 2D, or if the number of headers does not
        match the number of columns.
    """
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("array_to_table() expects a 1D or 2D array.")

    n_cols = arr.shape[1]
    if headers is None:
        headers = (
            ("value",)
            if n_cols == 1
            else tuple(f"col_{i}" for i in range(n_cols))
        )
    elif len(headers) != n_cols:
        raise ValueError(
            f"Expected {n_cols} headers for an array with {n_cols} columns, "
            f"got {len(headers)}."
        )

    columns = [
        Column(str(header), fmt=fmt, colalign=colalign, headalign=headalign)
        for header in headers  #type: ignore
    ]
    if index:
        columns.insert(
            0,
            Column(index_header, fmt="{}", colalign=">", headalign=headalign),
        )

    kwargs = {"border": border, "color": False}
    kwargs.update(table_kwargs)
    table = ANSITable(*columns, **kwargs)

    for i, row in enumerate(arr):
        values: tuple[Any, ...] = tuple(row.tolist())
        if index:
            values = (i, *values)
        table.row(*values)

    return str(table).rstrip()
