from __future__ import annotations


__all__ = [
    "DALoadByStep",
    "DSLoadByStep",
]


import itertools
import time
from collections.abc import Mapping, Iterable
from typing import Annotated, Any
from pathlib import Path

import numpy as np
import psutil
import xarray as xr
from pydantic import (
    AfterValidator,
    ByteSize,
    Field,
    NewPath,
    NonNegativeFloat,
    PositiveInt,
    TypeAdapter,
    validate_call,
)
from tqdm import tqdm

from ._options import OPTIONS
# OPTIONS = {"tqdm_disable": False}

validate_func_args_and_return = validate_call(
    config={"strict": True,
            "arbitrary_types_allowed": True,
            "validate_default": True},
    validate_return=True,
)


def validate_array_is_1d(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 1, "array must be 1D"
    return arr


def validate_array_size_is_ge_1(arr: np.ndarray) -> np.ndarray:
    assert arr.size >= 1, "array size must be >= 1"
    return arr


validator_1d_array = Annotated[
    np.ndarray,
    AfterValidator(validate_array_is_1d),
    AfterValidator(validate_array_size_is_ge_1),
]


@validate_func_args_and_return
def split_array(
    arr: validator_1d_array,
    n: PositiveInt,
) -> list[list[Any]]:
    """Split a 1D array in a list of lists with n elements in each list."""

    # Use list(x) instead of x.tolist() to preserve type of np.datetime64
    # Note: in Python >= 3.12 this can be done with itertools.batched
    return [list(x) for x in np.array_split(arr, range(n, arr.size, n))]


@validate_func_args_and_return
def bytesize_to_human_readable(size: int) -> str:
    return TypeAdapter(ByteSize).validate_python(size).human_readable(decimal=True)


@validate_func_args_and_return
def to_bytesize(size: int | str) -> int:
    """Convert int or str to bytesize."""
    return int(TypeAdapter(ByteSize).validate_python(size, strict=True))


class DsDaMixin:
    """Methods common to Datasets and DataArrays."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def dx(self) -> xr.Dataset | xr.DataArray:
        """Returns the Dataset or DataArray itself."""
        return self._obj

    @classmethod
    def _indexers_or_indexers_kwargs(
        cls,
        indexers: Mapping[Any, Any] | None,
        indexers_kwargs: Mapping[Any, Any],
    ) -> dict[str, Any]:
        """Check if indexers OR indexers_kwargs was given and return it."""

        indexers = {} if indexers is None else indexers

        input_args = [indexers, indexers_kwargs]

        m = [0 if len(x) == 0 else 1 for x in input_args]
        if sum(m) != 1:
            error_message = "indexers OR indexers_kwargs must be given"
            raise ValueError(error_message)

        return dict(input_args[m.index(1)])

    def _check_dims(self, dims: Iterable) -> None:
        """Check if the request dimensions exist."""

        dx_dims = [str(x) for x in self.dx.dims]

        for dim in dims:
            if dim not in self.dx.dims:
                raise ValueError(
                    f"'{dim}' is not a valid dimension. Valid"
                    " dimensions are: " + ", ".join(dx_dims)
                )

    def _check_available_memory(self) -> None:
        """Raise an error if there is no free memory to hold the full data."""

        if (free_memory := psutil.virtual_memory().available) < self.dx.nbytes:
            err_msg = (
                "The unpacked data needs"
                f" {bytesize_to_human_readable(self.da.nbytes)} but the"
                f" system only has {bytesize_to_human_readable(free_memory)}"
            )

            raise MemoryError(err_msg)

    def update_attrs(self, attrs: Mapping[Any, Any] | None = None) -> None:
        attrs = {} if attrs is None else attrs
        self.dx.attrs = {**self.dx.attrs, **attrs}

    def update_encoding(self, encoding: Mapping[Any, Any] | None = None) -> None:
        encoding = {} if encoding is None else encoding
        self.dx.encoding = {**self.dx.encoding, **encoding}

    def to_localfile(
            self,
            outfile: Annotated[Path, Field(strict=False)],
            **kwargs: Any,
    ) -> None:

        match outfile.suffix:

            case ".nc":
                self.dx.to_netcdf(outfile, **kwargs)

            case ".zarr":
                self.dx.to_zarr(outfile, **kwargs)

            case _:
                err_msg = (
                    "Invalid file extenion. File extension must be"
                    " .nc or .zarr"
                )
                raise ValueError(err_msg)


@xr.register_dataarray_accessor("lbs")
class DALoadByStep(DsDaMixin):

    @property
    def da(self) -> xr.DataArray:
        """Returns the DataArray itself."""
        return self._obj

    @property
    def name(self) -> str:
        return "unnamed" if self.da.name is None else str(self.da.name)

    @property
    def itemsize_packed(self) -> int:
        """Bytes consumed by one element of this DataArray's data in
        "packed" format."""

        # da.dtype is the dtype of the "unpacked" data, after applying
        #   scale_factor, add_offset, etc.
        # da.encoding.dtype is the "packed" dtype, the one that was actually
        #   used when writing the data to disk.

        dtype = self.da.encoding.get("dtype", self.da.dtype)
        return np.dtype(dtype).itemsize

    @validate_func_args_and_return
    def _create_dict_with_dims_and_subsets(
        self,
        dims_and_steps: dict[str, int],
    ) -> dict[str, list[list[Any]]]:
        """Return dict with dimensions as keys and list of lists of subsets
        as values."""

        dims_and_subsets = {}
        for dim, step in dims_and_steps.items():
            dims_and_subsets[dim] = split_array(self.da[dim].values, step)

        return dims_and_subsets

    @validate_func_args_and_return
    def _combine_dict_with_dims_and_subsets(
        self,
        dims_and_subsets: dict[str, list[list[Any]]],
    ) -> list[dict[str, list[Any]]]:
        """Return list of dicts with dimensions as keys and list of subsets
        as values."""

        return [
            dict(zip(dims_and_subsets.keys(), values, strict=False))
            for values in itertools.product(*dims_and_subsets.values())
        ]

    def _concat_list_of_dataarrays(self, das: list[xr.DataArray]) -> xr.DataArray:
        """Concatenate a list of DataArrays into a single DataArray."""

        da = xr.combine_by_coords(das)

        # xr.combine_by_coords returns a Dataset if da.name is not None
        if isinstance(da, xr.Dataset):
            da = da[next(iter(da.data_vars))]

        da.attrs, da.encoding = self.da.attrs, self.da.encoding

        return da

    def _loading_message(self, subset: dict[Any, list[Any]]) -> str:
        """Loading message."""

        bytesize = self.da.sel(subset).size * self.itemsize_packed

        prefix = (
            f"Loading '{bytesize_to_human_readable(bytesize)}' of"
            f" '{self.name}' between "
        )

        msg = ", ".join([f"{k}=[{v[0]}, {v[-1]}]" for k, v in subset.items()])

        return f"{prefix}{msg}"

    @validate_func_args_and_return
    def load_by_step(
        self,
        *,
        indexers: Mapping[Any, PositiveInt] | None = None,
        seconds_between_requests: NonNegativeFloat = 0,
        **indexers_kwargs: PositiveInt,
    ) -> xr.DataArray:
        """Load the DataArray in memory splitting the loading process along one
        or more dimensions.

        This is useful to download large quantities of data from a THREDDS
        server automatically breaking the large request in smaller requests
        to avoid server timeout.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by positive
            integers.
            One of indexers or indexers_kwargs must be provided.
        seconds_between_requests : float, optional
            Wait time in seconds between requests.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        xr.DataArray
            DatArray loaded in memory.

        Examples
        --------
        This example reads data from a local file just for demonstration
        purpose. A real application would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> da = ds["Tair"]
        >>> da._in_memory
        False
        >>> da2 = da.lbs.load_by_step(time=500, lon=30, seconds_between_requests=1)
        >>> da2._in_memory
        True

        """

        # return it if it was already loaded
        if self.da._in_memory:
            return self.da

        # check if the whole DataArray can fit in memory
        self._check_available_memory()

        # indexers OR indexers_kwargs is mandatory
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(
            indexers, indexers_kwargs
        )

        self._check_dims(dims_and_steps.keys())

        dims_and_subsets = self._create_dict_with_dims_and_subsets(dims_and_steps)

        # this is the main iterable
        subsets = self._combine_dict_with_dims_and_subsets(dims_and_subsets)

        das = []
        with tqdm(subsets, disable=OPTIONS["tqdm_disable"]) as pbar:
            for subset in pbar:
                # if tqdm is disable, just print it to the console
                msg = self._loading_message(subset)
                if OPTIONS["tqdm_disable"]:
                    print(msg)
                else:
                    pbar.set_description(msg)

                das.append(self.da.sel(**subset).compute())
                time.sleep(seconds_between_requests)

        return self._concat_list_of_dataarrays(das)

    @validate_func_args_and_return
    def load_by_bytesize(
        self,
        *,
        indexers: Mapping[Any, PositiveInt | str] | None = None,
        seconds_between_requests: NonNegativeFloat = 0,
        **indexers_kwargs: PositiveInt | str,
    ) -> xr.DataArray:
        """

        Examples
        --------
        This example reads data from a local file just for demonstration
        purpose. A real application would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> da = ds["Tair"]
        >>> da._in_memory
        False
        >>> da2 = da.lbs.load_by_bytesize(time="1MB", seconds_between_requests=1)
        >>> da2._in_memory
        True

        """

        # indexers OR indexers_kwargs is mandatory
        dim_and_bytesize = self.__class__._indexers_or_indexers_kwargs(
            indexers, indexers_kwargs
        )

        self._check_dims(dim_and_bytesize.keys())

        if len(dim_and_bytesize) != 1:
            error_message = (
                "When using load_by_bytesize only one dimension can be passed."
            )
            raise ValueError(error_message)

        dim, bytesize = next(iter(dim_and_bytesize.items()))

        # convert to int if bytesize is a str, e.g.: "10MB"
        bytesize = to_bytesize(bytesize)

        step = int(bytesize / (self.itemsize_packed * self.da.size / self.da[dim].size))

        if step > self.da[dim].size:
            return self.da.lbs.load()

        if step < 1:
            error_message = (
                "It is not possible to load blocks of"
                f" '{bytesize_to_human_readable(bytesize)}' along dimension"
                f" '{dim}' even when using step=1. Consider increasing the size"
                " or calling split_by_step and splitting along a second"
                " dimension."
            )
            raise ValueError(error_message)

        return self.load_by_step(
            **{dim: step},   # type: ignore
            seconds_between_requests=seconds_between_requests,
        )

    def load(self) -> xr.DataArray:
        """Same as the standard da.load()."""

        bytesize = self.da.size * self.itemsize_packed

        msg = (
            f"Downloading '{bytesize_to_human_readable(bytesize)}' of"
            f" '{self.name}' in a single call"
        )

        print(msg)

        return self.da.compute()


@xr.register_dataset_accessor("lbs")
class DSLoadByStep(DsDaMixin):

    @property
    def ds(self) -> xr.Dataset:
        """Returns the DataSet itself."""
        return self._obj

    @validate_func_args_and_return
    def load_by_step(
        self,
        *,
        indexers: Mapping[Any, PositiveInt] | None = None,
        seconds_between_requests: NonNegativeFloat = 0,
        **indexers_kwargs: PositiveInt,
    ) -> xr.Dataset:
        """Load the Dataset in memory splitting the loading process along one
        or more dimensions.

        This is useful to download large quantities of data from a THREDDS
        server automatically breaking the large request in smaller requests
        to avoid server timeout.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by positive
            integers.
            One of indexers or indexers_kwargs must be provided.
        seconds_between_requests : float, optional
            Wait time in seconds between requests.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        xr.Dataset
            Dataset loaded in memory

        Examples
        --------
        This example reads data from a local file just for demonstration
        purpose. A real application would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> ds2 = ds.lbs.load_by_step(time=500, lon=30, seconds_between_requests=1)

        """

        # check if the whole Dataset can fit in memory
        self._check_available_memory()

        # indexers OR indexers_kwargs is mandatory
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(
            indexers, indexers_kwargs
        )

        self._check_dims(dims_and_steps.keys())

        # use a copy so the state of self.ds is preserved
        ds = self.ds.copy()

        # apply load for each data variable
        for var in list(ds.data_vars):
            da = ds[var]

            # keep only the dimension that exists in the DataArray
            dims_and_steps_da = {
                dim: step for dim, step in dims_and_steps.items() if dim in da.dims
            }

            if dims_and_steps_da:
                da_in_memory = da.lbs.load_by_step(
                    seconds_between_requests=seconds_between_requests,
                    **dims_and_steps_da,
                )
            else:
                da_in_memory = da.lbs.load()

            ds[var] = da_in_memory

        return ds

    @validate_func_args_and_return
    def load_and_save_by_step(
        self,
        *,
        outfile: Annotated[NewPath, Field(strict=False)],
        indexers: Mapping[Any, PositiveInt] | None = None,
        attrs: Mapping[Any, Mapping[Any, Any]] | None = None,
        encoding: Mapping[Any, Mapping[Any, Any]] | None = None,
        to_outfile_kwargs: Mapping[Any, Any] | None = None,
        seconds_between_requests: NonNegativeFloat = 0,
        **indexers_kwargs: PositiveInt,
    ) -> None:
        """Loop through DataArrays in a Dataset, loading each in memory
        splitting the loading process along one or more dimensions and save
        to outfile before freeing the memory and repeating the process for the
        next DataArray.

        This is useful to download large quantities of data from a THREDDS
        server automatically breaking the large request in smaller requests
        to avoid server timeout.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by positive
            integers.
            One of indexers or indexers_kwargs must be provided.
        outfile : Path, str
            File to save the data into.
        attrs : dict, optional
            A dict of dicts where the key is a variable name and the value is
            a dict with key:value attrs pairs. To set the dataset attrs use
            variable "global".
        encoding : dict, optional
            A dict of dicts where the key is a variable name and the value is
            a dict with key:value encoding pairs. To set the dataset encoding
            use variable "global".
        seconds_between_requests : float, optional
            Wait time in seconds between requests.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        None

        Examples
        --------
        This example reads data from a local file just for demonstration
        purpose. A real application would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> attrs = {
        ...     "global": {"title": "My example dataset"},
        ...     "Tair": {"long_name": "My example variable"},
        ...     }
        >>> encoding = {
        ...     "Tair": {"dtype": "f4", "zlib": "True", "deflate": 1},
        ...     }
        >>> ds.lbs.load_and_save_by_step(time=500,
                                         lon=30,
                                         attrs=attrs,
                                         encoding=encoding,
                                         outfile="/tmp/foo.nc",
                                         seconds_between_requests=1)

        """

        # indexers OR indexers_kwargs is mandatory
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(
            indexers, indexers_kwargs
        )

        self._check_dims(dims_and_steps.keys())

        attrs = {} if attrs is None else attrs
        encoding = {} if encoding is None else encoding
        to_outfile_kwargs = {} if to_outfile_kwargs is None else to_outfile_kwargs

        # use a copy so the state of self.ds is preserved
        ds = self.ds.copy()

        # update Dataset and DataArray attrs and encoding
        ds.lbs.update_attrs(attrs.get("global", None))
        ds.lbs.update_encoding(encoding.get("global", None))
        for var in list(ds.data_vars):
            ds[var].lbs.update_attrs(attrs.get(var, None))
            ds[var].lbs.update_encoding(encoding.get(var, None))

        # create empty Dataset
        ds[[]].lbs.to_localfile(outfile, mode="w", **to_outfile_kwargs)

        # apply load for each data variable
        for var in list(ds.data_vars):
            da = ds[var]

            # keep only the dimension that exists in the DataArray
            dims_and_steps_da = {
                dim: step for dim, step in dims_and_steps.items() if dim in da.dims
            }

            if dims_and_steps_da:
                da_in_memory = da.lbs.load_by_step(
                    seconds_between_requests=seconds_between_requests,
                    **dims_and_steps_da,
                )
            else:
                da_in_memory = da.lbs.load()

            # append DataArray
            da_in_memory.lbs.to_localfile(outfile, mode="a", **to_outfile_kwargs)
