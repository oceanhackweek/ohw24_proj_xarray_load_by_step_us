#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __all__ = [

# ]


%reset -f

from typing import Any, Mapping, Annotated
from pydantic import validate_call, Field, ByteSize, AfterValidator
import itertools
import psutil
import numpy as np
import xarray as xr
from tqdm import tqdm
import time


# >>> url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
# >>> ds = xr.open_dataset(url, drop_variables="tau")
# >>> da = ds["water_u"]
# >>> da = da.sel(lon=slice(-54 + 360, -31 + 360), lat=slice(-36, 7))
# >>> da._in_memory
# False
# >>> da.load()
# MemoryError: Unable to allocate 386. GiB for an array with shape (16729, 40, 1076, 288) and data type int16
# >>> da = da.sel(time=slice("2024-01-01", "2024-01-10"))

# >>> da.load()
# # after a few (~7) minutes, you get this error message
# oc_open: server error retrieving url: code=? message="Error {
#     code = 500;
#     message = "java.net.SocketTimeoutException: Read timed out; water_u -- 14821:14900,0:39,1100:2175,3825:4112";
# }"
# # after some time, you get the same message again
# }"oc_open: server error retrieving url: code=? message="Error {
#     code = 500;
#     message = "java.net.SocketTimeoutException: Read timed out; water_u -- 14821:14900,0:39,1100:2175,3825:4112";
# }"


validate_func_args_and_return = validate_call(
    config=dict(strict=True,
                arbitrary_types_allowed=True,
                validate_default=True),
    validate_return=True)


# define the type int >= 1
int_ge_1 = Annotated[int, Field(ge=1)]


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
def split_array(arr: validator_1d_array,
                n: int_ge_1,
                ) -> list[list[Any]]:
    """Split a 1D array in lists with n elements in each list."""

    # Use: list(x) instead of x.tolist() to preserve type of np.datetime64
    return [list(x) for x in np.array_split(arr, range(n, arr.size, n))]


@validate_func_args_and_return
def bytesize_to_human_readable(size: int) -> str:
    return ByteSize(size).human_readable(decimal=True)


@xr.register_dataarray_accessor("lbs")
class DALoadByStep:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def da(self) -> xr.DataArray:
        """Returns the DataArray itself."""
        return self._obj

    @property
    def name(self) -> str:
        return "unnamed" if self.da.name is None else self.da.name

    @property
    def itemsize_packed(self) -> int:
        """Bytes consumed by one element of this DataArray's data in
        "packed" format."""

        # da.dtype is the dtype of the "unpacked" data, after applying
        #   scale_factor, add_offset, etc.
        # da.encoding.dtype is the "packed" dtype, the one that was actually
        #   used when writting the data to disk.

        dtype = self.da.encoding.get("dtype", self.da.dtype)
        return np.dtype(dtype).itemsize

    @property
    def nbytes_packed(self) -> int:
        """Total bytes consumed by the elements of this DataArray's data in
        "packed" format."""

        return self.da.size * self.itemsize_packed

    @classmethod
    def _indexers_or_indexers_kwargs(cls,
                                     indexers: dict[Any, Any] | None,
                                     indexers_kwargs: dict[Any, Any] | None,
                                     ) -> dict[Any, Any]:
        """Check if indexers OR indexers_kwargs was given and return it."""

        input_args = [indexers, indexers_kwargs]

        m = list(map(lambda x: 1 if x else 0, input_args))
        if sum(m) != 1:
            raise ValueError("indexers OR indexers_kwargs must be given")

        return input_args[m.index(1)]

    @validate_func_args_and_return
    def _create_dict_with_dims_and_subsets(
            self,
            dims_and_steps: dict[str, int],
            ) -> dict[str, list[list[Any]]]:
        """Return dict with dimensions as keys and list of lists of subsets
        as values."""

        dims_and_subsets = dict()
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

        return [dict(zip(dims_and_subsets.keys(), values))
                for values in itertools.product(*dims_and_subsets.values())]

    def _check_available_memory(self) -> None:
        """Raise an error if there is no free memory to hold the full DataArray."""

        if (free_memory := psutil.virtual_memory().available) < self.da.nbytes:

            err_msg = ("The unpacked DataArray needs"
                       f" {bytesize_to_human_readable(self.da.nbytes)} but the"
                       f" system only has {bytesize_to_human_readable(free_memory)}")

            raise MemoryError(err_msg)

    def _check_chunk_size(self, subset, chunk_max_size: int) -> None:
        """Raise an error if the chunk size is greater than the allowed
        chunk_max_size."""

        if (chunk_size := self.da.sel(**subset).lbs.nbytes_packed) > chunk_max_size:

            err_msg = ("With the requested `dim=step`, the chunk size is"
                       f" {bytesize_to_human_readable(chunk_size)}."
                       " This is more than the allowed chunk_max_size of"
                       f" {bytesize_to_human_readable(chunk_max_size)}."
                       " Consider reducing the step or subsetting in more"
                       " dimension: dim1=step1, dim2=step2, ...")

            raise ValueError(err_msg)

    def _concat_list_of_dataarrays(self, das: list[xr.DataArray]) -> xr.DataArray:
        """Concatenate a list of DataArrays into a single DataArray."""

        da = xr.combine_by_coords(das)

        # xr.combine_by_coords returns a Dataset if da.name is not None
        if isinstance(da, xr.Dataset):
            da = da[list(da.data_vars)[0]]

        da.attrs, da.encoding = self.da.attrs, self.da.encoding

        return da

    @validate_func_args_and_return
    def load_by_step(self,
                     indexers: Mapping[Any, int_ge_1] | None = None,
                     chunk_max_size: ByteSize = "50MB",
                     **indexers_kwargs: int_ge_1 | None,
                     ) -> xr.DataArray:

        # return it if it was already loaded
        if self.da._in_memory:
            return self.da

        self._check_available_memory()

        # indexers OR indexers_kwargs is mandatory
        indexers_kwargs = self.__class__._indexers_or_indexers_kwargs(indexers,
                                                                      indexers_kwargs)

        dims_and_subsets = self._create_dict_with_dims_and_subsets(indexers_kwargs)

        # this is the main iterable
        subsets = self._combine_dict_with_dims_and_subsets(dims_and_subsets)

        self._check_chunk_size(subsets[0], chunk_max_size)

        das = []
        for subset in tqdm(subsets):
            das.append(self.da.sel(**subset).compute())

        da = self._concat_list_of_dataarrays(das)

        return da


ds = xr.tutorial.open_dataset("air_temperature")
da = ds["air"]

da2 = da.lbs.load_by_step(time=100, lat=15)
