#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__all__ = [
    "DALoadByStep",
    "DSLoadByStep",
]


from typing import Any, Mapping, Annotated
from pydantic import (validate_call, Field, ByteSize, AfterValidator,
                      PositiveInt, NonNegativeFloat, NewPath)
import itertools
import psutil
import numpy as np
import xarray as xr
from tqdm import tqdm
import time


validate_func_args_and_return = validate_call(
    config=dict(strict=True,
                arbitrary_types_allowed=True,
                validate_default=True),
    validate_return=True)


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
                n: PositiveInt,
                ) -> list[list[Any]]:
    """Split a 1D array in a list of lists with n elements in each list."""

    # Use list(x) instead of x.tolist() to preserve type of np.datetime64
    # Note: in Python >= 3.12 this can be done with itertools.batched
    return [list(x) for x in np.array_split(arr, range(n, arr.size, n))]


@validate_func_args_and_return
def bytesize_to_human_readable(size: int) -> str:
    return ByteSize(size).human_readable(decimal=True)


class DsDaMixin:
    """Methods common to Datasets and DataArrays."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def dx(self) -> xr.Dataset | xr.DataArray:
        """Returns the Dataset or DataArray itself."""
        return self._obj

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

    def _check_dims(self, dims: list[str]) -> None:
        """Check if the request dimensions exist."""

        for dim in dims:
            if dim not in self.dx.dims:
                raise ValueError(f"'{dim}' is not a valid dimension. Valid"
                                 " dimensions are: " + ", ".join(self.dx.dims))

    def _check_available_memory(self) -> None:
        """Raise an error if there is no free memory to hold the full data."""

        if (free_memory := psutil.virtual_memory().available) < self.dx.nbytes:

            err_msg = ("The unpacked data needs"
                       f" {bytesize_to_human_readable(self.da.nbytes)} but the"
                       f" system only has {bytesize_to_human_readable(free_memory)}")

            raise MemoryError(err_msg)


@xr.register_dataarray_accessor("lbs")
class DALoadByStep(DsDaMixin):

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

    def _concat_list_of_dataarrays(self, das: list[xr.DataArray]) -> xr.DataArray:
        """Concatenate a list of DataArrays into a single DataArray."""

        da = xr.combine_by_coords(das)

        # xr.combine_by_coords returns a Dataset if da.name is not None
        if isinstance(da, xr.Dataset):
            da = da[list(da.data_vars)[0]]

        da.attrs, da.encoding = self.da.attrs, self.da.encoding

        return da

    def _pbar_message(self, subset: dict[Any, list[Any]]) -> str:
        """Progess bar message."""

        size = self.da.sel(subset).size * self.itemsize_packed

        preffix = (f"Donwloading '{bytesize_to_human_readable(size)}' of"
                   f" '{self.name}' between ")

        msg = ", ".join([f"{k}=[{v[0]}, {v[-1]}]"
                         for k, v in subset.items()])

        return f"{preffix}{msg}"

    @validate_func_args_and_return
    def load_by_step(self,
                     *,
                     indexers: Mapping[str, PositiveInt] | None = None,
                     seconds_between_requests: NonNegativeFloat = 0,
                     **indexers_kwargs: PositiveInt | None,
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
        purpose. A real aplication would be to read data from a THREDDS server.

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
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(indexers,
                                                                     indexers_kwargs)

        self._check_dims(dims_and_steps.keys())

        dims_and_subsets = self._create_dict_with_dims_and_subsets(dims_and_steps)

        # this is the main iterable
        subsets = self._combine_dict_with_dims_and_subsets(dims_and_subsets)

        das = []
        with tqdm(subsets) as pbar:
            for subset in pbar:
                pbar.set_description(self._pbar_message(subset))
                das.append(self.da.sel(**subset).compute())
                time.sleep(seconds_between_requests)

        da = self._concat_list_of_dataarrays(das)

        return da

    def load(self) -> xr.DataArray:
        """Same as the standard da.load()."""

        size = self.da.size * self.itemsize_packed

        msg = (f"Donwloading '{bytesize_to_human_readable(size)}' of"
               f" '{self.name}' in a single call")

        print(msg)

        return self.da.compute()


@xr.register_dataset_accessor("lbs")
class DSLoadByStep(DsDaMixin):

    @property
    def ds(self) -> xr.DataArray:
        """Returns the DataSet itself."""
        return self._obj

    @validate_func_args_and_return
    def load_by_step(self,
                     *,
                     indexers: Mapping[str, PositiveInt] | None = None,
                     seconds_between_requests: NonNegativeFloat = 0,
                     **indexers_kwargs: PositiveInt | None,
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
        purpose. A real aplication would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> ds2 = ds.lbs.load_by_step(time=500, lon=30, seconds_between_requests=1)

        """

        # check if the whole Dataset can fit in memory
        self._check_available_memory()

        # indexers OR indexers_kwargs is mandatory
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(indexers,
                                                                     indexers_kwargs)

        self._check_dims(dims_and_steps.keys())

        # use a copy so the state of self.ds is preserved
        ds = self.ds.copy()

        # apply load for each data variable
        for idx, var in enumerate(list(self.ds.data_vars)):
            da = self.ds[var]

            # keep only the dimension that exists in the DataArray
            dims_and_steps_da = {dim: step
                                 for dim, step in dims_and_steps.items()
                                 if dim in da.dims}

            if dims_and_steps_da:
                da_in_memory = da.lbs.load_by_step(
                    seconds_between_requests=seconds_between_requests,
                    **dims_and_steps_da)
            else:
                da_in_memory = da.lbs.load()

            ds[var] = da_in_memory

        return ds

    @validate_func_args_and_return
    def load_and_save_by_step(self,
                              *,
                              indexers: Mapping[str, PositiveInt] | None = None,
                              outfile: Annotated[NewPath, Field(strict=False)],
                              seconds_between_requests: NonNegativeFloat = 0,
                              **indexers_kwargs: PositiveInt | None,
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
        purpose. A real aplication would be to read data from a THREDDS server.

        >>> ds = xr.tutorial.open_dataset("air_temperature_gradient")
        >>> ds.lbs.load_and_save_by_step(time=500,
                                         lon=30,
                                         outfile="/tmp/foo.nc",
                                         seconds_between_requests=1)

        """

        # indexers OR indexers_kwargs is mandatory
        dims_and_steps = self.__class__._indexers_or_indexers_kwargs(indexers,
                                                                     indexers_kwargs)

        self._check_dims(dims_and_steps.keys())

        # apply load for each data variable
        for idx, var in enumerate(list(self.ds.data_vars)):
            da = self.ds[var]

            # keep only the dimension that exists in the DataArray
            dims_and_steps_da = {dim: step
                                 for dim, step in dims_and_steps.items()
                                 if dim in da.dims}

            if dims_and_steps_da:
                da_in_memory = da.lbs.load_by_step(
                    seconds_between_requests=seconds_between_requests,
                    **dims_and_steps_da)
            else:
                da_in_memory = da.lbs.load()

            # create file in first iteration and then append
            da_in_memory.to_netcdf(outfile,
                                   mode="a" if idx else "w")
