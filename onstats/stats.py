"""Calculated gridded stats using xarray and dask."""
import os
from pathlib import Path
import shutil
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from fsspec import get_mapper
from intake import open_catalog

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress, LocalCluster
from distributed.diagnostics.progressbar import get_scheduler

from ontake.ontake import Ontake
from oncore.dataio import put, isdir, exists, rm, get
from oncore.date import daterange, _parse, timedelta

from onstats.utils import uv_to_spddir, expand_time_group
import onstats.derived_variable as dv
from onstats.xarray_stats import (
    rpv,
    distribution,
    directional_stat,
    distribution_spddir,
)
from onstats.frequency_domain import hmo, BANDS


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")


class DerivedVar:
    def __init__(
        self,
        dset,
        mask=None,
        hs_threshold=0,
        clear_sky_threshold=0.0,
        covered_sky_threshold=1.0,
        var_depth="dpt",
        var_hs="hs",
        var_hs_sea="phs0",
        var_hs_sw1="phs1",
        var_hs_sw2="phs2",
        var_dir_sea="pdir0",
        var_dir_sw1="pdir1",
        var_dir_sw2="pdir2",
        var_lp_sw1="plp1",
        var_uwnd="uwnd",
        var_vwnd="vwnd",
        var_ucur="ucur",
        var_vcur="vcur",
        var_fp="fp",
        var_fp_sw1="pfp1",
        var_tp_sw1="ptp1",
        var_cloud_cover="tcc",
        var_uwnd150="u150",
        var_vwnd150="v150",
        logger=logging,
        **kwargs,
    ):
        """Wrapper around derived variable functions.

        The purpose of this class is defining the arguments required by the different
        derived variables so that they can be called as an attribute.

        """
        self.dset = dset
        self.hs_threshold = hs_threshold
        self.clear_sky_threshold = clear_sky_threshold
        self.covered_sky_threshold = covered_sky_threshold

        self.var_depth = var_depth
        self.var_hs = var_hs
        self.var_hs_sea = var_hs_sea
        self.var_hs_sw1 = var_hs_sw1
        self.var_hs_sw2 = var_hs_sw2
        self.var_dir_sea = var_dir_sea
        self.var_dir_sw1 = var_dir_sw1
        self.var_dir_sw2 = var_dir_sw2
        self.var_lp_sw1 = var_lp_sw1
        self.var_uwnd = var_uwnd
        self.var_vwnd = var_vwnd
        self.var_ucur = var_ucur
        self.var_vcur = var_vcur
        self.var_fp = var_fp
        self.var_fp_sw1 = var_fp_sw1
        self.var_tp_sw1 = var_tp_sw1
        self.var_cloud_cover = var_cloud_cover
        self.var_uwnd150 = var_uwnd150
        self.var_vwnd150 = var_vwnd150

    @property
    def tp(self):
        """Peak wave period data_var."""
        return dv.tp(fp=self.dset[self.var_fp])

    @property
    def winpow(self):
        """Wind Power for Wind Quarry turbine."""
        wp = dv.winpow(
            uwnd150=self.dset[self.var_uwnd150], vwnd150=self.dset[self.var_vwnd150]
        )
        return wp

    @property
    def douglas_sea(self):
        """Douglas sea scale data_var."""
        return dv.douglas_sea(hs_sea=self.dset[self.var_hs_sea])

    @property
    def douglas_swell(self):
        """Douglas swell scale data_var.

        Wavelength is attempted to be used from prescribed `var_lp_sw1`, falling back
            to derived_variable `wlen_sw1` if that var is not in dataset.

        """
        lp_sw1 = self.dset.data_vars.get(self.var_lp_sw1, None)
        if lp_sw1 is None:
            logger.info("Deriving swell wave length from swell period")
            lp_sw1 = self.wlen_sw1
        return dv.douglas_swell(hs_sw1=self.dset[self.var_hs_sw1], lp_sw1=lp_sw1)

    @property
    def wspd(self):
        """Wind speed data_var."""
        return dv.wspd(uwnd=self.dset[self.var_uwnd], vwnd=self.dset[self.var_vwnd])

    @property
    def wdir(self):
        """Wind from direction data_var."""
        return dv.wdir(uwnd=self.dset[self.var_uwnd], vwnd=self.dset[self.var_vwnd])

    @property
    def cspd(self):
        """Current speed data_var."""
        return dv.cspd(ucur=self.dset[self.var_ucur], vcur=self.dset[self.var_vcur])

    @property
    def crossing_seas(self):
        """Crossing seas occurrance data_var."""
        return dv.crossing_seas(
            hs=self.dset[self.var_hs],
            hs_sea=self.dset[self.var_hs_sea],
            hs_sw1=self.dset[self.var_hs_sw1],
            dir_sea=self.dset[self.var_dir_sea],
            dir_sw1=self.dset[self.var_dir_sw1],
            hs_sw2=self.dset[self.var_hs_sw2],
            dir_sw2=self.dset[self.var_dir_sw2],
            hs_threshold=self.hs_threshold,
        )

    @property
    def wlen_sw1(self):
        """Wevelength of the primary swell."""
        fp = self.dset.data_vars.get(self.var_fp_sw1, None)
        tp = self.dset.data_vars.get(self.var_tp_sw1, None)
        depth = self.dset.data_vars.get(self.var_depth, None)
        return dv.wlen_sw1(fp=fp, tp=tp, depth=depth)

    @property
    def clear_sky(self):
        """Clear sky based on cloud cover fraction."""
        return dv.clear_sky(
            cloud_cover=self.dset[self.var_cloud_cover],
            cover_threshold=self.clear_sky_threshold,
        )

    @property
    def covered_sky(self):
        """Covered sky based on cloud cover fraction."""
        return dv.covered_sky(
            cloud_cover=self.dset[self.var_cloud_cover],
            cover_threshold=self.covered_sky_threshold,
        )


class StatsDeprecated(DerivedVar):
    def __init__(
        self,
        dataset,
        master_url="gs://oceanum-catalog/oceanum.yml",
        namespace="hindcast",
        chunk=None,
        mask=None,
        mapping={},
        slice_dict={},
        chunks=None,
        persist=False,
        updir=None,
        localdir="/scratch",
        zarrfile=None,
        zarrmode="w",
        allow_split_large_chunks=False,
        local_cluster=0,
        **kwargs,
    ):
        """Gridded stats using dask arrays.

        Args:
            dataset (str, xr.Dataset, uri): Ontake dataset id if string, xarray dataset
                or a uri string starting with gs:// for the zarr store path.
            master_url (str): Ontake catalog master url path.
            namespace (str): Ontake catalog namespace.
            chunk (str): Chunk optimisation strategy, one of grid, timeseries, slab.
            mask (str): either a variable name or an expression to evaluate on one or
                more variables to define a mask array for masking output dataset. e.g.,
                `self.dset.hs==0`.
            mapping (dict): Dictionary for renaming dataset variables.
            slice_dict (dict): Dictionary specifying slicing arg.
            chunks (dict): Chunking dict to rechunk dataset after opening.
            persist (bool): If True, persist output dataset before saving as netcdf.
            updir (str): Upload direction to upload netcdf and zarr stats files to.
            zarrfile (str): Name of zarr file to write after each compute call.
            zarrmode (str): Write mode for output zarr archive with stats.
            allow_split_large_chunks (bool): Allow dask auto-resize of small chunks.
            local_cluster (int): Number of cores to scale a Local dask cluster.

        Note:
            Set compute=True in the method calls when you want to trigger computation
                and therefore avoid blowing up memory by computing everything at once.
            The idea behind zarrfile argument is to allow it to download existing zarr
                stores and update and upload it back after each compute call, therefore
                if the workflow breaks one can run only the remaining function calls.

        """
        dask.config.set({"array.slicing.split_large_chunks": allow_split_large_chunks})

        if local_cluster:
            cluster = LocalCluster()
            cluster.scale(local_cluster)
            logger.info(cluster)

        self.dataset = dataset
        self.master_url = master_url
        self.namespace = namespace
        self.chunk = chunk
        self.mask = mask
        self.mapping = mapping
        self.slice_dict = slice_dict
        self.chunks = chunks
        self.persist = persist
        self.updir = updir
        self.localdir = localdir
        self.zarrfile = zarrfile
        self.zarrmode = zarrmode

        self._hour_of_day = None

        # Download partial zarr file to be appended
        if self.updir and self.zarrfile:
            src = os.path.join(self.updir, os.path.basename(zarrfile))
            if isdir(src) and zarrmode == "a":
                if isdir(self.zarrfile):
                    logger.warning(
                        f"Removing existing tmp file {self.zarrfile} before pulling"
                    )
                    rm(self.zarrfile, recursive=True)
                logger.info(f"Downloading existing zarr")
                get(src, os.path.dirname(self.zarrfile), recursive=True)

        # Open dataset
        self._open_dataset(
            dataset=self.dataset,
            namespace=self.namespace,
            chunk=self.chunk,
            chunks=self.chunks,
        )

        self.dsout = xr.Dataset()

        # Instantiating DerivedVar
        super().__init__(self.dset, **kwargs)

        # Define mask
        self._set_mask(mask)

        # Encoding for output
        self.encoding = {}

    @property
    def dsout_dask(self):
        """Output dataset with only dask variables."""
        dask_vars = [v for v in self.dsout.data_vars if self.dsout[v].chunks]
        return self.dsout[dask_vars]

    @property
    def hour_of_day(self):
        """The time of the day data array accounting for time offset.

        Note:
            If the attribute has already been defined the existing value is returned
                to avoid unecessary computation.

        """
        if self._hour_of_day is not None:
            return self._hour_of_day
        logger.debug("Estimating hour offset and broadcasting to 3D")
        hour_offset = da.floor((self.dset.longitude + 7.7) / 15).chunk(
            {"longitude": self.dset.chunks["longitude"][0]}
        )
        hour = self.dset.time.dt.hour.chunk({"time": self.dset.chunks["time"][0]})
        self._hour_of_day = (hour + hour_offset) % 24
        self._hour_of_day.name = "hour_of_day"
        return self._hour_of_day

    def _slice_dset(self):
        """Masking dataset using slice_dict."""
        for slice_method, slice_kwargs in self.slice_dict.items():
            for dim, slicing in slice_kwargs.items():
                if (
                    isinstance(slicing, slice)
                    and not isinstance(slicing.start, str)
                    and not isinstance(slicing.stop, str)
                ):
                    # Ensure order in slice object is correct (era5)
                    sign_coord = np.sign(self.dset[dim][-1] - self.dset[dim][0])
                    sign_slice = np.sign(slicing.stop - slicing.start)
                    if sign_coord != sign_slice:
                        logger.warn(
                            f"Order in slice and coord {dim} differ, swapping slice."
                        )
                        slicing = slice(slicing.stop, slicing.start)
                self.dset = getattr(self.dset, slice_method)(**{dim: slicing})
                if self.dset[dim].size == 0:
                    raise ValueError(
                        f"Empty {dim} slicing from {slicing}, perhaps longitude "
                        "conventions in dataset and slice are different."
                    )

    def _set_mask(self, mask):
        """Define the mask data array."""
        # Define mask
        if isinstance(mask, xr.DataArray):
            self.dset["mask"] = mask
        elif mask in self.dset.data_vars:
            self.dset["mask"] = self.dset[mask]
        elif isinstance(mask, str):
            self.dset["mask"] = eval(mask)
        else:
            self.dset["mask"] = 1

    def _upload(self, filename):
        """Upload stats files.

        Args:
            filename (str): Name of file to upload.

        """
        outfile = os.path.join(self.updir, os.path.basename(filename))
        logger.info(f"Uploading {filename} --> {outfile}")
        if exists(outfile):
            logger.debug(f"Removing existing file {outfile} before uploading")
            rm(outfile, recursive=isdir(outfile))
        put(filename, outfile, recursive=isdir(filename))

    def _open_dataset(self, dataset, namespace, chunk, chunks):
        """Set dset attribute either from ontake dataset of from xarray dataset itself.

        Args:
            dataset (str, xr.Dataset): Dataset ontake id, URI or an xarray dataset.
            namespace (str): Ontake namespace in case dataset is an ontake id.
            chunk (str): Ontake chunk optimisation strategy in case dataset is an ontake id.
            chunks (dict): coordinates and sizes for chunking dataset after opening.

        Note:
            The arguments are provided from class attributes but can be provided from
                inside methods as well when a different dataset chunk strategy is to
                be used for a certain stat for instance.

        """
        logger.info("Open dataset")
        if isinstance(dataset, str) and dataset.endswith(".nc"):
            logger.debug(f"Opening netcdf file: {dataset}")
            self.dset = xr.open_dataset(dataset, chunks=chunks)
        elif isinstance(dataset, str):
            try:
                logger.debug(f"Try opening zarr store from URI: {dataset}")
                self.dset = xr.open_zarr(get_mapper(dataset), consolidated=True)
            except KeyError:
                logger.debug(f"Ontake dataset {dataset} {self.master_url} {namespace}")
                # Open catalog and ensure dataset is a substring of a catalog entry
                ot = Ontake(master_url=self.master_url, namespace=namespace)
                kwargs = {}
                if chunk is not None:
                    logger.debug(f"Chunk stragegy: {chunk}")
                    kwargs.update({"chunk": chunk})
                self.dset = ot.dataset(dataset, **kwargs)
        elif isinstance(dataset, xr.Dataset):
            self.dset = dataset
        else:
            raise ValueError(
                "dataset must be either a string specifying an ontake "
                "dataset id or bucket URI, or an xarray dataset."
            )
        # Renaming
        self.dset = self.dset.rename(self.mapping)
        # Slicing
        self._slice_dset()
        # Rechunking
        if chunks:
            logger.info(f"Re-chunking dataset as {chunks}")
            self.dset = self.dset.chunk(chunks)
        self.data_vars = list(self.dset.data_vars.keys())
        logger.info(f"{self.dset}")

    def _load(self):
        """Trigger computation and load output dataset in memory.

        If `self.persist==True`, Dataset.persist() is used to keep data as dask arrays.
            This will only be allowed here if running on a dask cluster so progress
            can be locked. If `self.persist==False`, Dataset.compute() is used instead.

        """
        if self.persist:
            try:
                get_scheduler(None)
            except ValueError as err:
                raise ValueError(
                    "Dask cluster is required when using the 'persist' argument."
                ) from err
            self.dsout = self.dsout.persist()
            progress(self.dsout)
        else:
            with ProgressBar():
                self.dsout = self.dsout.compute()

    def _sortby(self):
        """Sort output dataset by all coordinates."""
        for name, coords in self.dsout.coords.items():
            if coords[0] > coords[-1]:
                logger.info(f"Sorting by coordinate {name}")
                self.dsout = self.dsout.sortby(name)

    def _setattrs(self):
        """Define some attributes in output dataset."""
        if "quantile" in self.dsout.coords:
            self.dsout["quantile"].attrs = {
                "standard_name": "quantile",
                "long_name": "quantile",
                "units": "",
            }

    def _update_dset(self, derived_vars):
        """Append derived variables to dataset.

        Args:
            derived_vars (list): List of derived variable names to append. Each derived
                variable must (1) exist as a property in DerivedVar class, and (2) be
                able to be derived from existing variables in dataset.

        """
        for derived_var in derived_vars:
            if derived_var in self.dset.data_vars:
                logger.debug(f"{derived_var} already a variable in dataset")
                continue
            self._is_derived_variable(derived_var)
            logger.info(f"Updating dataset with derived variable: {derived_var}")
            self.dset[derived_var] = getattr(self, derived_var)

    def _is_derived_variable(self, name):
        """Check that derived variable has been properly prescribed."""
        if getattr(DerivedVar, name, None) is None:
            raise AttributeError(f"Variable '{name}' not in dataset nor in DerivedVar")
        if not isinstance(getattr(DerivedVar, name), property):
            raise TypeError(f"Derived var '{name}' must be a property in DerivedVar")
        if not isinstance(getattr(self, name), xr.DataArray):
            raise TypeError(f"Property '{name}' in DerivedVar must return DataArray.")

    def _compute(self, is_compute):
        """Trigger computation of dask variables and write to partial zarr."""
        if is_compute:
            logger.info(f"Computing dask variables: {self.dsout_dask}")
            self.dsout = self.dsout.compute()
            if self.zarrfile:
                if isdir(self.zarrfile):
                    dstmp = xr.open_zarr(self.zarrfile, consolidated=True)
                    mode = self.zarrmode
                else:
                    dstmp = xr.Dataset()
                    mode = "w"
                new_data_vars = list(set(self.dsout.data_vars) - set(dstmp.data_vars))
                if new_data_vars:
                    logger.info(
                        f"Writing variables {new_data_vars} to partial archive {self.zarrfile}"
                    )
                    dsout = self.dsout[new_data_vars]
                    self.to_zarr(self.zarrfile, dsout, mode=mode)
                    self.zarrmode = "a"

    def range_probability(self, data_ranges, dim="time", compute=False, **kwargs):
        """Calculate probability of specific ranges.

        Args:
            data_ranges (list): List of dictionaries to define each data range to
                calculate probabilities from with keys:
                - var (str): Variable name, can be a valid data_var or derived_var.
                - start (float): Minimum value for interval.
                - stop (float); Maximum value for interval.
                - left (closed | open): Define if minimum value should be included.
                - right (closed | open): Define if maximum value should be included.
            dim (str): Dimension name to calculate probabilities along.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        data_vars = list(set([data_range["var"] for data_range in data_ranges]))
        derived_vars = [v for v in data_vars if v not in self.dset.data_vars]
        self._update_dset(derived_vars)

        logger.debug(f"Calculating time-probability for vars: {data_vars}")

        dset = self.dset[data_vars]
        counts = dset.count(dim)

        # Probability for each range
        for data_range in data_ranges:

            # Data variable to compute
            dvar = data_range["var"]
            darray = dset[dvar]

            # Data range values
            start = data_range["start"] if data_range["start"] is not None else -np.inf
            stop = data_range["stop"] if data_range["stop"] is not None else np.inf

            # Functions for computing open and close intervals
            left = data_range.get("left", "closed")
            right = data_range.get("right", "closed")
            if left == "closed":
                lfunc = da.greater_equal
            elif left == "open":
                lfunc = da.greater
            if right == "closed":
                rfunc = da.less_equal
            elif right == "open":
                rfunc = da.less

            # Output label
            llabel = f"{start:g}" if data_range["start"] is not None else "min"
            rlabel = f"{stop:g}" if data_range["stop"] is not None else "max"
            varname = data_range.get("label", f"{dvar}_p_{llabel}-{rlabel}")

            # Probability
            in_range = lfunc(darray, start) & rfunc(darray, stop)
            self.dsout[varname] = in_range.sum(dim=dim) / counts[dvar]

        self._compute(is_compute=compute)
        return self.dsout

    def data_count(
        self,
        dim="time",
        data_vars=[],
        derived_vars=[],
        suffix="_pcount",
        compute=False,
        **kwargs,
    ):
        """Calculate the percentage of valid data along dimension.

        Args:
            dim (str): Dimension to calculate percentage count over.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            suffix (str): String to append to each variable name in output dataset.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        if not data_vars:
            raise ValueError("At least one data_var or derived_var should be provided")

        self._update_dset(data_vars)
        logger.debug(f"Calculating count percentage for vars: {data_vars}")

        dsout = 100 * self.dset[data_vars].count(dim) / self.dset[dim].size
        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        )

        self._compute(is_compute=compute)
        return dsout

    def value_probability(
        self,
        dim="time",
        data_vars=[],
        bins=[],
        bin_name="bin",
        suffix="_prob",
        compute=False,
        **kwargs,
    ):
        """Calculate the probability of specific values.

        This function is useful for integer-type data, use range_probability for float.

        Args:
            dim (str): Dimension name to calculate probabilities along.
            data_vars (list): Data vars to apply stats over, includes derived vars.
            bins (list): List of values for binning the data to calculate probability.
            bin_name (str): Name of bin coordinate in output probability dataset. Note
                that the bin coordinate is only created if there is more than one bin.
            suffix (str): String to append to each variable name in output dataset.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        if not data_vars:
            raise ValueError("At least one data_var or derived_var should be provided")

        self._update_dset(data_vars)
        bins = list(bins)
        logger.debug(f"Calculating {dim}-probability for vars: {data_vars}")

        dset = self.dset[data_vars]
        counts = dset.count(dim)

        # Probability for each variable
        for data_var in data_vars:
            dvar = dset[data_var]
            count = counts[data_var]
            darrays = []
            for bin_value in bins:
                in_bin = dvar == bin_value
                darrays.append(in_bin.sum(dim=dim) / count)
            # Create extra coordinate only if more than one bin
            if len(bins) == 1:
                self.dsout[f"{data_var}{suffix}"] = darrays[0].where(self.dset.mask)
            else:
                self.dsout[f"{data_var}{suffix}"] = xr.concat(darrays, bin_name).where(
                    self.dset.mask
                )
        if len(bins) > 1:
            self.dsout[bin_name] = bins

        self._compute(is_compute=compute)
        return self.dsout

    def time_probability_hour_of_day(
        self, derived_var, bin_value, suffix="_hprob", compute=False, **kwargs
    ):
        """Calculate the time probability for each hour with time offset accounted.

        Args:
            derived_var (str): Derived_vars to calculate before applying stats.
            bin (value): List of values for binning the data to calculate probability.
            suffix (str): String to append to each variable name in output dataset.
            compute (bool): Compute dask variables from output dataset before returning.

        Note:
            Only implemented atm for one single derived variable and one matching bin.

        """
        self._update_dset([derived_var])
        data_var = derived_var
        logger.debug(f"Calculating hourly time-probability for var: {data_var}")

        dset = self.dset[data_vars]
        counts = dset.count("time")

        darrays = []
        hours = range(24)
        for hour in hours:
            logger.info(f"Probability for hour {hour:0.0f}")
            dset_hour = self.dset.where(self.hour_of_day == hour)

            # Probability for each variable
            dvar = dset_hour[data_var]
            count = counts[data_var]
            in_bin = dvar == bin_value
            # Persist it here because code is blowing memory up.
            prob = in_bin.sum(dim="time") / count
            darrays.append(prob)

        self.dsout[f"{data_var}{suffix}"] = xr.concat(darrays, "hour_of_day")
        self.dsout["hour_of_day"] = hours

        self._compute(is_compute=compute)
        return self.dsout

    def rpv(
        self,
        return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
        percentile=95,
        distribution="gumbel_r",
        duration=24,
        dim="time",
        data_vars=[],
        derived_vars=[],
        group=None,
        suffix="_rpv",
        compute=False,
    ):
        """Return period values.

        Args:
            return_periods (list): Return period years to calculate rpv values for.
            percentile (float): Percentile above which peaks are selected.
            distribution (str): Statistical distribution to fit the data, any valid
                distribution in scipy.stats, e.g., "gumbel_r", "weibull_min", etc.
            duration (float): Hours in storm below which extra peaks are discarded.
            dim (str): Dimension to calculate rpv along.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        self._update_dset(data_vars)

        logger.debug(f"Calculating rpv for vars: {data_vars}")

        dset = self.dset[data_vars]

        if group is not None:
            logger.info(f"Grouping by {group}")
            suffix += f"_{group}"
            dset = dset.groupby(f"time.{group}")

        dsout = rpv(
            darr=dset,
            return_periods=return_periods,
            percentile=percentile,
            distribution=distribution,
            duration=duration,
            dim=dim,
        )
        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        )

        self._compute(is_compute=compute)
        return dsout

    def apply_func(
        self,
        func,
        dim="time",
        data_vars=[],
        derived_vars=[],
        group=None,
        suffix=None,
        compute=False,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            func (str): Name of valid xarray function to apply.
            dim (str): Dimension to apply function over.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset,
                defined as `f"_{func}"` if `suffix==None`.
            compute (bool): Compute dask variables from output dataset before returning.
            kwargs: kwargs for function func.

        """
        if suffix is None:
            suffix = f"_{func}"

        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        self._update_dset(data_vars)

        logger.debug(f"Calculating {dim}-{func} for vars: {data_vars}")

        dset = self.dset[data_vars]

        if group:
            logger.info(f"Grouping by {group}")
            suffix += f"_{group}"
            dset = dset.groupby(f"time.{group}")

        # Calculate dask stats
        dsout = getattr(dset, func)(dim=dim, **kwargs)

        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        )

        self._compute(is_compute=compute)
        return dsout

    def hsig(
        self,
        dim,
        data_vars=[],
        derived_vars=[],
        compute=False,
        **kwargs,
    ):
        """Time-domain significant wave height.

        Args:
            data_var (str): Data var to calculate Hsig over.
            dim (str): Name of time dimension to calculate Hsig over.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        self._update_dset(data_vars)

        logger.debug(f"Calculating hsig for vars: {data_vars}")

        dset = self.dset[data_vars]

        dsout = 4 * dset.std(dim)
        dsout = dsout.rename({v: f"hsig_{v}" for v in dsout.data_vars})
        dsout.attrs = {
            "standard_name": "sea_surface_wave_significant_height",
            "long_name": "time-domain significant wave height of sea and swell waves",
            "units": "m",
        }

        self.dsout = self.dsout.merge(dsout)
        self._compute(is_compute=compute)
        return dsout

    def hmo_stepwise(
        self,
        dim,
        fs,
        segsec,
        step_x,
        step_y,
        bands=BANDS,
        data_vars=[],
        derived_vars=[],
        xname="x",
        yname="y",
        tname="time",
        **kwargs,
    ):
        """Frequency domain significant wave height for frequency bands.

        Args:
            dim (str): Name of time dimension to calculate Hsig over.
            fs (float): Sampling frequency of data.
            segsec (int): Size of overlapping segments (s).
            step_x (int): x step size for loading slices in memory.
            step_y (int): y step size for loading slices in memory.
            bands (dict): Frequency bands, keys are band labels, values are [fmin, fmax].
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            xname (str): Name of x-coordinate in dataset.
            yname (str): Name of y-coordinate in dataset.
            tname (str): Name of time-coordinate in dataset.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        self._update_dset(data_vars)

        dset = self.dset[data_vars].rename({xname: "x", yname: "y", tname: "time"})

        logger.debug(f"Calculating hmo for vars: {data_vars}")

        # Box slices for looping over grid
        yend = dset.y.size
        xend = dset.x.size
        if yend % step_y != 0:
            yend += step_y
        if xend % step_x != 0:
            xend += step_x
        yarr = pd.interval_range(start=0, end=yend, freq=step_y)
        xarr = pd.interval_range(start=0, end=xend, freq=step_x)

        # Compute each spatial box slice loading before calculating stats
        i = 1
        dsout_list = []
        for iy, yint in enumerate(yarr):
            for xint in xarr:
                logger.info(f"Compute partial dataset {i}/{len(xarr) * len(yarr)}")
                ds = dset.isel(
                    y=slice(yint.left, yint.right),
                    x=slice(xint.left, xint.right),
                ).load()
                dsout = hmo(ds, fs=fs, segsec=segsec, bands=bands, dim=dim)
                dsout_list.append(dsout)
                i += 1
        dsout = xr.combine_by_coords(dsout_list)
        self.dsout = self.dsout.merge(dsout)
        return dsout

    def hmo(
        self,
        dim,
        fs,
        segsec,
        bands=BANDS,
        data_vars=[],
        derived_vars=[],
        compute=False,
        **kwargs,
    ):
        """Frequency domain significant wave height for frequency bands.

        Args:
            dim (str): Name of time dimension to calculate Hsig over.
            fs (float): Sampling frequency of data.
            segsec (int): Size of overlapping segments (s).
            bands (dict): Frequency bands, keys are band labels, values are [fmin, fmax].
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats,
                useful if data_vars=="all" and you also want derived vars.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        self._update_dset(data_vars)

        logger.debug(f"Calculating hmo for vars: {data_vars}")

        dsout = hmo(self.dset[data_vars], fs=fs, segsec=segsec, bands=bands, dim=dim)
        self.dsout = self.dsout.merge(dsout)
        self._compute(is_compute=compute)
        return dsout

    def apply_func_sector(
        self,
        func,
        data_vars,
        dir_var,
        nsector=4,
        dim="time",
        suffix=None,
        compute=False,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            func (str): Name of valid xarray function to apply.
            data_vars (list): Data vars to apply stats over, includes derived vars.
            dir_var (str): Directional data var to bin data over.
            nsector (int): Number of directional sectors.
            dim (str): Dimension to apply function over.
            suffix (str): String to append to each variable name in output dataset,
                defined as `f"_{func}"` if `suffix==None`.
            compute (bool): Compute dask variables from output dataset before returning.
            kwargs: kwargs for function func.

        """
        logger.debug(f"Calculating time-{func} for vars: {data_vars}")

        data_vars += [dir_var]
        self._update_dset(data_vars)

        if suffix is None:
            suffix = f"_{func}"
        suffix += "_dir"

        dsout = directional_stat(
            dset=self.dset[data_vars],
            func=func,
            dir_var=dir_var,
            nsector=nsector,
            dim=dim,
            **kwargs,
        )

        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars})
        )

        self._compute(is_compute=compute)
        return dsout

    def distribution(
        self,
        hs_range,
        tp_range,
        dp_range,
        dim="time",
        group="month",
        hs_name="hs",
        tp_name="tp",
        dp_name="dp",
        label="hs_tp_dp_dist",
        compute=False,
        **kwargs,
    ):
        """Distribution statistics.

        Args:
            hs_range (dict): Numpy arange kwargs defining Hs bins.
            tp_range (dict): Numpy arange kwargs defining Tp bins.
            dp_range (dict): Numpy arange kwargs defining Dp bins.
            dim (str): Dimension to calculate distribution along.
            group (str): Time grouping type, any valid time_{group} such month, season.
            hs_name (str): Name for Hs variable to use in dataset.
            tp_name (str): Name for Tp variable to use in dataset.
            dp_name (str): Name for Dp variable to use in dataset.
            label (str): Name for joint distribution variable.
            compute (bool): Compute dask variables from output dataset before returning.

        """
        data_vars = [hs_name, tp_name, dp_name]
        self._update_dset(data_vars)

        dsout = distribution(
            hs=self.dset[hs_name],
            tp=self.dset[tp_name],
            dp=self.dset[dp_name],
            hs_bins=np.hstack((np.arange(**hs_range), hs_range["stop"])),
            tp_bins=np.hstack((np.arange(**tp_range), tp_range["stop"])),
            dp_bins=np.hstack((np.arange(**dp_range), dp_range["stop"])),
            dim=dim,
            group=group,
            label=label,
        )

        self.dsout = self.dsout.merge(dsout)
        self._compute(is_compute=compute)
        return dsout

    def distribution_stepwise(
        self,
        hs_range,
        tp_range,
        dp_range,
        step_longitude,
        step_latitude,
        dim="time",
        group="month",
        hs_name="hs",
        tp_name="tp",
        dp_name="dp",
        range_from_data=True,
        label="hs_tp_dp_dist",
        compute=False,
        eager=True,
        **kwargs,
    ):
        """Stepwise Hs/Tp/Dp distribution statistics over spatial windows.

        This method is a workaround to avoid memory issues when calculating joint
            distributions over too many bins and is optimal to use with datasets
            chunked for timeseries reading.

        Args:
            hs_range (dict): Numpy arange kwargs defining Hs bins.
            tp_range (dict): Numpy arange kwargs defining Tp bins.
            dp_range (dict): Numpy arange kwargs defining Dp bins.
            step_longitude (int): Longitude step size for loading slices in memory.
            step_latitude (int): Longitude step size for loading slices in memory.
            dim (str): Dimension to calculate distribution along.
            group (str): Time grouping type, any valid time_{group} such month, season.
            hs_name (str): Name for Hs variable to use in dataset.
            tp_name (str): Name for Tp variable to use in dataset.
            dp_name (str): Name for Dp variable to use in dataset.
            range_from_data (bool): Use max Hs, Tp to define last data bin.
            label (str): Name for joint distribution variable.
            compute (bool): Compute and save to partial output dataset.
            eager (bool): Load each box slice before calculating distributions.

        Note:
            Best to choose spatial steps so that dataset to load is large while fitting
                into memory, optimal performance if they match file chunking on disk.

        """
        data_vars = [hs_name, tp_name, dp_name]
        self._update_dset(data_vars)
        dset = self.dset[data_vars]

        # Bins
        hs_bins = np.hstack((np.arange(**hs_range), hs_range["stop"]))
        tp_bins = np.hstack((np.arange(**tp_range), tp_range["stop"]))
        dp_bins = np.hstack((np.arange(**dp_range), dp_range["stop"]))

        # Compute upper bins for Hs, Tp from data
        if range_from_data:
            logger.info(f"Computing Hs, Tp bin edges from data")

            # Hs
            if f"{hs_name}_max" in self.dsout:
                hsmax = float(self.dsout[f"{hs_name}_max"].compute().max())
            else:
                hsmax = float(self.dset[hs_name].max())
            logger.debug(f"Max Hs: {hsmax} m")
            idmax = np.argmax(hs_bins >= hsmax) or len(hs_bins) - 1
            hs_bins = hs_bins[: idmax + 1]

            # Tp
            if f"{tp_name}_max" in self.dsout:
                tpmax = float(self.dsout[f"{tp_name}_max"].compute().max())
            else:
                tpmax = float(self.dset[tp_name].max().compute())
            logger.debug(f"Max Tp: {tpmax} s")
            idmax = np.argmax(tp_bins >= tpmax) or len(tp_bins) - 1
            tp_bins = tp_bins[: idmax + 1]

        logger.info(f"Hs bins: {hs_bins}, Tp bins: {tp_bins}, Dp bins: {dp_bins}")

        # Box slices for looping over grid
        yend = dset.latitude.size
        xend = dset.longitude.size
        if yend % step_latitude != 0:
            yend += step_latitude
        if xend % step_longitude != 0:
            xend += step_longitude
        lats = pd.interval_range(start=0, end=yend, freq=step_latitude)
        lons = pd.interval_range(start=0, end=xend, freq=step_longitude)

        # Compute each spatial box slice dumping each full latitude slice
        tmp_store = os.path.join(self.localdir, "tmpdist.zarr")
        i = 1
        for ilat, lat_interval in enumerate(lats):
            dslat = xr.Dataset()
            for lon_interval in lons:
                logger.info(f"Compute partial dataset {i}/{len(lons) * len(lats)}")
                ds = dset.isel(
                    latitude=slice(lat_interval.left, lat_interval.right),
                    longitude=slice(lon_interval.left, lon_interval.right),
                )
                if eager:
                    logger.debug(f"Loading sliced dataset into memory")
                    ds = ds.load()

                dist = distribution(
                    hs=ds[hs_name],
                    tp=ds[tp_name],
                    dp=ds[dp_name],
                    hs_bins=hs_bins,
                    tp_bins=tp_bins,
                    dp_bins=dp_bins,
                    dim=dim,
                    group=group,
                    label=label,
                )
                dslat = xr.combine_by_coords([dslat, dist])
                i += 1
            # Dump to temporary archive
            logger.info(f"Writing latitude slice {ilat + 1}/{len(lats)} to tmp archive")
            if ilat == 0:
                dslat.to_zarr(tmp_store, mode="w", consolidated=True)
            else:
                dslat.to_zarr(
                    tmp_store, mode="a", append_dim="latitude", consolidated=True
                )
            del dslat

        dsout = xr.open_zarr(tmp_store, consolidated=True)
        dsout[label].encoding.pop("chunks", None)

        self.dsout = self.dsout.merge(dsout)
        self._compute(is_compute=compute)
        return dsout

    def distribution_spddir_stepwise(
        self,
        spd_range,
        dir_range,
        step_longitude,
        step_latitude,
        dim="time",
        group="month",
        spd_name="wspd",
        dir_name="wdir",
        range_from_data=True,
        label="spd_dir_dist",
        compute=False,
        eager=True,
        **kwargs,
    ):
        """Stepwise speed/direction distribution statistics over spatial windows.

        This method is a workaround to avoid memory issues when calculating joint
            distributions over too many bins and is optimal to use with datasets
            chunked for timeseries reading.

        Args:
            spd_range (dict): Numpy arange kwargs defining spd bins.
            dir_range (dict): Numpy arange kwargs defining dir bins.
            step_longitude (int): Longitude step size for loading slices in memory.
            step_latitude (int): Longitude step size for loading slices in memory.
            dim (str): Dimension to calculate distribution along.
            group (str): Time grouping type, any valid time_{group} such month, season.
            spd_name (str): Name for spd variable to use in dataset.
            dir_name (str): Name for dir variable to use in dataset.
            range_from_data (bool): Use max Spd to define last data bin.
            label (str): Name for joint distribution variable.
            compute (bool): Compute and save to partial output dataset.
            eager (bool): Load each box slice before calculating distributions.

        Note:
            Best to choose spatial steps so that dataset to load is large while fitting
                into memory, optimal performance if they match file chunking on disk.

        """
        data_vars = [spd_name, dir_name]
        self._update_dset(data_vars)
        dset = self.dset[data_vars]

        # Bins
        spd_bins = np.hstack((np.arange(**spd_range), spd_range["stop"]))
        dir_bins = np.hstack((np.arange(**dir_range), dir_range["stop"]))

        # Compute upper bins for Spd from data
        if range_from_data:
            logger.info(f"Computing Spd bin edges from data")

            if f"{spd_name}_max" in self.dsout:
                spdmax = float(self.dsout[f"{spd_name}_max"].compute().max())
            else:
                spdmax = float(self.dset[spd_name].max())
            logger.debug(f"Max spd: {spdmax} m/s")
            idmax = np.argmax(spd_bins >= spdmax) or len(spd_bins) - 1
            spd_bins = spd_bins[: idmax + 1]

        logger.info(f"Spd bins: {spd_bins}, Dir bins: {dir_bins}")

        # Box slices for looping over grid
        yend = dset.latitude.size
        xend = dset.longitude.size
        if yend % step_latitude != 0:
            yend += step_latitude
        if xend % step_longitude != 0:
            xend += step_longitude
        lats = pd.interval_range(start=0, end=yend, freq=step_latitude)
        lons = pd.interval_range(start=0, end=xend, freq=step_longitude)

        # Compute each spatial box slice dumping each full latitude slice
        tmp_store = os.path.join(self.localdir, "tmp_spddir_dist.zarr")
        i = 1
        for ilat, lat_interval in enumerate(lats):
            dslat = xr.Dataset()
            for lon_interval in lons:
                logger.info(f"Compute partial dataset {i}/{len(lons) * len(lats)}")
                ds = dset.isel(
                    latitude=slice(lat_interval.left, lat_interval.right),
                    longitude=slice(lon_interval.left, lon_interval.right),
                )
                if eager:
                    logger.debug(f"Loading sliced dataset into memory")
                    ds = ds.load()

                dist = distribution_spddir(
                    spd=ds[spd_name],
                    dir=ds[dir_name],
                    spd_bins=spd_bins,
                    dir_bins=dir_bins,
                    dim=dim,
                    group=group,
                    label=label,
                )
                dslat = xr.combine_by_coords([dslat, dist])
                i += 1
            # Dump to temporary archive
            logger.info(f"Writing latitude slice {ilat + 1}/{len(lats)} to tmp archive")
            if ilat == 0:
                dslat.to_zarr(tmp_store, mode="w", consolidated=True)
            else:
                dslat.to_zarr(
                    tmp_store, mode="a", append_dim="latitude", consolidated=True
                )
            del dslat

        dsout = xr.open_zarr(tmp_store, consolidated=True)
        dsout[label].encoding.pop("chunks", None)

        self.dsout = self.dsout.merge(dsout)
        self._compute(is_compute=compute)
        return dsout

    def to_netcdf(self, outfile, format="NETCDF4", _FillValue=-32767):
        """Save output dataset as netcdf.

        Args:
            outfile (str): Name of output netcdf file.
            format (str): Output Netcdf file format.
            _FillValue (int): Fill Value.

        """
        logger.debug(f"Saving stats dataset into file: {outfile}")
        encoding = {}
        for data_var in self.dsout.data_vars:
            encoding[data_var] = {"zlib": True, "_FillValue": _FillValue}
            encoding[data_var].update(self.dsout[data_var].encoding)
        # Loading into memory before saving to disk. We may want to reassess this.
        self._load()
        self._sortby()
        self._setattrs()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)
        if self.updir:
            self._upload(outfile)

    def to_zarr(
        self,
        outfile,
        dsout=None,
        _FillValue=-32767,
        mode="w",
        chunksizes={"": {}},
        **kwargs,
    ):
        """Save output dataset as zarr.

        Args:
            outfile (str): Base name of output zarr file.
            dsout (str): Output dataset to write, self.dsout by default.
            _FillValue (int): Fill Value.
            mode (str): Zarr write mode.
            chunksizes (dict): Key is a suffix for outfile, values are chunks, one file
                is saved for each key-value in the dictionary.

        """
        logger.debug(f"Saving stats dataset into file: {outfile}")
        dsout = dsout or self.dsout
        for data_var in dsout.data_vars:
            dsout[data_var].encoding.update({"_FillValue": _FillValue})
            dsout[data_var].encoding.pop("zlib", None)
        self._sortby()
        self._setattrs()

        basename, ext = os.path.splitext(outfile)
        for suffix, chunks in chunksizes.items():
            if suffix and not suffix.startswith("_"):
                suffix = "_" + suffix
            store = f"{basename}{suffix}{ext}"
            logger.info(f"Writing zarr file {store} with chunks {chunks}")
            fsmap = get_mapper(store)
            included_chunks = {c: v for c, v in chunks.items() if c in dsout.dims}
            dsout.chunk(included_chunks).to_zarr(fsmap, consolidated=True, mode=mode)
            if self.updir:
                self._upload(store)


class Stats:
    def __init__(
        self,
        outfile,
        dset=None,
        urlpath=None,
        engine="zarr",
        catalog=None,
        dataset_id=None,
        mapping={},
        slice_dict={},
        updir=None,
        localdir="/scratch",
        allow_split_large_chunks=False,
        n_workers=None,
        calls=[],
        **kwargs,
    ):
        """Gridded stats using dask arrays.

        Args:
            outfile (str): Name or URI of output file, must end with ".nc" or ".zarr".
            dset (xr.Dataset): Source dataset to calculate stats from.
            urlpath (str): Path or URI of dataset file to calculate stats from.
            engine (str): Engine to use with xr.open_dataset when using urlpath.
            catalog (str): Path or URI of intake catalog with datasets to open.
            dataset_id (str): Intake dataset id to calculate stats from.
            mapping (dict): Dictionary for renaming dataset variables.
            slice_dict (dict): Dictionary specifying slicing arg.
            updir (str): Upload direction to upload netcdf and zarr stats files to.
            allow_split_large_chunks (bool): Allow dask auto-resize of small chunks.
            n_workers (int): Number of workers for local dask distributed cluster.
            calls (list): List of dicts defining each stats method to run with keys:
                method: name of stats method to run.
                kwargs: kwargs to run the stats method.

        Note:
            You must provide one of 'dset', 'urlpath' or ['catalog', 'dataset_id'].

        """
        dask.config.set({"array.slicing.split_large_chunks": allow_split_large_chunks})

        if not any([dset, urlpath, catalog and dataset_id]):
            raise ValueError(
                "You must provide one of dset, urlpath or [catalog, dataset_id]"
            )

        self.outfile = outfile
        self.dset = dset
        self.urlpath = urlpath
        self.catalog = catalog
        self.dataset_id = dataset_id
        self.engine = engine
        self.mapping = mapping
        self.slice_dict = slice_dict
        self.n_workers = n_workers
        self.updir = updir
        self.calls = calls

        self.dsout = xr.Dataset()

    def __call__(self):
        """Loop over list of dictionary to execute stats methods.

        Local dask clusters are set up for running each stats method.

        """
        # Execute each stats method
        for call in self.calls:
            method = call["method"]
            kwargs = call.get("kwargs", {})
            logger.info(f"Stat.{method}({kwargs})")
            with Client(processes=True, n_workers=self.n_workers) as client:
                logger.info(client)
                getattr(self, method)(**kwargs)
            self._clean_dask_worker_space()

        # Save output file
        if self.outfile.endswith(".nc"):
            self.to_netcdf(self.outfile)
        elif self.outfile.endswith(".zarr"):
            self.to_zarr(self.outfile)

    def _clean_dask_worker_space(self):
        """Remove existing dask worker space directory."""
        p = Path("./dask-worker-space")
        if p.is_dir():
            logger.info("Removing existing dask worker space")
            shutil.rmtree(p)

    def _open_dataset(self, chunks={}):
        """Open and slice dataset according to the init attributes provided.

        Args:
            chunks (dict): Mapping dim: size for chunking each dimension,
                only used if opening dataset from urlpath to avoid rechunking.

        """
        logger.info("Open dataset")

        # Open dataset
        if isinstance(self.dset, xr.Dataset):
            logger.debug("Dataset provided")
            dset = self.dset
        elif isinstance(self.dset, str):
            raise ValueError(
                f"dset must be a dataset instance, got {self.dset}. Use 'urlpath' "
                "and 'engine' to specify a dataset path to open"
            )
        elif self.urlpath:
            logger.debug(f"Open dataset from urlpath: {self.urlpath}")
            dset = xr.open_dataset(self.urlpath, engine=self.engine, chunks=chunks)
        elif self.catalog and self.dataset_id:
            logger.debug(f"Open dataset from intake: {self.catalog}-{self.dataset_id}")
            cat = open_catalog(self.catalog)
            if not cat:
                raise ValueError(f"Cannot open intake catalog from {self.catalog}")
            dset = cat[self.dataset_id].to_dask()
        else:
            raise ValueError("Cannot identify source dataset from input arguments")

        # Rename dataset
        mapping = {k: v for k, v in self.mapping.items() if k in dset}
        if mapping:
            logger.debug(f"Rename {dset} as {mapping}")
            dset = dset.rename(mapping)

        # Slice dataset
        dset = self._slice(dset)

        logger.info(f"Source dataset:\n{dset}")
        return dset

    def _slice(self, dset):
        """Slice dataset from slice_dict attribute."""
        for slice_method, slice_kwargs in self.slice_dict.items():
            dset = getattr(dset, slice_method)(**slice_kwargs)
            sizes = {k: v.size for k, v in ds.coords.items()}
            for k, v in sizes.items():
                if v == 0:
                    raise ValueError(
                        f"Slicing dataset {dset} with slice_dict {self.slice_dict}"
                        f" resulted in coordinate {k} with zero size"
                    )
        return dset

    def _upload(self, filename):
        """Upload stats files.

        Args:
            filename (str): Name of file to upload.

        """
        outfile = os.path.join(self.updir, os.path.basename(filename))
        logger.info(f"Uploading {filename} --> {outfile}")
        if exists(outfile):
            logger.debug(f"Removing existing file {outfile} before uploading")
            rm(outfile, recursive=isdir(outfile))
        put(filename, outfile, recursive=isdir(filename))

    def _sortby(self):
        """Sort output dataset by all coordinates."""
        for name, coords in self.dsout.coords.items():
            if coords[0] > coords[-1]:
                logger.info(f"Sorting by coordinate {name}")
                self.dsout = self.dsout.sortby(name)

    def _setattrs(self):
        """Define some attributes in output dataset."""
        if "quantile" in self.dsout.coords:
            self.dsout["quantile"].attrs = {
                "standard_name": "quantile",
                "long_name": "quantile",
                "units": "",
            }

    def apply_func(
        self,
        func,
        dim="time",
        data_vars=[],
        group=None,
        suffix=None,
        compute=True,
        chunks={},
        **kwargs,
    ):
        """apply xarray function.

        Args:
            func (str): Name of valid xarray function to apply.
            dim (str): Dimension to apply function over.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset,
                defined as `f"_{func}"` if `suffix==None`.
            compute (bool): Compute dask variables from output dataset before returning.
            kwargs: kwargs for function func.

        """
        if suffix is None:
            suffix = f"_{func}"

        # Open dataset
        dset = self._open_dataset(chunks=chunks)

        # Selecting variables
        if data_vars == "all":
            data_vars = [k for k in dset.data_vars]
        logger.debug(f"Calculating {dim}-{func} for vars: {data_vars}")
        dset = dset[data_vars]

        # Grouping by
        if group:
            logger.info(f"Grouping by {group}")
            suffix += f"_{group}"
            dset = dset.groupby(f"time.{group}")

        # Calculate stats
        dsout = getattr(dset, func)(dim=dim, **kwargs)
        if compute:
            dsout = dsout.load()

        # Rename
        dsout = dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})

        # Merge onto output
        self.dsout = self.dsout.merge(dsout)

        return dsout

    def to_netcdf(self, outfile, format="NETCDF4", _FillValue=-32767):
        """Save output dataset as netcdf.

        Args:
            outfile (str): Name of output netcdf file.
            format (str): Output Netcdf file format.
            _FillValue (int): Fill Value.

        """
        logger.debug(f"Saving stats dataset into file: {outfile}")
        encoding = {v: {"zlib": True} for v in self.dsout.data_vars}
        self._sortby()
        self._setattrs()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)
        if self.updir:
            self._upload(outfile)

    def to_zarr(
        self,
        outfile,
        dsout=None,
        _FillValue=-32767,
        mode="w",
        chunksizes={"": {}},
        **kwargs,
    ):
        """Save output dataset as zarr.

        Args:
            outfile (str): Base name of output zarr file.
            dsout (str): Output dataset to write, self.dsout by default.
            _FillValue (int): Fill Value.
            mode (str): Zarr write mode.
            chunksizes (dict): Key is a suffix for outfile, values are chunks, one file
                is saved for each key-value in the dictionary.

        """
        logger.debug(f"Saving stats dataset into file: {outfile}")
        dsout = dsout or self.dsout
        for data_var in dsout.data_vars:
            dsout[data_var].encoding.update({"_FillValue": _FillValue})
            dsout[data_var].encoding.pop("zlib", None)
        self._sortby()
        self._setattrs()

        basename, ext = os.path.splitext(outfile)
        for suffix, chunks in chunksizes.items():
            if suffix and not suffix.startswith("_"):
                suffix = "_" + suffix
            store = f"{basename}{suffix}{ext}"
            logger.info(f"Writing zarr file {store} with chunks {chunks}")
            fsmap = get_mapper(store)
            included_chunks = {c: v for c, v in chunks.items() if c in dsout.dims}
            dsout.chunk(included_chunks).to_zarr(fsmap, consolidated=True, mode=mode)
            if self.updir:
                self._upload(store)
