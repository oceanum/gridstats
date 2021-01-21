"""Calculated gridded stats using xarray and dask."""
import os
import shutil
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from fsspec import get_mapper

from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from distributed.diagnostics.progressbar import get_scheduler

from ontake.ontake import Ontake
from oncore.dataio import put, isdir, exists, rm

from onstats.utils import uv_to_spddir
import onstats.derived_variable as dv
from onstats.xarray_stats import rpv, distribution


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


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
        self.var_fp_sw1 = var_fp_sw1
        self.var_tp_sw1 = var_tp_sw1
        self.var_cloud_cover = var_cloud_cover
        self.var_uwnd150 = var_uwnd150
        self.var_vwnd150 = var_vwnd150

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


class Stats(DerivedVar):
    def __init__(
        self,
        dataset,
        master_url="gs://oceanum-catalog/oceanum.yml",
        namespace="hindcast",
        chunk=None,
        mask=None,
        slice_dict={},
        chunks=None,
        persist=False,
        updir=None,
        logger=logging,
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
            slice_dict (dict): Dictionary specifying slicing arg.
            chunks (dict): Chunking dict to rechunk dataset after opening.
            persist (bool): If True, persist output dataset before saving as netcdf.
            updir (str): Upload direction to upload netcdf and zarr stats files to.

        Tips:
            Run the calculations on a dask distributed cluster. This optimise
                distribution of computation efficiently.
            Check out computation on dashboard hosted on http://localhost:8787/status.
            When running with hyperthreading on (i.e., google VMs), limit cluster to
                half the number of cores if having memory issues.
            If still having memory issues, trigger computation after each method call,
                this will be slower but will avoid blowing up memory resources.

        """
        self.dataset = dataset
        self.master_url = master_url
        self.namespace = namespace
        self.chunk = chunk
        self.mask = mask
        self.slice_dict = slice_dict
        self.chunks = chunks
        self.persist = persist
        self.updir = updir

        self._hour_of_day = None

        # Open dataset
        self._open_dataset()
        self.dsout = xr.Dataset()

        # Instantiating DerivedVar
        super().__init__(self.dset, **kwargs)

        # Define mask
        self._set_mask(mask)

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

    def _open_dataset(self):
        """Set dset attribute either from ontake dataset of from xarray dataset itself.

        If self.dataset is a string, it should be a valid ontake dataset and the
            ontake master_url and namespace arguments must be provided at initialisation.
            If self.dataset is an xarray dataset then it is just assigned to self.dset attribute.

        Note: there is a point of failure here is the dataset string is a substring of
            more than one intake dataset in catalog. This should be fixed in the future.

        """
        logger.info("Open dataset")
        if isinstance(self.dataset, str) and self.dataset.endswith(".nc"):
            logger.debug(f"Opening netcdf file: {self.dataset}")
            self.dset = xr.open_dataset(self.dataset, chunks=self.chunks)
        elif isinstance(self.dataset, str):
            try:
                logger.debug(f"Try opening zarr store from URI: {self.dataset}")
                self.dset = xr.open_zarr(get_mapper(self.dataset), consolidated=True)
            except KeyError:
                logger.debug(
                    f"Ontake dataset {self.dataset} {self.master_url} {self.namespace}"
                )
                # Open catalog and ensure dataset is a substring of a catalog entry
                ot = Ontake(master_url=self.master_url, namespace=self.namespace)
                kwargs = {}
                if self.chunk is not None:
                    kwargs.update({"chunk": self.chunk})
                self.dset = ot.dataset(self.dataset, **kwargs)
        elif isinstance(self.dataset, xr.Dataset):
            self.dset = self.dataset
        else:
            raise ValueError(
                "dataset must be either a string specifying an ontake "
                "dataset id or bucket URI, or an xarray dataset."
            )
        # Slicing
        self._slice_dset()
        # Rechunking
        if self.chunks:
            logger.info(f"Re-chunking dataset as {self.chunks}")
            self.dset = self.dset.chunk(self.chunks)
        self.data_vars = list(self.dset.data_vars.keys())

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
            raise AttributeError(f"Derived var {name} must be defined in DerivedVar")
        if not isinstance(getattr(DerivedVar, name), property):
            raise TypeError(f"Derived var {name} must be a property in DerivedVar")
        if not isinstance(getattr(self, name), xr.DataArray):
            raise TypeError(f"Property {name} in DerivedVar must return a DataArray.")

    def _count(self, data_var, dim="time"):
        """Returns the count array over dimension dim accounting for missing values."""
        logger.debug(f"Calculating {data_var} count over {dim}")
        if isinstance(data_var, str):
            dvar = self.dset[data_var]
        else:
            dvar = data_var
        count = (0 * dvar + 1).sum(dim=dim, skipna=True)
        return count.where(count > 0)

    def range_probability(self, data_ranges, dim="time", **kwargs):
        """Calculate probability of specific ranges.

        Args:
            data_ranges (list): List of dictionaries to define each data range to
                calculate probabilities from with keys:
                - var (str): Variable name, can be a valid data_var or derived_var.
                - start (float): Minimum value for interval.
                - stop (float); Maximum value for interval.
                - left (closed | open): Define if minimum value should be included.
                - right (closed | open): Define if maximum value should be included.
            dim (str): Dimension name to calculate probabilities over.
            kwargs: Not used here, ignored.

        """
        data_vars = list(set([data_range["var"] for data_range in data_ranges]))
        derived_vars = [v for v in data_vars if v not in self.dset.data_vars]
        self._update_dset(derived_vars)

        logger.debug(f"Calculating time-probability for vars: {data_vars}")

        # Probability for each range
        counts = {}
        for data_range in data_ranges:

            # Data variable to compute
            dvar = data_range["var"]
            darray = self.dset[dvar]

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

            # Count array
            if dvar not in counts:
                counts[dvar] = self._count(data_var=dvar, dim=dim)

            # Probability
            in_range = lfunc(darray, start) & rfunc(darray, stop)
            self.dsout[varname] = in_range.sum(dim=dim) / counts[dvar]

        self.dset = self.dset.where(self.dset.mask)

    def data_count(
        self, dim="time", data_vars=[], derived_vars=[], suffix="_pcount", **kwargs
    ):
        """Calculate the percentage of valid data along dimension.

        Args:
            dim (str): Dimension to calculate percentage count over.
            data_vars (list): Data vars to apply stats over.
            derived_vars (list): Derived_vars to calculate before applying stats.
            suffix (str): String to append to each variable name in output dataset.

        """
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        if not data_vars:
            raise ValueError("At least one data_var or derived_var should be provided")

        self._update_dset(derived_vars)
        logger.debug(f"Calculating count percentage for vars: {data_vars}")

        dsout = 100 * self._count(self.dset[data_vars], dim) / self.dset[dim].size
        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        )
        return dsout

    def value_probability(
        self,
        dim="time",
        data_vars=[],
        derived_vars=[],
        bins=[],
        bin_name="bin",
        suffix="_prob",
        **kwargs,
    ):
        """Calculate the probability of specific values.

        This function is useful for integer-type data, use range_probability for float.

        Args:
            data_vars (list): Data vars to apply stats over.
            derived_vars (list): Derived_vars to calculate before applying stats.
            bins (list): List of values for binning the data to calculate probability.
            bin_name (str): Name of bin coordinate in output probability dataset. Note
                that the bin coordinate is only created if there is more than one bin.
            suffix (str): String to append to each variable name in output dataset.

        Note:
            At least one `data_var` or `derived_var` should be provided.
            The output dataset has an extra coordinate with name defaulting to `bin`,
                representing the bin values over which probabilities are calculated.

        """
        assert list(data_vars) + list(
            derived_vars
        ), "At least one data_var or derived_var should be provided."

        bins = list(bins)
        self._update_dset(derived_vars)
        data_vars = list(data_vars) + list(derived_vars)
        logger.debug(f"Calculating time-probability for vars: {data_vars}")

        # Probability for each variable
        for data_var in data_vars:
            dvar = self.dset[data_var]
            count = self._count(data_var=data_var, dim=dim)
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

    def time_probability_hour_of_day(
        self, derived_var, bin_value, suffix="_hprob", **kwargs
    ):
        """Calculate the time probability for each hour with time offset accounted.

        Args:
            derived_var (str): Derived_vars to calculate before applying stats.
            bin (value): List of values for binning the data to calculate probability.
            suffix (str): String to append to each variable name in output dataset.

        Note:
            Only implemented atm for one single derived variable and one matching bin.

        """
        self._update_dset([derived_var])
        data_var = derived_var
        logger.debug(f"Calculating hourly time-probability for var: {data_var}")

        darrays = []
        hours = range(24)
        for hour in hours:
            logger.info(f"Probability for hour {hour:0.0f}")
            dset_hour = self.dset.where(self.hour_of_day == hour)

            # Probability for each variable
            dvar = dset_hour[data_var]
            count = self._count(data_var=dvar, dim="time")
            in_bin = dvar == bin_value
            # Persist it here because code is blowing memory up.
            prob = in_bin.sum(dim="time") / count
            darrays.append(prob)

        self.dsout[f"{data_var}{suffix}"] = xr.concat(darrays, "hour_of_day")
        self.dsout["hour_of_day"] = hours

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
            derived_vars (list): Derived_vars to calculate before applying stats.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset.

        """
        self._update_dset(derived_vars)
        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars

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
        return dsout

    def apply_func(
        self,
        func,
        dim="time",
        data_vars=[],
        derived_vars=[],
        group=None,
        suffix=None,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            func (str): Name of valid xarray function to apply.
            dim (str): Dimension to apply function over.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            derived_vars (list): Derived_vars to calculate before applying stats.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset,
                defined as `f"_{func}"` if `suffix==None`.

        """
        self._update_dset(derived_vars)
        if suffix is None:
            suffix = f"_{func}"

        if data_vars == "all":
            data_vars = self.data_vars
        data_vars += derived_vars
        logger.debug(f"Calculating time-{func} for vars: {data_vars}")

        dset = self.dset[data_vars]

        if group is not None:
            logger.info(f"Grouping by {group}")
            suffix += f"_{group}"
            dset = dset.groupby(f"time.{group}")

        # Calculate dask stats
        dsout = getattr(dset, func)(dim=dim, **kwargs)

        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        )
        return dsout

    def apply_func_sector(
        self,
        func,
        data_vars,
        dir_var,
        nsector=4,
        dim="time",
        suffix=None,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            data_vars (list): Data vars to apply stats over, includes derived vars.
            func (str): Name of valid xarray function to apply.
            dim (str): Dimension to apply function over.
            suffix (str): String to append to each variable name in output dataset,
                defined as `f"_{func}"` if `suffix==None`.

        """
        logger.debug(f"Calculating time-{func} for vars: {data_vars}")

        derived_vars = [
            v for v in data_vars + [dir_var] if v not in self.dset.data_vars
        ]
        self._update_dset(list(set(derived_vars)))

        if suffix is None:
            suffix = f"_{func}"
        suffix += "_dir"

        dset = self.dset[data_vars]
        dirs = self.dset[dir_var]

        ds = 360 / nsector
        sectors = np.linspace(0, 360 - ds, nsector)
        starts = (sectors - ds / 2) % 360
        stops = (sectors + ds / 2) % 360
        dsout = []
        for start, stop in zip(starts, stops):
            if stop > start:
                mask = (dirs >= start) & (dirs < stop)
            else:
                mask = (dirs >= start) | (dirs < stop)
            dsout.append(dset.where(mask))
        dsout = xr.concat(dsout, dim="direction").assign_coords({"direction": sectors})
        dsout["direction"].attrs = {
            "standard_name": dirs.attrs.get("standard_name", "direction"),
            "long_name": dirs.attrs.get("standard_name", "direction sector"),
            "units": dirs.attrs.get("units", "degree"),
            "variable_name": dir_var,
        }

        # Calculate dask stats
        dsout = getattr(dsout, func)(dim=dim, **kwargs)

        self.dsout = self.dsout.merge(
            dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars})
        )
        return dsout

    def distribution(
        self,
        ranges,
        dim="time",
        mask_var=None,
        mapping={},
        suffix="_dist",
        **kwargs,
    ):
        """Distribution statistics.

        Args:
            ranges (dict): Bins definition, keys are the variable names, values
                are the kwargs for pandas.interval_range function to define bins, e.g.
                ranges={"hs": {"start": 0, "end": 3, "freq": 0.5}, "tp": {"start": 0, "end": 20, "freq": 5}}.
            dim (str): Dimension to calculate distribution along.
            mask_var (str): Name of variable to use for masking land.
            mapping (dict): Mapping to rename distribution variables.
            suffix (str): String to append to each variable name in output dataset.

        """

        data_vars = list(ranges.keys())
        self._update_dset(data_vars)

        dsout = distribution(
            dset=self.dset, ranges=ranges, dim=dim, mask_var=mask_var, mapping=mapping
        )

        self.dsout = self.dsout.merge(
            dsout.rename(
                {v: f"{v}{suffix}" for v in dsout.data_vars if v != "data_count"}
            )
        )
        dsout

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
            encoding.update({data_var: {"zlib": True, "_FillValue": _FillValue}})
        # Loading into memory before saving to disk. We may want to reassess this.
        self._load()
        self._sortby()
        self._setattrs()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)
        if self.updir:
            self._upload(outfile)

    def to_zarr(self, outfile, _FillValue=-32767, **kwargs):
        """Save output dataset as zarr.

        Args:
            outfile (str): Name of output zarr file.
            _FillValue (int): Fill Value.

        """
        logger.debug(f"Saving stats dataset into file: {outfile}")
        self._sortby()
        self._setattrs()
        encoding = {}
        for data_var in self.dsout.data_vars:
            encoding.update({data_var: {"_FillValue": _FillValue}})
        fsmap = get_mapper(outfile)
        self.dsout.to_zarr(fsmap, consolidated=True, encoding=encoding, mode="w")
        if self.updir:
            self._upload(outfile)
