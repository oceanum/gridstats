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

from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from distributed.diagnostics.progressbar import get_scheduler

from ontake.ontake import Ontake

from stats.util import uv_to_spddir
import stats.derived_variable as dv


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
        logger=logging,
        **kwargs
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
        mask=None,
        slice_dict={},
        chunks=None,
        persist=False,
        logger=logging,
        **kwargs
    ):
        """Gridded stats using dask arrays.

        Args:
            dataset (str, xr.Dataset): An ontake dataset id if string or an xarray dataset.
            master_url (str): Ontake catalog master url path.
            namespace (str): Ontake catalog namespace.
            mask (str): either a variable name or an expression to evaluate on one or
                more variables to define a mask array for masking output dataset. e.g.,
                `self.dset.hs==0`.
            slice_dict (dict): Dictionary specifying slicing arg.
            chunks (dict): Chunking dict to rechunk dataset after opening.
            persist (bool): if True, persist output dataset before saving as netcdf.

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
        self.mask = mask
        self.chunks = chunks
        self.persist = persist

        self._hour_of_day = None

        # Open dataset
        self._open_dataset()
        self.dsout = xr.Dataset()

        # Instantiating DerivedVar
        super().__init__(self.dset, **kwargs)

        # Define mask
        self._set_mask(mask)

        # Slicing
        self._slice_dset(slice_dict or {})

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

    def _slice_dset(self, slice_dict):
        """Masking dataset using slice_dict."""
        for slice_method, slice_kwargs in slice_dict.items():
            self.dset = getattr(self.dset, slice_method)(**slice_kwargs)
        logger.debug("Processing dataset: {}".format(self.dset))

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

    def _open_dataset(self):
        """Set dset attribute either from ontake dataset of from xarray dataset itself.

        If self.dataset is a string, it should be a valid ontake dataset and the
            ontake master_url and namespace arguments must be provided at initialisation.
            If self.dataset is an xarray dataset then it is just assigned to self.dset attribute.

        Note: there is a point of failure here is the dataset string is a substring of
            more than one intake dataset in catalog. This should be fixed in the future.

        """
        logger.info("Open dataset")
        if isinstance(self.dataset, str):
            logger.debug(
                "Opening ontake dataset from: {} {} {}".format(
                    self.dataset, self.master_url, self.namespace
                )
            )
            # Open catalog and ensure dataset is a substring of a catalog entry
            ot = Ontake(master_url=self.master_url, namespace=self.namespace)
            self.dset = ot.dataset(self.dataset)
        elif isinstance(self.dataset, xr.Dataset):
            self.dset = self.dataset
        else:
            raise ValueError(
                "dataset must be either a string specifying an ontake dataset id or an xarray dataset."
            )
        if self.chunks:
            logger.info("Re-chunking dataset as {}".format(self.chunks))
            self.dset = self.dset.chunk(self.chunks)

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

    def _update_dset(self, derived_vars):
        """Append derived variables to dataset.

        Args:
            derived_vars (list): List of derived variable names to append. Each derived
                variable must (1) exist as a property in DerivedVar class, and (2) be
                able to be derived from existing variables in dataset.

        """
        for derived_var in derived_vars:
            self._is_derived_variable(derived_var)
            if derived_var in self.dset.data_vars:
                continue
            logger.info(
                "Updating dataset with derived variable: {}".format(derived_var)
            )
            self.dset[derived_var] = getattr(self, derived_var)

    def _is_derived_variable(self, name):
        """Check that derived variable has been properly prescribed."""
        assert (
            getattr(DerivedVar, name, None) is not None
        ), "Derived variable {} must be defined in DerivedVar class".format(name)
        assert isinstance(
            getattr(DerivedVar, name), property
        ), "Derived variable {} must be defined as a @property in the DerivedVar class".format(
            name
        )
        assert isinstance(
            getattr(self, name), xr.DataArray
        ), "Property {} in DerivedVar class must return a DataArray object.".format(
            name
        )

    def _time_count(self, data_var, dim="time"):
        """Returns the count array over dimension dim accounting for missing values."""
        logger.debug("Calculating {} count over {}".format(data_var, dim))
        if isinstance(data_var, str):
            dvar = self.dset[data_var]
        else:
            dvar = data_var
        count = (0 * dvar + 1).sum(dim=dim, skipna=True)
        return count.where(count > 0)

    def range_probability(
        self,
        data_ranges,
        dim="time",
        **kwargs
    ):
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
            varname = f"{dvar}_{llabel}-{rlabel}"

            # Count array
            if dvar not in counts:
                counts[dvar] = self._time_count(data_var=dvar, dim=dim)

            # Probability
            in_range = lfunc(darray, start) & rfunc(darray, stop)
            self.dsout[varname] = (in_range.sum(dim=dim) / counts[dvar])

        self.dset = self.dset.where(self.dset.mask)

    def value_probability(
        self,
        dim="time",
        data_vars=[],
        derived_vars=[],
        bins=[],
        bin_name="bin",
        prefix="prob_",
        **kwargs
    ):
        """Calculate the probability of specific values.

        This function is useful for integer-type data, use range_probability for float.

        Args:
            data_vars (list): Data vars to apply stats over.
            derived_vars (list): Derived_vars to calculate before applying stats.
            bins (list): List of values for binning the data to calculate probability.
            bin_name (str): Name of bin coordinate in output probability dataset. Note
                that the bin coordinate is only created if there is more than one bin.
            prefix (str): String to prepend to each variable name in output dataset.

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
        logger.debug("Calculating time-probability for vars: {}".format(data_vars))

        # Probability for each variable
        for data_var in data_vars:
            dvar = self.dset[data_var]
            count = self._time_count(data_var=data_var, dim=dim)
            darrays = []
            for bin_value in bins:
                in_bin = dvar == bin_value
                darrays.append(in_bin.sum(dim=dim) / count)
            # Create extra coordinate only if more than one bin
            if len(bins) == 1:
                self.dsout["{}{}".format(prefix, data_var)] = darrays[0].where(
                    self.dset.mask
                )
            else:
                self.dsout["{}{}".format(prefix, data_var)] = xr.concat(
                    darrays, bin_name
                ).where(self.dset.mask)
        if len(bins) > 1:
            self.dsout[bin_name] = bins

    def time_probability_hour_of_day(
        self, derived_var, bin_value, prefix="hprob_", **kwargs
    ):
        """Calculate the time probability for each hour with time offset accounted.

        Args:
            derived_var (str): Derived_vars to calculate before applying stats.
            bin (value): List of values for binning the data to calculate probability.
            prefix (str): String to prepend to each variable name in output dataset.

        Note:
            Only implemented atm for one single derived variable and one matching bin.

        """
        self._update_dset([derived_var])
        data_var = derived_var
        logger.debug(
            "Calculating hourly time-probability for var: {}".format(data_var)
        )

        darrays = []
        hours = range(24)
        for hour in hours:
            logger.info("Probability for hour {:0.0f}".format(hour))
            dset_hour = self.dset.where(self.hour_of_day == hour)

            # Probability for each variable
            dvar = dset_hour[data_var]
            count = self._time_count(data_var=dvar, dim="time")
            in_bin = dvar == bin_value
            # Persist it here because code is blowing memory up.
            prob = in_bin.sum(dim="time") / count
            darrays.append(prob)

        self.dsout["{}{}".format(prefix, data_var)] = xr.concat(darrays, "hour_of_day")
        self.dsout["hour_of_day"] = hours

    def apply_func(self, func, dim="time", data_vars=None, derived_vars=[], prefix=None, **kwargs):
        """apply xarray function.

        Args:
            func (str): Name of valid xarray function to apply.
            dim (str): Dimension to apply function over.
            data_vars (list): Data vars to apply stats over, all by default.
            derived_vars (list): Derived_vars to calculate before applying stats.
            prefix (str): String to prepend to each variable name in output dataset,
                defined as `"{}_".format(func)` if `prefix==None`.

        """
        self._update_dset(derived_vars)
        if prefix is None:
            prefix = "{}_".format(func)

        # Variables to apply
        if data_vars is None:
            data_vars = self.dset.data_vars.keys()
        elif isinstance(data_vars, str):
            data_vars = [data_vars]
        data_vars += derived_vars
        logger.debug("Calculating time-{} for vars: {}".format(func, data_vars))

        dsout = getattr(self.dset[data_vars], func)(dim=dim, **kwargs)
        self.dsout = self.dsout.merge(
            dsout.rename({v: "{}{}".format(prefix, v) for v in dsout.data_vars.keys()})
        )
        return dsout

    def to_netcdf(
        self, outfile, format="NETCDF4", zlib=True, _FillValue=-32767
    ):
        """Save output dataset as netcdf.

        Args:
            outfile (str):
            format (str):
            zlib (bool):
            _FillValue (number):
            dtype (int, object):

        """
        logger.debug("Saving stats dataset into file: {}".format(outfile))
        encoding = {}
        for data_var in self.dsout.data_vars:
            encoding.update({data_var: {"zlib": zlib, "_FillValue": _FillValue}})
        # Loading into memory before saving to disk. We may want to reassess this.
        self._load()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)

