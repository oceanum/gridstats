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

import intake
import intake_xarray

from stats.util import uv_to_spddir
import stats.derived_variable as dv


logging.basicConfig(level="INFO")

# TODO: Generalise input partitions in crossing_seas


class DerivedVar(object):
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
        self.logger = logger

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
            self.logger.info("Deriving swell wave length from swell period")
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
            cover_threshold=self.clear_sky_threshold
        )

    @property
    def covered_sky(self):
        """Covered sky based on cloud cover fraction."""
        return dv.covered_sky(
            cloud_cover=self.dset[self.var_cloud_cover],
            cover_threshold=self.covered_sky_threshold
        )

class Stats(DerivedVar):
    def __init__(
        self,
        dataset,
        intake_catalog="gs://oceanum-catalog/oceanum.yml",
        mask=None,
        slice_dict={},
        chunks=None,
        persist=False,
        logger=logging,
        **kwargs
    ):
        """Gridded stats using dask arrays.

        Args:
            dataset (str, xr.Dataset): Either an intake dataset id if string or
                an xarray dataset.
            intake_catalog (str): Path for intake catalog in case dataset is a intake
                id, can be either a local or a url path.
            mask (str): either a variable name or an expression to evaluate on one or
                more variables to define a mask array for masking output dataset. e.g.,
                `self.dset.hs==0`.
            slice_dict (dict): Dictionary specifying slicing arg.
            chunks (dict): Chunking dict to rechunk dataset after opening.
            persist (bool): if True, persist output dataset before saving as netcdf.
            logger (object): Logging object.

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
        self.intake_catalog = intake_catalog
        self.mask = mask
        self.chunks = chunks
        self.persist = persist
        self.logger = logger

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
        self.logger.debug("Estimating hour offset and broadcasting to 3D")
        hour_offset = da.floor((self.dset.longitude + 7.7) / 15).chunk({'longitude': self.dset.chunks['longitude'][0]})
        hour = self.dset.time.dt.hour.chunk({'time': self.dset.chunks['time'][0]})
        self._hour_of_day = (hour + hour_offset) % 24
        self._hour_of_day.name = "hour_of_day"
        return self._hour_of_day

    def _slice_dset(self, slice_dict):
        """Masking dataset using slice_dict."""
        for slice_method, slice_kwargs in slice_dict.items():
            self.dset = getattr(self.dset, slice_method)(**slice_kwargs)
        self.logger.info("Processing dataset: {}".format(self.dset))

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
        """Set dset attribute either from intake dataset of from xarray dataset itself.

        If self.dataset is a string, it should be a valid intake dataset and the
            intake_catalog argument must be provided at initialisation so that the
            catalog can be built. If self.dataset is an xarray dataset then it is just
            assigned to self.dset attribute.

        Note: there is a point of failure here is the dataset string is a substring of
            more than one intake dataset in catalog. This should be fixed in the future.

        """
        self.logger.info("Open dataset")
        if isinstance(self.dataset, str):
            assert (
                self.intake_catalog is not None
            ), "dataset is a string but intake_catalog has not been provided"
            self.logger.debug(
                "Opening dataset {} from intake catalog {}".format(
                    self.dataset, self.intake_catalog
                )
            )
            # Open catalog and ensure dataset is a substring of a catalog entry
            self.cat = intake.open_catalog(self.intake_catalog)
            intake_dataset = ""
            for intake_dataset in self.cat.walk().keys():
                if self.dataset in intake_dataset:
                    break
            assert (
                self.dataset in intake_dataset
            ), "Cannot locate dataset {} in intake catalog [{}]".format(
                self.dataset, ", ".join(self.cat.walk().keys())
            )
            self.dset = getattr(self.cat, intake_dataset).to_dask()
        elif isinstance(self.dataset, xr.Dataset):
            self.dset = self.dataset
        else:
            raise ValueError(
                "dataset must be either a string specifying an intake dataset id or an xarray dataset itself"
            )
        if self.chunks:
            self.logger.info("Re-chunking dataset as {}".format(self.chunks))
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

    def _grouper(self, iterable, n, fillvalue=None):
        """Iterates over variable chunk sizes.

        Args:
            - iterable: sequence to iterate over.
            - n (int): size of chunks to iterate over.
            - fillvalue: value to use to complete last chunk if necessary.

        """
        args = [iter(iterable)] * n
        return izip_longest(*args, fillvalue=fillvalue)

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
            self.logger.info(
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
        self.logger.debug("Calculating {} count over {}".format(data_var, dim))
        if isinstance(data_var, str):
            dvar = self.dset[data_var]
        else:
            dvar = data_var
        count = (0 * dvar + 1).sum(dim=dim, skipna=True)
        return count.where(count > 0)

    def time_probability(
        self,
        data_vars=[],
        derived_vars=[],
        bins=[],
        bin_type="exact",
        bin_name="bin",
        prefix="prob_",
        **kwargs
    ):
        """Calculate the time probability.

        Args:
            data_vars (list): Data vars to apply stats over.
            derived_vars (list): Derived_vars to calculate before applying stats.
            bins (list): List of values for binning the data to calculate probability.
            bin_type (str): Either `"exact"` or `"bounds"`. If `bin_type=="exact"`,
                the probability is calculated for the exact maches in the `bins` list
                (suitable for integer types such as douglas scales). If
                `bin_type=="bounds"`, the data are binned within each two neighbours in
                the `bins` list and the probability is calculated over these bins.
            bin_name (str): Name of bin coordinate in output probability dataset. Note
                that the bin coordinate is only created if there is more than one bin.
            prefix (str): String to prepend to each variable name in output dataset.

        Note:
            At least one `data_var` or `derived_var` should be provided.
            The output dataset has an extra coordinate with name defaulting to `bin`,
                representing the bin values over which probabilities are calculated.
                If `bin_type=="exact"`, the coordinate values are defined by the values
                in the `bins` list. If `bin_type=="bounds"`, the coordinate values are
                the central values of each bin.

        """
        assert list(data_vars) + list(
            derived_vars
        ), "At least one data_var or derived_var should be provided."
        assert bin_type in [
            "exact",
            "bounds",
        ], "bin_type must be either 'exact' or 'bounds'"

        if bin_type == "bounds":
            raise NotImplementedError("bounds option not yet implemented.")

        bins = list(bins)
        self._update_dset(derived_vars)
        data_vars = list(data_vars) + list(derived_vars)
        self.logger.debug("Calculating time-probability for vars: {}".format(data_vars))

        # Probability for each variable
        for data_var in data_vars:
            dvar = self.dset[data_var]
            count = self._time_count(data_var=data_var, dim="time")
            darrays = []
            for bin_value in bins:
                in_bin = dvar == bin_value
                darrays.append(in_bin.sum(dim="time") / count)
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
        self,
        derived_var,
        bin_value,
        prefix="hprob_",
        **kwargs
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
        self.logger.debug("Calculating hourly time-probability for var: {}".format(data_var))

        darrays = []
        hours = range(24)
        for hour in hours:
            self.logger.info("Probability for hour {:0.0f}".format(hour))
            dset_hour = self.dset.where(self.hour_of_day == hour)

            # Probability for each variable
            dvar = dset_hour[data_var]
            count = self._time_count(data_var=dvar, dim="time")
            in_bin = dvar == bin_value
            # Persist it here because code is blowing memory up.
            prob = (in_bin.sum(dim="time") / count)
            darrays.append(prob)


        self.dsout["{}{}".format(prefix, data_var)] = xr.concat(darrays, "hour_of_day")
        self.dsout["hour_of_day"] = hours

    def time_probability_10_14(
        self,
        derived_var,
        bin_value,
        prefix="hprob_10h14h_",
        **kwargs
    ):
        """Calculate the time probability for hours 10-14h.

        Because the time_probability_hour_of_day method is blowing memory up...

        Args:
            derived_var (str): Derived_vars to calculate before applying stats.
            bin (value): List of values for binning the data to calculate probability.
            prefix (str): String to prepend to each variable name in output dataset.

        Note:
            Only implemented atm for one single derived variable and one matching bin.

        """
        self._update_dset([derived_var])
        data_var = derived_var
        self.logger.debug("Calculating hourly time-probability for var: {}".format(data_var))

        self.logger.info("Probability for hours 10-14h")
        hour_of_day = self.hour_of_day
        dset_hour = self.dset.where((hour_of_day >= 10)&(hour_of_day <= 14))
        dvar = dset_hour[data_var]
        count = self._time_count(data_var=dvar, dim="time")
        self.dsout["{}{}".format(prefix, data_var)] = (dvar == bin_value).sum(dim="time") / count

    def time_quantile(
        self,
        data_vars=None,
        derived_vars=[],
        q=[0.5, 0.9, 0.95, 0.99],
        prefix="q_",
        **kwargs
    ):
        """Calculate quantile levels.

        Args:
            data_vars (list): Data vars to apply stats over, all by default.
            derived_vars (list): Derived_vars to calculate before applying stats.
            q (list): Quantile levels to apply.
            prefix (str): String to prepend to each variable name in output dataset.

        Note: it is not possible (as of xarray 0.12.3) to calculate quantile on dask
              arrays and so it is calculated over latitude bands at a time. time_stat
              will be able to be used once quantile is implemented in dask.

        """
        self._update_dset(derived_vars)

        # Variables to apply
        if data_vars is None:
            data_vars = self.dset.data_vars.keys()
        elif isinstance(data_vars, str):
            data_vars = [data_vars]
        data_vars += derived_vars
        self.logger.debug(
            "Calculating time-quantile for vars: {} - {}".format(data_vars, q)
        )

        # Calculating over each latitude
        dsets = []
        for ilat in range(self.dset.latitude.size):
            self.logger.info(
                "Calculating quantile for latitude {} of {}".format(
                    ilat, self.dset.latitude.size
                )
            )
            dslat = self.dset[data_vars].isel(latitude=ilat, drop=True).compute()
            dsets.append(dslat.quantile(q=q, dim="time"))

        # Merging over latitude dimension
        dsout = xr.concat(dsets, dim="latitude").transpose(
            "quantile", "latitude", "longitude"
        )
        dsout["latitude"] = self.dset.latitude
        dsout = dsout.rename(
            {v: "{}{}".format(prefix, v) for v in dsout.data_vars.keys()}
        )
        self.dsout = self.dsout.merge(dsout)
        return dsout

    def time_stat(self, func, data_vars=None, derived_vars=[], prefix=None, **kwargs):
        """apply stat defined by func.

        Args:
            func (str): Name of valid stat to apply.
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
        self.logger.debug("Calculating time-{} for vars: {}".format(func, data_vars))

        dsout = getattr(self.dset[data_vars], func)(dim="time")
        self.dsout = self.dsout.merge(
            dsout.rename({v: "{}{}".format(prefix, v) for v in dsout.data_vars.keys()})
        )
        return dsout

    def to_netcdf(
        self, outfile, format="NETCDF4_CLASSIC", zlib=True, _FillValue=-32768
    ):
        """Save output dataset as netcdf.

        Args:
            outfile (str):
            format (str):
            zlib (bool):
            _FillValue (number):
            dtype (int, object):

        """
        self.logger.debug("Saving stats dataset into file: {}".format(outfile))
        encoding = {}
        for data_var in self.dsout.data_vars:
            encoding.update({data_var: {"zlib": zlib, "_FillValue": _FillValue}})
        # Loading into memory before saving to disk. We may want to reassess this.
        self._load()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)


def main(yaml_file, load_after_call=False, verbose=False, logger=logging):
    """Calculate stats from configuration file.

    Args:
        load_after_call (bool): If True, computation is triggered after running each
            stats method, as opposed to only at the end before saving to disk.
        verbose (bool): Add some extra logging if True.
        logger (object): Logging instance.

    Required keys in yaml config:
        init (dict): kwarg options for initialising Stat class.
        calls (dict): key is name of method to call, values are kwargs.

    """
    conf = yaml.load(open(yaml_file))
    stats = Stats(**conf["init"])
    for call in conf.get("calls", []):
        if verbose:
            logging.info("Stat.{}({})".format(call["method"], call.get("kwargs", {})))
        getattr(stats, call["method"])(**call.get("kwargs", {}))
        if load_after_call:
            logging.info("Trigerring computations")
            stats._load()
    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate gridded stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Usage: python stats.py stats.yml",
    )
    parser.add_argument("config", help="yaml config file stats")
    parser.add_argument(
        "-l",
        "--load_after_call",
        action="store_true",
        help="Choose it to compute after each call method, slower but memory-friendly",
    )
    args = vars(parser.parse_args())

    self = main(args["config"], load_after_call=args["load_after_call"], verbose=True)
