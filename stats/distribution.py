"""Calculated distribution stats using xarray and dask.

Unfinished work moved from stats module.

"""
import os
import shutil
import yaml
import logging
import argparse
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tempfile import mkdtemp
from nco import Nco
from tqdm import tqdm

import intake
import intake_xarray

from util import uv_to_spddir
import derived_variable as dv


logging.basicConfig(level="INFO")
NCO = Nco()

# TODO: Generalise input partitions in crossing_seas


class DerivedVar(object):
    def __init__(
        self,
        dset,
        mask=None,
        hs_threshold=0,
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
        **kwargs
    ):
        """Wrapper around derived variable functions.

        The purpose of this class is defining the arguments required by the different
        derived variables so that they can be called as an attribute.

        """
        self.dset = dset
        self.hs_threshold = hs_threshold
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

    @property
    def douglas_sea(self):
        """Douglas sea scale data_var."""
        return dv.douglas_sea(hs_sea=self.dset[self.var_hs_sea])

    @property
    def douglas_swell(self):
        """Douglas swell scale data_var."""
        return dv.douglas_swell(
            hs_sw1=self.dset[self.var_hs_sw1], lp_sw1=self.dset[self.var_hs_sw1]
        )

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


class Stats(DerivedVar):
    def __init__(
        self,
        dataset,
        intake_catalog=None,
        mask=None,
        slice_dict={},
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
                `self.dset.hs==0`
            slice_dict (dict): dictionary specifying slicing arg
            logger (object): Logging object.

        """
        self.dataset = dataset
        self.intake_catalog = intake_catalog
        self.mask = mask
        self.logger = logger

        # Open dataset
        self._open_dataset()
        self.dsout = xr.Dataset()

        # Instantiating DerivedVar
        super().__init__(self.dset, **kwargs)

        # Define mask
        self._set_mask(mask)

        # Slicing
        self._slice_dset(slice_dict or {})

    def _slice_dset(self, slice_dict):
        """Masking dataset using slice_dict."""
        for slice_method, slice_kwargs in slice_dict.items():
            self.dset = getattr(self.dset, slice_method)(**slice_kwargs)

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

    def _load(self):
        """Load output dataset into memory and show the progress."""
        self.logger.info("Loading stats dataset into memory")
        with ProgressBar():
            self.dsout.load()

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
        count = (0 * self.dset[data_var] + 1).sum(dim=dim, skipna=True)
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
                # import ipdb; ipdb.set_trace()
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
            self.logger.debug(
                "Calculating quantile for latitude {} of {}".format(
                    ilat, self.dset.latitude.size
                )
            )
            dslat = self.dset[data_vars].isel(latitude=ilat, drop=True).load()
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

    def time_distribution(
        self,
        outfile,
        hs_bins,
        tp_bins,
        dir_bins,
        wsp_bins=None,
        hs_name="hs",
        tp_name="tp",
        dir_name="dp",
        u_name=None,
        v_name=None,
        nlat=10,
        memory_safe=True,
        maskval=np.int16(0),
        attrs={},
        **kwargs
    ):
        """Distribution statistics.

        Args:
            - outfile (str): name for output distribution stats file.
            - hs_bins (dict): `start`, `stop` and `step` to define Hs bins.
            - tp_bins (dict): `start`, `stop` and `step` to define Tp bins.
            - dir_bins (dict): `start`, `stop` and `step` to define wave/wind
              direction bins.
            - wsp_bins (dict): `start`, `stop` and `step` to define wind speed
              bins.
            - hs_name (str): name of Hs variable in input datasets.
            - tp_name (str): name of Tp variable in input datasets.
            - dir_name (str): name of Dp variable in input datasets.
            - u_name (str): name of u-component of wind, if available.
            - v_name (str): name of v-component of wind, if available.
            - nlat (int): number of latitude rows to load into memory.
            - memory_safe (bool): choose it to save each site into disk
              and concatenate after to avoid blowing up memory (olny option
              currently available, in-memory not yet implemented).
            - maskval (digit): used for _FillValue.
            - attrs (dict): used for global attributes in output file.

        Note:
            - waves and winds share the same direction bins dimension.

        """
        tmpfiles = []
        tmpdir = mkdtemp(dir=os.path.dirname(outfile))
        self.logger.debug(
            "Temporary directory for intermediate files: {}".format(tmpdir)
        )

        # Load on memory for speed
        with ProgressBar():
            self.dset.load().fillna(-1e10)
            nsites = self.dset.latitude.size * self.dset.longitude.size

        # Wave coordinates and bins
        wave_coords = {
            "site": np.arange(nsites),
            "month": np.arange(1, 13),
            "hs_bins": np.arange(**hs_bins) + hs_bins["step"] / 2.0,
            "tp_bins": np.arange(**tp_bins) + tp_bins["step"] / 2.0,
            "dir_bins": np.arange(**dir_bins) + dir_bins["step"] / 2.0,
        }

        wave_bins = (
            np.hstack((np.arange(**hs_bins), hs_bins["stop"])),
            np.hstack((np.arange(**tp_bins), tp_bins["stop"])),
            np.hstack((np.arange(**dir_bins), dir_bins["stop"])),
        )

        # Wind coordinates and bins
        if u_name is not None and v_name is not None:
            assert wsp_bins is not None, "wsp_bins must be provided."
            is_wind = True
            self.logger.info("Wind stats will be included")
            wind_coords = {
                "month",
                np.arange(1, 13),
                "wsp_bins",
                np.arange(**wsp_bins) + wsp_bins["step"] / 2.0,
                "dir_bins",
                np.arange(**dir_bins) + dir_bins["step"] / 2.0,
            }

            wind_bins = (
                np.hstack((np.arange(**wsp_bins), wsp_bins["stop"])),
                np.hstack((np.arange(**dir_bins), dir_bins["stop"])),
            )
        else:
            is_wind = False
            self.logger.info("Wind stats will NOT be included")

        # Initialise output
        data_shape = [c.size for c in wave_coords.values()]
        self.dsout["wave_count"] = xr.DataArray(
            data=np.empty(data_shape, dtype=np.int16),
            coords=wave_coords,
            dims=wave_coords.keys(),
            attrs={
                "standard_name": "sea_surface_wave_count",
                "units": "",
                "missing_value": maskval,
            },
        )
        grid_lon, grid_lat = np.meshgrid(
            self.dset.longitude.values, self.dset.latitude.values
        )
        self.dsout["lon"] = xr.DataArray(
            grid_lon.ravel(), coords={"site": self.dsout.site}, dims=("site")
        )
        self.dsout["lat"] = xr.DataArray(
            grid_lat.ravel(), coords={"site": self.dsout.site}, dims=("site")
        )

        isite = 0
        with tqdm(total=nsites) as pbar:
            # ------------------------------------
            # Looping over latitudes and loading
            # ------------------------------------
            for lat in self.dset.latitude:
                # self.logger.info('{}\nProcessing lat={}\n{}'.format(100*'=', float(lat), 100*'='))
                dset_lon = self.dset.sel(latitude=lat, drop=True)
                # if not (dset_lon[hs_name]>=0).any():
                #     self.logger.debug('No valid point in lat={}, skipping.'.format(float(lat)))
                #     pbar.update(self.dset.latitude.size)
                #     isite += self.dset.latitude.size
                #     continue

                # -------------------------------------------------------
                # Looping over longitudes and reset lon/lat coordinates
                # -------------------------------------------------------
                for lon in dset_lon.longitude:
                    ds_site = dset_lon.sel(longitude=lon, drop=True).reset_coords()
                    if not (ds_site[hs_name] >= 0).any():
                        pbar.update(1)
                        isite += 1
                        continue
                    if is_wind:
                        ds_site["wsp"], ds_site["wdir"] = uv_to_spddir(
                            ds_site[u_name], ds_site[v_name], coming_from=True
                        )

                    # ---------------------
                    # Looping over months
                    # ---------------------
                    monthly_wave_dict = {}
                    monthly_wind_dict = {}
                    for month, ds in ds_site.groupby("time.month"):

                        # Calculate waves distribution
                        mask = (
                            ds[dir_name] > dir_bins["stop"]
                        )  # to allow directions will be bin-centred
                        ds[dir_name][mask] = ds[dir_name][mask] - 360
                        hdd = np.histogramdd(
                            sample=(
                                ds[hs_name].values,
                                ds[tp_name].values,
                                ds[dir_name].values,
                            ),
                            normed=False,
                            bins=wave_bins,
                        )
                        wave_count = hdd[0].astype("int32")
                        monthly_wave_dict.update({month: wave_count})

                        # Calculate wind distribution
                        if is_wind:
                            mask = (
                                ds["wsp"] > dir_bins["stop"]
                            )  # to allow directions will be bin-centred
                            ds["wsp"][mask] = ds["wsp"][mask] - 360
                            hdd = np.histogramdd(
                                sample=(ds["wsp"].values, ds["wdir"].values),
                                normed=False,
                                bins=wind_bins,
                            )
                            wind_count = hdd[0].astype("int32")
                            monthly_wind_dict.update({month: wind_count})

                    # Ensure there are arrays for every month
                    monthly_wave_arrays = [
                        monthly_wave_dict.get(m, 0 * wave_count) for m in range(1, 13)
                    ]
                    if is_wind:
                        monthly_wind_arrays = [
                            monthly_wind_dict.get(m, 0 * wind_count)
                            for m in range(1, 13)
                        ]

                    # Fill in current site
                    self.dsout["wave_count"].loc[
                        dict(site=isite)
                    ] += monthly_wave_arrays
                    # # Distribution wave stats for current site
                    # self.dsout['wave_count'] = xr.DataArray(
                    #     data=monthly_wave_arrays,
                    #     coords=wave_coords,
                    #     dims=wave_coords.keys(),
                    #     name='wave_count',
                    #     attrs={
                    #         'standard_name': 'sea_surface_wave_count',
                    #         'units': '',
                    #         'missing_value': maskval,
                    #     },
                    # )

                    # Distribution wind stats for current site
                    if is_wind:
                        self.dsout["wind_count"] = xr.DataArray(
                            data=monthly_wind_arrays,
                            coords=wind_coords,
                            dims=wind_coords.keys(),
                            name="wind_count",
                            attrs={
                                "standard_name": "wind_count_at_10m_above_ground_level",
                                "units": "",
                                "missing_value": maskval,
                            },
                        )

                    # # Aditional wave stats (required by stats api)
                    # self.dsout['msum'] = (ds_site.hs>=0).groupby('time.month').sum()
                    # self.dsout['msum'] = self.dsout['msum'].fillna(0).astype('int32')
                    # ds_site['eflx'] = 0.42 * ds_site[hs_name] * ds_site[tp_name]
                    # stats = [hs_name, tp_name, dir_name, 'eflx']
                    # if is_wind:
                    #     stats += ['wsp']
                    # ds_monthly = ds_site[stats].groupby('time.month')
                    # self.dsout = xr.merge((
                    #     self.dsout,
                    #     ds_monthly.mean().rename({s: s+'-mean' for s in stats}),
                    #     ds_monthly.max().rename({s: s+'-max' for s in stats}),
                    #     ds_monthly.min().rename({s: s+'-min' for s in stats}),
                    # ))

                    # self.dsout.attrs.update({'totsum': np.int32(self.dsout.msum.sum())})

                    # # Save to disk if memory_safe (only option available atm)
                    # if memory_safe:
                    #     site_string = str(isite).zfill(len(str(nsites)))
                    #     tmpfile = os.path.join(tmpdir, 'diststats_s{}.nc'.format(site_string))
                    #     self.logger.debug('Saving temporary distribution: {}'.format(site_string))
                    #     self.dsout.to_netcdf(
                    #         path=tmpfile,
                    #         encoding={v: {'zlib': True, '_FillValue': maskval} for v in self.dsout.data_vars},
                    #     )
                    #     tmpfiles.append(tmpfile)

                    isite += 1
                    pbar.update(1)

        # Set global attributes
        self.dsout.attrs = attrs

        # Saving output
        self.encoding = {
            v: {"zlib": True, "_FillValue": maskval, "dtype": np.int16}
            for v in self.dsout.data_vars
        }
        self.to_netcdf(outfile)

        # # Eliminate site from msum to support API (this makes it not valid for time-varying masks)
        # self.logger.info(' {}'.format(outfile))
        # dset = xr.open_dataset(outfile)

        # # Concatenate output
        # self.logger.info('Merging individual sites into {}'.format(outfile))
        # tmpfile = self.nco.ncecat(
        #     input=tmpfiles,
        #     output=outfile,
        #     options=['-h'],
        #     ulm_nm='site'
        # )

        # shutil.rmtree(tmpdir)

    def to_netcdf(
        self,
        outfile,
        format="NETCDF4_CLASSIC",
        zlib=True,
        _FillValue=-32768,
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
            encoding.update(
                {data_var: {"zlib": zlib, "_FillValue": _FillValue}}
            )
        with ProgressBar():
            self.dsout.to_netcdf(outfile, format=format, encoding=encoding)


def main(yaml_file, verbose=False, logger=logging):
    """Calculate stats from configuration file.

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
    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate gridded stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Usage: python stats.py stats.yml",
    )
    parser.add_argument("config", help="yaml config file stats")
    args = vars(parser.parse_args())

    self = main(args["config"], verbose=True)
