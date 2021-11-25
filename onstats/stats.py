"""Calculated gridded stats using xarray and dask."""
import os
from pathlib import Path
from importlib import import_module
from inspect import getmembers, isfunction
import shutil
import logging
import numpy as np
import xarray as xr
from intake import open_catalog
import dask
from dask.distributed import Client
import multiprocessing as mp

from oncore.dataio import put, isdir, exists, rm, get

logger = logging.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")


class Plugin(type):
    """Add stats functions as bound methods at class creation."""

    def __new__(cls, name, bases, dct):
        modules = [p.stem for p in (Path(__file__).parent / "functions").glob("*.py")]
        modules = [import_module(f"onstats.functions.{name}") for name in modules]
        for module in modules:
            for func_name, func in getmembers(module, isfunction):
                if not func_name.startswith("_"):
                    dct[func_name] = func
        return type.__new__(cls, name, bases, dct)


class Stats(metaclass=Plugin):
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
        n_workers="half",
        cluster_kwargs={},
        calls=[],
    ):
        """Gridded stats using dask arrays.

        Args:
            - outfile (str): Name or URI of output file, must end with '.nc' or '.zarr'.
            - dset (xr.Dataset): Source dataset to calculate stats from.
            - urlpath (str): Path or URI of dataset file to calculate stats from.
            - engine (str): Engine to use with xr.open_dataset when using urlpath.
            - catalog (str): Path or URI of intake catalog to calculate stats from.
            - dataset_id (str): Intake dataset id to calculate stats from.
            - mapping (dict): Dictionary for renaming dataset variables.
            - slice_dict (dict): Dictionary specifying slicing parameters.
            - updir (str): Path or URI to upload output stats file to.
            - allow_split_large_chunks (bool): Allow dask auto-resize of small chunks.
            - cluster_kwargs (dict): Keyword arguments for the local dask cluster.
            - calls (list): List of dicts defining each stat to run with keys
              specifying keyword arguments for the `apply_func` method.

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
        self.updir = updir
        self.cluster_kwargs = cluster_kwargs
        self.calls = calls

        self.dsout = xr.Dataset()

    def __call__(self):
        """Loop over list of dictionary to execute stats methods.

        Local dask clusters are set up for running each stats method.

        """
        self._clean_dask_worker_space()

        # Execute each stats method
        for call in self.calls:
            with Client(**self.cluster_kwargs) as client:
                logger.info(client)
                logger.info(f"Stat.apply_func(**{call})")
                self.apply_func(**call)
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
            logger.debug("Removing existing dask worker space")
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

        logger.debug(f"Dataset chunks: {dset.chunks}")
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
        data_vars="all",
        chunks={},
        group=None,
        suffix=None,
        compute=True,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            func (str): Name of valid function defined in functions package.
            data_vars (list): Data vars to apply stats over, "all" for all variables.
            chunks (dict): Mapping {dim_name: dim_size} for chunking dataset.
            group (str): Time grouping type, any valid time_{group} such month, season.
            suffix (str): String to append to each variable name in output dataset.
            compute (bool): Compute dask variables in output dataset before returning.
            kwargs: Aditional keyword arguments for function `func`.

        """
        if suffix is None:
            suffix = f"_{func}"

        # Open dataset
        dset = self._open_dataset(chunks=chunks)

        # Selecting variables
        if data_vars == "all":
            data_vars = [k for k in dset.data_vars]
        logger.debug(f"Calculating {func} for vars: {data_vars}")
        dset = dset[data_vars]

        # Replace suffix if grouping by
        if group:
            suffix += f"_{group}"

        # Calculate stats
        dsout = getattr(self, func)(dset, group=group, **kwargs)
        if compute:
            logger.info("Compute dask stats")
            dsout = dsout.load()
        else:
            logger.info("Dask stats will be computed when saving dataset to disk")

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
            included_chunks = {c: v for c, v in chunks.items() if c in dsout.dims}
            dsout.chunk(included_chunks).to_zarr(store, consolidated=True, mode=mode)
            if self.updir:
                self._upload(store)
