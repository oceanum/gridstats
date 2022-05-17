"""Calculated gridded stats using xarray and dask."""
import os
from pathlib import Path
from importlib import import_module
from inspect import getmembers, isfunction
import shutil
import logging.config
import numpy as np
import xarray as xr
from intake import open_catalog
import dask
from dask.distributed import Client
import multiprocessing as mp
from contextlib import contextmanager

from oncore.dataio import put, isdir, exists, rm, get
from oncore import LOGGING_CONFIG

from onstats.utils import stepwise, cd


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


_FILLVALUE = np.int32(2 ** 32 / 2)


@contextmanager
def DummyClient(*args, **kwargs):
    """Dummy context manager when not using local dask cluster."""
    yield "<No Dask Cluster>"


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
        chunks={},
        updir=None,
        localdir="/tmp/run/stats",
        allow_split_large_chunks=False,
        cluster_kwargs={},
        disable_cluster_logging=True,
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
            - chunks(dict): Chunking sizes in output dataset.
            - updir (str): Path or URI to upload output stats file to.
            - localdir (str): Path to calculate stats and create dask workspace.
            - allow_split_large_chunks (bool): Allow dask auto-resize of small chunks.
            - cluster_kwargs (dict): Keyword arguments for the local dask cluster.
            - disable_cluster_logging (bool): Disable cluster logging below CRITICAL.
            - calls (list): List of dicts defining each stat to run with keys
              specifying keyword arguments for the `apply_func` method.

        Note:
            - You must provide one of 'dset', 'urlpath' or ['catalog', 'dataset_id'].

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
        self.chunks = chunks
        self.updir = updir
        self.localdir = localdir
        self.cluster_kwargs = cluster_kwargs
        self.calls = calls

        self._client = True

        self.dsout = xr.Dataset()

        if disable_cluster_logging:
            logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)
            logging.getLogger("distributed.client").setLevel(logging.CRITICAL)

    @property
    def client(self):
        """Define cluster client to use or not."""
        if self._client is True:
            return Client
        elif self._client is False:
            return DummyClient
        else:
            raise ValueError(f"_client must be bool, got {self.use_dask_cluster}")

    def __call__(self):
        """Loop over list of dictionary to execute stats methods."""
        Path(self.localdir).mkdir(parents=True, exist_ok=True)
        with cd(self.localdir):
            self._clean_dask_worker_space()

            # Execute each stats method
            for call in self.calls:
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
            - chunks (dict): Mapping dim: size for chunking each dimension.

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
            dset = cat[self.dataset_id](chunks=chunks).to_dask()
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
            sizes = {k: v.size for k, v in dset.coords.items()}
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
            - filename (str): Name of file to upload.

        """
        outfile = os.path.join(self.updir, os.path.basename(filename))
        logger.info(f"Uploading {filename} --> {outfile}")
        if exists(outfile):
            logger.debug(f"Removing existing file {outfile} before uploading")
            rm(outfile, recursive=isdir(outfile))
        put(filename, outfile, recursive=isdir(filename))

    def _finalise(self):
        """Sort dimensions, define attributes, transpose."""
        self._sortby()
        self._chunk()
        self._transpose()
        self._setattrs()
        self._setdtype()

    def _sortby(self):
        """Sort output dataset by all coordinates."""
        logger.debug("Sorting output by dims")
        for name, coords in self.dsout.coords.items():
            if coords[0] > coords[-1]:
                logger.info(f"Sorting by coordinate {name}")
                self.dsout = self.dsout.sortby(name)

    def _chunk(self):
        """.Chunk output dataset."""
        logger.debug("Chunking output")
        chunks = {k: v for k, v in self.chunks.items() if k in self.dsout.dims}
        self.dsout = self.dsout.chunk(chunks)
        for dvar in self.dsout.values():
            dvar.encoding.pop("chunks", None)

    def _transpose(self):
        logger.debug("Transposing output")
        """Transpose to ensure some predefined ordering."""
        if "quantile" in self.dsout.coords:
            self.dsout = self.dsout.transpose("quantile", ...)

    def _setattrs(self):
        """Define some attributes in output dataset."""
        logger.debug("Defining attributes in output")
        if "quantile" in self.dsout.coords:
            self.dsout["quantile"].attrs = {
                "standard_name": "quantile",
                "long_name": "quantile",
                "units": "",
            }

    def _setdtype(self):
        """Ensure correct data types."""
        # Season is written with object data type which causes problems when rewriting
        if "season" in self.dsout.coords:
            self.dsout["season"] = self.dsout.season.astype("U")
        # Ensure float32
        for varname in self.dsout.data_vars:
            if self.dsout[varname].dtype == "float64":
                self.dsout[varname] = self.dsout[varname].astype("float32")

    def _directional_stat(self, dset, func, dirs, nsector, group, **kwargs):
        """Calculate func over directional sectors.

        Args:
            - dset (Dataset): Dataset to sectorise.
            - func (str): Name of valid function defined in functions package.
            - dirs (DataArray): Directional data array to use for binning dset.
            - nsector (int): Number of directional sectors.
            - group (str): Time grouping type, any valid time_{group} such month, season.
            - kwargs: Aditional keyword arguments for function `func`.

        Notes:
            - dirs must share same dimensions as variables in dset.

        """
        # Binning data per directional sector
        dsector = 360 / nsector
        sectors = np.linspace(0, 360 - dsector, nsector)
        starts = (sectors - dsector / 2) % 360
        stops = (sectors + dsector / 2) % 360
        dsout = []
        for start, stop in zip(starts, stops):
            logger.info(
                f"Calculate directional {func} for sector [{start:5.1f}, {stop:5.1f})"
            )
            if stop > start:
                mask = (dirs >= start) & (dirs < stop)
            else:
                mask = (dirs >= start) | (dirs < stop)
            dsout.append(getattr(self, func)(dset.where(mask), group=group, **kwargs))

        # Concat directional bins into new dimension
        dsout = xr.concat(dsout, dim="direction").assign_coords({"direction": sectors})
        dsout["direction"].attrs = {
            "standard_name": dirs.attrs.get("standard_name", "direction") + "_sector",
            "long_name": dirs.attrs.get("long_name", "direction") + " sector",
            "units": dirs.attrs.get("units", "degree"),
            "variable_name": dirs.name,
        }

        return dsout

    @stepwise
    def apply_func(
        self,
        func,
        data_vars="all",
        chunks={},
        group=None,
        suffix=None,
        compute=True,
        dset=None,
        nsector=None,
        dir_var="dpm",
        use_dask_cluster=True,
        **kwargs,
    ):
        """apply xarray function.

        Args:
            - func (str): Name of valid function defined in functions package.
            - data_vars (list): Data vars to apply stats over, "all" for all variables.
            - chunks (dict): Mapping {dim_name: dim_size} for chunking dataset.
            - group (str): Time grouping type, any valid time_{group} such month, season.
            - suffix (str): String to append to each variable name in output dataset.
            - compute (bool): Compute dask variables in output dataset before returning.
            - dset (xr.Dataset): Dataset to reduce if other than what is defined from
              init, only necessary because of the decorators which modify the dataset.
            - nsector (int): Number of directional sectors if sectorising dataset.
            - dir_var (str): Name of directional variable to use if sectorising dataset.
            - use_dask_cluster (bool): Use local dask cluster to calculate stats.
            - kwargs: Aditional keyword arguments for function `func`.

        Note:
            - Some stats, e.g., `quantile` or `rpv` require specifying single chunks
              for the core dimension, usually `time`.

        """
        # Local cluster definition
        self._client = use_dask_cluster

        # Define variable suffix
        if suffix is None:
            suffix = f"_{func}"
        if group:
            suffix += f"_{group}"

        # Open dataset
        if dset is None:
            dset = self._open_dataset(chunks=chunks)

        # Variable to sectorise directions
        if nsector:
            try:
                dirs = dset[dir_var]
            except KeyError:
                raise ValueError(
                    "Attempting to sectorise over directions "
                    f"but dset does not have dir_var '{dir_var}'"
                )

        # Selecting variables
        if data_vars == "all":
            data_vars = [k for k in dset.data_vars]
        logger.debug(f"Calculating {func} for vars: {data_vars}")
        dset = dset[data_vars]

        # Calculating stats
        with self.client(**self.cluster_kwargs) as client:
            logger.info(client)

            # Create graphs
            if nsector:
                suffix += "_direc"
                dsout = self._directional_stat(dset, func, dirs, nsector, group, **kwargs)
            else:
                dsout = getattr(self, func)(dset, group=group, **kwargs)

            # Computing
            if compute:
                logger.info(f"Compute dask {func} stat")
                dsout = dsout.load()
            else:
                logger.info(f"Dask {func} stat will be computed when saving to disk")

        # Rename and merge onto output dataset
        dsout = dsout.rename({v: f"{v}{suffix}" for v in dsout.data_vars.keys()})
        self.dsout = self.dsout.merge(dsout)

        return dsout

    def to_netcdf(self, outfile, format="NETCDF4", _FillValue=-32767):
        """Save output dataset as netcdf.

        Args:
            - outfile (str): Name of output netcdf file.
            - format (str): Output Netcdf file format.
            - _FillValue (int): Fill Value.

        """
        logger.info(f"Saving stats file: {Path(outfile).absolute()}")
        encoding = {v: {"zlib": True} for v in self.dsout.data_vars}
        self._finalise()
        self.dsout.to_netcdf(outfile, format=format, encoding=encoding)
        if self.updir:
            self._upload(outfile)

    def to_zarr(
        self,
        outfile,
        _FillValue=_FILLVALUE,
        mode="w",
        **kwargs,
    ):
        """Save output dataset as zarr.

        Args:
            - outfile (str): Base name of output zarr file.
            - _FillValue (int): Fill Value.
            - kwargs: Keyword arguments to pass to Dataset.to_zarr.

        """
        logger.info(f"Saving stats file: {Path(outfile).absolute()}")
        self._finalise()
        for data_var in self.dsout.data_vars:
            self.dsout[data_var].encoding.update({"_FillValue": _FillValue})
            self.dsout[data_var].encoding.pop("zlib", None)
        self.dsout.to_zarr(outfile, **kwargs)
        if self.updir:
            self._upload(outfile)


if __name__ == "__main__":
    st = Stats(
        outfile="tmp.nc",
        urlpath="/source/onstats/tests/tasman.nc",
        engine="netcdf4",
        mapping={"tps": "tp"},
        localdir="/scratch",
        cluster_kwargs={},
        calls=[],
    )

    # dsout1 = st.apply_func(func="mean", data_vars=["hs", "tp", "dpm"], compute=True, dim="time")
    dsout2 = st.apply_func(func="mean", data_vars=["hs", "tp", "dpm"], compute=True, dim="time", xstep=10, ystep=10)
