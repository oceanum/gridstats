"""Create kmz files from netcdf."""
import datetime
import sys
import os
import copy
import yaml
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
from dateutil.parser import parse
import warnings

import oncore

from simplekml import (
    Kml,
    OverlayXY,
    ScreenXY,
    Units,
    RotationXY,
    AltitudeMode,
    Color,
    Camera,
)
from simplekml.featgeom import GroundOverlay


warnings.filterwarnings("ignore")


plt.switch_backend("agg")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

outdir = "product"


class KMZ:
    """Make kmz from netcdf files."""

    def __init__(self, config, **kwargs):
        self.camera = self._read_config(config, "camera")
        self.layers = self._read_config(config, "layer")
        self.charts = self._read_config(config, "chart")
        self.groundoverlay = self._read_config(config, "groundoverlay")
        self.logo = self._read_config(config, "logo")
        self.pixels = kwargs.pop("pixels", 1024)
        self.kmls = {}
        self.outdir = kwargs.pop("outdir", "./tmp")
        self.kmzfile = kwargs.pop("kmzfile", "southernocean.kmz")
        self.resolution = self._read_config(config, "resolution")
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        self.kml = Kml(open=1)

        try:
            self.mask_file = self._read_config(config, "mask_file")
        except KeyError:
            self.mask_file = None

    def __repr__(self):
        return "KMZ Object"

    @property
    def llcrnrlon(self):
        return float(self.ds.lon.min())

    @property
    def llcrnrlat(self):
        return float(self.ds.lat.min())

    @property
    def urcrnrlon(self):
        return float(self.ds.lon.max())

    @property
    def urcrnrlat(self):
        return float(self.ds.lat.max())

    @property
    def layer_name(self):
        return self.layer_val.get("desc", self.layer_val["var"])

    @property
    def visibility(self):
        return self.layer_val.get("visibility", 0)

    @property
    def figname(self):
        dimname = "_{}".format(self.dim_name) or ""
        dimindex = "_{}".format(self.dim_index) or ""
        return os.path.join(
            self.outdir, "{}{}{}_map.png".format(self.layer_id, dimname, dimindex)
        )

    @property
    def colorbar_name(self):
        dimname = "_{}".format(self.dim_name) or ""
        dimindex = "_{}".format(self.dim_index) or ""
        return os.path.join(
            self.outdir, "{}{}{}_colorbar.png".format(self.layer_id, dimname, dimindex)
        )

    @property
    def colorbar_label(self):
        name = self.layer_name
        if "units" in self.layer_val and self.layer_val["units"]:
            name += " [{}]".format(self.layer_val["units"])
        return name

    @property
    def plot_kwargs(self):
        kwargs = self.chart.copy()
        kwargs.pop("type", None)
        if (
            "levels" not in kwargs
            and "inc" in kwargs
            and "vmin" in kwargs
            and "vmax" in kwargs
        ):
            if "levels" not in kwargs:
                inc = kwargs.pop("inc", 0.1)
                vmin = kwargs.pop("vmin", 0.0)
                vmax = kwargs.pop("vmax", np.ceil(self.darr.max())) + inc
                kwargs.update({"levels": np.arange(vmin, vmax, inc)})
        if "cmap" in kwargs:
            kwargs.update({"cmap": self._get_cmap(kwargs["cmap"])})
        return kwargs

    @property
    def _is_global(self):
        """Check if global wrapping longitudes."""
        lon_step = self.ds.lon[1] - self.ds.lon[0]
        lon_range = self.ds.lon[-1] - self.ds.lon[0]
        lon_buff = lon_step * 0.5
        if lon_range + lon_step + lon_buff >= 360:
            return True
        else:
            return False

    @property
    def depth(self):
        """DataArray used to extract depth contours."""
        try:
            depth_var = self.layer_val["depth_var"]
        except KeyError():
            raise ValueError(f"'depth_var' must be defined in layer {self.layer_name}")
        return self.ds[depth_var].where(self.ds[depth_var] > 0).fillna(0.)

    def _set_mask(self, dset):  # , ncfile, var, vmin=None, vmax=None):
        import geopandas as gpd
        import rioxarray
        from shapely.geometry import mapping

        if isinstance(self.mask_file, str):
            gdf = gpd.read_file(self.mask_file)
        elif isinstance(self.mask_file, list):
            gdf = pd.concat([gpd.read_file(file) for file in self.mask_file]).pipe(
                gpd.GeoDataFrame
            )
        else:
            raise ValueError(f"mask_file must be string or list")
        gdf = gdf.to_crs("EPSG:4326")
        tmp = dset.copy(deep=True)
        tmp.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        tmp.rio.write_crs(gdf.crs, inplace=True)
        tmp = tmp.rio.clip(
            gdf.geometry.apply(mapping), gdf.crs, drop=False, invert=True
        )
        # Depth not masked in gebco for some reason, quick fix here
        if "depth" in tmp.data_vars:
            tmp2 = dset.copy(deep=True)
            logger.warning(
                f"Masking depth values below zero because gebco does not mask for some reason"
            )
            tmp = tmp.assign({"depth": tmp2.depth.where(tmp2.depth >= 0)})
        return tmp

    def _get_cmap(self, cmap):
        """Get colormap from cmocean if available otherwise matplotlib."""
        try:
            return getattr(cmocean.cm, cmap)
        except:
            try:
                return getattr(oncore.cm, cmap)
            except:
                return plt.cm.get_cmap(cmap)

    def _plot_layer_figures(self, layer_type="groundoverlay"):
        """Plotting and saving all figures from layer.

        layer_type (str):
            - `groundoverlay`: for raster layers.
            - `linestring`: for contour layers.

        The number of figures depend on the indices provided.

        """
        self._read_dataset()
        indices = self.layer_val.get("indices", None)
        self.dim_names = []
        if indices is not None:
            # Loop through each index to be sliced away
            for self.dim_name, dim_slice in indices.items():
                # Loop through slicing dictionary defined by 'index'
                for self.dim_index, self.dim_label in dim_slice.items():
                    self.dim_names.append(str(self.dim_label))
                    if layer_type == "groundoverlay":
                        self._make_plot()
                    elif layer_type == "linestring":
                        self.add_contour()
                    elif layer_type == "cyclone":
                        self._make_cyclone()
                    elif layer_type == "statscontour":
                        self._make_statscontour()
        else:
            self.dim_name = None
            self.dim_index = 0
            self.dim_label = self.layer_val.get("name", "Annual")
            self.dim_names.append(str(self.dim_label))
            if layer_type == "groundoverlay":
                self._make_plot()
            elif layer_type == "linestring":
                self.add_contour()
            elif layer_type == "cyclone":
                self._make_cyclone()
            elif layer_type == "statscontour":
                self._make_statscontour()

    def _make_plot(self):
        """Here the map and colorbar from each layer are generated."""
        self.darr = self.ds[self.layer_val["var"]]
        if self.dim_name is not None:
            self.darr = self.darr.isel(**{self.dim_name: self.dim_index})
        fig, ax = self.gearth_fig()
        # Plotting
        logger.debug("Plotting png for layer: {}".format(self.figname))
        cs = getattr(ax, self.chart["type"])(
            self.darr.lon, self.darr.lat, self.darr, **self.plot_kwargs
        )
        # # Watermark
        # ax.text(
        #     x=0.05,
        #     y=0.95,
        #     s="www.moanaproject.org",
        #     transform=ax.transAxes,
        #     fontsize=15,
        #     fontweight="bold",
        #     color="black",
        #     alpha=0.5,
        #     ha="left",
        #     va="center",
        #     rotation="0"
        # )
        if self.chart["type"] == "contour":
            cs.clabel(fontsize=10, fmt=f"%i{self.layer_val.get('units', 'm')}")
        fig.savefig(self.figname, transparent=True, format="png")
        plt.close(fig)
        if self.chart["type"] in ["contourf", "pcolormesh"]:
            # Colorbar
            fig = plt.figure(figsize=(1.2, 4.0), facecolor=None, frameon=False)
            ax = fig.add_axes([0.0, 0.05, 0.2, 0.9])
            cb = fig.colorbar(cs, cax=ax)
            cb.set_label(self.colorbar_label, rotation=-90, color="white", labelpad=20)

            # set colorbar tick color
            cb.ax.yaxis.set_tick_params(color="white")
            # set colorbar edgecolor
            cb.outline.set_edgecolor("white")
            # set colorbar ticklabels
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
            fig.savefig(
                self.colorbar_name, transparent=True, format="png"
            )  # Change transparent to True if your colorbar is not on space
            plt.close(fig)
            self.colorbar = True
        else:
            self.colorbar = False
        self.add_raster_to_kmz()

    def _gray_to_hex(self, scale):
        """Convert gray scale (0,1) into hex code"""
        rgb = int(round(scale * 255))
        return "ff{0:02x}{1:02x}{2:02x}".format(rgb, rgb, rgb)

    def _rgb_to_hex(self, rgb_scaled):
        """Convert scaled rgb tuple (0-1, 0-1, 0-1) into hex code"""
        rgb = [int(round(v * 255)) for v in rgb_scaled]
        return "ff{0:02x}{1:02x}{2:02x}".format(*rgb)

    def _add_colorbar(self, group):
        screen = group.newscreenoverlay(name="Colorbar")
        screen.icon.href = self.colorbar_name
        screen.overlayxy = OverlayXY(
            x=0, y=0, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.screenxy = ScreenXY(
            x=0.015, y=0.075, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.rotationXY = RotationXY(
            x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = self.visibility

    def _make_statscontour(self):
        """Make linestring for stats over depth layers."""
        logger.debug(f"Plotting statscontour for layer: {self.figname}")

        self.darr = self.ds[self.layer_val["var"]]

        levels = self.chart["depth_contours"]
        step = self.chart.get("step", 1)
        units = self.layer_val.get("units", "m")

        # Extract coordinates of depth contours, each contour will have multiple paths
        fig = plt.figure()
        depths = {}
        for level in levels:
            ax = fig.add_subplot()
            cs = ax.contour(self.depth.lon, self.depth.lat, self.depth, levels=[level])
            paths = cs.collections[0].get_paths()
            # multipoint = self.group.newmultigeometry(name="{:0.0f} m".format(level))
            # Collect all contour paths for level
            all_coords = []
            for ipath, path in enumerate(paths):
                coords = [
                    (path.vertices[ivert][0], path.vertices[ivert][1], 0)
                    for ivert in range(len(path.vertices))
                ]
                all_coords.append(coords)
            depths.update({level: all_coords})
            plt.clf()
        plt.close()

        norm = colors.Normalize(vmin=self.chart["vmin"], vmax=self.chart["vmax"], clip=True)
        cmap = cm.get_cmap(self.chart.get("cmap", "turbo"))

        # Define base container
        containers = [c.name for c in self.group.containers]
        if self.layer_name not in containers:
            group = self.group.newfolder(name=self.layer_name)
        else:
            index = containers.index(self.layer_name)
            group = self.group.containers[index]
        group1 = group.newfolder(name=f"{self.layer_val['name']} at depths")

        # Colorbar
        # if self.colorbar:
        #     self._add_colorbar(group1)

        # Loop over depth contours
        npoints = {}
        for depth, all_contours in depths.items():
            npoints[depth] = 0
            logger.debug(f"Stat for depth layer {depth}")
            subgroup = group1.newfolder(name=f"{depth:0.0f}m")
            # Loop over all paths for current contour
            for contour_coords in all_contours:
                x = xr.DataArray([c[0] for c in contour_coords], dims=("stat",))
                y = xr.DataArray([c[1] for c in contour_coords], dims=("stat",))
                z = self.darr.interp(lon=x, lat=y).fillna(0.)
                # Subsetting
                x = x[::step]
                y = y[::step]
                z = z[::step]
                coords = [(xc, yc, zc) for xc, yc, zc in zip(x.values, y.values, z.values)]
                npoints[depth] += x.size
                for ind in range(len(coords) - 1):
                    # Current segment to plot
                    segment = coords[ind:ind+2]
                    # Color of segment
                    value = (segment[0][-1] + segment[1][-1]) / 2
                    rgba = cmap(norm(value), bytes=True)
                    alpha = 255 if value > 0 else 0
                    color = Color.rgb(rgba[0], rgba[1], rgba[2], alpha)
                    # Define linestring for segment
                    linestring = subgroup.newlinestring(
                        name=f"{value:0.2f} {units}",
                        coords=segment,
                        altitudemode=AltitudeMode.relativetoground,
                    )
                    linestring.style.linestyle.color = color
                    linestring.style.linestyle.width = self.plot_kwargs.get(
                        "linewidths", 2.0
                    )
                    linestring.visibility = self.visibility

    def _make_cyclone_raster(self):
        """Make linestring for cyclone path layers."""
        logger.info("Plotting cyclone for layer: {}".format(self.figname))
        # Parse ibtracs dataset
        self._parse_ibtracs(
            tstart=self.layer_val.get("tstart", None),
            tend=self.layer_val.get("tend", None),
        )

        # Plotting all cyclones over the glob
        x0 = 0
        x1 = 360
        y0 = -80
        y1 = 80

        self.group = self.kml.newfolder(name=self.layer_name)
        # Loop over cat groups
        for cat, ds in self.ds.groupby(self.ds.catmax):
            logger.info("Adding cat {:0.0f} layer".format(cat))
            subgroup = self.group.newfolder(name="Cat {:0.0f}".format(cat))
            # pnt = subgroup.newpoint(name="point test")
            # pnt.coords = [(100+cat, 10+cat)]

            # Force attributes here to prescribe figure name
            self.dim_name = "cat"
            self.dim_index = cat

            fig, ax = self.gearth_fig(
                x0=x0, x1=x1, y0=y0, y1=y1, dpi=max(self.pixels, 512)
            )

            ax.scatter(
                ds.lon,
                ds.lat,
                0.5,
                ds.usa_sshs,
                vmin=0,
                vmax=5,
                cmap="viridis",
                edgecolors="none",
            )
            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))
            fig.savefig(self.figname, transparent=True, format="png")
            plt.close(fig)

            ground = subgroup.newgroundoverlay(name="Cat {:0.0f}".format(cat))
            ground.visibility = 0
            ground.icon.href = self.figname
            ground.latlonbox.west = x0
            ground.latlonbox.east = x1
            ground.latlonbox.south = y0
            ground.latlonbox.north = y1
            # break

        # levels = self.plot_kwargs.pop('levels')
        # gray_scales = np.linspace(0.2, 0.0, len(levels))
        # for level, gray_scale in zip(levels, gray_scales):
        #     logger.debug('Drawing contour level {}'.format(level))
        #     kwargs = self.plot_kwargs.copy()
        #     kwargs.update({'levels': [level]})

        #     cs = getattr(ax, self.chart['type'])(
        #         self.darr.lon, self.darr.lat, self.darr, **kwargs)

        #     multipoint = self.group.newmultigeometry(name='{:0.0f} m'.format(level))
        #     paths = cs.collections[0].get_paths()
        #     for ipath, path in enumerate(paths):
        #         coords = [
        #             (path.vertices[ivert][0], path.vertices[ivert][1], 0)
        #             for ivert in range(len(path.vertices))
        #         ]
        #         linestring = multipoint.newlinestring(
        #             coords=coords,
        #             altitudemode=AltitudeMode.relativetoground,
        #         )
        #         linestring.style.linestyle.color = self._gray_to_hex(gray_scale)
        #         linestring.style.linestyle.width = kwargs.get('linewidths', 2.0)

    def _make_cyclone_points(self):
        """Make poins for cyclone path layers."""
        logger.info("Plotting cyclone for layer: {}".format(self.figname))
        # Parse ibtracs dataset
        self._parse_ibtracs(
            tstart=self.layer_val.get("tstart", None),
            tend=self.layer_val.get("tend", None),
        )

        symbols = {
            1: "symbols/circle_blue.png",
            2: "symbols/circle_green.png",
            3: "symbols/circle_yellow.png",
            4: "symbols/circle_orange.png",
            5: "symbols/circle_red.png",
        }
        self.group = self.kml.newfolder(name=self.layer_name)
        # Loop over cat groups
        for cat, ds in self.ds.groupby(self.ds.catmax):
            logger.info("Adding cat {:0.0f} layer".format(cat))
            subgroup = self.group.newfolder(name="Cat {:0.0f}".format(cat))

            # ds = ds.dropna(dim='date_time')
            for istorm in range(ds.storm.size):
                logger.info("Storm {}".format(istorm))
                dstorm = ds.isel(storm=istorm)
                lat = dstorm.lat.values
                lon = dstorm.lon.values
                cat = dstorm.usa_sshs.values
                for x, y, z in zip(lon, lat, cat):
                    if not any(np.isnan((x, y, z))):
                        pnt = subgroup.newpoint()
                        pnt.coords = [(lon, lat)]
                        pnt.style.iconstyle.icon.href = symbols[z]

    def _make_cyclone(self):
        """Make linestring for cyclone path layers."""
        logger.info("Plotting cyclone for layer: {}".format(self.figname))
        # Parse ibtracs dataset
        self._parse_ibtracs(
            tstart=self.layer_val.get("tstart", None),
            tend=self.layer_val.get("tend", None),
        )

        colors = {
            1: Color.blue,
            2: Color.green,
            3: Color.yellow,
            4: Color.orange,
            5: Color.red,
        }
        self.group = self.kml.newfolder(name=self.layer_name)
        # Loop over cat groups
        for cat, ds in self.ds.groupby(self.ds.catmax):
            logger.info("Adding cat {:0.0f} layer".format(cat))
            subgroup = self.group.newfolder(name="Cat {:0.0f}".format(cat))
            for istorm in range(ds.storm.size):
                df = ds.isel(storm=istorm).to_dataframe().dropna()
                if df.size < 1:
                    continue
                else:
                    logger.info("Storm {}".format(istorm))
                storm_name = "{:%Y-%m-%d}".format(df.iloc[0].storm)
                coords = [(x, y, 0) for x, y in zip(df.lon, df.lat)]
                linestring = subgroup.newlinestring(
                    name=storm_name,
                    coords=coords,
                    altitudemode=AltitudeMode.relativetoground,
                )
                linestring.style.linestyle.color = colors[df["catmax"].max()]
                linestring.style.linestyle.width = self.plot_kwargs.get(
                    "linewidths", 2.0
                )
                linestring.visibility = 0

    def _append_kmls(self):
        """Append kml layer information."""
        self.kmls.setdefault(self.layer_name, [])
        self.kmls[self.layer_name].append(
            {
                "dim": self.dim_label,
                "map": self.figname,
                "colorbar": self.colorbar_name,
                "visibility": self.visibility,
            }
        )

    def _to_180(self):
        """Convert longitudes in grid from 0-360 to -180--180 convention.

        Only done for global grids, to ensure consistent coordinates convention.

        """
        # Ignore if lon is not a coordinate
        if "lon" not in self.ds.dims:
            return
        # Ignore if not global
        if not self._is_global:
            return
        self.ds = self.ds.assign_coords({"lon": (self.ds.lon.values + 180) % 360 - 180})
        self.ds = self.ds.sortby("lon")
        # Make it wrap
        dsi = self.ds.isel(lon=[0])
        dsi = dsi.assign_coords({"lon": [180]})
        self.ds = xr.concat((self.ds, dsi), dim="lon")

    def _parse_ibtracs(self, tstart="2019-01-01", tend=None):
        """Read and parse relevant data from ibtracs file."""
        self.ds["storm"] = self.ds.time.isel(date_time=0)
        self.ds = self.ds.reset_coords()
        # Sort it so it can be sliced
        self.ds = self.ds.isel(storm=np.argsort(self.ds.storm).values)
        self.ds = self.ds.sel(storm=slice(tstart, tend))

        self.ds = self.ds.where(self.ds.usa_sshs > 0)[["lat", "lon", "usa_sshs"]]
        self.ds["catmax"] = self.ds.usa_sshs.max(dim="date_time")
        return self.ds

    def _read_dataset(self):
        x0 = self.layer_val.get("x0", None)
        x1 = self.layer_val.get("x1", None)
        y0 = self.layer_val.get("y0", None)
        y1 = self.layer_val.get("y1", None)
        filename = self.layer_val["filename"]
        if filename.endswith(".nc"):
            self.ds = xr.open_dataset(self.layer_val["filename"]).squeeze(drop=True)
        elif filename.endswith(".zarr"):
            self.ds = xr.open_zarr(
                self.layer_val["filename"], consolidated=True
            ).squeeze(drop=True)
        else:
            raise ValueError(
                f"File {filename} not recognised, only .nc and .zarr are supported"
            )
        # Slicing
        for method, kwarg in self.layer_val.get("slice_dict", {}).items():
            self.ds = getattr(self.ds, method)(**kwarg)

        # Rename coordinates
        if "longitude" in self.ds and "latitude" in self.ds:
            self.ds = self.ds.rename({"longitude": "lon", "latitude": "lat"})
        self.ds = self.ds.sortby("lat").sortby("lon")
        self._to_180()
        self.ds = self.ds.sel(lon=slice(x0, x1), lat=slice(y0, y1))

        # Hacked for now, interpolating to improve land mask
        res = float(np.diff(self.ds.lon[[0, 1]]))
        if self.resolution is not None and res > self.resolution:
            lons = np.arange(
                self.ds.lon[0], self.ds.lon[-1] + self.resolution, self.resolution
            )
            lats = np.arange(
                self.ds.lat[0], self.ds.lat[-1] + self.resolution, self.resolution
            )
            self.ds = self.ds.interp(lon=lons, lat=lats)

        if self.mask_file is not None:
            self.ds = self._set_mask(self.ds)
            # mask = self.dset_mask.interp_like(self.ds)
            # self.ds = self.ds.where(mask.notnull())

    def _read_config(self, filename, what):
        with open(filename, "r") as stream:
            return yaml.load(stream, Loader=yaml.Loader).get(what, None)

    def run(self):
        """Generate KMZ file based on yml config."""
        if self.logo is not None:
            self.add_logo(
                "Oceanum",
                self.logo,
                0.0350,
                0.90,
                0.20,
            )

        for group0, layers in self.layers.items():
            self.group0 = self.kml.newfolder(name=group0, open=1)
            for self.layer_id, self.layer_val in layers.items():
                if not self.layer_val.get("active", True):
                    logger.info(f"Layer {self.layer_id} not active, skipping.")
                    continue
                subgroups = [c.name for c in self.group0.containers]
                subgroup = self.layer_val.get("subgroup", None)
                if subgroup is not None:
                    logger.debug(f"Subgroup {subgroup} key defined in layer")
                    if subgroup in subgroups:
                        logger.debug(f"Subgroup {subgroup} already created")
                        self.group = self.group0.containers[subgroups.index(subgroup)]
                    else:
                        logger.debug(f"Creating subgroup {subgroup}")
                        self.group = self.group0.newfolder(name=subgroup)
                else:
                    logger.debug("subgroup key not defined in layer or None")
                    self.group = self.group0

                self.chart = self.charts[self.layer_val["chart"]]
                logger.info("Creating layer: {}".format(self.layer_id))
                logger.debug("Chart config: \n{}".format(self.chart))
                if self.chart.get("type", None) == "isoline":
                    layer_type = "linestring"
                elif self.chart.get("type", None) == "cyclone":
                    layer_type = "cyclone"
                elif self.chart.get("type", None) == "statscontour":
                    layer_type = "statscontour"
                else:
                    layer_type = "groundoverlay"
                self._plot_layer_figures(layer_type=layer_type)
        self.add_camera(**self.camera)
        self.save_kmz()

    def gearth_fig(self, x0=None, x1=None, y0=None, y1=None, dpi=None):
        """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image.

        Args:
            x0 (float): West limit for figure, by default self.llcrnlon.
            x1 (float): East limit for figure, by default self.urcrnlon.
            y0 (float): South limit for figure, by default self.llcrnlat.
            y1 (float): North limit for figure, by default self.urcrnlat.
            dpi (int): Figure resolution, by default self.dpi.

        """
        x0 = x0 or self.llcrnrlon
        x1 = x1 or self.urcrnrlon
        y0 = y0 or self.llcrnrlat
        y1 = y1 or self.urcrnrlat
        dpi = dpi or self.pixels

        aspect = np.cos(np.mean([y0, y1]) * np.pi / 180.0)
        xsize = np.ptp([x1, x0]) * aspect
        ysize = np.ptp([y1, y0])
        aspect = ysize / xsize
        if aspect > 1.0:
            figsize = (10.0 / aspect, 10.0)
        else:
            figsize = (10.0, 10.0 * aspect)
        # if False:
        #     plt.ioff()  # Make `True` to prevent the KML components from poping-up.
        fig = plt.figure(figsize=figsize, frameon=False, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")
        return fig, ax

    def add_camera(self, **kw):
        """Add camera settings to kmz."""
        self.kml.document.camera = Camera(
            latitude=kw.pop("latitude", np.mean([self.urcrnrlat, self.llcrnrlat])),
            longitude=kw.pop("longitude", np.mean([self.urcrnrlon, self.llcrnrlon])),
            altitude=kw.pop("altitude", 2e7),
            roll=kw.pop("roll", 0),
            tilt=kw.pop("tilt", 0),
            altitudemode=kw.pop("altitudemode", AltitudeMode.relativetoground),
        )

    def add_raster_to_kmz(self, **kw):
        """Create kmz file."""
        containers = [c.name for c in self.group.containers]
        if self.layer_name not in containers:
            group = self.group.newfolder(name=self.layer_name)
        else:
            index = containers.index(self.layer_name)
            group = self.group.containers[index]

        subgroup = group.newfolder(name=self.dim_label)
        ground = subgroup.newgroundoverlay(name="Map")
        # ground = group.newgroundoverlay()
        for key, val in self.groundoverlay.items():
            setattr(ground, key, val)
        # ground.draworder = draworder
        ground.latlonbox.rotation = self.groundoverlay["rotation"]
        ground.icon.href = self.figname
        ground.latlonbox.east = self.llcrnrlon
        ground.latlonbox.south = self.llcrnrlat
        ground.latlonbox.north = self.urcrnrlat
        ground.latlonbox.west = self.urcrnrlon
        ground.visibility = self.visibility

        if self.colorbar:
            self._add_colorbar(subgroup)

    def add_contour(self):
        """Make linestring contour layers."""
        logger.debug("Plotting linestring for layer: {}".format(self.figname))
        fig, ax = self.gearth_fig()
        self.darr = self.ds[self.layer_val["var"]]

        if self.dim_name is not None:
            self.darr = self.darr.isel(**{self.dim_name: self.dim_index})

        # group = self.group.newfolder(name=self.layer_name)
        containers = [c.name for c in self.group.containers]
        if self.layer_name not in containers:
            group = self.group.newfolder(name=self.layer_name)
        else:
            index = containers.index(self.layer_name)
            group = self.group.containers[index]
            group = group.newfolder(name=f"Isoline {self.dim_label}")

        rgb = self.plot_kwargs.pop("rgb", [0, 0, 0])
        levels = self.plot_kwargs.pop("levels")
        gray_scales = np.linspace(0.2, 0.0, len(levels))
        for level, gray_scale in zip(levels, gray_scales):
            logger.debug("Drawing contour level {}".format(level))
            kwargs = self.plot_kwargs.copy()
            kwargs.update({"levels": [level]})

            # Remove kwargs not used by contour
            kwargs.pop("precision", None)
            kwargs.pop("rgb", None)

            cs = ax.contour(self.darr.lon, self.darr.lat, self.darr, **kwargs)

            units = self.layer_val.get("units", "m")
            precision = self.chart.get("precision", "0.0f")
            paths = cs.collections[0].get_paths()
            if not paths:
                continue
            multipoint = group.newmultigeometry(name=f"{level:{precision}} {units}")
            for ipath, path in enumerate(paths):
                coords = [
                    (path.vertices[ivert][0], path.vertices[ivert][1], 0)
                    for ivert in range(len(path.vertices))
                ]
                linestring = multipoint.newlinestring(
                    coords=coords, altitudemode=AltitudeMode.relativetoground
                )
                # linestring.style.linestyle.color = self._gray_to_hex(gray_scale)
                linestring.style.linestyle.color = self._rgb_to_hex(rgb)
                linestring.style.linestyle.width = kwargs.get("linewidths", 2.0)
            multipoint.visibility = self.visibility
        plt.close(fig)

    def add_logo(self, name, icon_href, x, y, size_x):
        screen = self.kml.newscreenoverlay(name=name)
        screen.icon.href = icon_href
        screen.overlayxy = OverlayXY(
            x=0, y=0, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.screenxy = ScreenXY(
            x=x, y=y, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.rotationXY = RotationXY(
            x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.size.x = size_x
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    def save_kmz(self):
        self.kml.savekmz(os.path.join(self.outdir, self.kmzfile))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make KMZ file from netcdf stats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=None,
    )
    parser.add_argument("config", help="yaml config file specifying layers")
    parser.add_argument(
        "-o", "--outdir", help="output directory for saving files", default="./outdir"
    )
    parser.add_argument(
        "-k", "--kmzfile", help="output name for kmz", default="southernocean.kmz"
    )
    parser.add_argument("-p", "--pixels", help="Figure dpi", default=1024, type=int)
    args = vars(parser.parse_args())

    self = KMZ(**args)
    self.run()
