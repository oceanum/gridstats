"""Create kmz files from netcdf."""
import datetime
import sys
import os
import copy
import yaml
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import cmocean
from dateutil.parser import parse

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
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        self.kml = Kml()

        try:
            self.mask = self._set_mask(**self._read_config(config, "mask"))
        except KeyError:
            self.mask = None

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
        for self.layer_id, self.layer_val in self.layers.items():
            self.chart = self.charts[self.layer_val["chart"]]
            logger.info("Creating layer: {}".format(self.layer_id))
            logger.debug("Chart config: \n{}".format(self.chart))
            if self.chart.get("type", None) == "isoline":
                layer_type = "linestring"
            elif self.chart.get("type", None) == "cyclone":
                layer_type = "cyclone"
            else:
                layer_type = "groundoverlay"
            self._plot_layer_figures(layer_type=layer_type)
        self.add_camera(**self.camera)
        self.save_kmz()

    def _set_mask(self, ncfile, var, vmin=None, vmax=None):
        darr = xr.open_dataset(ncfile)[var]
        self.dset_mask = (darr * 0 + 1)
        if vmin is not None:
            self.dset_mask = self.dset_mask.where(darr >= vmin)
        if vmax is not None:
            self.dset_mask = self.dset_mask.where(darr <= vmax)
        if "longitude" in self.dset_mask and "latitude" in self.dset_mask:
            self.dset_mask = self.dset_mask.rename({"longitude": "lon", "latitude": "lat"})
        return self.dset_mask

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
            for self.dim_name, dim_slice in indices.items():
                for self.dim_index, self.dim_label in dim_slice.items():
                    self.dim_names.append(str(self.dim_label))
                    if layer_type == "groundoverlay":
                        self._make_plot()
                    elif layer_type == "linestring":
                        self._make_contour()
                    elif layer_type == "cyclone":
                        self._make_cyclone()
        else:
            self.dim_name = None
            self.dim_index = 0
            self.dim_label = "Annual"
            self.dim_names.append(str(self.dim_label))
            if layer_type == "groundoverlay":
                self._make_plot()
            elif layer_type == "linestring":
                self._make_contour()
            elif layer_type == "cyclone":
                self._make_cyclone()

    def _make_plot(self):
        """Here the map and colorbar from each layer are generated."""
        self.darr = self.ds[self.layer_val["var"]]
        if self.dim_name is not None:
            self.darr = self.darr.isel(**{self.dim_name: self.dim_index})
        fig, ax = self.gearth_fig()
        # Plotting
        logger.info("Plotting png for layer: {}".format(self.figname))
        cs = getattr(ax, self.chart["type"])(
            self.darr.lon, self.darr.lat, self.darr, **self.plot_kwargs
        )
        if self.chart["type"] == "contour":
            cs.clabel(fontsize=10, fmt=f"%i{self.layer_val.get('units', 'm')}")
        fig.savefig(self.figname, transparent=True, format="png")
        plt.close(fig)
        if self.chart["type"] in ["contourf", "pcolormesh"]:
            # Colorbar
            fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)
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

    def _make_contour(self):
        """Make linestring contour layers."""
        logger.info("Plotting linestring for layer: {}".format(self.figname))
        fig, ax = self.gearth_fig()
        self.darr = self.ds[self.layer_val["var"]]

        if self.dim_name is not None:
            self.darr = self.darr.isel(**{self.dim_name: self.dim_index})

        self.group = self.kml.newfolder(name=self.layer_name)

        levels = self.plot_kwargs.pop("levels")
        gray_scales = np.linspace(0.2, 0.0, len(levels))
        for level, gray_scale in zip(levels, gray_scales):
            logger.info("Drawing contour level {}".format(level))
            kwargs = self.plot_kwargs.copy()
            kwargs.update({"levels": [level]})

            cs = ax.contour(self.darr.lon, self.darr.lat, self.darr, **kwargs)

            multipoint = self.group.newmultigeometry(name="{:0.0f} m".format(level))
            paths = cs.collections[0].get_paths()
            for ipath, path in enumerate(paths):
                coords = [
                    (path.vertices[ivert][0], path.vertices[ivert][1], 0)
                    for ivert in range(len(path.vertices))
                ]
                linestring = multipoint.newlinestring(
                    coords=coords, altitudemode=AltitudeMode.relativetoground, name="aaaaa"
                )
                linestring.style.linestyle.color = self._gray_to_hex(gray_scale)
                linestring.style.linestyle.width = kwargs.get("linewidths", 2.0)
            multipoint.visibility = self.visibility

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
            # import ipdb; ipdb.set_trace()

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
        #     logger.info('Drawing contour level {}'.format(level))
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
                # import ipdb; ipdb.set_trace()
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
        self.ds = xr.open_dataset(self.layer_val["filename"]).squeeze(drop=True)
        if "longitude" in self.ds and "latitude" in self.ds:
            self.ds = self.ds.rename({"longitude": "lon", "latitude": "lat"})
        self.ds = self.ds.sortby("lat").sortby("lon")
        self.ds = self.ds.sel(lon=slice(x0, x1), lat=slice(y0, y1))
        self._to_180()
        if self.mask is not None:
            mask = self.dset_mask.interp_like(self.ds)
            self.ds = self.ds.where(mask.notnull())

    def _read_config(self, filename, what):
        with open(filename, "r") as stream:
            return yaml.load(stream)[what]

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

        # import ipdb; ipdb.set_trace()
        if self.layer_name not in [c.name for c in self.kml.containers]:
            self.group = self.kml.newfolder(name=self.layer_name)

        subgroup = self.group.newfolder(name=self.dim_label)
        ground = subgroup.newgroundoverlay(name="Map")
        # ground = self.group.newgroundoverlay()
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
            # screen = self.group.newscreenoverlay(name='Colorbar')
            screen = subgroup.newscreenoverlay(name="Colorbar")
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
            screen.visibility = ground.visibility

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
