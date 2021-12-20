"""Kmz file of points."""
import datetime
import pandas as pd
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


def read_points(filename):
    df = pd.read_excel(filename)
    df.columns = ["lat", "lon", "dtime"]
    times = [datetime.datetime.strptime(s, "%d/%m/%Y  %H%M") for s in df["dtime"]]
    df.index = times
    return df.drop(columns="dtime")


# Table with points to plot
df = read_points("ROA route v3.xlsx")


# Start kml
kml = Kml(open=True)

# Add each location
ind = 1
for time, row in df.iterrows():
    pnt = kml.newpoint()
    pnt.name = f"{time:%Y-%m-%d %H:%m}"
    pnt.description = f"Point {ind}"
    pnt.coords = [(row.lon, row.lat)]
    ind += 1

# Save
kml.save("points.kml")

# single_point = kml.newpoint(name="The World", coords=[(0.0, 0.0)])

# group = kml.newfolder(name="Locations")

