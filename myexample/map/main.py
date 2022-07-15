import json

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from os.path import dirname, join

data = json.loads('data/china_provinces.geojson')
data
curdoc().add_root()