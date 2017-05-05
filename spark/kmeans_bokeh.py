import json
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Range1d,  Slider, HoverTool
from bokeh.models.glyphs import Ellipse, Circle
from bokeh.io import curdoc, show

with open("output/sample_clusters.json", "r") as f:
	cluster_hist = json.load(f)
f.close()

source = ColumnDataSource(cluster_hist["0"])

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

kmeans_viz = figure(title="K-Means Visualization", plot_width=600, plot_height=600,
					toolbar_location = "above", toolbar_sticky=False, tools=TOOLS,
					x_range=Range1d(cluster_hist["min_long"], cluster_hist["max_long"]),
					y_range=Range1d(cluster_hist["min_lat"], cluster_hist["max_lat"]))

clusters = kmeans_viz.add_glyph(source, Ellipse(x="longitude", y="latitude", width="width", height="height", fill_color="LightSeaGreen"))
centroids = kmeans_viz.add_glyph(source, Circle(x="longitude", y="latitude", size=10, fill_alpha=1))
hover = kmeans_viz.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
	("Count", "@count"),
	("(Long, Lat)", "($x.2, $y.2)"),
	("Width", "@width"),
	("Height", "@height"),

]


iter_slider = Slider(start=0, end=len(cluster_hist) - 5, value=0, step=1, title="Iteration")


def update(attr, old, new):
	clusters.data_source.data = cluster_hist[str(iter_slider.value)]
	centroids.data_source.data = cluster_hist[str(iter_slider.value)]

iter_slider.on_change("value", update)


curdoc().add_root(layout([[kmeans_viz,iter_slider]]))
