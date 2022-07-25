import holoviews as hv
import holoviews.operation.datashader as hd
from bokeh.plotting import show ,curdoc
from bokeh.models import Button, CustomJS
from bokeh.layouts import column, row
from bokeh import events
hd.shade.cmap=["lightblue", "darkblue"]
hv.extension("bokeh", "matplotlib") 

import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict as odict

# messageButton = Button(label="message",name=str(0))
# num=100000
# np.random.seed(1)

# dists = {cat: pd.DataFrame(odict([('x',np.random.normal(x,s,num)), 
#                                   ('y',np.random.normal(y,s,num)), 
#                                   ('val',val), 
#                                   ('cat',cat)]))      
#          for x,  y,  s,  val, cat in 
#          [(  2,  2, 0.03, 10, "d1"), 
#           (  2, -2, 0.10, 20, "d2"), 
#           ( -2, -2, 0.50, 30, "d3"), 
#           ( -2,  2, 1.00, 40, "d4"), 
#           (  0,  0, 3.00, 50, "d5")] }

# df = pd.concat(dists,ignore_index=True)
# df["cat"]=df["cat"].astype("category")


# points = hv.Points(df.sample(10000))
# dd = hd.datashade(points)
# renderer = hv.renderer('bokeh')
# renderer = renderer.instance(mode='server')
# hvplot = renderer.get_plot(dd)
# p = hvplot.state
# p.js_on_event(events.Tap, CustomJS(code="""
#         console.log('1111')
#     """))
# curdoc().add_root(column(messageButton,p))
# renderer.server_doc(dd)

# # Declare some points
# points = hv.Points(np.random.randn(1000,2 ))

# # Declare points as source of selection stream
# selection = hv.streams.Selection1D(source=points)

# # Write function that uses the selection indices to slice points and compute stats
# def selected_info(index):
#     arr = points.array()[index]
#     if index:
#         label = 'Mean x, y: %.3f, %.3f' % tuple(arr.mean(axis=0))
#     else:
#         label = 'No selection'
#     return points.clone(arr, label=label).opts(color='red')

# # Combine points and DynamicMap
# selected_points = hv.DynamicMap(selected_info, streams=[selection])
# layout = points.opts(tools=['box_select', 'lasso_select']) + selected_points
# renderer = hv.renderer('bokeh')
# # renderer = renderer.instance(mode='server')
# # hvplot = renderer.get_plot(layout)
# # curdoc().add_root(hvplot.state)
# doc = renderer.server_doc(layout)


# 1. 用图还是分开，用图的话怎么添加工具，怎么转成像素
# 2. hover用到的数据怎么获得，数据来源？
# 3. 尝试用图


