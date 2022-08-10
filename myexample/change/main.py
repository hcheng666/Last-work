import holoviews as hv
import holoviews.operation.datashader as hd
from bokeh.plotting import show ,curdoc,figure
from bokeh.models import HoverTool,ColumnDataSource,TapTool,Button, CustomJS,GeoJSONDataSource,Select,CheckboxButtonGroup
from bokeh.transform import linear_cmap
from bokeh.palettes import OrRd9
from bokeh.layouts import column ,row
from bokeh.events import ButtonClick
from yaml import load
# hd.shade.cmap=["lightblue", "darkblue"]
hv.extension("bokeh", "matplotlib")
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import geopandas as gpd
from holoviews.operation.datashader import datashade, bundle_graph
from holoviews import opts
import psycopg2
isLoad = False
def update():
    '''数据筛选'''
    # 大小透明度
    global isLoad
    active1 = np.array(checkbox_button_group1.active) + 1
    active2 = np.array(checkbox_button_group2.active) + 1
    if active1.size==0 or active2.size==0:
        # 可以放个消息
        return

    flow_ = flow[(flow['S']>=50) & (((flow['origin_line'].apply(lambda r:r in active1)) & (flow['destination_line'].apply(lambda r:r in active2)) | ((flow['origin_line'].apply(lambda r:r in active2) & (flow['destination_line'].apply(lambda r:r in active1))))))].reset_index(drop=True)
    # 优化不了。。
    citys['size'] = pd.Series([0]*362)
    for i in range(len(flow_)):
        citys.loc[city2id[flow_['origin'][i]],'size'] += 1
        citys.loc[city2id[flow_['destination'][i]], 'size'] += 1
    size = np.array(citys['size'])/ np.max(citys['size'])
    citys.loc[:,'size'] = pd.Series(size*10)
    citys['alpha'] = pd.Series(size)
    if isLoad==False:
        isLoad=True
    else:
        plot.tools.pop()
        plot.tools.pop()
        plot.renderers.pop()

    

    nodes = hv.Nodes(citys,['x','y','index'],vdims=['name','line','size','alpha'])
    graph = hv.Graph(((flow_), nodes), vdims=['flow','origin','destination'])
    graph.opts(node_size='size', edge_line_width=0.1,node_color='line',cmap=colors,node_alpha='alpha')
    bundled = bundle_graph(graph,initial_bandwidth=0.05, decay=0.3) 

    hvplot = renderer.get_plot(bundled)



    plot.renderers.append(hvplot.state.renderers[0])
    tap_tool = TapTool(renderers=[hvplot.state.renderers[0]])
    hover_tool = HoverTool(renderers = [hvplot.state.renderers[0]],tooltips=[("name", "@name"),("line","@line")])
    plot.add_tools(hover_tool,tap_tool)
    

LABELS = ["一线城市", "二线城市", "三线城市", "四线城市", "五线城市"]

checkbox_button_group1 = CheckboxButtonGroup(labels=LABELS)
checkbox_button_group2 = CheckboxButtonGroup(labels=LABELS)
get_button = Button(label='Get')
get_button.on_click(update)
# 导入数据
citys_lines = np.load('data/citys_lines.npy',allow_pickle=True)
# 把地图数据读进来
connect = create_engine(
f'postgresql://postgres:774165@127.0.0.1:5432/first')
nine_line = gpd.read_postgis('select * from nine_line_3857',con=connect,geom_col='geometry')
province = gpd.read_postgis('select * from province_3857',con=connect, geom_col='geometry')
citys = gpd.read_postgis('select * from citys_3857', con=connect, geom_col='geometry')
sql_query = "select * from "
pg_table_name = "edge_list"
connect = psycopg2.connect(database="first", user="postgres", password="774165", host="127.0.0.1", port="5432")
try:
    flow = pd.read_sql_query(sql_query+pg_table_name, con=connect)
except Exception as e:
    print(F'查询失败，详情:{e}')
finally:
    connect.close()
flow.columns = ['origin','destination','flow']
# 构造查询字典
city2id = {}
id2city = {}
for i in range(len(citys)):
    city2id[citys['name'][i]] = citys['index'][i]
    id2city[citys['index'][i]] = citys['name'][i]

geosource_nineline = GeoJSONDataSource(geojson = nine_line.to_json())
geosource_province = GeoJSONDataSource(geojson = province.to_json())
plot = figure()
plot.axis.visible = False
plot.grid.visible = False
plot.patches('xs','ys', source = geosource_province,
                            fill_color = None,
                            line_color = 'grey', 
                            line_width = 1, 
                            fill_alpha = 1)
plot.patches('xs','ys', source = geosource_nineline,
                            fill_color = None,
                            line_color = 'gray', 
                            line_width = 5, 
                            fill_alpha = 1)

def name_line(name):
    if name in citys_lines[0]:
        return 1
    elif name in citys_lines[1]:
        return 2
    elif name in citys_lines[2]:
        return 3
    elif name in citys_lines[3]:
        return 4
    elif name in citys_lines[4]:
        return 5
    else:
        return 5

# 数据格式修改
citys['x'] = citys['geometry'].apply(lambda r: r.x)
citys['y'] = citys['geometry'].apply(lambda r: r.y)
citys['line'] = citys['name'].apply(name_line)
citys = citys[['x','y','index','name','line']]
flow['source'] = flow['origin'].apply(lambda r: city2id[r])
flow['target'] = flow['destination'].apply(lambda r: city2id[r])
flow['origin_line'] = flow['origin'].apply(name_line)
flow['destination_line'] = flow['destination'].apply(name_line)

from numpy import NaN
S = np.load('data/S.npy')
SS = []
for i in range(362):
    for j in range(362):
        if S[i][j] == NaN:
            continue
        SS.append(S[i][j])

flow['S'] = pd.Series(SS)


# # 数据筛选
# def update():00 
#     flow_ = flow[(flow['S']>=50) & (((flow['origin_line']==1) & (flow['destination_line']==2)) | ((flow['origin_line']==2) & (flow['destination_line']==1)))].reset_index(drop=True)

#     citys['size'] = pd.Series([0]*362)
#     for i in range(len(flow_)):
#         citys.loc[city2id[flow_['origin'][i]],'size'] += 0.5
#         citys.loc[city2id[flow_['destination'][i]], 'size'] += 0.5
    
# flow_ = flow
# datashader
colors = ['#000000']+hv.Cycle('Category20').values
kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
# nodes = hv.Nodes(citys,['x','y','index'],vdims=['name','line','size'])

renderer = hv.renderer('bokeh')
renderer = renderer.instance(mode='server')

# graph = hv.Graph(((flow_), nodes), vdims=['flow','origin','destination'])
# graph.opts(node_size='size', edge_line_width=0.1,node_color='line',cmap=colors)
# tooltips = [
#     ('flow', '@flow'),
#     ('origin', '@origin'),
#     ('destination', '@destination')
# ]
# hover = HoverTool(tooltips=tooltips)
# bundled = bundle_graph(graph,initial_bandwidth=0.05, decay=0.3)

# hvplot = renderer.get_plot(bundled)
# plot.renderers.append(hvplot.state.renderers[0])
# tap_tool = TapTool(renderers=[hvplot.state.renderers[0]])
# hover_tool = HoverTool(renderers = [hvplot.state.renderers[0]],tooltips=[("name", "@name"),("line","@line")])
# plot.add_tools(hover_tool,tap_tool)
curdoc().add_root(row(column(checkbox_button_group1,checkbox_button_group2,plot),get_button))
# (datashade(bundled, normalization='linear', width=800, height=800) * bundled.nodes).opts(
#     opts.Nodes(color='line', size=10, width=1000, cmap=colors, legend_position='right'))