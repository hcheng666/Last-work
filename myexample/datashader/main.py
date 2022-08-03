# import holoviews as hv
# from holoviews import opts
# import holoviews.operation.datashader as hd
from bokeh.plotting import curdoc, figure
from bokeh.models import (HoverTool,ColumnDataSource,MultiLine,EdgesAndLinkedNodes, Circle,TapTool,Button, 
                            CustomJS,Spinner,TextInput,GeoJSONDataSource,DataTable,NumberFormatter,TableColumn,
                            GraphRenderer, StaticLayoutProvider)
from bokeh.transform import linear_cmap
from bokeh.palettes import OrRd9,Spectral4,Reds8
from bokeh.layouts import column ,row
from bokeh.events import ButtonClick
# hd.shade.cmap=["lightblue", "darkblue"]
# hv.extension("bokeh", "matplotlib") 
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import numpy as np
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from os.path import dirname, join
# 全局变量
flow = pd.DataFrame()
nine_line = gpd.GeoDataFrame()
province = gpd.GeoDataFrame()
citys = gpd.GeoDataFrame()
geosource_nineline = GeoJSONDataSource()
geosource_province = GeoJSONDataSource()
geosource_citys = GeoJSONDataSource()
city2id = {}
id2city = {}

# 判断数据是否导入
isLoad  = False

# 消息按钮
messageButton = Button(label="message",name=str(0),visible=False)
messageButton.js_on_change("name", CustomJS(args=dict(type='success'),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))

# 数据导入按钮
database_button = Button(visible=True, name='Database')
def load_data(event):
    global isLoad,flow,nine_line,province,citys,geosource_citys,geosource_nineline,geosource_province,city2id,id2city
    sql_query = "select * from "
    pg_table_name = "edge_list"
    connect = psycopg2.connect(database="first", user="postgres", password="774165", host="127.0.0.1", port="5432")
    try:
        flow = pd.read_sql_query(sql_query+pg_table_name, con=connect)
    except Exception as e:
        print(F'查询失败，详情:{e}')
    finally:
        connect.close()
    # 每次加载文件要清空查询和筛选组件
    searchTextOrigin.value = ''
    searchTextDestination.value = ''
    spinnerMin.value = 0.1
    spinnerMax.value = 50
    
    # 把地图数据读进来
    connect = create_engine(
    f'postgresql://postgres:774165@127.0.0.1:5432/first')
    nine_line = gpd.read_postgis('select * from nine_line_3857',con=connect,geom_col='geometry')
    province = gpd.read_postgis('select * from province_3857',con=connect, geom_col='geometry')
    citys = gpd.read_postgis('select * from citys_3857', con=connect, geom_col='geometry')
    geosource_nineline = GeoJSONDataSource(geojson = nine_line.to_json())
    geosource_province = GeoJSONDataSource(geojson = province.to_json())
    geosource_citys = GeoJSONDataSource(geojson = citys.to_json())

    # 构造查询字典
    for i in range(len(citys)):
        city2id[citys['name'][i]] = citys['index'][i]
        id2city[citys['index'][i]] = citys['name'][i]
    
    # 构造边数据
    flow.columns = ['origin','destination','flow']
    flow.loc[:,'flow'] = flow['flow'].apply(lambda r: float(r))
    flow['start'] = flow['origin'].apply(lambda r: city2id[r])
    flow['end'] = flow['destination'].apply(lambda r: city2id[r])
    alpha = flow['flow']/np.quantile(flow['flow'],0.99,interpolation='lower')
    flow['alpha'] = pd.Series([a if a<=1 else 1 for a in alpha])
    flow['width'] = flow['alpha'].apply(lambda r:r*2)
    graph.node_renderer.data_source.add(citys['index'], 'index')
    graph_layout = dict(zip(citys['index'], zip(citys['geometry'].apply(lambda r: r.x), citys['geometry'].apply(lambda r: r.y))))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    update()
    isLoad = True
    
    messageButton.name = str(int(messageButton.name)+1)
database_button.on_event(ButtonClick, load_data)

# 模糊查找函数
def fuzzyfinder(user_input, collection):
        suggestions = []
        pattern = '.*'.join(user_input) # Converts 'djm' to 'd.*j.*m'
        regex = re.compile(pattern)     # Compiles a regex.
        for item in collection:
            match = regex.search(item)  # Checks if the current item matches the regex.
            if match:
                suggestions.append(True)
            else :
                suggestions.append(False)
        return suggestions

def update():
    flow_ = flow[(flow['flow'] >= spinnerMin.value) & (flow['flow'] <= spinnerMax.value)]
    if searchTextOrigin.value!='':
        flow_ = flow_[fuzzyfinder(searchTextOrigin.value,flow_['origin'])]
    if searchTextDestination.value!='':
        flow_ = flow_[fuzzyfinder(searchTextDestination.value,flow_['destination'])]
    mapper = linear_cmap(field_name="flow", palette=Reds8, low=min(flow_['flow']), high=max(flow_['flow']))
    alpha = flow_['flow']/np.max(flow_['flow'])#np.quantile(flow_['flow'],0.999,interpolation='lower')
    flow_.loc[:,'alpha'] = pd.Series([a if a<=1 else 1 for a in alpha])
    flow_.loc[:,'width'] = flow['alpha'].apply(lambda r:r*2)
    # edge_source.data = flow_.to_dict(orient='dict')
    edge_source.data = {
        'index'             : flow_.index,
        'origin'            : flow_.origin,
        'destination'       : flow_.destination,
        'flow'              : flow_.flow,
        'start'             : flow_.start,
        'end'               : flow_.end,
        'alpha'             : flow_.alpha,
        'width'             : flow_.width
    }
    graph.edge_renderer.glyph = MultiLine(line_color=mapper, line_alpha='alpha', line_width='width')
    if isLoad==False:
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
        
        plot.renderers.append(graph)
        plot.add_tools(hover_tool, TapTool())

# 查找和筛选组件
models = []
searchTextOrigin = TextInput(value = '', title='Origin: ')
searchTextDestination = TextInput(value = '', title='Destination: ')
spinnerMin = Spinner(title="MinFlow", low=0, high=100, step=1, value=0.1, format="0,0.000")
spinnerMax = Spinner(title="MaxFlow", low=0, high=100, step=1, value=50, format="0,0.000")
models.append(searchTextOrigin)
models.append(searchTextDestination)
models.append(spinnerMin)
models.append(spinnerMax)

# 绑定事件
for model in models:
    model.on_change('value', lambda attr, old, new: update())

# datashader

# nodes = hv.Nodes(citys,['x','y','id'],['name'])
# graph = hv.Graph(((flow), nodes), vdims=['flow','origin','destination','alpha','width'])
# tooltips = [
#     ('flow', '@flow'),
#     ('origin', '@origin'),
#     ('destination', '@destination')
# ]
# hover = HoverTool(tooltips=tooltips)
# graph.opts(inspection_policy='edges').opts(
#     opts.Graph(tools=[hover,'tap'],node_size=3, edge_cmap='blues', edge_color='flow',edge_alpha='alpha', edge_line_width='width',
#     edge_hover_color='green',edge_hover_line_width=5,edge_hover_line_alpha=1,
#     edge_selection_line_width=5,edge_selection_line_color='red',edge_selection_line_alpha=1,
#     node_selection_color='red',edge_visible=False,node_visible=False))
# renderer = hv.renderer('bokeh')
# # renderer = renderer.instance(mode='server')
# # hvplot = renderer.get_plot(graph)
# # show(hvplot.state)
# doc = renderer.server_doc(graph)
 
 # 流量图
plot = figure(title="Graph Layout Demonstration")
plot.axis.visible = False
plot.grid.visible = False
# node_source = ColumnDataSource(citys)
#edge_source = ColumnDataSource(flow)
edge_source = ColumnDataSource(data=dict())
graph = GraphRenderer()
graph.edge_renderer.data_source = edge_source
graph.node_renderer.selection_glyph = Circle(size=15, fill_color='green')
graph.node_renderer.hover_glyph = Circle(size=15, fill_color='red')
graph.edge_renderer.selection_glyph = MultiLine(line_color='green', line_width=5, line_alpha=1)
graph.edge_renderer.hover_glyph = MultiLine(line_color='red', line_width=5,line_alpha=1)
graph.inspection_policy = EdgesAndLinkedNodes()
graph.selection_policy = EdgesAndLinkedNodes()
hover_tool = HoverTool(tooltips=[("origin", "@origin"), ("destination", "@destination"), ("flow",'@flow')])


# 数据表
columns = [
    TableColumn(field="origin", title="Origin"),
    TableColumn(field="destination", title="Destination"),
    TableColumn(field="flow", title="Flow", formatter=NumberFormatter(format="0,0.000"))
]
data_table = DataTable(source=edge_source, columns=columns, width=500)


    #button.label = str(int(button.label)+1)
#button.on_event(ButtonClick, update)

curdoc().add_root(column(database_button,row(spinnerMin,spinnerMax),row(searchTextOrigin,searchTextDestination),row(data_table,plot)))