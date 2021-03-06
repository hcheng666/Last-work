''' A column salary chart with minimum and maximum values.
This example shows the capability of exporting a csv file from ColumnDataSource.

'''
from os.path import dirname, join
import geopandas as gpd
import numpy as np
import psycopg2
import pandas as pd
import base64
from sqlalchemy import create_engine
import re

from bokeh.events import ButtonClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.palettes import OrRd9, Spectral4
from bokeh.transform import linear_cmap
from bokeh.plotting import figure
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn, Spinner, FileInput, Dropdown,
                          TextInput, GeoJSONDataSource, HoverTool, MultiLine, Circle, TapTool)
# 定义变量
df = pd.DataFrame()
source = ColumnDataSource(data=dict())
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
mapper = linear_cmap(field_name="flow", palette=OrRd9, low=0, high=0)
# 消息按钮
messageButton = Button(label="message",name=str(0),visible=False)
messageButton.js_on_change("name", CustomJS(args=dict(type='success'),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))


# 文件按钮，包括两个操作，从数据库导入，从本地文件选择
# 数据库导入按钮
database_button = Button(visible=False, name='Database')
def load_data(event):
    global isLoad,df,nine_line,province,citys,geosource_citys,geosource_nineline,geosource_province,city2id,id2city
    sql_query = "select * from "
    pg_table_name = "edge_list"
    connect = psycopg2.connect(database="first", user="postgres", password="774165", host="127.0.0.1", port="5432")
    try:
        df = pd.read_sql_query(sql_query+pg_table_name, con=connect)
    except Exception as e:
        print(F'查询失败，详情:{e}')
    finally:
        connect.close()
    df.columns  = ['origin', 'destination', 'flow']
    df['flow'] = df['flow'].apply(lambda r: float(r))
    df = df.head(10000)
    # 每次加载文件要清空查询和筛选组件
    searchTextOrigin.value = ''
    searchTextDestination.value = ''
    spinnerMin.value = 1
    spinnerMax.value = 10
    
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
    
    update()
    isLoad = True
    messageButton.name = str(int(messageButton.name)+1)
database_button.on_event(ButtonClick, load_data)


# 本地导入按钮
fileinput = FileInput(visible=False, name='FileInput')
def upload_csv_to_server(attr, old, new):
    global df 

    #decode base64 format (from python base24 website)
    base64_message = fileinput.value
    base64_bytes = base64_message.encode('utf-8')
    message_bytes = base64.b64decode(base64_bytes)
    message = message_bytes.decode('utf-8')

    #convert string to csv and save it on the server side
    message_list = message.splitlines()
    message = [mes.split(',') for mes in message_list]
    
    df = pd.DataFrame(message[1:10000])
    
    df.columns = ['origin', 'destination', 'flow']
    df['flow'] = df['flow'].apply(lambda r: float(r))
    #print(df.columns)
    # 每次加载文件要清空查询和筛选组件
    searchTextOrigin.value = ''
    searchTextDestination.value = ''
    spinnerMin.value = 1
    spinnerMax.value = 10
    update()
    messageButton.name = str(int(messageButton.name)+1)
fileinput.on_change('filename', upload_csv_to_server)


menu = [('Open File','1'), ('Load From Database','2')]
dropdown = Dropdown(label='File', menu=menu, name='Dropdown', tags=['hi'])

dropdown.js_on_click(CustomJS(code="""
    if (this.item == '1') {
        document.getElementById('FileInputElement').querySelector('input[type=file]').dispatchEvent(new MouseEvent('click'))
    }
    else {
        document.getElementById('DataBaseElement').querySelector('button[type=button]').dispatchEvent(new MouseEvent('click'))
    }
    """))
# ----------------------------------------------------------------------------------------
# def function_to_call(attr, old, new):
#     print dropdown.value

# dropdown.on_change('value', function_to_call)
# dropdown = Dropdown(label='File', menu=['Open'], name='Dropdown', tags=['hi'])

# # Set up callback for Dropdown widget
# dropdown.js_on_click(CustomJS(code="""
#     if (this.item == 'Open') {
#         document.getElementById('FileInputElement').querySelector('input[type=file]').dispatchEvent(new MouseEvent('click'))
#     }
#     """))
#--------------------------------------------------

# 查找和筛选组件
models = []
searchTextOrigin = TextInput(value = '', title='Origin: ')
searchTextDestination = TextInput(value = '', title='Destination: ')
spinnerMin = Spinner(title="MinFlow", low=0, high=100, step=1, value=1, format="0,0.000")
spinnerMax = Spinner(title="MaxFlow", low=0, high=100, step=1, value=10, format="0,0.000")
models.append(searchTextOrigin)
models.append(searchTextDestination)
models.append(spinnerMin)
models.append(spinnerMax)

# 查找和筛选更新函数    待更新：精确查找和模糊查找
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
    global mapper
    current = df[(df['flow'] >= spinnerMin.value) & (df['flow'] <= spinnerMax.value)].dropna().reset_index(drop=True)
    if searchTextOrigin.value!='':
        current = current[fuzzyfinder(searchTextOrigin.value,current['origin'])].reset_index(drop=True)
    if searchTextDestination.value!='':
        current = current[fuzzyfinder(searchTextDestination.value,current['destination'])].reset_index(drop=True)
    if not current.empty:
        # 数据预处理，为构造OD数据做准备
        coordinate = citys['geometry'].apply(lambda r: (r.x,r.y))
        points = list(coordinate) # 点数据
        edges = [] # 边数据
        for i in range(len(current)):
            edges.append((city2id[current['origin'][i]], city2id[current['destination'][i]]))

        # 构造画线的数据
        # x, y, namex, namey, flow
        xs = []
        ys = []
        ori_name = []
        des_name = []

        for i in range(len(current)):
            # if edges[i][2]<3:
            #     continue
            xs_i = [0]*2
            ys_i = [0]*2
            xs_i[0] = points[edges[i][0]][0]
            xs_i[1] = points[edges[i][1]][0]
            ys_i[0] = points[edges[i][0]][1]
            ys_i[1] = points[edges[i][1]][1]
            xs.append(xs_i.copy())
            ys.append(ys_i.copy())
            ori_name.append(current['origin'][i])
            des_name.append(current['destination'][i])

        # 地图数据和线数据
        mapper = linear_cmap(field_name="flow", palette=tuple(reversed(OrRd9)), low=min(current['flow']), high=max(current['flow']))
        alpha = current['flow']/np.quantile(current['flow'],0.999,interpolation='lower')
        alpha = [a if a<=1 else 1 for a in alpha]
        width = np.array(alpha) * 2
        source.data = {
            'origin'            : current.origin,
            'destination'       : current.destination,
            'flow'              : current.flow,
            'xs'                : xs,
            'ys'                : ys,
            'alpha'             : alpha,
            'width'             : width
        }
    
        if isLoad == False:
            p.patches('xs','ys', source = geosource_nineline,
                            fill_color = None,
                            line_color = 'gray', 
                            line_width = 5, 
                            fill_alpha = 1)
            p.patches('xs','ys', source = geosource_province,
                            fill_color = None,
                            line_color = 'grey', 
                            line_width = 1, 
                            fill_alpha = 1)
            citys_renderer = p.circle(x='x', y='y', size=3, color='#46A3FF', alpha=0.7,
             hover_color=Spectral4[1],selection_color=Spectral4[2], source=geosource_citys)
            p.add_tools(HoverTool(renderers = [citys_renderer],
                            tooltips = [('name','@name'),
                                        ]))
            p.add_tools(TapTool(renderers = [citys_renderer]))
            #citys_renderer.selection_glyph = Circle(size=10, fill_color=Spectral4[2])
            #citys_renderer.hover_glyph = Circle(size=10, fill_color=Spectral4[1])
        else:
            if len(p.renderers) == 4:
                p.tools.pop(-1)
                p.tools.pop(-1)
                p.renderers.pop(-1)
                
        lines = p.multi_line('xs', 'ys', source=source,line_alpha='alpha',line_color=mapper,line_width='width',
        hover_line_color = Spectral4[1],selection_line_color=Spectral4[2],selection_line_width=3, hover_line_width=3,
        hover_line_alpha=1,selection_line_alpha=1)
        p.add_tools(HoverTool(renderers = [lines],tooltips = [('origin','@origin'),('destination','@destination'),('flow','@flow')]))
        p.add_tools(TapTool(renderers = [lines]))
        #lines.selection_glyph =  MultiLine(line_color=Spectral4[2], line_width=10)
        #lines.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=10)
    else:
        p.tools.pop(-1)
        p.tools.pop(-1)
        p.renderers.pop(-1)
        source.data = {}
        
# 绑定事件
for model in models:
    model.on_change('value', lambda attr, old, new: update())

# def update():
#     #df = pd.DataFrame(source_row.data)
#     current = df[(df['flow'] >= slider.value[0]) & (df['flow'] <= slider.value[1])].dropna()
#     source.data = {
#         'origin'            : current.origin,
#         'destination'       : current.destination,
#         'flow'              : current.flow,
#     }

# slider = RangeSlider(title="Max flow", start=0, end=100, value=(0, 100), step=0.5, format="0,0")
# slider.on_change('value', lambda attr, old, new: update())



# def update2():
#     #df = pd.DataFrame(source_row.data)
#     current = df[(df['flow'] >= spinnerMin.value) & (df['flow'] <= spinnerMax.value)].dropna()
#     source.data = {
#         'origin'            : current.origin,
#         'destination'       : current.destination,
#         'flow'              : current.flow,
#     }
# spinnerMin = Spinner(title="MinSalary", low=0, high=100, step=1, value=1, format="0,0.000")
# spinnerMin.on_change('value',lambda attr, old, new: update2())
# spinnerMax = Spinner(title="MaxSalary", low=0, high=100, step=1, value=10, format="0,0.000")
# spinnerMax.on_change('value',lambda attr, old, new: update2())

# 下载按钮组件
DownloadButton = Button(label="Download", button_type="success")
DownloadButton.js_on_event("button_click", CustomJS(args=dict(source=source),
                            code=open(join(dirname(__file__), "download.js")).read()))

# -------------------------------------------------------------------------------------
# 地图展示模块
# # 数据导入
# connect = create_engine(
#     f'postgresql://postgres:774165@127.0.0.1:5432/first')
# nine_line = gpd.read_postgis('select * from nine_line_3857',con=connect,geom_col='geometry')
# province = gpd.read_postgis('select * from province_3857',con=connect, geom_col='geometry')
# citys = gpd.read_postgis('select * from citys_3857', con=connect, geom_col='geometry')

# # 数据预处理，为构造OD数据做准备
# coordinate = citys['geometry'].apply(lambda r: (r.x,r.y))
# points = list(coordinate) # 点数据
# edges = [] # 边数据
# edge_list = df
# for i in range(362):
#     for j in range(361):
#         k = j+1 if i==j else j
#         #edges.append((i,k,edge_list['flow'][i*361+j]))
#         edges.append((i,k))

# # 构造画线的数据
# # x, y, namex, namey, flow
# xs = []
# ys = []
# flows = []
# ori_name = []
# des_name = []

# for i in range(len(edge_list)):
#     # if edges[i][2]<3:
#     #     continue
#     xs_i = [0]*2
#     ys_i = [0]*2
#     xs_i[0] = points[edges[i][0]][0]
#     xs_i[1] = points[edges[i][1]][0]
#     ys_i[0] = points[edges[i][0]][1]
#     ys_i[1] = points[edges[i][1]][1]
#     xs.append(xs_i.copy())
#     ys.append(ys_i.copy())
#     flows.append(edge_list['flow'][i])
#     ori_name.append(edge_list['origin'][i])
#     des_name.append(edge_list['destination'][i])

# # 地图数据和线数据
# mapper = linear_cmap(field_name="flow", palette=OrRd9, low=min(flows), high=max(flows))
# alpha = flows/np.quantile(flows,0.999,interpolation='lower')
# alpha = [a if a<=1 else 1 for a in alpha]
# width = np.array(alpha) * 2
# multi_line_source = ColumnDataSource({
#     'xs': xs,
#     'ys': ys,
#     'flow': flows,
#     'alpha': alpha,
#     'width': width
# })



# 画图
p = figure(background_fill_color="lightgrey",tools = "pan,wheel_zoom,zoom_in,zoom_out,reset")
p.axis.visible = False
p.grid.visible = False


#widget
columns = [
    TableColumn(field="origin", title="Origin"),
    TableColumn(field="destination", title="Destination"),
    TableColumn(field="flow", title="Flow", formatter=NumberFormatter(format="0,0.000"))
]

data_table = DataTable(source=source, columns=columns, width=500)
#spinner = row(spinnerMin,spinnerMax)
#controls = column(slider, DownloadButton, spinner)
searchFilter = column(row(spinnerMin,spinnerMax),row(searchTextOrigin,searchTextDestination),messageButton)
rows = row(column(searchFilter,data_table,DownloadButton), p)
rows.name = "Row"
curdoc().add_root(fileinput)
curdoc().add_root(database_button)
curdoc().add_root(dropdown)
curdoc().add_root(rows)
curdoc().title = "Export CSV"

# update()
