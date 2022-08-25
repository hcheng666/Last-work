import holoviews as hv
from holoviews import opts
import holoviews.operation.datashader as hd
from holoviews.operation.datashader import datashade, bundle_graph
from bokeh.plotting import curdoc, figure
from bokeh.models import (HoverTool,ColumnDataSource,MultiLine,EdgesAndLinkedNodes, Circle,TapTool,Button, 
                            CustomJS,Spinner,TextInput,GeoJSONDataSource,DataTable,NumberFormatter,TableColumn,
                            GraphRenderer, StaticLayoutProvider,Select,CheckboxButtonGroup)
from bokeh.transform import linear_cmap
from bokeh.palettes import OrRd9,Spectral4,Reds8
from bokeh.layouts import column ,row
from bokeh.events import ButtonClick



from holoviews import opts

hv.extension("bokeh")
renderer = hv.renderer('bokeh')
renderer = renderer.instance(mode='server')
hd.shade.cmap=["grey"]
hv.extension("bokeh", "matplotlib") 

import re
import pandas as pd
import numpy as np
from numpy import random
import geopandas as gpd
import numpy as np
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from os.path import dirname, join
from numpy import NaN

import tensorflow as tf
from tensorflow.keras import layers
from sklearn import linear_model
import nest_asyncio
import multiprocessing
nest_asyncio.apply()

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

#Part1 数据展示和导入-----------------------------------------------------------------------------------------------------
# 判断数据是否导入
isLoad  = False

# 消息按钮    根据改变的不同设置不同的消息
messageButton = Button(label=str(0),name=str(0),visible=False)
# 数据加载消息
messageButton.js_on_change("name", CustomJS(args=dict(type='success',content='数据加载成功！',duration=800),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))
# 开始提示消息
messageButton.js_on_change("label", CustomJS(args=dict(type='success',content='数据更新成功！',duration=500),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))

# 数据导入按钮
database_button = Button(visible=True, name='Database',label='从数据库导入数据')
database_button.js_on_click(CustomJS(args=dict(type='warning',content='数据正在加载中，请稍等',duration=3000),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))

def load_data(event):
    #  数据加载太慢了，想想怎么优化
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
    # searchTextOrigin.value = ''
    # searchTextDestination.value = ''
    # spinnerMin.value = 0.1
    # spinnerMax.value = 50
    
    if isLoad==False:
        # 把地图数据读进来
        connect = create_engine(
        f'postgresql://postgres:774165@127.0.0.1:5432/first')
        nine_line = gpd.read_postgis('select * from nine_line_3857',con=connect,geom_col='geometry')
        province = gpd.read_postgis('select * from province_3857',con=connect, geom_col='geometry')
        citys = gpd.read_postgis('select * from citys_3857', con=connect, geom_col='geometry')
        #province['geometry'] = province.geometry.boundary
        geosource_nineline = GeoJSONDataSource(geojson = nine_line.to_json())
        geosource_province = GeoJSONDataSource(geojson = province.to_json())
        geosource_citys = GeoJSONDataSource(geojson = citys.to_json())

        # 构造查询字典
        for i in range(len(citys)):
            city2id[citys['name'][i]] = citys['index'][i]
            id2city[citys['index'][i]] = citys['name'][i]
        citys['x'] = citys['geometry'].apply(lambda r: r.x)
        citys['y'] = citys['geometry'].apply(lambda r: r.y)
        citys['line'] = citys['name'].apply(name_line)
        citys = citys[['x','y','index','name','line']]
        graph.node_renderer.data_source.add(citys['index'], 'index')
        graph_layout = dict(zip(citys['index'], zip(citys['x'], citys['y'])))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    # 构造边数据
    flow.columns = ['origin','destination','flow']
    flow.loc[:,'flow'] = flow['flow'].apply(lambda r: float(r))
    flow['start'] = flow['origin'].apply(lambda r: city2id[r])
    flow['end'] = flow['destination'].apply(lambda r: city2id[r])
    alpha = flow['flow']/np.quantile(flow['flow'],0.99,interpolation='lower')
    flow['alpha'] = pd.Series([a if a<=1 else 1 for a in alpha])
    flow['width'] = flow['alpha'].apply(lambda r:r*2)
    # flow['source'] = flow['origin'].apply(lambda r: city2id[r])
    # flow['target'] = flow['destination'].apply(lambda r: city2id[r])
    flow['origin_line'] = flow['origin'].apply(name_line)
    flow['destination_line'] = flow['destination'].apply(name_line)

    S = np.load('data/S.npy')
    SS = []
    for i in range(362):
        for j in range(362):
            if S[i][j] == NaN:
                continue
            SS.append(S[i][j])

    flow['S'] = pd.Series(SS)
    
    update()
    isLoad = True
    messageButton.name = str(int(messageButton.name)+1)
    #showMessage(tp='success', ct='数据加载成功！', dur=800)
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

def update():
    flow_ = flow[(flow['flow'] >= spinnerMin.value) & (flow['flow'] <= spinnerMax.value)].copy()
    if searchTextOrigin.value!='':
        flow_ = flow_[fuzzyfinder(searchTextOrigin.value,flow_['origin'])]
    if searchTextDestination.value!='':
        flow_ = flow_[fuzzyfinder(searchTextDestination.value,flow_['destination'])]
    if flow_.empty:
        # message
        return 
    mapper = linear_cmap(field_name="flow", palette=Reds8, low=min(flow_['flow']), high=max(flow_['flow']))
    alpha = flow_['flow']/np.max(flow_['flow'])#np.quantile(flow_['flow'],0.999,interpolation='lower')
    flow_.loc[:,'alpha'] = pd.Series([a if a<=1 else 1 for a in alpha]).values
    flow_.loc[:,'width'] = flow_['alpha'].apply(lambda r:r*2)
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
        # plot.patches('xs','ys', source = geosource_province,
        #                     fill_color = None,
        #                     line_color = 'grey', 
        #                     line_width = 1, 
        #                     fill_alpha = 1)
        dd = hv.Polygons(province).opts(line_alpha=0,fill_color='#CDCDCD')
        dd2 = datashade(hv.Path(province))
        #dd2 = datashade(hv.Path(province).opts(opts.Path(color='grey')))
        hvplot = renderer.get_plot(dd)
        hvplot2 = renderer.get_plot(dd2)
        plot.renderers.append(hvplot.state.renderers[0])
        plot.renderers.append(hvplot2.state.renderers[0])
        plot.patches('xs','ys', source = geosource_nineline,
                            fill_color = None,
                            line_color = 'gray', 
                            line_width = 5, 
                            fill_alpha = 1)
        
        plot.renderers.append(graph)
        plot.add_tools(tap_tool)
    messageButton.label = str(int(messageButton.label)+1)

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
#hover_tool = HoverTool(tooltips=[("origin", "@origin"), ("destination", "@destination"), ("flow",'@flow')])
# hover_tool = HoverTool(renderers = [graph],tooltips = [('origin','@origin'),('destination','@destination'),('flow','@flow')])
# tap_tool = TapTool(renderers=[graph])
#hover_tool = HoverTool(tooltips = [('origin','@origin'),('destination','@destination'),('flow','@flow')])
tap_tool = TapTool()
# 数据表
columns = [
    TableColumn(field="origin", title="Origin"),
    TableColumn(field="destination", title="Destination"),
    TableColumn(field="flow", title="Flow", formatter=NumberFormatter(format="0,0.000"))
]
data_table = DataTable(source=edge_source, columns=columns, width=500)

# 下载按钮组件
DownloadButton = Button(label="Download", button_type="success")
DownloadButton.js_on_event("button_click", CustomJS(args=dict(source=edge_source),
                            code=open(join(dirname(__file__), "download.js")).read()))
#-----------------------------------------------------------------------------------------------------

#Part2 模型训练-----------------------------------------------------------------------------------------------------
flows = np.load(r"data/data_avg.npy")
citys_name = pd.read_csv(r"data/citys_coordiante.csv")
geo_distance = np.load(r'data/geo_distance.npy')
n = len(flows)
# 流量标准化函数
def T_ij(flows, m):
    # 对流量进行标准化
    flow_re = np.zeros_like(flows)
    j_sum = flows.sum(axis=1)
    for i in range(n):
        flow_re[i,:]= flows[i,:] / j_sum[i] * m[i]
    return flow_re
# 相似度计算函数
def cos_sim(vec1, vec2):
    # cosine simility
    dot = vec1.dot(vec2)
    L2 = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    return dot / L2

def dot_sim(vec1, vec2):
    # dot simility
    dot = vec1.dot(vec2)
    return dot

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, negative_sample_size):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=negative_sample_size+1,
                                       name="w2v_context")

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


# 提前数据准备
m = flows.sum(axis=1)
# 让每一行的数据归一化
data = pd.DataFrame(flows)
data = data.apply(lambda r:r/r.sum(), axis=1)

# 再计算累积分布
data_cum = np.array(data)
for i in range(1, data_cum.shape[1]):
    data_cum[:,i] = data_cum[:, i-1]+data_cum[:,i]
#pd.DataFrame(data_cum)

# 起点
start_cum= m / m.sum()
for i in range(1, start_cum.shape[0]):
    start_cum[i] = start_cum[i]+start_cum[i-1]

np.random.seed(486626)

# 调参应用 参数有：1 目标数量， 2 每个目标应用的负样本数量（正样本数量默认为1）3 向量维度 4 训练迭代次数
spinner_target = Spinner(title="Target Size:", low=10000, high=120000, step=5000, value=60000, format="0,0")
spinner_negative = Spinner(title="Negative Sample Size:", low=1, high=10, step=1, value=5)
spinner_dimension = Spinner(title="Dimension Size:", low=10, high=150, step=10, value=80)
spinner_epoch = Spinner(title="Epochs:", low=1, high=5, step=1, value=1)
button_train = Button(label='Get Vector',button_type="primary")
#searchCity = TextInput(value = '', title='Chose city: ')

def train():
    # 获取参数
    target_size = spinner_target.value
    negative_sample_size = spinner_negative.value
    vocab_size = len(m)
    # 正样本采样
    targets = [] #(n,)
    postive_context = [] #(n,)
    starts = random.uniform(size=(target_size,2))
    for i in range(target_size):
        targets.append(np.searchsorted(start_cum, starts[i][0]))
        postive_context.append(np.searchsorted(data_cum[targets[i]], starts[i][1]))
    
    data_reconstruct = np.zeros((n,n),dtype=float)
    for i in range(target_size):
        data_reconstruct[targets[i]][postive_context[i]]+=1
    # 流量标准化
    data_reconstruct_re = T_ij(data_reconstruct, m)
    np.save('data_reconstruct_re.npy',data_reconstruct_re)
    # 画出相关性图
    # if p1.renderers != []:
    #     p1.renderers.pop()
    # p1_source.data = {
    #     'x': flow.flatten(),
    #     'y': data_reconstruct_re.flatten()
    # }
    # p1.circle(y='y', x='x',color='red',size=3,source=p1_source)
        
    start_cum_negative = np.array([m]*vocab_size)
    start_cum_negative = np.where(data_reconstruct_re==0, start_cum_negative, 0)
    for i in range(vocab_size):
        start_cum_negative[i][i]=0
    # 让每一行的数据归一化
    start_cum_negative = pd.DataFrame(start_cum_negative)
    start_cum_negative = start_cum_negative.apply(lambda r:r/r.sum(), axis=1)

    # 再计算累积分布
    start_cum_negative = np.array(start_cum_negative)
    for i in range(1, start_cum_negative.shape[1]):
        start_cum_negative[:,i] = start_cum_negative[:, i-1]+start_cum_negative[:,i]

    # 负样本采样
    
    negative_context = []
    rand = random.uniform(size=(target_size,negative_sample_size))
    for i in range(target_size):
        negative_context_i = []
        for j in range(negative_sample_size):
            negative_context_temp = np.searchsorted(start_cum_negative[targets[i]], rand[i][j])
            negative_context_i.append(negative_context_temp)
        negative_context.append(negative_context_i)
    # 构建完整训练集
    labels = []
    contexts = []
    for i in range(target_size):
        contexts.append([postive_context[i]] + negative_context[i])
        labels.append([1]+[0]*negative_sample_size)

    # 训练数据预处理
    SEED = 486626
    AUTOTUNE = tf.data.AUTOTUNE

    BATCH_SIZE = 128
    BUFFER_SIZE = 10000
    targets_use = np.array(targets, dtype='float32')
    contexts_use = np.array(contexts, dtype='float32')
    labels_use = np.array(labels, dtype='float32')
    dataset = tf.data.Dataset.from_tensor_slices(((targets_use, contexts_use), labels_use))
    dataset = dataset.shuffle(BUFFER_SIZE,seed=SEED).batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 开始训练
    embedding_dim = spinner_dimension.value
    word2vec = Word2Vec(vocab_size, embedding_dim, negative_sample_size)
    word2vec.compile(optimizer='adam',
                    loss=custom_loss,
                    metrics=['accuracy'])
    word2vec.fit(dataset, epochs=spinner_epoch.value)
    weights_v = word2vec.get_layer('w2v_embedding').get_weights()[0]
    
    dot_simility_i = []
    cos_simility_i = []
    geo = []
    for i in range(n):
        for j in range(i+1,n):
            dot_simility_i.append(-dot_sim(weights_v[i], weights_v[j]))
            cos_simility_i.append(1-cos_sim(weights_v[i], weights_v[j]))
            geo.append(geo_distance[i][j])
    # if p2.renderers != []:
    #     p2.renderers.pop()
    # p2_source.data = {
    #     'x': geo,
    #     'y': dot_simility_i
    # }
    # p2.circle(y='y', x='x',color='red',size=3,source=p2_source)
    np.save('geo.npy',geo)
    np.save('dot_simility_i.npy',dot_simility_i)
    # d' 和 d 的相关性散点图
    # 首先转换
    x = np.expand_dims(np.array(geo), axis=1)
    y = np.expand_dims(np.array(cos_simility_i), axis=1)

    # 调用线性回归函数，让常数项为0
    clf = linear_model.LinearRegression(fit_intercept=False)
    # 开始线性回归计算
    clf.fit(x,y)
    d_p = np.array(cos_simility_i) / clf.coef_[0][0]
    
    # if p3.renderers != []:
    #         p3.renderers.pop()
    # p3_source.data = {
    #     'x': geo,
    #     'y': d_p
    # }
    # p3.circle(y='y', x='x',color='red',size=3,source=p3_source)
    np.save('d_p.npy',d_p)

    table_data = {}
    for i in range(len(citys)):
        table_data[citys['name'][i]] = weights_v[i]
    np.save('table_data.npy',table_data)

def callback():
    pool = multiprocessing.Pool()
    pool.apply_async(train)
    pool.close()
    pool.join()
    messageButton2.name = str(int(messageButton2.name)+1)
    if p1.renderers != []:
        p1.renderers.pop()
    p1_data = (flows.flatten(),np.load('data.npy').flatten())
    p1d = datashade(hv.Points(p1_data).opts(size=3))
    hvp1 = renderer.get_plot(p1d)
    p1.renderers.append(hvp1.state.renderers[0])
    # p1_source.data = {
    #     'x': flow.flatten(),
    #     'y': np.load('data.npy').flatten()
    # }
    if p2.renderers != []:
        p2.renderers.pop()
    p2_data = (np.load('geo.npy').flatten(),np.load('dot_simility_i.npy').flatten())
    p2d = datashade(hv.Points(p2_data).opts(size=3))
    hvp2 = renderer.get_plot(p2d)
    p2.renderers.append(hvp2.state.renderers[0])
    # p2_source.data = {
    #     'x': np.load('geo.npy').flatten(),
    #     'y': np.load('dot_simility_i.npy').flatten()
    # }
    if p3.renderers != []:
        p3.renderers.pop()
    p3_data = (np.load('geo.npy').flatten(),np.load('d_p.npy').flatten())
    p3d = datashade(hv.Points(p3_data).opts(size=3))
    hvp3 = renderer.get_plot(p3d)
    p3.renderers.append(hvp3.state.renderers[0])
    # p3_source.data = {
    #     'x': np.load('geo.npy').flatten(),
    #     'y': np.load('d_p.npy').flatten()
    # }

    # p1.circle(y='y', x='x',color='red',size=3,source=p1_source)
    # p2.circle(y='y', x='x',color='red',size=3,source=p2_source)
    # p3.circle(y='y', x='x',color='red',size=3,source=p3_source)
    table_source2.data = np.load('table_data.npy',allow_pickle=True).item()
    # 加一个按钮的点击，提示训练完成


# button_train.js_on_click(CustomJS(code="""
#         console.log('1111')
#     """))
button_train.js_on_click(CustomJS(args=dict(type='warning',content='开始训练，请稍等', duration='3000'),
                        code=open(join(dirname(__file__), "newMessage.js")).read()))
button_train.on_event(ButtonClick, callback)
# 正样本采样和原始网络的散点图
p1 = figure(background_fill_color="lightgrey", width=300,height=200)
#p1_source = ColumnDataSource(data=dict())
# 两个相关性散点图
p2 = figure(background_fill_color="lightgrey", width=300,height=200)
#p2_source = ColumnDataSource(data=dict())

p3 = figure(background_fill_color="lightgrey", width=300,height=200)
#p3_source = ColumnDataSource(data=dict())
# 向量表
columns2 = []
for i in range(len(citys_name)):
    columns2.append(TableColumn(field=citys_name['name'][i],title=citys_name['name'][i], formatter=NumberFormatter(format="0.0000"),width=30))

table_source2 = ColumnDataSource(data=dict())
data_table2 = DataTable(source=table_source2, columns=columns2, width=900,height=600,autosize_mode="force_fit")
# def selectCity(attr, old, new):
#      table_source.selected['1d'].indices = [0]
# searchCity.on_change('value',selectCity)
# 消息按钮
messageButton2 = Button(label="message",name=str(0),visible=False)
messageButton2.js_on_change("name", CustomJS(args=dict(type='success',content='训练完成', duration='1000'),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))


#Part3 变形图----------------------------------------------------------------------------------------------
isLoad2 = False
LABELS = ["一线城市", "二线城市", "三线城市", "四线城市", "五线城市"]

checkbox_button_group1 = CheckboxButtonGroup(labels=LABELS)
checkbox_button_group2 = CheckboxButtonGroup(labels=LABELS)
get_button = Button(label='Get')

def update2():
    '''数据筛选'''
    # 大小透明度
    global isLoad2
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
    if isLoad2==False:
        plot2.patches('xs','ys', source = geosource_province,
                            fill_color = None,
                            line_color = 'grey', 
                            line_width = 1, 
                            fill_alpha = 1)
        plot2.patches('xs','ys', source = geosource_nineline,
                                    fill_color = None,
                                    line_color = 'gray', 
                                    line_width = 5, 
                                    fill_alpha = 1)
        isLoad2=True
    else:
        plot2.tools.pop()
        plot2.tools.pop()
        plot2.renderers.pop()

    

    nodes = hv.Nodes(citys,['x','y','index'],vdims=['name','line','size','alpha'])

    graph = hv.Graph(((flow_), nodes), vdims=['flow','origin','destination'])
    graph.opts(node_size='size', edge_line_width=0.1,node_color='line',cmap=colors,node_alpha='alpha')
    bundled = bundle_graph(graph,initial_bandwidth=0.05, decay=0.3) 

    hvplot = renderer.get_plot(bundled)

    plot2.renderers.append(hvplot.state.renderers[0])
    tap_tool = TapTool(renderers=[hvplot.state.renderers[0]])
    hover_tool = HoverTool(renderers = [hvplot.state.renderers[0]],tooltips=[("name", "@name"),("line","@line")])
    plot2.add_tools(hover_tool,tap_tool)

get_button.on_click(update2)
# 导入数据
citys_lines = np.load('data/citys_lines.npy',allow_pickle=True)



plot2 = figure()
plot2.axis.visible = False
plot2.grid.visible = False


colors = ['#000000']+hv.Cycle('Category20').values
kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

curdoc().add_root(database_button)
curdoc().add_root(row(column(messageButton,row(spinnerMin,spinnerMax),row(searchTextOrigin,searchTextDestination),data_table,DownloadButton),plot))
curdoc().add_root(row(column(row(messageButton2,spinner_target,spinner_negative),row(spinner_dimension,spinner_epoch),button_train,row(data_table2,column(p1,p2,p3)))))
curdoc().add_root(row(column(checkbox_button_group1,checkbox_button_group2,plot2),get_button))