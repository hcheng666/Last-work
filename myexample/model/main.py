from dataclasses import field
from turtle import width
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from numpy import random, source
import tensorflow as tf
from tensorflow.keras import layers
# 导入sklearn库下的linear_model类
from sklearn import linear_model
import asyncio
import nest_asyncio
import multiprocessing
nest_asyncio.apply()
from os.path import dirname, join


from bokeh.io import curdoc
from bokeh.models import HoverTool,ColumnDataSource, TableColumn,TextInput
from bokeh.models.widgets import DataTable
from bokeh.models import Button, CustomJS, Spinner, NumberFormatter, Div
from bokeh.plotting import figure
from bokeh.layouts import row, column
import pandas as pd
import numpy as np
from bokeh.palettes import OrRd9
from bokeh.events import ButtonClick
import holoviews as hv
from holoviews.operation.datashader import datashade
import holoviews.operation.datashader as hd
hv.extension("bokeh")
hd.shade.cmap=["lightblue", "darkblue"]
renderer = hv.renderer('bokeh')
renderer = renderer.instance(mode='server')

flow = np.load(r"data/data_avg.npy")
citys = pd.read_csv(r"data/citys_coordiante.csv")
geo_distance = np.load(r'data/geo_distance.npy')
n = len(flow)
# 流量标准化函数
def T_ij(flow, m):
    # 对流量进行标准化
    flow_re = np.zeros_like(flow)
    j_sum = flow.sum(axis=1)
    for i in range(n):
        flow_re[i,:]= flow[i,:] / j_sum[i] * m[i]
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
m = flow.sum(axis=1)
# 让每一行的数据归一化
data = pd.DataFrame(flow)
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
searchCity = TextInput(value = '', title='Chose city: ')

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
    messageButton.name = str(int(messageButton.name)+1)
    if p1.renderers != []:
        p1.renderers.pop()
    p1_data = (flow.flatten(),np.load('data.npy').flatten())
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
    table_source.data = np.load('table_data.npy',allow_pickle=True).item()
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
columns = []
for i in range(len(citys)):
    columns.append(TableColumn(field=citys['name'][i],title=citys['name'][i], formatter=NumberFormatter(format="0.0000"),width=30))

table_source = ColumnDataSource(data=dict())
data_table = DataTable(source=table_source, columns=columns, width=900,height=600,autosize_mode="force_fit")
# def selectCity(attr, old, new):
#      table_source.selected['1d'].indices = [0]
# searchCity.on_change('value',selectCity)
# 消息按钮
messageButton = Button(label="message",name=str(0),visible=False)
messageButton.js_on_change("name", CustomJS(args=dict(type='success',content='训练完成', duration='1000'),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))

curdoc().add_root(row(column(row(messageButton,spinner_target,spinner_negative),row(spinner_dimension,spinner_epoch,searchCity),button_train,data_table),column(p1,p2,p3)))


# 功能已近实现，但还有几个问题
# 1. 加载太慢，需要异步
# 2. 数据显示太丑