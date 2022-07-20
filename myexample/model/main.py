import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from numpy import random, source
import tensorflow as tf
from tensorflow.keras import layers


from bokeh.io import show, curdoc
from bokeh.models import HoverTool,ColumnDataSource
from bokeh.models import Button, CustomJS, Spinner
from bokeh.plotting import figure
from bokeh.layouts import row, column
import pandas as pd
import numpy as np
from bokeh.palettes import OrRd9
from bokeh.events import ButtonClick


flow = np.load(r"data/data_avg.npy")
n = len(flow)
# 流量标准化函数
def T_ij(flow, m):
    # 对流量进行标准化
    flow_re = np.zeros_like(flow)
    j_sum = flow.sum(axis=1)
    for i in range(n):
        flow_re[i,:]= flow[i,:] / j_sum[i] * m[i]
    return flow_re

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
targets = [] #(n,)
postive_context = [] #(n,)
target_size = 100000
starts = random.uniform(size=(target_size,2))

# 调参应用 参数有：1 目标数量， 2 每个目标应用的负样本数量（正样本数量默认为1）3 向量维度 4 训练迭代次数
spinner_target = Spinner(title="Target Siz:", low=10000, high=120000, step=5000, value=60000, format="0,0")
spinner_negative = Spinner(title="Negative Sample Size:", low=1, high=10, step=1, value=5)
spinner_dimension = Spinner(title="Dimension Size:", low=10, high=150, step=10, value=80)
spinner_epoch = Spinner(title="Epochs:", low=1, high=5, step=1, value=1)
button_train = Button(label='Get Vector',button_type="primary")

def train():
    # 获取参数
    target_size = spinner_target.value
    negative_sample_size = spinner_negative.value
    vocab_size = len(m)
    # 正样本采样
    for i in range(target_size):
        targets.append(np.searchsorted(start_cum, starts[i][0]))
        postive_context.append(np.searchsorted(data_cum[targets[i]], starts[i][1]))
    
    data_reconstruct = np.zeros((n,n),dtype=float)
    for i in range(target_size):
        data_reconstruct[targets[i]][postive_context[i]]+=1
    # 流量标准化
    data_reconstruct_re = T_ij(data_reconstruct, m)
    # 画出相关性图
    if p1.renderers != []:
        p1.renderers.pop()
    p1_source.data = {
        'x': flow.flatten(),
        'y': data_reconstruct_re.flatten()
    }
    p1.circle(y='y', x='x',color='red',size=3,source=p1_source)
        
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
    print(dataset)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 更换损失函数
    embedding_dim = spinner_dimension.value
    word2vec = Word2Vec(vocab_size, embedding_dim, negative_sample_size)
    word2vec.compile(optimizer='adam',
                    loss=custom_loss,
                    metrics=['accuracy'])
    word2vec.fit(dataset, epochs=spinner_epoch.value)

button_train.on_event(ButtonClick, train)
# 正样本采样和原始网络的散点图
p1 = figure(background_fill_color="lightgrey")
p1_source = ColumnDataSource(data=dict())
curdoc().add_root(column(row(column(spinner_target,spinner_negative),column(spinner_dimension,spinner_epoch)),button_train,p1))
