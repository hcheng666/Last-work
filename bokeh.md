# Bokeh学习笔记

## 第一步 基础

### 创建一个简单的折线图

```python
# 导入相关的库
from bokeh.plotting import figure, output_file, save, show
# 准备数据
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 2, 4, 5]
y2 = [2, 3, 4, 5, 6]
y3 = [4, 5, 5, 7, 2]
# 创建画板
# create a new plot with a title and axis labels
p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")
# 添加渲染器
# add multiple renderers
p.line(x, y1, legend_label="Temp.", color="blue", line_width=2)
p.line(x, y2, legend_label="Rate", color="red", line_width=2)
p.line(x, y3, legend_label="Objects", color="green", line_width=2)
# 用 show 或者 save 函数保存结果
show(p)
# 或
output_file("test.html")
save(p)
```



### 多个图放在一起

```python
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 2, 4, 5]
y2 = [2, 3, 4, 5, 6]
y3 = [4, 5, 5, 7, 2]

# create a new plot with a title and axis labels
p = figure(title="Multiple glyphs example", x_axis_label="x", y_axis_label="y")

# add multiple renderers
p.line(x, y1, legend_label="Temp.", color="blue", line_width=2)
p.line(x, y2, legend_label="Rate", color="red", line_width=2)
# 散点图 ，可以扩展成圆
p.circle(x, y3, legend_label="Objects", color="yellow", size=12)
# 柱状图
p.vbar(x=x, top=y2, legend_label="Rate", width=0.5, bottom=0, color="red")

# show the results
show(p)

# 图的各种属性
# add circle renderer with additional arguments
p.circle(
    x,
    y,
    legend_label="Objects",
    fill_color="red",
    fill_alpha=0.5,
    line_color="blue",
    size=80,
)

# 对已有的图表进行更改需要定义一个变量，然后重写
# change color of previously created object's glyph
glyph = circle.glyph
glyph.fill_color = "blue"
```

### 自定义图例和标注

```python
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 2]
y2 = [2, 3, 4, 5, 6]

# create a new plot
p = figure(title="Legend example")

# add circle renderer with legend_label arguments
line = p.line(x, y1, legend_label="Temp.", line_color="blue", line_width=2)
circle = p.circle(
    x,
    y2,
    legend_label="Objects",
    fill_color="red",
    fill_alpha=0.5,
    line_color="blue",
    size=80,
)
# 图例的各种属性
# display legend in top left corner (default is top right corner)
p.legend.location = "top_left"
# add a title to your legend
p.legend.title = "Obervations"
# change appearance of legend text
p.legend.label_text_font = "times"
p.legend.label_text_font_style = "italic"
p.legend.label_text_color = "navy"

# change border and background of legend
p.legend.border_line_width = 3
p.legend.border_line_color = "navy"
p.legend.border_line_alpha = 0.8
p.legend.background_fill_color = "navy"
p.legend.background_fill_alpha = 0.2

# show the results
show(p)



# 自定义头线
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# create new plot
p = figure(title="Headline example")
# add line renderer with a legend
p.line(x, y, legend_label="Temp.", line_width=2)
# change headline location to the left
p.title_location = "left"
# change headline text
p.title.text = "Changing headline text example"

# style the headline
p.title.text_font_size = "25px"
p.title.align = "right"
p.title.background_fill_color = "darkgrey"
p.title.text_color = "white"

# show the results
show(p)



# 添加箱型标注
import random

from bokeh.models import BoxAnnotation
from bokeh.plotting import figure, show

# generate some data (1-50 for x, random values for y)
x = list(range(0, 51))
y = random.sample(range(0, 100), 51)

# create new plot
p = figure(title="Box annotation example")

# add line renderer
line = p.line(x, y, line_color="#000000", line_width=2)

# add box annotations
low_box = BoxAnnotation(top=20, fill_alpha=0.2, fill_color="#F0E442")
mid_box = BoxAnnotation(bottom=20, top=80, fill_alpha=0.2, fill_color="#009E73")
high_box = BoxAnnotation(bottom=80, fill_alpha=0.2, fill_color="#F0E442")

# add boxes to existing figure
p.add_layout(low_box)
p.add_layout(mid_box)
p.add_layout(high_box)

# show the results
show(p)

```

### 自定义画图

```python
# 使用主题
from bokeh.io import curdoc
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# apply theme to current document
# 使用现有主题，也可以自定主题
curdoc().theme = "dark_minimal"
# create a plot
p = figure(sizing_mode="stretch_width", max_width=500, height=250)
# add a renderer
p.line(x, y)
# show the results
show(p)

# 重新定义图的大小
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# create a new plot with a specific size
p = figure(
    title="Plot sizing example",
  # 可以设置固定大小
    width=350,
    height=250,
  # 也可以自适应屏幕大小
    sizing_mode = "stretch_width",
  	height=250,
  
    x_axis_label="x",
    y_axis_label="y",
)
# 
# 可以在后面重设
p.width = 450
p.height = 150

# add circle renderer
circle = p.circle(x, y, fill_color="red", size=15)

# show the results
show(p)
```

### 自定义坐标轴

```python
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# create a plot
p = figure(
  # 控制刻度的范围
  	y_range=(0,25),
    title="Customized axes example",
    sizing_mode="stretch_width",
    max_width=500,
    height=350,
)

# add a renderer
p.circle(x, y, size=10)

# change some things about the x-axis
p.xaxis.axis_label = "Temp"
p.xaxis.axis_line_width = 3
p.xaxis.axis_line_color = "red"

# change some things about the y-axis
p.yaxis.axis_label = "Pressure"
p.yaxis.major_label_text_color = "orange"
p.yaxis.major_label_orientation = "vertical"

# change things on all axes
# 控制刻度的位置
p.axis.minor_tick_in = -3
p.axis.minor_tick_out = 6

# 对刻度进行格式标准化
from bokeh.models import NumeralTickFormatter
p.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
# 还有其他的标准格式，例如坐标轴以时间展开
p.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

# show the results
show(p)


# 还可以使用对数坐标轴
from bokeh.plotting import figure, show

# prepare some data
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y0 = [i**2 for i in x]
y1 = [10**i for i in x]
y2 = [10**(i**2) for i in x]

# create a new plot with a logarithmic axis type
p = figure(
    title="Logarithmic axis example",
    sizing_mode="stretch_width",
    height=300,
    max_width=500,
  # 坐标轴以对数形式展示
    y_axis_type="log",
    y_range=[0.001, 10 ** 11],
    x_axis_label="sections",
    y_axis_label="particles",
)

# add some renderers
p.line(x, x, legend_label="y=x")
p.circle(x, x, legend_label="y=x", fill_color="white", size=8)
p.line(x, y0, legend_label="y=x^2", line_width=3)
p.line(x, y1, legend_label="y=10^x", line_color="red")
p.circle(x, y1, legend_label="y=10^x", fill_color="red", line_color="red", size=6)
p.line(x, y2, legend_label="y=10^x^2", line_color="orange", line_dash="4 4")

# show the results
show(p)
```

### 自定义格网

```python
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# create a plot
p = figure(
    title="Customized grid lines example",
    sizing_mode="stretch_width",
    max_width=500,
    height=250,
)

# add a renderer
p.line(x, y, line_color="green", line_width=2)

# change things only on the x-grid
p.xgrid.grid_line_color = "red"

# change things only on the y-grid
p.ygrid.grid_line_alpha = 0.8
p.ygrid.grid_line_dash = [6, 4]

# show the results
show(p)


# 自定义边界和带
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# create a plot
p = figure(
    title="Bands and bonds example",
    sizing_mode="stretch_width",
    max_width=500,
    height=250,
)

# add a renderer
p.line(x, y, line_color="green", line_width=2)

# add bands to the y-grid
# 按颜色分带，更清晰
p.ygrid.band_fill_color = "olive"
p.ygrid.band_fill_alpha = 0.1

# define vertical bonds
# 加上边界
p.xgrid.bounds = (2, 4)

# show the results
show(p)
```

### 设置背景颜色

```python
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# create a plot
p = figure(
    title="Background colors example",
    sizing_mode="stretch_width",
    max_width=500,
    height=250,
)

# add a renderer
p.line(x, y, line_color="green", line_width=2)

# change the fill colors
p.background_fill_color = (204, 255, 255)
p.border_fill_color = (102, 204, 255)
p.outline_line_color = (0, 0, 255)

# show the results
show(p)
```

### 工具栏设置

```python
# 在画板上设置
p= figure(title="Toolbar positioning example", toolbar_location="below")
# 或者 直接设置属性
p.toolbar_location = "below"

# 隐藏工具栏
p.toolbar_location = None
# 设置自动隐藏
p.toolbar.autohide = True
# 把Bokeh的logo隐藏掉
p.toolbar.logo = None

# 自定义提供的工具栏
from bokeh.models.tools import BoxZoomTool, ResetTool
# 只添加缩放和重设工具
p = figure(tools = [BoxZoomTool(), ResetTool()])
# 再添加平移工具
p.add_tools(PanTool(dimensions="width"))

# HoverTool, 提示工具，当鼠标移到图上的某个位置时，给出提示
from bokeh.models import HoverTool
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

p = figure(
    y_range=(0, 10),
    toolbar_location=None,
  # 添加工具
    tools=[HoverTool()],
  # 并且可以自定义提示语
    tooltips="Data point @x has the value @y",
    sizing_mode="stretch_width",
    max_width=500,
    height=250,
)

# add renderers
p.circle(x, y, size=10)
p.line(x, y, line_width=2)

# show the results
show(p)
```

### 根据数据对颜色和形状大小进行映射

```python
import random

from bokeh.plotting import figure, show

# generate some data (1-10 for x, random values for y)
x = list(range(0, 26))
y = random.sample(range(0, 100), 26)

# generate list of rgb hex colors in relation to y
# 以RGB的形式对颜色进行映射
colors = ["#%02x%02x%02x" % (255, int(round(value * 255 / 100)), 255) for value in y]

# generate radii based on data
# 对大小进行映射
radii = y / 100 * 2

# 以色带的形式对颜色进行映射
# create linear color mapper
# 1
from bokeh.palettes import Turbo256
from bokeh.transform import linear_cmap
mapper = linear_cmap(field_name="y", palette=Turbo256, low=min(y), high=max(y))

# create new plot
p = figure(
    title="Vectorized colors example",
    sizing_mode="stretch_width",
    max_width=500,
    height=250,
)

# add circle and line renderers
line = p.line(x, y, line_color="blue", line_width=1)
# 这里的color可以是单值，也可以是跟数据同样大小的list，
circle = p.circle(x, y, radius=radii, fill_color=colors, line_color="blue", size=15)

# create circle renderer with color mapper
# 1
p.circle(x, y, color=mapper, size=10)

# show the results
show(p)
```

For more information about color mapping and other similar operations, see [Using mappers](https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#userguide-styling-using-mappers) and [Transforming data](https://docs.bokeh.org/en/latest/docs/user_guide/data.html#userguide-data-transforming) in the user guide. In addition to `linear_cmap`, this includes `log_cmap` and `factor_cmap`, for example.

### 多个图的排列

```python
# 使用column和row对图表进行排列
from bokeh.layouts import row
from bokeh.plotting import figure, show

# prepare some data
x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
y2 = [abs(i - 5) for i in x]

# create three plots with one renderer each
s1 = figure(width=250, height=250, background_fill_color="#fafafa")
s1.circle(x, y0, size=12, color="#53777a", alpha=0.8)

s2 = figure(width=250, height=250, background_fill_color="#fafafa")
s2.triangle(x, y1, size=12, color="#c02942", alpha=0.8)

s3 = figure(width=250, height=250, background_fill_color="#fafafa")
s3.square(x, y2, size=12, color="#d95b43", alpha=0.8)

# put the results in a row and show
show(row(s1, s2, s3))
# 使用sizing_mode对大小进行合理调节，填充整个浏览器
show(row(children=[s1, s2, s3], sizing_mode="scale_width"))
```

### 图表的展示和保存

```python
from bokeh.plotting import figure, output_file, save

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]

# set output to static HTML file
# 保存到指定文件
output_file(filename="custom_filename.html", title="Static HTML file")

# create a new plot with a specific size
p = figure(sizing_mode="stretch_width", max_width=500, height=250)

# add a circle renderer
circle = p.circle(x, y, fill_color="red", size=15)

# save the results to a file
save(p)

# 直接在网页中展示
show()

# 在notebook中展示
output_notebook()

# 保存成图片
from bokeh.io import export_png
from bokeh.plotting import figure

# prepare some data
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 2]
# create a new plot with fixed dimensions
p = figure(width=350, height=250)
# add a circle renderer
circle = p.circle(x, y, fill_color="red", size=15)
# save the results to a file
export_png(p, filename="plot.png")
```

### 数据格式

```python
# ColumnDataSource 是Bokeh专有的数据格式
# 使用方式：
# 首先导入包
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# create dict as basis for ColumnDataSource
# 创建字典数据为ColumnDataSource作基础
data = {'x_values': [1, 2, 3, 4, 5],
        'y_values': [6, 7, 2, 3, 6]}

# create ColumnDataSource based on dict
source = ColumnDataSource(data=data)

# create a plot and renderer with ColumnDataSource data
p = figure()
p.circle(x='x_values', y='y_values', source=source)

# 也可以先定义一个空的数据源
source = ColumnDataSource(data=dict())
# 后面再从数据中赋予
def update():
    current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
    source.data = {
        'name'             : current.name,
        'salary'           : current.salary,
        'years_experience' : current.years_experience,
    }
    
# 将其他数据源都转换成ColumnDataSource的形式来使用，DataFrame
source = ColumnDataSource(df)

# 数据筛选
from bokeh.layouts import gridplot
from bokeh.models import CDSView, ColumnDataSource, IndexFilter
from bokeh.plotting import figure, show

# create ColumnDataSource from a dict
source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]))

# create a view using an IndexFilter with the index positions [0, 2, 4]
view = CDSView(source=source, filters=[IndexFilter([0, 2, 4])])

# setup tools
tools = ["box_select", "hover", "reset"]

# create a first plot with all data in the ColumnDataSource
p = figure(height=300, width=300, tools=tools)
p.circle(x="x", y="y", size=10, hover_color="red", source=source)

# create a second plot with a subset of ColumnDataSource, based on view
p_filtered = figure(height=300, width=300, tools=tools)
p_filtered.circle(x="x", y="y", size=10, hover_color="red", source=source, view=view)

# show both plots next to each other in a gridplot layout
show(gridplot([[p, p_filtered]]))
```

### widgets

```python
from bokeh.layouts import layout
from bokeh.models import Div, RangeSlider, Spinner
from bokeh.plotting import figure, show

# prepare some data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [4, 5, 5, 7, 2, 6, 4, 9, 1, 3]

# create plot with circle glyphs
p = figure(x_range=(1, 9), width=500, height=250)
points = p.circle(x=x, y=y, size=30, fill_color="#21a7df")

# set up textarea (div)
div = Div(
    text="""
          <p>Select the circle's size using this control element:</p>
          """,
    width=200,
    height=30,
)

# set up spinner
spinner = Spinner(
    title="Circle size",
    low=0,
    high=60,
    step=5,
    value=points.glyph.size,
    width=200,
)
spinner.js_link("value", points.glyph, "size")

# set up RangeSlider
range_slider = RangeSlider(
    title="Adjust x-axis range",
    start=0,
    end=10,
    step=1,
    value=(p.x_range.start, p.x_range.end),
)
range_slider.js_link("value", p.x_range, "start", attr_selector=0)
range_slider.js_link("value", p.x_range, "end", attr_selector=1)

# create layout
layout = layout(
    [
        [div, spinner],
        [range_slider],
        [p],
    ]
)

# show result
show(layout)
```

## 第二步 例子

### Export CSV（数据的展示和下载）

```python
from os.path import dirname, join

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn)

df = pd.read_csv(join(dirname(__file__), 'salary_data.csv'))

source = ColumnDataSource(data=dict())

def update():
    current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
    source.data = {
        'name'             : current.name,
        'salary'           : current.salary,
        'years_experience' : current.years_experience,
    }

slider = RangeSlider(title="Max Salary", start=10000, end=110000, value=(10000, 50000), step=1000, format="0,0")
slider.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success")
button.js_on_event("button_click", CustomJS(args=dict(source=source),
                            code=open(join(dirname(__file__), "download.js")).read()))

columns = [
    TableColumn(field="name", title="Employee Name"),
    TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="years_experience", title="Experience (years)")
]

data_table = DataTable(source=source, columns=columns, width=800)

controls = column(slider, button)

curdoc().add_root(row(controls, data_table))
curdoc().title = "Export CSV"

update()
```

### Movies（数据的筛选和查询）

```python
''' An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

'''
import sqlite3 as sql
from os.path import dirname, join

import numpy as np
import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.plotting import figure
from bokeh.sampledata.movies_data import movie_path
# 首先从数据库导出全部数据
conn = sql.connect(movie_path)
query = open(join(dirname(__file__), 'query.sql')).read()
movies = psql.read_sql(query, conn)
# 根据初始条件选择和生成想要的数据列
movies["color"] = np.where(movies["Oscars"] > 0, "orange", "grey")
movies["alpha"] = np.where(movies["Oscars"] > 0, 0.9, 0.25)
movies.fillna(0, inplace=True)  # just replace missing values with zero
movies["revenue"] = movies.BoxOffice.apply(lambda x: '{:,d}'.format(int(x)))

with open(join(dirname(__file__), "razzies-clean.csv")) as f:
    razzies = f.read().splitlines()
movies.loc[movies.imdbID.isin(razzies), "color"] = "purple"
movies.loc[movies.imdbID.isin(razzies), "alpha"] = 0.9

# 可以转换的坐标轴
axis_map = {
    "Tomato Meter": "Meter",
    "Numeric Rating": "numericRating",
    "Number of Reviews": "Reviews",
    "Box Office (dollars)": "BoxOffice",
    "Length (minutes)": "Runtime",
    "Year": "Year",
}
# 事先准备好的html页面，也就是说一些文字描述之类的显示可以事先准备好
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

# Create Input controls  创建筛选的组件，包括滑动组件、选择组件、文字输入组件
reviews = Slider(title="Minimum number of reviews", value=80, start=10, end=300, step=10)
min_year = Slider(title="Year released", start=1940, end=2014, value=1970, step=1)
max_year = Slider(title="End Year released", start=1940, end=2014, value=2014, step=1)
oscars = Slider(title="Minimum number of Oscar wins", start=0, end=4, value=0, step=1)
boxoffice = Slider(title="Dollars at Box Office (millions)", start=0, end=800, value=0, step=1)
genre = Select(title="Genre", value="All",
               options=open(join(dirname(__file__), 'genres.txt')).read().split())
director = TextInput(title="Director name contains")
cast = TextInput(title="Cast names contains")
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Tomato Meter")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Number of Reviews")

# Create Column Data Source that will be used by the plot，创建数据源
source = ColumnDataSource(data=dict(x=[], y=[], color=[], title=[], year=[], revenue=[], alpha=[]))

# 提示工具栏，定义要显示的信息
TOOLTIPS=[
    ("Title", "@title"),
    ("Year", "@year"),
    ("$", "@revenue")
]
# 画布
p = figure(height=600, width=700, title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
# renderer
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")

# 定义数据更新函数
def select_movies():
    genre_val = genre.value
    director_val = director.value.strip()
    cast_val = cast.value.strip()
    selected = movies[
        (movies.Reviews >= reviews.value) &
        (movies.BoxOffice >= (boxoffice.value * 1e6)) &
        (movies.Year >= min_year.value) &
        (movies.Year <= max_year.value) &
        (movies.Oscars >= oscars.value)
    ]
    if (genre_val != "All"):
        selected = selected[selected.Genre.str.contains(genre_val) is True]
    if (director_val != ""):
        selected = selected[selected.Director.str.contains(director_val) is True]
    if (cast_val != ""):
        selected = selected[selected.Cast.str.contains(cast_val) is True]
    return selected

# 定义数据更新响应事件
def update():
    df = select_movies()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d movies selected" % len(df)
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        title=df["Title"],
        year=df["Year"],
        revenue=df["revenue"],
        alpha=df["alpha"],
    )
# 为控件绑定事件
controls = [reviews, boxoffice, genre, min_year, max_year, oscars, director, cast, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())
# 页面布局
inputs = column(*controls, width=320)

l = column(desc, row(inputs, p), sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Movies"

```

