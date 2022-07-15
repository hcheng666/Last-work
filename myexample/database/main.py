from tkinter import Menu
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, CustomJS, Dropdown, Div)
from os.path import dirname, join

message = CustomJS(code=open(join(dirname(__file__), "newMessage.js")).read())
button = Button(label="message", button_type="success",name=str(0))
# button.js_on_event("button_click", CustomJS(
#                             code=open(join(dirname(__file__), "newMessage.js")).read()))
button.js_on_change("name", CustomJS(args=dict(type='message'),
                            code=open(join(dirname(__file__), "newMessage.js")).read()))
button.visible = False                            
menu = [('Open File','1'), ('Load From Database','2')]
dropdown = Dropdown(label="File", menu=menu)
def function_to_call(event):
    if event.item == '1':
        button.name = str(int(button.name)+1)
dropdown.on_event("menu_item_click", function_to_call)

desc = Div(text=open(join(dirname(__file__), "import.html")).read())

curdoc().add_root(column(button,dropdown))