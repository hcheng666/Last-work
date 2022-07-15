from bokeh.io import curdoc
from bokeh.models import Dropdown, FileInput, CustomJS
from bokeh.events import ButtonClick

# Set up widgets
dropdown = Dropdown(label='File', menu=['Open'], name='Dropdown', tags=['hi'])
fileinput = FileInput(visible=False, name='FileInput')

# Set up callback for Dropdown widget
dropdown.js_on_click(CustomJS(code="""
    if (this.item == 'Open') {
        document.getElementById('FileInputElement').querySelector('input[type=file]').dispatchEvent(new MouseEvent('click'))
    }
    """))

# Set up callback for hidden FileInput widget
def loadfile(attr, old, new):
    print(fileinput.filename)
fileinput.on_change('filename', loadfile)

# Set up layouts and add to document
curdoc().add_root(dropdown)
curdoc().add_root(fileinput)