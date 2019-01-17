import os
import math
import dash
import base64
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
from dash.dependencies import Input, Output, State
import dash_table

X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)


app = dash.Dash('UrbanSound8k')
external_css = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
#external_css = ["https://codepen.io/ainalem/pen/QzogPe.css"]
for css in external_css:
    app.css.append_css({"external_url": css})
#External JavaScript
external_js = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js"]
for js in external_js:
    app.scripts.append_script({"external_url": js})

app.layout = html.Div(
    [
        html.H3('A.C.E Control Panel'),
        dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
            dcc.Tab(label='Sector 1', value='tab-1-example'),
            dcc.Tab(label='Sector 2', value='tab-2-example'),
            dcc.Tab(label='Sector 3', value='tab-3-example'),
            dcc.Tab(label='Sector 4', value='tab-4-example'),
            dcc.Tab(label='Sector 5', value='tab-5-example'),
            dcc.Tab(label='Sector 6', value='tab-6-example'),
            dcc.Tab(label='Sector 7', value='tab-7-example'),
            dcc.Tab(label='Sector 8', value='tab-8-example'),

    ]),
        dcc.Graph(id='live-graph', animate=True, style={'height': '300px'}),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
        html.H5('Live alerts:'),
        html.Div(id='alerts-output',
             children='No alerts'),
        html.H6('Drag and drop test samples:'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },

        ),
    ]
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(name, "wb") as fp:
        fp.write(base64.decodebytes(data))


def parse_contents(content, filename):
    try:
        if 'wav' in filename:
            save_file(filename,content)
        else:
            return html.Div([
            'Only works with *.wav files'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    os.system('ffmpeg -i ' + filename + ' -ar 48000 ' + '_' + filename)
    os.system('ffmpeg -i _' + filename + ' -ss 0 -t 4 __' + filename) 
    os.system('python3 passtoAI.py -i __' + filename + '>res') #I know that it's not the best way
    f=open("res", "r")
    if f.mode == 'r':
        contents =f.read()
    #os.system('rm res')
    os.system('rm *.wav')
    print('==================contents',contents)
    return html.Div([
        (contents),
        ])





@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    X.append(X[-1]+1)
    Y.append(random.uniform(-1,1))

    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )

    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[-1,1]),)}




@app.callback(Output('alerts-output', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents(list_of_contents,list_of_names)]
        return children





if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0')
