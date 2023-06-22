from dash import dcc, html, dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from data_analysis import layout as data_analysis_layout
from modeling import layout as modeling_layout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from shared_data import float_columns , data ,columns 

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css',dbc.themes.BOOTSTRAP],suppress_callback_exceptions = True)

app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(
            className='navbar',  
            children=[
                dcc.Link('Visualisation', href='/data-analysis', className='navbarLink'),
                html.Span(' | '),
                dcc.Link('Modeling', href='/modeling', className='navbarLink'),
            ]
        ),
        html.Hr(),
        html.Div(id='page-content')
    ]
)


@app.callback(Output('page-content','children'),
            [Input('url','pathname')])
def display_page(pathname):
    if pathname == '/data-analysis':
        return data_analysis_layout()
    elif pathname == '/modeling':
        return modeling_layout()
    else:
        return data_analysis_layout()
@app.callback(Output('pie-chart', 'figure'), [Input('pie-dropdown', 'value')])
def update_pie_chart(selected_value):
    column_name = selected_value
    
    value_counts = data[column_name].value_counts()

    labels = value_counts.index.tolist()
    values = value_counts.values.tolist()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    return fig

@app.callback(
    Output('hist-plot', 'figure'), 
    [Input('hist-dropdown', 'value')]  # Input: id of the dropdown component
    
)
def update_histogram_plot(selected_value):

    column_name = selected_value
    hist_data = data[column_name]

    fig = go.Figure(data=[go.Histogram(x=hist_data, histnorm='probability density')])

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
