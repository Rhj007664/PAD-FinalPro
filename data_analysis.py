from dash import dcc, html , dash
from shared_data import data ,float_columns ,columns 
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
def layout():
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    fig = go.Figure(data=[go.Bar(x=missing_percentage.index, y=missing_percentage.values)])
    fig.update_layout(title="Percentage of Missing Values by Column",title_x=0.5, xaxis_title="Columns", yaxis_title="Percentage")
    
    dtype_counts = data.dtypes.value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values)])
    fig_pie.update_layout(title="Data type",title_x=0.5)
    return html.Div(
        className='titleCss',
        children=[
            dcc.Markdown(
                """
                The dataset used in this project has been taken from kaggel link below:
                
                """
            ),

            dcc.Link(
                "Diagnosis of COVID-19 and its clinical spectrum",
                href="https://www.kaggle.com/datasets/einsteindata4u/covid19",
                target="_blank"
            ),
            dcc.Graph(figure=fig_pie),
            dbc.Row(
                children=[
                    dbc.Col(dcc.Graph(figure=fig), width=12)
                ],
                style={'margin': '20px'}  # Add margin around the heatmap
            ),
            
            dbc.Row(
                children=[
                    dbc.Col(dcc.Dropdown(columns, id='pie-dropdown',value=columns[2] ),width=6),  # Dropdown for pie chart
                    dbc.Col(dcc.Dropdown(float_columns, id='hist-dropdown',value=float_columns[1]), width=6),  # Dropdown for histogram
                ],
                style={'margin': '20px'}  # Add margin around the dropdowns
            ),
            dbc.Row(
                children=[
                    dbc.Col(dcc.Graph(id='pie-chart'), width=6),  # Pie chart
                    dbc.Col(dcc.Graph(id='hist-plot'), width=6),  # Histogram plot
                ],
                style={'margin': '20px'}  # Add margin around the plots
            ),


        ]
    )

