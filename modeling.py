from dash import dcc, html, dash
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import shared_data  


def layout():
   
    return html.Div(
        className='titleCss',
        children=[
            dcc.Markdown(
                """
                Displaying classification reports for different models:
                """
            ),
            dbc.Row(
                children=[
                    dbc.Col(dcc.Graph(figure=shared_data.tree_fig),width=4), 
                    dbc.Col(dcc.Graph(figure=shared_data.knn_fig), width=4),  
                    dbc.Col(dcc.Graph(figure=shared_data.randomF_fig), width=4), 
                ],
               
            ),
            dcc.Markdown(
                """
                Feature selection:
                """
            ),
            dcc.Graph(figure=shared_data.fig_feature_importance),
            dcc.Markdown(
                """
                Score after feature selection:
                """
            ),
            dcc.Graph(figure=shared_data.randomF_fig_best),
            dcc.Markdown(
                """
                Learning curve:
                """
            ),
            dbc.Row(
                children=[
                    dbc.Col(dcc.Graph(figure=shared_data.fig),width=4), 
                    dbc.Col(dcc.Graph(figure=shared_data.fig_knn), width=4),  
                    dbc.Col(dcc.Graph(figure=shared_data.fig_randomF), width=4), 
                ],
               
            ),
            
            
        ]
    )