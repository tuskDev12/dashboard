import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np

# === Datos de ejemplo ===
# Regresión Logística
conf_matrix_log = np.array([[12445, 711],
                            [1783, 1987]])
coef_names = [f'Feature {i+1}' for i in range(20)]
coef_values = np.random.randn(20)  # Reemplaza con coef reales

# Árbol de Decisión
conf_matrix_tree = np.array([[21972, 5608],
                             [1850, 6119]])
feature_names = [f'Feature {i+1}' for i in range(10)]
importances = np.random.rand(10)  # Reemplaza con importancias reales

# === App ===
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Dashboard Ejecutivo - Modelos Predictivos", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    dcc.Tabs(id="tabs", value='tab-logistic', children=[
        dcc.Tab(label='Regresión Logística', value='tab-logistic'),
        dcc.Tab(label='Árbol de Decisión', value='tab-tree'),
    ], colors={'border': 'grey', 'primary': 'navy', 'background': 'lightgrey'}),

    html.Div(id='tabs-content')
], style={'padding': '20px'})


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-logistic':
        return html.Div([
            html.H2("Regresión Logística", style={'marginTop': '20px'}),
            html.P("Reporte: Accuracy 85% | Precision: 0.87/0.74 | Recall: 0.95/0.53 | F1: 0.91/0.61", style={'fontSize': '16px'}),

            html.H4("Matriz de Confusión"),
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_log,
                    x=['Predicho 0', 'Predicho 1'],
                    y=['Real 0', 'Real 1'],
                    colorscale='Blues'
                )
            ),

            html.H4("Coeficientes del Modelo"),
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(x=coef_names, y=coef_values)
                ).update_layout(xaxis_title='Variable', yaxis_title='Coeficiente')
            )
        ])

    elif tab == 'tab-tree':
        return html.Div([
            html.H2("Árbol de Decisión", style={'marginTop': '20px'}),
            html.P("Reporte: Accuracy 79% | Precision: 0.92/0.52 | Recall: 0.80/0.77 | F1: 0.85/0.62", style={'fontSize': '16px'}),

            html.H4("Matriz de Confusión"),
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_tree,
                    x=['Predicho 0', 'Predicho 1'],
                    y=['Real 0', 'Real 1'],
                    colorscale='Greens'
                )
            ),

            html.H4("Importancia de Variables"),
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(x=feature_names, y=importances)
                ).update_layout(xaxis_title='Variable', yaxis_title='Importancia')
            )
        ])


if __name__ == '__main__':
    app.run_server(debug=True)

