import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np

# === Datos REALES ===
# Matriz de Confusión Regresión Logística
conf_matrix_log = np.array([[12445, 711],
                            [1783, 1987]])

# Coeficientes Regresión Logística
coef_names = [f'Feature {i+1}' for i in range(20)]
coef_values = np.random.randn(20)  # Sustituye por model.coef_

# Matriz de Confusión Árbol de Decisión
conf_matrix_tree = np.array([[21972, 5608],
                             [1850, 6119]])

# Importancia de variables Árbol de Decisión
feature_names = [f'Feature {i+1}' for i in range(10)]
importances = np.random.rand(10)  # Sustituye por model.feature_importances_

# === Crear app ===
app = dash.Dash(__name__)
server = app.server

# === Estilos globales ===
colors = {
    'background': '#1a1a2e',
    'text': '#ffffff',
    'primary': '#00c3ff',
    'secondary': '#21e6c1'
}

app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '30px'}, children=[
    html.H1("Dashboard Ejecutivo - Modelos Predictivos", style={'textAlign': 'center', 'color': colors['primary']}),

    dcc.Tabs(
        id="tabs",
        value='tab-logistic',
        children=[
            dcc.Tab(label='Regresión Logística', value='tab-logistic', style={'backgroundColor': colors['background'], 'color': colors['text']}),
            dcc.Tab(label='Árbol de Decisión', value='tab-tree', style={'backgroundColor': colors['background'], 'color': colors['text']})
        ],
        colors={
            "border": colors['primary'],
            "primary": colors['secondary'],
            "background": colors['background']
        }
    ),

    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-logistic':
        return html.Div([
            html.H2("Regresión Logística", style={'color': colors['secondary'], 'marginTop': '20px'}),

            html.P("Este bloque muestra las métricas principales del modelo de Regresión Logística: cómo clasifica casos positivos y negativos.", style={'color': colors['text']}),

            html.H4("Matriz de Confusión", style={'color': colors['primary']}),
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_log,
                    x=['Predicho 0', 'Predicho 1'],
                    y=['Real 0', 'Real 1'],
                    colorscale='Teal'
                )
            ),

            html.P("Esta matriz resume cuántos casos fueron correctamente o incorrectamente clasificados.", style={'color': colors['text']}),

            html.H4("Coeficientes del Modelo", style={'color': colors['primary']}),
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(
                        x=coef_names,
                        y=coef_values,
                        marker=dict(color=colors['secondary'])
                    )
                ).update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font=dict(color=colors['text']),
                    xaxis_title='Variable',
                    yaxis_title='Valor del Coeficiente',
                    title='Impacto de cada variable en la probabilidad de lluvia'
                )
            ),

            html.P("Cada barra representa cuánto impacta cada variable en la probabilidad estimada de lluvia. Valores positivos aumentan la probabilidad, valores negativos la reducen.", style={'color': colors['text']})
        ])

    elif tab == 'tab-tree':
        return html.Div([
            html.H2("Árbol de Decisión", style={'color': colors['secondary'], 'marginTop': '20px'}),

            html.P("Este bloque muestra las métricas principales del modelo de Árbol de Decisión.", style={'color': colors['text']}),

            html.H4("Matriz de Confusión", style={'color': colors['primary']}),
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_tree,
                    x=['Predicho 0', 'Predicho 1'],
                    y=['Real 0', 'Real 1'],
                    colorscale='Greens'
                )
            ),

            html.P("Muestra cuántos días fueron correctamente clasificados como lluvia o no lluvia.", style={'color': colors['text']}),

            html.H4("Importancia de Variables", style={'color': colors['primary']}),
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(
                        x=feature_names,
                        y=importances,
                        marker=dict(color=colors['primary'])
                    )
                ).update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font=dict(color=colors['text']),
                    xaxis_title='Variable',
                    yaxis_title='Importancia Relativa',
                    title='Variables más influyentes en la decisión'
                )
            ),

            html.P("Cada barra muestra cuánta influencia tiene cada variable en la decisión del árbol: mientras más alta, más importante.", style={'color': colors['text']})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
