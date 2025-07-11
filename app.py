import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# === Datos REALES de tus modelos ===
# Resultados Regresión Logística
conf_matrix_log = np.array([[12445, 711],
                           [1783, 1987]])
logistic_metrics = {
    'accuracy': 0.85,
    'precision_0': 0.87,
    'recall_0': 0.95,
    'precision_1': 0.74,
    'recall_1': 0.53
}

# Coeficientes más importantes (ejemplo basado en tu análisis)
top_logistic_features = [
    'Humidity3pm', 'Cloud3pm', 'Sunshine', 
    'Pressure3pm', 'RainToday', 'WindGustSpeed'
]
top_logistic_coefs = [0.82, 0.45, -0.38, -0.35, 0.32, 0.28]

# Resultados Árbol de Decisión
conf_matrix_tree = np.array([[21972, 5608],
                            [1850, 6119]])
tree_metrics = {
    'accuracy': 0.79,
    'precision_0': 0.92,
    'recall_0': 0.80,
    'precision_1': 0.52,
    'recall_1': 0.77
}

# Variables más importantes del árbol
top_tree_features = [
    'Humidity3pm', 'Sunshine', 'Pressure3pm',
    'Cloud3pm', 'RainToday', 'WindGustSpeed'
]
top_tree_importances = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]

# === Crear app ===
app = dash.Dash(__name__)
server = app.server

# === Estilos globales ===
colors = {
    'background': '#f8f9fa',
    'text': '#343a40',
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545'
}

app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    html.H1("Análisis Predictivo de Lluvia en Australia", 
            style={'textAlign': 'center', 'color': colors['primary'], 'marginBottom': '30px'}),
    
    html.Div([
        html.P("Esta herramienta visualiza el desempeño de dos modelos para predecir si lloverá mañana:", 
               style={'color': colors['text'], 'fontSize': '18px'}),
        html.Ul([
            html.Li("Regresión Logística: Modelo estadístico que calcula probabilidades"),
            html.Li("Árbol de Decisión: Modelo que hace preguntas secuenciales sobre las condiciones climáticas")
        ], style={'color': colors['secondary']})
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    dcc.Tabs(
        id="tabs",
        value='tab-logistic',
        children=[
            dcc.Tab(label='Regresión Logística', value='tab-logistic', 
                   style={'fontWeight': 'bold'}),
            dcc.Tab(label='Árbol de Decisión', value='tab-tree',
                   style={'fontWeight': 'bold'})
        ],
        colors={
            "border": colors['primary'],
            "primary": colors['primary'],
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
            html.H3("Resultados del Modelo de Regresión Logística", 
                   style={'color': colors['primary'], 'marginTop': '20px'}),
            
            html.Div([
                html.Div([
                    html.H4("Exactitud General", style={'color': colors['success']}),
                    html.Div(f"{logistic_metrics['accuracy']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['success']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H4("Precisión (Días sin lluvia)", style={'color': colors['primary']}),
                    html.Div(f"{logistic_metrics['precision_0']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['primary']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H4("Precisión (Días con lluvia)", style={'color': colors['danger']}),
                    html.Div(f"{logistic_metrics['precision_1']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['danger']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            html.H4("Matriz de Confusión", style={'color': colors['primary'], 'marginTop': '20px'}),
            html.P("Comparación entre las predicciones del modelo y lo que realmente ocurrió:", 
                   style={'color': colors['secondary']}),
            
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_log,
                    x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                    y=['Real: No lluvia', 'Real: Lluvia'],
                    colorscale='Blues',
                    showscale=True,
                    hoverinfo='z'
                ).update_layout(
                    xaxis_title='Predicción del modelo',
                    yaxis_title='Realidad',
                    margin={'t': 30}
                )
            ),
            
            html.Div([
                html.Div([
                    html.P("Correctamente predicho:", style={'fontWeight': 'bold'}),
                    html.P("12,445 días sin lluvia", style={'color': colors['success']}),
                    html.P("1,987 días con lluvia", style={'color': colors['success']})
                ], style={'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.P("Errores de predicción:", style={'fontWeight': 'bold'}),
                    html.P("711 días predichos como lluvia cuando no llovió", style={'color': colors['danger']}),
                    html.P("1,783 días predichos como no lluvia cuando sí llovió", style={'color': colors['danger']})
                ], style={'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            html.H4("Factores Clave que Influyen en la Predicción", 
                   style={'color': colors['primary'], 'marginTop': '20px'}),
            html.P("Variables que más afectan la probabilidad de lluvia según el modelo:", 
                   style={'color': colors['secondary']}),
            
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(
                        x=top_logistic_coefs,
                        y=top_logistic_features,
                        orientation='h',
                        marker_color=colors['primary'],
                        hoverinfo='x'
                    )
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title='Importancia (coeficiente)',
                    yaxis_title='Variable climática',
                    margin={'t': 30}
                )
            ),
            
            html.Div([
                html.P("Interpretación:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li("Valores positivos aumentan la probabilidad de lluvia"),
                    html.Li("Valores negativos disminuyen la probabilidad de lluvia"),
                    html.Li("Por ejemplo: alta humedad a las 3pm aumenta la probabilidad de lluvia")
                ])
            ], style={'backgroundColor': '#e9f5ff', 'padding': '15px', 'borderRadius': '10px'})
        ])
    
    elif tab == 'tab-tree':
        return html.Div([
            html.H3("Resultados del Modelo de Árbol de Decisión", 
                   style={'color': colors['primary'], 'marginTop': '20px'}),
            
            html.Div([
                html.Div([
                    html.H4("Exactitud General", style={'color': colors['success']}),
                    html.Div(f"{tree_metrics['accuracy']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['success']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H4("Precisión (Días sin lluvia)", style={'color': colors['primary']}),
                    html.Div(f"{tree_metrics['precision_0']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['primary']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H4("Precisión (Días con lluvia)", style={'color': colors['danger']}),
                    html.Div(f"{tree_metrics['precision_1']*100:.1f}%", 
                             style={'fontSize': '32px', 'fontWeight': 'bold', 'color': colors['danger']})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            html.H4("Matriz de Confusión", style={'color': colors['primary'], 'marginTop': '20px'}),
            html.P("Comparación entre las predicciones del modelo y lo que realmente ocurrió:", 
                   style={'color': colors['secondary']}),
            
            dcc.Graph(
                figure=ff.create_annotated_heatmap(
                    z=conf_matrix_tree,
                    x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                    y=['Real: No lluvia', 'Real: Lluvia'],
                    colorscale='Greens',
                    showscale=True,
                    hoverinfo='z'
                ).update_layout(
                    xaxis_title='Predicción del modelo',
                    yaxis_title='Realidad',
                    margin={'t': 30}
                )
            ),
            
            html.Div([
                html.Div([
                    html.P("Correctamente predicho:", style={'fontWeight': 'bold'}),
                    html.P("21,972 días sin lluvia", style={'color': colors['success']}),
                    html.P("6,119 días con lluvia", style={'color': colors['success']})
                ], style={'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.P("Errores de predicción:", style={'fontWeight': 'bold'}),
                    html.P("5,608 días predichos como lluvia cuando no llovió", style={'color': colors['danger']}),
                    html.P("1,850 días predichos como no lluvia cuando sí llovió", style={'color': colors['danger']})
                ], style={'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            html.H4("Factores Clave que Influyen en la Predicción", 
                   style={'color': colors['primary'], 'marginTop': '20px'}),
            html.P("Variables más importantes que el árbol considera para tomar decisiones:", 
                   style={'color': colors['secondary']}),
            
            dcc.Graph(
                figure=go.Figure(
                    go.Bar(
                        x=top_tree_importances,
                        y=top_tree_features,
                        orientation='h',
                        marker_color=colors['success'],
                        hoverinfo='x'
                    )
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title='Importancia relativa',
                    yaxis_title='Variable climática',
                    margin={'t': 30}
                )
            ),
            
            html.Div([
                html.P("Interpretación:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li("Muestra qué variables el modelo considera más importantes"),
                    html.Li("Por ejemplo: humedad a las 3pm es el factor más decisivo"),
                    html.Li("El árbol hace preguntas secuenciales sobre estas variables")
                ])
            ], style={'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '10px'}),
            
            html.Div([
                html.H4("Comparación de Modelos", style={'color': colors['primary']}),
                html.P("El Árbol de Decisión vs. Regresión Logística:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li("El árbol tiene menor exactitud general (79% vs 85%)"),
                    html.Li("Pero detecta mejor los días de lluvia (77% vs 53%)"),
                    html.Li("La regresión logística es mejor prediciendo días sin lluvia"),
                    html.Li("El árbol comete más errores 'falsas alarmas' de lluvia")
                ])
            ], style={'backgroundColor': '#fff3cd', 'padding': '15px', 'borderRadius': '10px', 'marginTop': '20px'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
