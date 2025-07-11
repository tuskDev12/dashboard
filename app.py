import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np

# === Datos ACTUALIZADOS ===
# Resultados Árbol de Decisión (se mantienen iguales)
conf_matrix_tree = np.array([[21972, 5608],
                           [1850, 6119]])
tree_metrics = {
    'accuracy': 0.79,
    'precision_0': 0.92,
    'recall_0': 0.80,
    'precision_1': 0.52,
    'recall_1': 0.77
}

# Variables más importantes del árbol (actualizadas)
top_tree_features = [
    'Humidity3pm', 'Sunshine', 'Pressure3pm',
    'Cloud3pm', 'RainToday', 'WindGustSpeed'
]
top_tree_importances = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]

# NUEVOS datos Regresión Logística
conf_matrix_log = np.array([[31436, 1697],
                          [5304, 4221]])
logistic_metrics = {
    'accuracy': 0.84,
    'precision_0': 0.86,
    'recall_0': 0.95,
    'precision_1': 0.71,
    'recall_1': 0.44
}

# Coeficientes actualizados (16 características)
top_logistic_features = [
    'Humidity3pm', 'Cloud3pm', 'Sunshine',
    'Pressure3pm', 'RainToday', 'WindGustSpeed',
    'WindSpeed3pm', 'Temp3pm'
]
top_logistic_coefs = [0.75, 0.42, -0.35, -0.32, 0.30, 0.25, 0.18, -0.12]

# === Crear app ===
app = dash.Dash(__name__)
server = app.server

# === Estilos globales ===
colors = {
    'background': '#f5f7fa',
    'text': '#2d3748',
    'card_bg': '#ffffff',
    'primary': '#3182ce',
    'secondary': '#718096',
    'success': '#38a169',
    'danger': '#e53e3e',
    'border': '#e2e8f0',
    'header_bg': '#ebf4ff'
}

app.layout = html.Div(style={
    'backgroundColor': colors['background'],
    'padding': '20px',
    'minHeight': '100vh'
}, children=[
    html.Div(style={
        'backgroundColor': colors['header_bg'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'border': f'1px solid {colors["border"]}'
    }, children=[
        html.H1("Análisis Predictivo de Lluvia en Australia", 
                style={'textAlign': 'center', 'color': colors['primary']}),
        html.P("Resultados actualizados con el último entrenamiento de modelos:", 
               style={'color': colors['text'], 'textAlign': 'center'}),
        
        html.Div(style={
            'display': 'flex',
            'justifyContent': 'center',
            'marginTop': '15px'
        }, children=[
            html.Div(style={
                'padding': '10px 20px',
                'backgroundColor': '#feebc8',
                'borderRadius': '5px',
                'margin': '0 10px'
            }, children=[
                html.P("Nuevos datos analizados", style={
                    'color': '#b77905',
                    'margin': '0',
                    'fontWeight': 'bold'
                })
            ])
        ])
    ]),

    dcc.Tabs(id="tabs", value='tab-logistic', children=[
        dcc.Tab(label='Regresión Logística', value='tab-logistic'),
        dcc.Tab(label='Árbol de Decisión', value='tab-tree')
    ]),

    html.Div(id='tabs-content')
])

# Componente reutilizable para métricas
def create_metric_card(title, value, color, explanation):
    return html.Div(style={
        'backgroundColor': colors['card_bg'],
        'borderRadius': '8px',
        'padding': '15px',
        'margin': '10px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.05)',
        'border': f'1px solid {colors["border"]}',
        'flex': '1',
        'minWidth': '200px'
    }, children=[
        html.H4(title, style={'color': color, 'marginTop': '0'}),
        html.Div(f"{value}", style={
            'fontSize': '28px',
            'fontWeight': 'bold',
            'color': color,
            'margin': '10px 0'
        }),
        html.P(explanation, style={
            'color': colors['secondary'],
            'fontSize': '14px',
            'marginBottom': '0'
        })
    ])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-logistic':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H3("Regresión Logística - Resultados Actualizados", 
                       style={'color': colors['primary']}),
                html.P("Modelo reentrenado con 42,658 registros y 16 variables predictoras", 
                       style={'color': colors['secondary']}),
                
                html.Div(style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-around',
                    'margin': '20px 0'
                }, children=[
                    create_metric_card(
                        "Exactitud", 
                        f"{logistic_metrics['accuracy']*100:.1f}%", 
                        colors['success'],
                        "Porcentaje total de predicciones correctas"
                    ),
                    create_metric_card(
                        "Días sin lluvia (Precisión)", 
                        f"{logistic_metrics['precision_0']*100:.1f}%", 
                        colors['primary'],
                        "Cuando predice 'no lluvia', acierta el 86%"
                    ),
                    create_metric_card(
                        "Días con lluvia (Detección)", 
                        f"{logistic_metrics['recall_1']*100:.1f}%", 
                        colors['danger'],
                        "Solo detecta el 44% de los días de lluvia reales"
                    )
                ]),
                
                html.H4("Matriz de Confusión Actualizada", 
                       style={'color': colors['primary'], 'marginTop': '20px'}),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_log,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Blues',
                        annotation_text=conf_matrix_log,
                        hoverinfo='z'
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        margin={'t': 30}
                    )
                ),
                
                html.Div(style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-around',
                    'marginTop': '20px'
                }, children=[
                    html.Div(style={
                        'backgroundColor': '#f0fff4',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'flex': '1',
                        'minWidth': '250px',
                        'margin': '10px',
                        'border': f'1px solid {colors["border"]}'
                    }, children=[
                        html.H5("Aciertos", style={'color': colors['success']}),
                        html.P("31,436 días sin lluvia correctos", style={'marginBottom': '5px'}),
                        html.P("4,221 días con lluvia correctos")
                    ]),
                    html.Div(style={
                        'backgroundColor': '#fff5f5',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'flex': '1',
                        'minWidth': '250px',
                        'margin': '10px',
                        'border': f'1px solid {colors["border"]}'
                    }, children=[
                        html.H5("Errores", style={'color': colors['danger']}),
                        html.P("1,697 falsas alarmas (predijo lluvia pero no llovió)"),
                        html.P("5,304 días de lluvia no detectados", style={'marginBottom': '0'})
                    ])
                ]),
                
                html.H4("Variables más influyentes (16 características)", 
                       style={'color': colors['primary'], 'marginTop': '30px'}),
                dcc.Graph(
                    figure=go.Figure(
                        go.Bar(
                            x=top_logistic_coefs,
                            y=top_logistic_features,
                            orientation='h',
                            marker_color=colors['primary']
                        )
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        xaxis_title='Fuerza de influencia',
                        margin={'t': 30}
                    )
                ),
                
                html.Div(style={
                    'backgroundColor': '#ebf8ff',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginTop': '20px',
                    'border': f'1px solid {colors["border"]}'
                }, children=[
                    html.H5("¿Qué significan estos cambios?", style={'color': colors['primary']}),
                    html.Ul(children=[
                        html.Li("El modelo ahora usa menos variables (16 vs 20 originales)"),
                        html.Li("Mantiene buena precisión para días sin lluvia (86%)"),
                        html.Li("Pero detecta menos días de lluvia reales (44% vs 53% original)"),
                        html.Li("La humedad a las 3pm sigue siendo el factor más importante")
                    ])
                ])
            ])
        ])
    
    elif tab == 'tab-tree':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H3("Árbol de Decisión - Resultados Consistentes", 
                       style={'color': colors['primary']}),
                html.P("Mismo desempeño que en análisis anteriores", 
                       style={'color': colors['secondary']}),
                
                html.Div(style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-around',
                    'margin': '20px 0'
                }, children=[
                    create_metric_card(
                        "Exactitud", 
                        f"{tree_metrics['accuracy']*100:.1f}%", 
                        colors['success'],
                        "Porcentaje total de predicciones correctas"
                    ),
                    create_metric_card(
                        "Días sin lluvia (Precisión)", 
                        f"{tree_metrics['precision_0']*100:.1f}%", 
                        colors['primary'],
                        "Cuando predice 'no lluvia', acierta el 92%"
                    ),
                    create_metric_card(
                        "Días con lluvia (Detección)", 
                        f"{tree_metrics['recall_1']*100:.1f}%", 
                        colors['danger'],
                        "Detecta el 77% de los días de lluvia reales"
                    )
                ]),
                
                html.H4("Matriz de Confusión", 
                       style={'color': colors['primary'], 'marginTop': '20px'}),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_tree,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Greens',
                        annotation_text=conf_matrix_tree,
                        hoverinfo='z'
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        margin={'t': 30}
                    )
                ),
                
                html.Div(style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-around',
                    'marginTop': '20px'
                }, children=[
                    html.Div(style={
                        'backgroundColor': '#f0fff4',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'flex': '1',
                        'minWidth': '250px',
                        'margin': '10px',
                        'border': f'1px solid {colors["border"]}'
                    }, children=[
                        html.H5("Aciertos", style={'color': colors['success']}),
                        html.P("21,972 días sin lluvia correctos", style={'marginBottom': '5px'}),
                        html.P("6,119 días con lluvia correctos")
                    ]),
                    html.Div(style={
                        'backgroundColor': '#fff5f5',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'flex': '1',
                        'minWidth': '250px',
                        'margin': '10px',
                        'border': f'1px solid {colors["border"]}'
                    }, children=[
                        html.H5("Errores", style={'color': colors['danger']}),
                        html.P("5,608 falsas alarmas (predijo lluvia pero no llovió)"),
                        html.P("1,850 días de lluvia no detectados", style={'marginBottom': '0'})
                    ])
                ]),
                
                html.H4("Variables más importantes", 
                       style={'color': colors['primary'], 'marginTop': '30px'}),
                dcc.Graph(
                    figure=go.Figure(
                        go.Bar(
                            x=top_tree_importances,
                            y=top_tree_features,
                            orientation='h',
                            marker_color=colors['success']
                        )
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        xaxis_title='Importancia relativa',
                        margin={'t': 30}
                    )
                ),
                
                html.Div(style={
                    'backgroundColor': '#e6ffed',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginTop': '20px',
                    'border': f'1px solid {colors["border"]}'
                }, children=[
                    html.H5("Comparación con Regresión Logística", style={'color': colors['success']}),
                    html.Ul(children=[
                        html.Li("🔹 El árbol detecta mejor los días de lluvia (77% vs 44%)"),
                        html.Li("🔹 Pero tiene más falsas alarmas (5,608 vs 1,697)"),
                        html.Li("🔹 Ambos coinciden en las variables clave (Humedad 3pm, Sol, Presión)"),
                        html.Li("🔹 La regresión logística es mejor para días sin lluvia (95% vs 80%)")
                    ])
                ])
            ])
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
