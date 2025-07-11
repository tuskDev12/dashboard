import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np

# === Datos ===
# Random Forest
conf_matrix_rf = np.array([
    [31854, 1279],
    [5216, 4309]
])
rf_metrics = {
    'accuracy': 0.85,
    'precision_0': 0.86,
    'recall_0': 0.96,
    'precision_1': 0.77,
    'recall_1': 0.45
}
top_rf_features = [
    'Humidity3pm', 'Cloud3pm', 'Sunshine',
    'Pressure3pm', 'RainToday', 'WindGustSpeed'
]
top_rf_importances = [0.32, 0.25, 0.18, 0.10, 0.09, 0.06]

# Árbol de Decisión
conf_matrix_tree = np.array([
    [21972, 5608],
    [1850, 6119]
])
tree_metrics = {
    'accuracy': 0.79,
    'precision_0': 0.92,
    'recall_0': 0.80,
    'precision_1': 0.52,
    'recall_1': 0.77
}
top_tree_features = [
    'Humidity3pm', 'Sunshine', 'Pressure3pm',
    'Cloud3pm', 'RainToday', 'WindGustSpeed'
]
top_tree_importances = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]

# === App ===
app = dash.Dash(__name__)
server = app.server

colors = {
    'background': '#f5f7fa',
    'text': '#2d3748',
    'card_bg': '#ffffff',
    'primary': '#2b6cb0',
    'secondary': '#4a5568',
    'success': '#2f855a',
    'danger': '#c53030',
    'border': '#e2e8f0',
    'header_bg': '#ebf4ff'
}

app.layout = html.Div(style={
    'backgroundColor': colors['background'],
    'padding': '20px',
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
}, children=[
    html.Div(style={
        'backgroundColor': colors['header_bg'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'border': f'1px solid {colors["border"]}'
    }, children=[
        html.H1("Comparación de Modelos Predictivos de Lluvia", style={
            'textAlign': 'center',
            'color': colors['primary'],
            'marginBottom': '10px'
        }),
        html.P("Este panel presenta una comparación detallada entre dos modelos de clasificación para predecir si lloverá mañana.",
               style={
                   'color': colors['text'],
                   'fontSize': '18px',
                   'textAlign': 'center'
               }),
        html.Ul([
            html.Li("Random Forest: Modelo de ensamblado que combina múltiples árboles para mejorar la precisión."),
            html.Li("Árbol de Decisión: Modelo interpretativo que realiza preguntas secuenciales sobre variables climáticas.")
        ], style={
            'color': colors['secondary'],
            'listStyleType': 'disc',
            'paddingLeft': '40px'
        })
    ]),

    dcc.Tabs(
        id="tabs",
        value='tab-rf',
        children=[
            dcc.Tab(
                label='Random Forest',
                value='tab-rf',
                style={
                    'fontWeight': 'bold',
                    'padding': '10px',
                    'border': f'1px solid {colors["border"]}',
                    'backgroundColor': colors['background']
                },
                selected_style={
                    'backgroundColor': colors['primary'],
                    'color': 'white',
                    'border': f'1px solid {colors["primary"]}'
                }
            ),
            dcc.Tab(
                label='Árbol de Decisión',
                value='tab-tree',
                style={
                    'fontWeight': 'bold',
                    'padding': '10px',
                    'border': f'1px solid {colors["border"]}',
                    'backgroundColor': colors['background']
                },
                selected_style={
                    'backgroundColor': colors['primary'],
                    'color': 'white',
                    'border': f'1px solid {colors["primary"]}'
                }
            )
        ]
    ),

    html.Div(id='tabs-content')
])

def create_metric_card(title, value, color):
    return html.Div(style={
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': colors['card_bg'],
        'borderRadius': '8px',
        'margin': '10px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.05)',
        'border': f'1px solid {colors["border"]}',
        'flex': '1',
        'minWidth': '200px'
    }, children=[
        html.H4(title, style={'color': color, 'marginBottom': '10px'}),
        html.Div(value, style={
            'fontSize': '32px',
            'fontWeight': 'bold',
            'color': color
        })
    ])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-rf':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H3("Resultados del Modelo Random Forest", style={
                    'color': colors['primary'],
                    'marginTop': '0',
                    'borderBottom': f'1px solid {colors["border"]}',
                    'paddingBottom': '10px'
                }),
                html.P("Random Forest combina múltiples árboles de decisión para lograr una mayor precisión general y reducir la varianza.",
                       style={'color': colors['secondary']})
            ]),

            html.Div(style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'flexWrap': 'wrap',
                'marginBottom': '20px'
            }, children=[
                create_metric_card("Exactitud", f"{rf_metrics['accuracy']*100:.1f}%", colors['success']),
                create_metric_card("Precisión (No lluvia)", f"{rf_metrics['precision_0']*100:.1f}%", colors['primary']),
                create_metric_card("Precisión (Lluvia)", f"{rf_metrics['precision_1']*100:.1f}%", colors['danger'])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H4("Matriz de Confusión", style={'color': colors['primary']}),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_rf,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Blues',
                        showscale=True
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        xaxis_title='Predicción del modelo',
                        yaxis_title='Observación real',
                        margin={'t': 30}
                    )
                )
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H4("Variables más relevantes", style={'color': colors['primary']}),
                dcc.Graph(
                    figure=go.Figure(
                        go.Bar(
                            x=top_rf_importances,
                            y=top_rf_features,
                            orientation='h',
                            marker_color=colors['primary']
                        )
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        xaxis_title='Importancia relativa',
                        yaxis_title='Variable',
                        margin={'t': 30}
                    )
                )
            ])
        ])

    elif tab == 'tab-tree':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H3("Resultados del Modelo Árbol de Decisión", style={
                    'color': colors['primary'],
                    'marginTop': '0',
                    'borderBottom': f'1px solid {colors["border"]}',
                    'paddingBottom': '10px'
                }),
                html.P("El Árbol de Decisión segmenta el conjunto de datos en base a preguntas secuenciales para clasificar si lloverá o no.",
                       style={'color': colors['secondary']})
            ]),

            html.Div(style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'flexWrap': 'wrap',
                'marginBottom': '20px'
            }, children=[
                create_metric_card("Exactitud", f"{tree_metrics['accuracy']*100:.1f}%", colors['success']),
                create_metric_card("Precisión (No lluvia)", f"{tree_metrics['precision_0']*100:.1f}%", colors['primary']),
                create_metric_card("Precisión (Lluvia)", f"{tree_metrics['precision_1']*100:.1f}%", colors['danger'])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H4("Matriz de Confusión", style={'color': colors['primary']}),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_tree,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Greens',
                        showscale=True
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        xaxis_title='Predicción del modelo',
                        yaxis_title='Observación real',
                        margin={'t': 30}
                    )
                )
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '20px',
                'borderRadius': '8px',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.H4("Variables más relevantes", style={'color': colors['primary']}),
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
                        yaxis_title='Variable',
                        margin={'t': 30}
                    )
                )
            ])
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
