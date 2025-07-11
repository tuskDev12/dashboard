import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objects as go
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

# Paleta de colores moderna
colors = {
    'background': '#1a1a2e',
    'text': '#e6f7ff',
    'card_bg': '#16213e',
    'primary': '#4cc9f0',
    'secondary': '#a5b4fc',
    'success': '#4ade80',
    'danger': '#f472b6',
    'border': '#2a3a5e',
    'header_bg': '#0f3460',
    'highlight': '#3a86ff'
}

app.layout = html.Div(style={
    'backgroundColor': colors['background'],
    'padding': '20px',
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif',
    'color': colors['text']
}, children=[
    html.Div(style={
        'backgroundColor': colors['header_bg'],
        'padding': '25px',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'border': f'1px solid {colors["border"]}',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'
    }, children=[
        html.H1("Comparación de Modelos Predictivos de Lluvia", style={
            'textAlign': 'center',
            'color': colors['primary'],
            'marginBottom': '15px'
        }),
        html.P("Análisis entre Random Forest y Árbol de Decisión para predecir precipitaciones.", 
               style={
                   'textAlign': 'center',
                   'fontSize': '18px',
                   'marginBottom': '20px'
               }),
    ]),

    dcc.Tabs(
        id="tabs",
        value='tab-rf',
        children=[
            dcc.Tab(label='Random Forest', value='tab-rf'),
            dcc.Tab(label='Árbol de Decisión', value='tab-tree'),
            dcc.Tab(label='Comparación', value='tab-compare'),
        ],
        colors={
            "border": colors['border'],
            "primary": colors['primary'],
            "background": colors['card_bg']
        },
        style={'borderRadius': '8px'}
    ),

    html.Div(id='tabs-content', style={'marginTop': '20px'})
])

def create_metric_card(title, value, color, tooltip=None):
    return html.Div(style={
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': colors['card_bg'],
        'borderRadius': '8px',
        'margin': '10px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.2)',
        'border': f'1px solid {colors["border"]}',
        'flex': '1',
        'minWidth': '200px',
        'position': 'relative'
    }, children=[
        html.H4(title, style={'color': color, 'marginBottom': '10px'}),
        html.Div(value, style={
            'fontSize': '32px',
            'fontWeight': 'bold',
            'color': color,
            'marginBottom': '5px'
        }),
        html.Small(tooltip if tooltip else "", style={
            'color': colors['secondary'],
            'fontSize': '12px',
            'position': 'absolute',
            'bottom': '8px',
            'left': '0',
            'right': '0'
        })
    ])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-rf':
        # Lógica Random Forest (igual que antes)
        pass

    elif tab == 'tab-tree':
        # Lógica Árbol de Decisión (igual que antes)
        pass

    elif tab == 'tab-compare':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H3("Comparación Directa de Modelos", style={
                    'color': colors['highlight'],
                    'marginTop': '0'
                }),
                html.P("Comparación clara de métricas clave, errores y variables importantes.",
                       style={'color': colors['secondary']})
            ]),

            html.Div(style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '20px',
                'marginBottom': '25px'
            }, children=[
                html.Div(style={
                    'flex': '1',
                    'minWidth': '300px',
                    'backgroundColor': colors['card_bg'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'border': f'1px solid {colors["border"]}',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                }, children=[
                    html.H4("Métricas Clave", style={'color': colors['highlight']}),
                    dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    name='Random Forest',
                                    x=['Exactitud', 'Precisión Lluvia', 'Detección Lluvia'],
                                    y=[rf_metrics['accuracy'], rf_metrics['precision_1'], rf_metrics['recall_1']],
                                    marker_color=colors['primary'],
                                    hovertemplate='%{x}: %{y:.1%}<extra></extra>'
                                ),
                                go.Bar(
                                    name='Árbol de Decisión',
                                    x=['Exactitud', 'Precisión Lluvia', 'Detección Lluvia'],
                                    y=[tree_metrics['accuracy'], tree_metrics['precision_1'], tree_metrics['recall_1']],
                                    marker_color=colors['success'],
                                    hovertemplate='%{x}: %{y:.1%}<extra></extra>'
                                )
                            ],
                            layout=go.Layout(
                                barmode='group',
                                plot_bgcolor=colors['card_bg'],
                                paper_bgcolor=colors['card_bg'],
                                font={'color': colors['text']},
                                yaxis={'tickformat': ',.0%', 'range': [0, 1]}
                            )
                        )
                    )
                ]),

                html.Div(style={
                    'flex': '1',
                    'minWidth': '300px',
                    'backgroundColor': colors['card_bg'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'border': f'1px solid {colors["border"]}',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                }, children=[
                    html.H4("Errores Comparados", style={'color': colors['highlight']}),
                    dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    name='Random Forest',
                                    x=['Falsas Alarmas', 'Lluvias No Detectadas'],
                                    y=[conf_matrix_rf[0][1], conf_matrix_rf[1][0]],
                                    marker_color=colors['primary'],
                                    hovertemplate='%{x}: %{y:,}<extra></extra>'
                                ),
                                go.Bar(
                                    name='Árbol de Decisión',
                                    x=['Falsas Alarmas', 'Lluvias No Detectadas'],
                                    y=[conf_matrix_tree[0][1], conf_matrix_tree[1][0]],
                                    marker_color=colors['success'],
                                    hovertemplate='%{x}: %{y:,}<extra></extra>'
                                )
                            ],
                            layout=go.Layout(
                                barmode='group',
                                plot_bgcolor=colors['card_bg'],
                                paper_bgcolor=colors['card_bg'],
                                font={'color': colors['text']}
                            )
                        )
                    )
                ])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Variables Importantes", style={'color': colors['highlight']}),
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                name='Random Forest',
                                x=top_rf_importances,
                                y=top_rf_features,
                                orientation='h',
                                marker_color=colors['primary']
                            ),
                            go.Bar(
                                name='Árbol de Decisión',
                                x=top_tree_importances,
                                y=top_tree_features,
                                orientation='h',
                                marker_color=colors['success']
                            )
                        ],
                        layout=go.Layout(
                            barmode='group',
                            plot_bgcolor=colors['card_bg'],
                            paper_bgcolor=colors['card_bg'],
                            font={'color': colors['text']},
                            xaxis_title='Importancia',
                            yaxis_title='Variables',
                            margin={'l': 120}
                        )
                    )
                )
            ])
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
