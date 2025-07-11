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

# Paleta de colores oscura
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
        html.P("Análisis comparativo entre dos enfoques de machine learning para predecir precipitaciones", 
               style={
                   'textAlign': 'center',
                   'fontSize': '18px',
                   'marginBottom': '20px'
               }),
        html.Div(style={
            'display': 'flex',
            'justifyContent': 'center',
            'gap': '30px',
            'flexWrap': 'wrap'
        }, children=[
            html.Div(style={
                'padding': '15px 25px',
                'backgroundColor': '#2a3a5e',
                'borderRadius': '8px',
                'borderLeft': f'4px solid {colors["primary"]}'
            }, children=[
                html.H4("Random Forest", style={'color': colors['primary'], 'marginBottom': '5px'}),
                html.P("Modelo de ensamblado con múltiples árboles para mayor precisión", 
                       style={'fontSize': '14px', 'marginBottom': '0'})
            ]),
            html.Div(style={
                'padding': '15px 25px',
                'backgroundColor': '#2a3a5e',
                'borderRadius': '8px',
                'borderLeft': f'4px solid {colors["success"]}'
            }, children=[
                html.H4("Árbol de Decisión", style={'color': colors['success'], 'marginBottom': '5px'}),
                html.P("Modelo interpretativo basado en reglas de decisión", 
                       style={'fontSize': '14px', 'marginBottom': '0'})
            ])
        ])
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
                    'padding': '12px',
                    'border': f'1px solid {colors["border"]}',
                    'backgroundColor': colors['card_bg']
                },
                selected_style={
                    'backgroundColor': colors['primary'],
                    'color': colors['background'],
                    'border': f'1px solid {colors["primary"]}'
                }
            ),
            dcc.Tab(
                label='Árbol de Decisión',
                value='tab-tree',
                style={
                    'fontWeight': 'bold',
                    'padding': '12px',
                    'border': f'1px solid {colors["border"]}',
                    'backgroundColor': colors['card_bg']
                },
                selected_style={
                    'backgroundColor': colors['success'],
                    'color': colors['background'],
                    'border': f'1px solid {colors["success"]}'
                }
            ),
            dcc.Tab(
                label='Comparación',
                value='tab-compare',
                style={
                    'fontWeight': 'bold',
                    'padding': '12px',
                    'border': f'1px solid {colors["border"]}',
                    'backgroundColor': colors['card_bg']
                },
                selected_style={
                    'backgroundColor': colors['highlight'],
                    'color': colors['background'],
                    'border': f'1px solid {colors["highlight"]}'
                }
            )
        ],
        colors={
            "border": colors['border'],
            "primary": colors['primary'],
            "background": colors['card_bg']
        }
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

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
    prevent_initial_call=True
)
def render_content(tab):
    if tab == 'tab-rf':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H3("Random Forest: Rendimiento Detallado", style={
                    'color': colors['primary'],
                    'marginTop': '0',
                    'borderBottom': f'1px solid {colors["border"]}',
                    'paddingBottom': '12px'
                }),
                html.P("Este modelo combina múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste.", 
                       style={'color': colors['secondary']})
            ]),

            html.Div(style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'flexWrap': 'wrap',
                'marginBottom': '25px'
            }, children=[
                create_metric_card(
                    "Exactitud", 
                    f"{rf_metrics['accuracy']*100:.1f}%", 
                    colors['primary'],
                    "Porcentaje de predicciones correctas"
                ),
                create_metric_card(
                    "Precisión (No lluvia)", 
                    f"{rf_metrics['precision_0']*100:.1f}%", 
                    colors['secondary'],
                    "Cuando predice no lluvia, acierta el 86%"
                ),
                create_metric_card(
                    "Detección (Lluvia)", 
                    f"{rf_metrics['recall_1']*100:.1f}%", 
                    colors['danger'],
                    "Identifica el 45% de días de lluvia reales"
                )
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Matriz de Confusión", style={
                    'color': colors['primary'],
                    'marginTop': '0'
                }),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_rf,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Blues',
                        showscale=True,
                        hoverinfo='z',
                        annotation_text=[[f"{val:,}" for val in row] for row in conf_matrix_rf],
                        font_colors=['white']
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font={'color': colors['text']},
                        xaxis_title='Predicción del modelo',
                        yaxis_title='Observación real',
                        margin={'t': 40}
                    )
                ),
                html.Div(style={
                    'display': 'flex',
                    'justifyContent': 'space-around',
                    'flexWrap': 'wrap',
                    'marginTop': '20px'
                }, children=[
                    html.Div(style={
                        'flex': '1',
                        'minWidth': '250px',
                        'padding': '15px',
                        'margin': '10px',
                        'backgroundColor': '#1e3a8a',
                        'borderRadius': '8px',
                        'borderLeft': f'4px solid {colors["primary"]}'
                    }, children=[
                        html.H5("Aciertos", style={'color': colors['primary']}),
                        html.P(f"{conf_matrix_rf[0][0]:,} días sin lluvia correctos", 
                               style={'marginBottom': '5px'}),
                        html.P(f"{conf_matrix_rf[1][1]:,} días con lluvia correctos")
                    ]),
                    html.Div(style={
                        'flex': '1',
                        'minWidth': '250px',
                        'padding': '15px',
                        'margin': '10px',
                        'backgroundColor': '#831843',
                        'borderRadius': '8px',
                        'borderLeft': f'4px solid {colors["danger"]}'
                    }, children=[
                        html.H5("Errores", style={'color': colors['danger']}),
                        html.P(f"{conf_matrix_rf[0][1]:,} falsas alarmas"),
                        html.P(f"{conf_matrix_rf[1][0]:,} lluvias no detectadas", 
                               style={'marginBottom': '0'})
                    ])
                ])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Variables Clave", style={
                    'color': colors['primary'],
                    'marginTop': '0'
                }),
                html.P("Factores más influyentes en las predicciones del modelo:", 
                       style={'color': colors['secondary']}),
                dcc.Graph(
                    figure=go.Figure(
                        go.Bar(
                            x=top_rf_importances,
                            y=top_rf_features,
                            orientation='h',
                            marker_color=colors['primary'],
                            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>'
                        )
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font={'color': colors['text']},
                        xaxis_title='Importancia relativa',
                        yaxis_title='Variable climática',
                        margin={'t': 40, 'l': 150}
                    )
                )
            ])
        ])

    elif tab == 'tab-tree':
        return html.Div([
            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H3("Árbol de Decisión: Rendimiento Detallado", style={
                    'color': colors['success'],
                    'marginTop': '0',
                    'borderBottom': f'1px solid {colors["border"]}',
                    'paddingBottom': '12px'
                }),
                html.P("Modelo basado en reglas de decisión que segmenta los datos mediante preguntas secuenciales.", 
                       style={'color': colors['secondary']})
            ]),

            html.Div(style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'flexWrap': 'wrap',
                'marginBottom': '25px'
            }, children=[
                create_metric_card(
                    "Exactitud", 
                    f"{tree_metrics['accuracy']*100:.1f}%", 
                    colors['success'],
                    "Porcentaje de predicciones correctas"
                ),
                create_metric_card(
                    "Precisión (No lluvia)", 
                    f"{tree_metrics['precision_0']*100:.1f}%", 
                    colors['secondary'],
                    "Cuando predice no lluvia, acierta el 92%"
                ),
                create_metric_card(
                    "Detección (Lluvia)", 
                    f"{tree_metrics['recall_1']*100:.1f}%", 
                    colors['danger'],
                    "Identifica el 77% de días de lluvia reales"
                )
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Matriz de Confusión", style={
                    'color': colors['success'],
                    'marginTop': '0'
                }),
                dcc.Graph(
                    figure=ff.create_annotated_heatmap(
                        z=conf_matrix_tree,
                        x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                        y=['Real: No lluvia', 'Real: Lluvia'],
                        colorscale='Greens',
                        showscale=True,
                        hoverinfo='z',
                        annotation_text=[[f"{val:,}" for val in row] for row in conf_matrix_tree],
                        font_colors=['white']
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font={'color': colors['text']},
                        xaxis_title='Predicción del modelo',
                        yaxis_title='Observación real',
                        margin={'t': 40}
                    )
                ),
                html.Div(style={
                    'display': 'flex',
                    'justifyContent': 'space-around',
                    'flexWrap': 'wrap',
                    'marginTop': '20px'
                }, children=[
                    html.Div(style={
                        'flex': '1',
                        'minWidth': '250px',
                        'padding': '15px',
                        'margin': '10px',
                        'backgroundColor': '#1e3a8a',
                        'borderRadius': '8px',
                        'borderLeft': f'4px solid {colors["success"]}'
                    }, children=[
                        html.H5("Aciertos", style={'color': colors['success']}),
                        html.P(f"{conf_matrix_tree[0][0]:,} días sin lluvia correctos", 
                               style={'marginBottom': '5px'}),
                        html.P(f"{conf_matrix_tree[1][1]:,} días con lluvia correctos")
                    ]),
                    html.Div(style={
                        'flex': '1',
                        'minWidth': '250px',
                        'padding': '15px',
                        'margin': '10px',
                        'backgroundColor': '#831843',
                        'borderRadius': '8px',
                        'borderLeft': f'4px solid {colors["danger"]}'
                    }, children=[
                        html.H5("Errores", style={'color': colors['danger']}),
                        html.P(f"{conf_matrix_tree[0][1]:,} falsas alarmas"),
                        html.P(f"{conf_matrix_tree[1][0]:,} lluvias no detectadas", 
                               style={'marginBottom': '0'})
                    ])
                ])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Variables Clave", style={
                    'color': colors['success'],
                    'marginTop': '0'
                }),
                html.P("Factores más influyentes en las decisiones del árbol:", 
                       style={'color': colors['secondary']}),
                dcc.Graph(
                    figure=go.Figure(
                        go.Bar(
                            x=top_tree_importances,
                            y=top_tree_features,
                            orientation='h',
                            marker_color=colors['success'],
                            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>'
                        )
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font={'color': colors['text']},
                        xaxis_title='Importancia relativa',
                        yaxis_title='Variable climática',
                        margin={'t': 40, 'l': 150}
                    )
                )
            ])
        ])

    elif tab == 'tab-compare':
        # Contenido de comparación - ahora como una variable predefinida
        compare_content = html.Div([
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
                    'marginTop': '0',
                    'borderBottom': f'1px solid {colors["border"]}',
                    'paddingBottom': '12px'
                }),
                html.P("Análisis comparativo de las fortalezas y debilidades de cada enfoque", 
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
                    html.H4("Métricas Clave Comparadas", style={
                        'color': colors['highlight'],
                        'marginTop': '0'
                    }),
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
                                yaxis={'tickformat': ',.0%', 'range': [0, 1]},
                                margin={'t': 40}
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
                    html.H4("Errores Comparados", style={
                        'color': colors['highlight'],
                        'marginTop': '0'
                    }),
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
                                font={'color': colors['text']},
                                margin={'t': 40}
                            )
                        )
                    )
                ])
            ]),

            html.Div(style={
                'backgroundColor': colors['card_bg'],
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '25px',
                'border': f'1px solid {colors["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Variables Importantes", style={
                    'color': colors['highlight'],
                    'marginTop': '0'
                }),
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                name='Random Forest',
                                x=top_rf_importances,
                                y=top_rf_features,
                                orientation='h',
                                marker_color=colors['primary'],
                                hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>'
                            ),
                            go.Bar(
                                name='Árbol de Decisión',
                                x=top_tree_importances,
                                y=top_tree_features,
                                orientation='h',
                                marker_color=colors['success'],
                                hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>'
                            )
                        ],
                        layout=go.Layout(
                            barmode='group',
                            plot_bgcolor=colors['card_bg'],
                            paper_bgcolor=colors['card_bg'],
                            font={'color': colors['text']},
                            xaxis_title='Importancia relativa',
                            margin={'t': 40, 'l': 150}
                        )
                    )
                )
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
                    'backgroundColor': '#1e3a8a',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'borderLeft': f'4px solid {colors["primary"]}',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                }, children=[
                    html.H4("Cuándo usar Random Forest", style={
                        'color': colors['primary'],
                        'marginTop': '0'
                    }),
                    html.Ul([
                        html.Li("Cuando la precisión general es más importante"),
                        html.Li("Para reducir falsas alarmas en predicciones"),
                        html.Li("En sistemas donde la consistencia es clave"),
                        html.Li("Cuando se necesita mejor rendimiento con datos complejos")
                    ], style={'color': colors['text']})
                ]),
                
                html.Div(style={
                    'flex': '1',
                    'minWidth': '300px',
                    'backgroundColor': '#1e3a8a',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'borderLeft': f'4px solid {colors["success"]}',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                }, children=[
                    html.H4("Cuándo usar Árbol de Decisión", style={
                        'color': colors['success'],
                        'marginTop': '0'
                    }),
                    html.Ul([
                        html.Li("Cuando detectar lluvia es más importante que evitar falsas alarmas"),
                        html.Li("Para sistemas que requieren explicaciones simples"),
                        html.Li("Cuando la interpretabilidad del modelo es clave"),
                        html.Li("En implementaciones donde la velocidad es prioritaria")
                    ], style={'color': colors['text']})
                ])
            ]),

            html.Div(style={
                'backgroundColor': '#0f3460',
                'padding': '25px',
                'borderRadius': '10px',
                'border': f'1px solid {colors["highlight"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            }, children=[
                html.H4("Estrategia Recomendada", style={
                    'color': colors['highlight'],
                    'marginTop': '0'
                }),
                html.P("Para maximizar los beneficios de ambos modelos:", 
                       style={'color': colors['secondary']}),
                html.Ol([
                    html.Li("Usar el Árbol de Decisión como sistema de alerta temprana"),
                    html.Li("Confirmar las predicciones positivas con Random Forest"),
                    html.Li("Priorizar acciones basadas en la intersección de ambas predicciones"),
                    html.Li("Ajustar umbrales según el costo relativo de falsos positivos/negativos")
                ], style={'color': colors['text']})
            ])
        ])
        
        return compare_content

    return html.Div()  # Fallback por si acaso

if __name__ == '__main__':
    app.run_server(debug=True)
