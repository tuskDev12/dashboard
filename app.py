import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff

app = dash.Dash(__name__)

# Datos de los modelos seleccionados
model_data = {
    "Random Forest": {
        "accuracy": 0.85,
        "precision_rain": 0.77,
        "recall_rain": 0.45,
        "false_alarms": 1279,
        "missed_rain": 5216,
        "conf_matrix": [[31854, 1279], [5216, 4309]],
        "color": "#4CAF50"  # Verde
    },
    "Árbol de Decisión": {
        "accuracy": 0.79,
        "precision_rain": 0.52,
        "recall_rain": 0.77,
        "false_alarms": 5608,
        "missed_rain": 1850,
        "conf_matrix": [[21972, 5608], [1850, 6119]],
        "color": "#FF9800"  # Naranja
    }
}

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Comparación de Modelos Predictivos de Lluvia", style={'textAlign': 'center', 'color': '#333'}),
    
    html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}, children=[
        html.H3("¿Por qué estos dos modelos?", style={'color': '#333'}),
        html.P("Hemos seleccionado los dos modelos que mejor se complementan para predecir lluvia:"),
        html.Ul([
            html.Li("🌳 Árbol de Decisión: Detecta mejor los días de lluvia reales (menos días de lluvia pasan desapercibidos)"),
            html.Li("🌲 Random Forest: Es más preciso en general y tiene menos falsas alarmas"),
            html.Li("💡 Juntos ofrecen un panorama más completo para la toma de decisiones")
        ])
    ]),
    
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '30px'}, children=[
        html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
            html.H3("Efectividad General", style={'textAlign': 'center'}),
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Bar(
                            name=model_name,
                            x=['Exactitud', 'Precisión (Lluvia)', 'Detección (Lluvia)'],
                            y=[data['accuracy'], data['precision_rain'], data['recall_rain']],
                            marker_color=data['color']
                        ) for model_name, data in model_data.items()
                    ],
                    layout=go.Layout(
                        barmode='group',
                        plot_bgcolor='white',
                        yaxis={'tickformat': ',.0%', 'range': [0, 1]}
                    )
                )
            )
        ]),
        
        html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
            html.H3("Errores Importantes", style={'textAlign': 'center'}),
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Bar(
                            name=model_name,
                            x=['Falsas Alarmas', 'Lluvias No Detectadas'],
                            y=[data['false_alarms'], data['missed_rain']],
                            marker_color=data['color']
                        ) for model_name, data in model_data.items()
                    ],
                    layout=go.Layout(
                        barmode='group',
                        plot_bgcolor='white'
                    )
                )
            )
        ])
    ]),
    
    html.Div(style={'marginBottom': '30px'}, children=[
        html.H3("¿Cómo interpretar estos resultados?", style={'color': '#333'}),
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '8px'}, children=[
                html.H4("Cuando usar Random Forest", style={'color': '#2E7D32'}),
                html.Ul([
                    html.Li("✅ Para alertas públicas donde las falsas alarmas son costosas"),
                    html.Li("✅ Cuando la precisión general es más importante"),
                    html.Li("🔍 Ejemplo: Planificación de eventos al aire libre")
                ])
            ]),
            html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': '#fff3e0', 'padding': '15px', 'borderRadius': '8px'}, children=[
                html.H4("Cuando usar Árbol de Decisión", style={'color': '#E65100'}),
                html.Ul([
                    html.Li("✅ Para sistemas de alerta temprana de lluvias"),
                    html.Li("✅ Cuando es crucial no pasar por alto días de lluvia"),
                    html.Li("🔍 Ejemplo: Agricultura o prevención de inundaciones")
                ])
            ])
        ])
    ]),
    
    html.Div(style={'marginBottom': '20px'}, children=[
        html.H3("Visualización de Aciertos y Errores", style={'textAlign': 'center'}),
        dcc.Tabs(id='model-tabs', value='Random Forest', children=[
            dcc.Tab(label='Random Forest', value='Random Forest', style={'fontWeight': 'bold'}),
            dcc.Tab(label='Árbol de Decisión', value='Árbol de Decisión', style={'fontWeight': 'bold'})
        ]),
        html.Div(id='tabs-content')
    ]),
    
    html.Div(style={'backgroundColor': '#e3f2fd', 'padding': '20px', 'borderRadius': '10px'}, children=[
        html.H3("¿Cómo pueden complementarse?", style={'color': '#0D47A1'}),
        html.P("Estos modelos trabajan mejor juntos que por separado:"),
        html.Ol([
            html.Li("Usar el Árbol de Decisión para detectar posibles días de lluvia"),
            html.Li("Pasar esos días por el Random Forest para reducir falsas alarmas"),
            html.Li("Tomar acciones basadas en la intersección de ambas predicciones")
        ]),
        html.P("Esta estrategia combinada aprovecha lo mejor de ambos modelos.")
    ])
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('model-tabs', 'value')
)
def render_content(tab):
    data = model_data[tab]
    return html.Div([
        dcc.Graph(
            figure=ff.create_annotated_heatmap(
                z=data['conf_matrix'],
                x=['Predicción: No lluvia', 'Predicción: Lluvia'],
                y=['Real: No lluvia', 'Real: Lluvia'],
                colorscale=[[0, '#f5f5f5'], [1, data['color']]],
                annotation_text=data['conf_matrix'],
                hoverinfo='z'
            ).update_layout(
                title=f"Matriz de Confusión - {tab}",
                xaxis_title='Predicción del Modelo',
                yaxis_title='Realidad'
            )
        ),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'}, children=[
            html.Div(style={'textAlign': 'center'}, children=[
                html.P("Días correctamente predichos", style={'fontWeight': 'bold'}),
                html.P(f"{data['conf_matrix'][0][0]:,} sin lluvia", style={'color': '#4CAF50'}),
                html.P(f"{data['conf_matrix'][1][1]:,} con lluvia", style={'color': '#4CAF50'})
            ]),
            html.Div(style={'textAlign': 'center'}, children=[
                html.P("Errores importantes", style={'fontWeight': 'bold'}),
                html.P(f"{data['false_alarms']:,} falsas alarmas", style={'color': '#F44336'}),
                html.P(f"{data['missed_rain']:,} lluvias no detectadas", style={'color': '#F44336'})
            ])
        ])
    ])

server = app.server  # 👈 ESTO ES CLAVE

if __name__ == '__main__':
    app.run_server(debug=True)
