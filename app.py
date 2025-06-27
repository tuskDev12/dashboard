import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# === Cargar y preparar datos ===
df = pd.read_csv("Starcraft 2.csv", sep=";")
df = df[df['APM'].astype(str).str.isnumeric()]
df['APM'] = df['APM'].astype(float)
df['NumberOfPACs'] = df['NumberOfPACs'].astype(float)
df['ActionLatency'] = df['ActionLatency'].astype(float)

X = df[['APM', 'NumberOfPACs', 'ActionLatency']].copy()

# === Precalcular métricas para distintos eps ===
eps_values = np.arange(12.6, 13.4, 0.1)
metrics = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X)
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    mask = clusters != -1
    if num_clusters > 1 and mask.any():
        sil = silhouette_score(X[mask], clusters[mask])
        ch = calinski_harabasz_score(X[mask], clusters[mask])
        db = davies_bouldin_score(X[mask], clusters[mask])
    else:
        sil, ch, db = np.nan, np.nan, np.nan

    metrics.append({
        'eps': round(eps, 1),
        'clusters': num_clusters,
        'silhouette': sil,
        'calinski': ch,
        'davies': db
    })

metrics_df = pd.DataFrame(metrics)

# === App DASH ===
app = dash.Dash(__name__)
server = app.server
app.title = "DBSCAN Dashboard"

app.layout = html.Div(style={'backgroundColor': '#111', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("DBSCAN Clustering - Métricas por Eps", style={'textAlign': 'center', 'color': '#00e6e6'}),

    html.Div([
        html.Label("Selecciona valor de eps:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='eps-selector',
            options=[{'label': f"eps = {row['eps']}", 'value': row['eps']} for _, row in metrics_df.iterrows()],
            value=12.6,
            style={'width': '300px', 'color': '#000'}
        )
    ], style={'marginBottom': '30px'}),

    html.Div(id='metricas-output', style={'marginBottom': '40px'}),

    html.Div([
        dcc.Graph(
            figure=go.Figure([
                go.Scatter(x=metrics_df['eps'], y=metrics_df['silhouette'], mode='lines+markers', name='Silhouette', line=dict(color='aqua'))
            ]).update_layout(title='Silhouette Score', xaxis_title='eps', yaxis_title='Score', paper_bgcolor='#111', plot_bgcolor='#111', font_color='white')
        )
    ]),

    html.Div([
        dcc.Graph(
            figure=go.Figure([
                go.Scatter(x=metrics_df['eps'], y=metrics_df['calinski'], mode='lines+markers', name='Calinski-Harabasz', line=dict(color='gold'))
            ]).update_layout(title='Índice Calinski-Harabasz', xaxis_title='eps', yaxis_title='Score', paper_bgcolor='#111', plot_bgcolor='#111', font_color='white')
        )
    ]),

    html.Div([
        dcc.Graph(
            figure=go.Figure([
                go.Scatter(x=metrics_df['eps'], y=metrics_df['davies'], mode='lines+markers', name='Davies-Bouldin', line=dict(color='magenta'))
            ]).update_layout(title='Índice Davies-Bouldin', xaxis_title='eps', yaxis_title='Score', paper_bgcolor='#111', plot_bgcolor='#111', font_color='white')
        )
    ])
])

@app.callback(
    Output('metricas-output', 'children'),
    Input('eps-selector', 'value')
)
def actualizar_metricas(eps):
    fila = metrics_df[metrics_df['eps'] == eps].iloc[0]
    return html.Div([
        html.H4(f"Resultados para eps = {eps}", style={"color": "#00ffcc"}),
        html.P(f"Clusters encontrados: {fila['clusters']}", style={"marginBottom": "5px"}),
        html.P(f"Silhouette Score: {fila['silhouette']:.3f}" if pd.notna(fila['silhouette']) else "Silhouette Score: N/A"),
        html.P(f"Calinski-Harabasz: {fila['calinski']:.2f}" if pd.notna(fila['calinski']) else "Calinski-Harabasz: N/A"),
        html.P(f"Davies-Bouldin: {fila['davies']:.3f}" if pd.notna(fila['davies']) else "Davies-Bouldin: N/A")
    ], style={
        "padding": "15px",
        "backgroundColor": "#1f2235",
        "borderRadius": "10px",
        "boxShadow": "0 0 10px #00bfff66",
        'width': '300px'
    })

if __name__ == '__main__':
    app.run_server(debug=True)
