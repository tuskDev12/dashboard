import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- Cargar datos ---
df = pd.read_csv('Starcraft 2.csv', sep=';')
df = df[df['APM'] > 0]  # filtrar inválidos
features = ['APM', 'NumberOfPACs', 'ActionLatency']
X = df[features].values

# --- Parámetros DBSCAN ---
eps_values = np.round(np.arange(12.6, 13.4, 0.1), 2)
min_samples = 5

# --- Precalcular clusters y métricas para cada eps ---
clusters_dict = {}
sil_scores = []
ch_scores = []
db_scores = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    clusters_dict[eps] = clusters
    
    mask = clusters != -1
    if len(set(clusters[mask])) > 1:
        sil_scores.append(silhouette_score(X[mask], clusters[mask]))
        ch_scores.append(calinski_harabasz_score(X[mask], clusters[mask]))
        db_scores.append(davies_bouldin_score(X[mask], clusters[mask]))
    else:
        sil_scores.append(np.nan)
        ch_scores.append(np.nan)
        db_scores.append(np.nan)

# --- Iniciar app Dash ---
app = dash.Dash(__name__)
server = app.server
app.title = "Dashboard DBSCAN Starcraft 2"

# --- Layout ---
app.layout = html.Div(
    style={
        'backgroundColor': '#0f1125',
        'color': 'white',
        'minHeight': '100vh',
        'fontFamily': 'Segoe UI, sans-serif',
        'padding': '20px',
    },
    children=[
        html.H1("🎮 Dashboard DBSCAN - Starcraft 2 Clustering", style={
            "color": "#00e6e6",
            "textAlign": "center",
            "marginBottom": "30px"
        }),
        
        html.Div([
            html.Label("Seleccione valor de eps:", style={"fontWeight": "bold"}),
            dcc.Slider(
                id='eps-slider',
                min=eps_values.min(),
                max=eps_values.max(),
                value=eps_values[0],
                marks={float(eps): str(eps) for eps in eps_values},
                step=None,
            ),
        ], style={"width": "80%", "margin": "auto", "paddingBottom": "30px"}),
        
        html.Div(id='metrics-output', style={
            "textAlign": "center",
            "fontSize": "18px",
            "marginBottom": "30px"
        }),
        
        dcc.Tabs([
            dcc.Tab(label='Silhouette Score', children=[
                dcc.Graph(id='silhouette-plot')
            ]),
            dcc.Tab(label='Calinski-Harabasz Index', children=[
                dcc.Graph(id='ch-plot')
            ]),
            dcc.Tab(label='Davies-Bouldin Index', children=[
                dcc.Graph(id='db-plot')
            ]),
        ]),
        
        html.H3("🧠 Visualización 3D de clusters DBSCAN", style={
            "textAlign": "center",
            "marginTop": "40px",
            "color": "#00aaff"
        }),
        
        dcc.Graph(id='cluster-3d-scatter')
    ]
)

# --- Callbacks ---
@app.callback(
    Output('metrics-output', 'children'),
    Output('silhouette-plot', 'figure'),
    Output('ch-plot', 'figure'),
    Output('db-plot', 'figure'),
    Output('cluster-3d-scatter', 'figure'),
    Input('eps-slider', 'value')
)
def update_dashboard(eps):
    clusters = clusters_dict[eps]
    mask = clusters != -1

    # Métricas para eps seleccionado
    sil = silhouette_score(X[mask], clusters[mask]) if np.sum(mask)>1 else np.nan
    ch = calinski_harabasz_score(X[mask], clusters[mask]) if np.sum(mask)>1 else np.nan
    db = davies_bouldin_score(X[mask], clusters[mask]) if np.sum(mask)>1 else np.nan
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    metrics_text = (
        f"Para eps = {eps}: Número de clusters = {num_clusters} | "
        f"Silhouette Score = {sil:.3f} | "
        f"Calinski-Harabasz Index = {ch:.0f} | "
        f"Davies-Bouldin Index = {db:.3f}"
    )
    
    # Gráfico Silhouette
    sil_fig = go.Figure()
    sil_fig.add_trace(go.Scatter(x=eps_values, y=sil_scores, mode='lines+markers', marker=dict(size=8)))
    sil_fig.add_vline(x=eps, line_dash="dash", line_color="cyan")
    sil_fig.update_layout(
        title='Silhouette Score según eps',
        xaxis_title='eps',
        yaxis_title='Silhouette Score',
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white')
    )

    # Gráfico Calinski-Harabasz
    ch_fig = go.Figure()
    ch_fig.add_trace(go.Scatter(x=eps_values, y=ch_scores, mode='lines+markers', marker=dict(size=8)))
    ch_fig.add_vline(x=eps, line_dash="dash", line_color="cyan")
    ch_fig.update_layout(
        title='Calinski-Harabasz Index según eps',
        xaxis_title='eps',
        yaxis_title='Calinski-Harabasz Index',
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white')
    )

    # Gráfico Davies-Bouldin
    db_fig = go.Figure()
    db_fig.add_trace(go.Scatter(x=eps_values, y=db_scores, mode='lines+markers', marker=dict(size=8)))
    db_fig.add_vline(x=eps, line_dash="dash", line_color="cyan")
    db_fig.update_layout(
        title='Davies-Bouldin Index según eps',
        xaxis_title='eps',
        yaxis_title='Davies-Bouldin Index',
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white')
    )

    # Gráfico 3D clusters
    df['Cluster'] = clusters
    fig3d = go.Figure()
    for cl in sorted(df['Cluster'].unique()):
        group = df[df['Cluster'] == cl]
        name = f"Ruido (-1)" if cl == -1 else f"Cluster {cl}"
        fig3d.add_trace(go.Scatter3d(
            x=group['APM'],
            y=group['NumberOfPACs'],
            z=group['ActionLatency'],
            mode='markers',
            marker=dict(size=4),
            name=name
        ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title='APM',
            yaxis_title='NumberOfPACs',
            zaxis_title='ActionLatency',
        ),
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white'),
        margin=dict(t=40, b=30, l=30, r=30)
    )
    
    return metrics_text, sil_fig, ch_fig, db_fig, fig3d

# --- Run server ---
if __name__ == '__main__':
    app.run_server(debug=True)
