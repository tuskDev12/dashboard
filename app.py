import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# --- Cargar dataset y preparar datos ---
df = pd.read_csv('Starcraft 2.csv', sep=',')

# Variables numéricas a usar para clustering
cols_numericas = ['APM', 'NumberOfPACs', 'ActionLatency']
df = df.dropna(subset=cols_numericas)
X = df[cols_numericas].copy()

# Escalar
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Usaremos 70% datos para "entrenar"
np.random.seed(42)
indices = np.random.choice(len(X_scaled), size=int(len(X_scaled)*0.7), replace=False)
X_train = X_scaled[indices]

# Rango eps para explorar métricas
eps_values = np.round(np.arange(0.1, 2.1, 0.1), 2)

# Precomputar métricas para todo rango eps
sil_scores = []
ch_scores = []
db_scores = []
num_clusters_list = []
noise_perc_list = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X_train)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    num_clusters_list.append(n_clusters)

    noise_perc = np.sum(clusters == -1) / len(clusters) * 100
    noise_perc_list.append(noise_perc)

    mask = clusters != -1
    if n_clusters > 1 and mask.any():
        sil = silhouette_score(X_train[mask], clusters[mask])
        ch = calinski_harabasz_score(X_train[mask], clusters[mask])
        db = davies_bouldin_score(X_train[mask], clusters[mask])
    else:
        sil = np.nan
        ch = np.nan
        db = np.nan

    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)

# --- Dash App ---
app = dash.Dash(__name__)
app.title = "Dashboard DBSCAN - Starcraft 2"

app.layout = html.Div(style={'backgroundColor': '#0f1125', 'color': 'white', 'padding': '20px', 'fontFamily': 'Arial'}, children=[
    html.H1("🎮 Dashboard DBSCAN - Parámetro eps", style={'textAlign': 'center', 'color': '#00e6e6'}),

    html.Div([
        html.Label("Selecciona valor de eps:", style={'fontSize': '20px'}),
        dcc.Slider(
            id='eps-slider',
            min=float(eps_values.min()),
            max=float(eps_values.max()),
            step=0.1,
            value=0.5,
            marks={float(eps): str(eps) for eps in eps_values},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'margin': '30px 40px'}),

    html.Div([
        html.Div([
            html.H4("Número de Clusters", style={'textAlign': 'center'}),
            html.H2(id='num-clusters', style={'color': '#00FF00', 'textAlign': 'center'})
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.H4("Porcentaje Ruido", style={'textAlign': 'center'}),
            html.H2(id='porc-ruido', style={'color': '#FF4500', 'textAlign': 'center'})
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.H4("Silhouette Score", style={'textAlign': 'center'}),
            html.H2(id='sil-score', style={'color': '#00FF00', 'textAlign': 'center'})
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'marginBottom': '40px'}),

    dcc.Graph(id='clusters-scatter', style={'height': '600px'}),

    dcc.Graph(id='metrics-history', style={'height': '350px'}),
])


@app.callback(
    Output('num-clusters', 'children'),
    Output('porc-ruido', 'children'),
    Output('sil-score', 'children'),
    Output('clusters-scatter', 'figure'),
    Output('metrics-history', 'figure'),
    Input('eps-slider', 'value')
)
def update_dashboard(eps):
    # DBSCAN con eps actual
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X_train)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    noise_perc = np.sum(clusters == -1) / len(clusters) * 100

    mask = clusters != -1
    if n_clusters > 1 and mask.any():
        sil = silhouette_score(X_train[mask], clusters[mask])
    else:
        sil = np.nan

    # KPIs
    num_clusters_text = str(n_clusters)
    porc_ruido_text = f"{noise_perc:.2f} %"
    sil_score_text = f"{sil:.3f}" if not np.isnan(sil) else "N/A"

    # Scatter clusters (2 primeras features)
    scatter_fig = go.Figure()
    for label in np.unique(clusters):
        pts = X_train[clusters == label]
        scatter_fig.add_trace(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode='markers',
            name='Ruido' if label == -1 else f'Cluster {label}',
            marker=dict(
                size=7,
                line=dict(width=0.5, color='DarkSlateGrey'),
                symbol='circle'
            )
        ))

    scatter_fig.update_layout(
        title=f'Clusters DBSCAN (eps={eps})',
        xaxis_title=cols_numericas[0],
        yaxis_title=cols_numericas[1],
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white'),
        legend=dict(title="Clusters")
    )

    # Gráfico historial métricas
    metrics_fig = go.Figure()
    metrics_fig.add_trace(go.Scatter(x=eps_values, y=sil_scores, mode='lines+markers', name='Silhouette'))
    metrics_fig.add_trace(go.Scatter(x=eps_values, y=ch_scores, mode='lines+markers', name='Calinski-Harabasz'))
    metrics_fig.add_trace(go.Scatter(x=eps_values, y=db_scores, mode='lines+markers', name='Davies-Bouldin'))

    metrics_fig.update_layout(
        title='Métricas según eps',
        xaxis_title='eps',
        yaxis_title='Valor métrica',
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#0f1125',
        font=dict(color='white'),
        legend=dict(x=0, y=1)
    )

    return num_clusters_text, porc_ruido_text, sil_score_text, scatter_fig, metrics_fig


if __name__ == '__main__':
    app.run_server(debug=True)
