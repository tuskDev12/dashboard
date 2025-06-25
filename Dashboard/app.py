import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# === Cargar y preparar los datos reales ===
df = pd.read_csv('Starcraft 2.csv', sep=';')
df = df[df['APM'] > 0]  # Filtrar valores inválidos

# Convertir LeagueIndex a binario
df['HighLeague'] = df['LeagueIndex'].apply(lambda x: 1 if x > 4 else 0)

# Seleccionar variables predictoras
features = ['APM', 'NumberOfPACs', 'ActionLatency']
X = df[features]
y = df['HighLeague']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

# === Entrenar modelo ===
clf = DecisionTreeClassifier(max_depth=4, random_state=11, criterion="entropy")
clf.fit(X_train, y_train)

# === Predicciones ===
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# === Métricas ===
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_proba)

# === Curva ROC ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_curve_fig = go.Figure()
roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', name='ROC', line=dict(color='lime')))
roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
roc_curve_fig.update_layout(
    title='Curva ROC',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    plot_bgcolor='#1e1e2f',
    paper_bgcolor='#1e1e2f',
    font=dict(color='white'),
    margin=dict(t=50, b=40, l=50, r=30)
)

# === Gráfico 3D con Plotly ===
scatter_3d_fig = go.Figure()

for label in df['HighLeague'].unique():
    grupo = df[df['HighLeague'] == label]
    scatter_3d_fig.add_trace(go.Scatter3d(
        x=grupo['APM'],
        y=grupo['NumberOfPACs'],
        z=grupo['ActionLatency'],
        mode='markers',
        name=f"Liga {'Alta' if label == 1 else 'Baja'}",
        marker=dict(size=4)
    ))

scatter_3d_fig.update_layout(
    title='Distribución 3D - Jugadores Starcraft 2',
    scene=dict(
        xaxis_title='APM',
        yaxis_title='NumberOfPACs',
        zaxis_title='ActionLatency',
    ),
    plot_bgcolor='#1e1e2f',
    paper_bgcolor='#1e1e2f',
    font=dict(color='white'),
    margin=dict(t=40, b=30, l=30, r=30)
)

# === App DASH ===
app = dash.Dash(__name__)
app.title = "Dashboard Starcraft - Árbol de Decisión"

app.layout = html.Div(
    style={
        'backgroundColor': '#0f1125',
        'color': 'white',
        'minHeight': '100vh',
        'fontFamily': 'Segoe UI, sans-serif',
        'padding': '30px',
    },
    children=[
        html.H1("🎮 Dashboard de Clasificación de Ligas - Starcraft 2", style={
            "color": "#00e6e6",
            "textAlign": "center",
            "marginBottom": "40px"
        }),

        html.Div([
            # Columna izquierda (Métricas)
            html.Div([
                html.H4("📊 Métricas del Modelo", style={"color": "gold"}),
                html.Div([
                    html.P(f"Accuracy: {acc*100:.2f}%", style={"margin": "5px"}),
                    html.P(f"Precision: {prec:.2f}", style={"margin": "5px"}),
                    html.P(f"Recall (clase 0): {rec:.2f}", style={"margin": "5px"}),
                    html.P(f"F1-score: {f1:.2f}", style={"margin": "5px"}),
                    html.P(f"AUC: {auc:.2f}", style={"margin": "5px"}),
                ], style={
                    "padding": "15px",
                    "backgroundColor": "#1f2235",
                    "borderRadius": "10px",
                    "boxShadow": "0 0 10px #00bfff66"
                })
            ], style={"width": "30%", "padding": "10px"}),

            # Columna derecha (Gráfica ROC)
            html.Div([
                html.H4("📈 Curva ROC", style={"color": "#00ff88", "textAlign": "center"}),
                dcc.Graph(figure=roc_curve_fig, config={'displayModeBar': False})
            ], style={"width": "70%", "padding": "10px"}),

        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "space-between"
        }),

        # === Gráfico 3D ===
        html.Div([
            html.H4("🧠 Gráfico 3D de APM vs. PACs vs. Latencia", style={"color": "#00aaff", "textAlign": "center"}),
            dcc.Graph(figure=scatter_3d_fig, config={'displayModeBar': False})
        ], style={"paddingTop": "40px"})
    ]
)

if __name__ == '__main__':
    app.run(debug=True)

