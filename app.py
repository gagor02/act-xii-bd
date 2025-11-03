"""
Purchase Predictor Web Application
Flask + Dash implementation - Logistic Regression with PostgreSQL
"""
from flask import Flask
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
import joblib
import numpy as np
import psycopg2
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'dbname': 'purchase_predictor_db9b',
    'user': 'postgres',  
    'password': '0322103724',  
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"⚠️ Error conectando a la base de datos: {e}")
        print("La aplicación funcionará sin guardar predicciones.")
        return None

def save_prediction(gender, age, salary, predicted_purchase, probability):
    """Save prediction to database"""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions 
            (gender, age, estimated_salary, predicted_purchase, purchase_probability, prediction_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (gender, int(age), float(salary), bool(predicted_purchase), 
             float(probability), datetime.now())
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✓ Predicción guardada: {gender}, {age} años, ${salary:,.0f} -> {probability*100:.2f}%")
        return True
    except Exception as e:
        print(f"⚠️ Error guardando predicción: {e}")
        return False

# Load the trained model, scaler, and encoder
model = joblib.load('modelo_regresion_logistica.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server, url_base_pathname='/')

# App layout con diseño Flex
app.layout = html.Div([
    html.Div([
        html.H1("🛒 Predictor de Compra", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Modelo de Regresión Logística - Con Base de Datos", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
    ]),
    
    # Contenedor principal en flex
    html.Div([
        # Columna izquierda - Formulario
        html.Div([
            html.Label("Género:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='gender-input',
                options=[
                    {'label': '👨 Masculino', 'value': 'Male'},
                    {'label': '👩 Femenino', 'value': 'Female'}
                ],
                value='Male',
                style={'marginBottom': '15px'}
            ),
            
            html.Label("Edad:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Input(
                id='age-input',
                type='number',
                placeholder='Ingrese edad (ej: 30)',
                value=30,
                min=18,
                max=100,
                style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'marginBottom': '15px'}
            ),
            
            html.Label("Salario Estimado ($):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Input(
                id='salary-input',
                type='number',
                placeholder='Ingrese salario (ej: 50000)',
                value=50000,
                min=0,
                max=200000,
                step=1000,
                style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'marginBottom': '15px'}
            ),
            
            html.Button('Predecir Compra', id='predict-button', n_clicks=0,
                       style={
                           'width': '100%', 'padding': '12px', 
                           'backgroundColor': "#57e300", 
                           'color': 'white', 'border': 'none',
                           'borderRadius': '6px', 'fontSize': '16px', 
                           'cursor': 'pointer', 'fontWeight': 'bold',
                           'transition': '0.3s'
                       }),
        ], style={
            'flex': '1',
            'maxWidth': '380px',
            'backgroundColor': '#f8f9fa',
            'padding': '25px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'marginRight': '30px'
        }),

        # Columna derecha - Resultados y gráfico
        html.Div([
            html.Div(id='prediction-output', 
                    style={'textAlign': 'center', 'fontSize': '22px', 'fontWeight': 'bold',
                           'marginBottom': '10px', 'minHeight': '30px'}),
            html.Div(id='probability-output', 
                    style={'textAlign': 'center', 'fontSize': '18px',
                           'color': '#7f8c8d', 'marginBottom': '20px', 'minHeight': '30px'}),
            html.Div(id='db-status', 
                    style={'textAlign': 'center', 'fontSize': '12px', 'color': '#95a5a6', 
                           'marginBottom': '20px'}),
            dcc.Graph(id='probability-gauge', style={'height': '350px'})
        ], style={
            'flex': '1',
            'maxWidth': '600px',
            'backgroundColor': '#ffffff',
            'padding': '25px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'textAlign': 'center'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'flex-start',
        'flexWrap': 'wrap',
        'gap': '20px',
        'marginBottom': '40px'
    }),

    html.Div([
        html.P("📊 ACT XII - Implementación Web del Modelo de Regresión Logística con PostgreSQL", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'marginTop': '30px', 'fontSize': '12px'}),
    ])
], style={
    'padding': '30px',
    'fontFamily': 'Segoe UI, sans-serif',
    'backgroundColor': '#f4f6f7',
    'minHeight': '100vh'
})


# Callback for prediction
@app.callback(
    [Output('prediction-output', 'children'),
     Output('probability-output', 'children'),
     Output('probability-gauge', 'figure'),
     Output('db-status', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('gender-input', 'value'),
     State('age-input', 'value'),
     State('salary-input', 'value')]
)
def predict_purchase(n_clicks, gender, age, salary):
    if age is None or salary is None:
        return "⚠️ Por favor complete todos los campos", "", {}, ""
    
    # Encode gender (Male=1, Female=0)
    gender_enc = 1 if gender == 'Male' else 0
    
    # Prepare input
    X_input = np.array([[gender_enc, age, salary]])
    X_scaled = scaler.transform(X_input)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Results
    will_purchase = prediction == 1
    purchase_prob = probability[1] * 100
    
    # Save to database
    db_saved = save_prediction(gender, age, salary, will_purchase, probability[1])
    db_status = "✓ Guardado en base de datos" if db_saved else "⚠️ No se pudo guardar en BD (la app sigue funcionando)"
    
    # Output text
    if will_purchase:
        prediction_text = "✅ Predicción: COMPRARÁ"
        color = '#27ae60'
    else:
        prediction_text = "❌ Predicción: NO COMPRARÁ"
        color = '#e74c3c'
    
    probability_text = f"Probabilidad de compra: {purchase_prob:.2f}%"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=purchase_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilidad de Compra (%)", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    
    return prediction_text, probability_text, fig, db_status

if __name__ == '__main__':
    print("\n🚀 Starting Purchase Predictor Application...")
    print("📍 Access at: http://localhost:8051")
    print("🗄️  Database: purchase_predictor_db")
    
    # Test database connection
    conn = get_db_connection()
    if conn:
        print("✓ Conexión a PostgreSQL exitosa")
        conn.close()
    else:
        print("⚠️ No se pudo conectar a PostgreSQL (la app funcionará sin BD)")
    
    print("\n")
    app.run(debug=True, host='localhost', port=8051)