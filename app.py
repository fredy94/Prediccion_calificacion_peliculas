import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------- CARGA DE ARCHIVOS -------------------

final_model = tf.keras.models.load_model("modelo_final.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

columnas_modelo = pd.read_csv("columnas_modelo.csv")["0"].tolist()

with open("y_range.pkl", "rb") as f:
    y_range = pickle.load(f)
    y_min, y_max = y_range["y_min"], y_range["y_max"]

df_actor_stats = pd.read_csv("df_actor_stats.csv")
df_actriz_stats = pd.read_csv("df_actriz_stats.csv")
df_director_stats = pd.read_csv("df_director_stats.csv")
df_escritor_stats = pd.read_csv("df_escritor_stats.csv")
df_productor_stats = pd.read_csv("df_productor_stats.csv")

personas_dict = {
    "actor": df_actor_stats,
    "actriz": df_actriz_stats,
    "director": df_director_stats,
    "escritor": df_escritor_stats,
    "productor": df_productor_stats
}

# ------------------- FUNCIONES -------------------

def preparar_input_desde_nombres(input_usuario, columnas_modelo, personas_dict):
    datos = {col: 0 for col in columnas_modelo}

    for var in [
        'start_Year', 'runtime_Minutes', 'Presupuesto', 'Ingresos',
        'popularity_tmdb', 'Número_de_votos_imdb',
        'Other_Nominaciones', 'Other_Win',
        'Score Ponderado ', 'Score log'
    ]:
        if var in input_usuario:
            datos[var] = input_usuario[var]

    for genero in input_usuario.get("generos", []):
        col = f"genero_{genero}"
        if col in datos:
            datos[col] = 1

    for campo in ["idioma", "pais", "productora"]:
        valor = input_usuario.get(campo, None)
        if valor:
            col = f"{campo}_{valor}"
            if col in datos:
                datos[col] = 1

    for rol in ["actor", "actriz", "director", "escritor", "productor"]:
        nombres = input_usuario.get(rol + "es", [])
        if not isinstance(nombres, list):
            nombres = [nombres]

        df_personas = personas_dict.get(rol, None)
        if df_personas is not None:
            df_filtrado = df_personas[df_personas["primaryName"].isin(nombres)]
            if not df_filtrado.empty:
                for stat in ["score_mean", "score_max", "score_std", "ganador_oscar_prom", "nominado_oscar_prom"]:
                    colname = f"{rol}_{stat}"
                    datos[colname] = df_filtrado[colname].mean()
                datos[f"tiene_{rol}"] = 1
            else:
                datos[f"tiene_{rol}"] = 0
        else:
            datos[f"tiene_{rol}"] = 0

    return pd.DataFrame([datos])


def alinear_columnas_para_modelo(df_input, columnas_modelo):
    for col in columnas_modelo:
        if col not in df_input:
            df_input[col] = 0
    return df_input[columnas_modelo]


def predecir_calificacion_pelicula(input_usuario, columnas_modelo, personas_dict, scaler, modelo, y_min=None, y_max=None):
    df_input = preparar_input_desde_nombres(input_usuario, columnas_modelo, personas_dict)
    df_input = alinear_columnas_para_modelo(df_input, columnas_modelo)

    df_input_scaled = df_input.copy()
    df_input_scaled[scaler.feature_names_in_] = scaler.transform(df_input_scaled[scaler.feature_names_in_])

    pred = modelo.predict(df_input_scaled, verbose=0)

    if y_min is not None and y_max is not None:
        pred_real = pred[0][0] * (y_max - y_min) + y_min
    else:
        pred_real = pred[0][0]

    return float(round(pred_real, 2))

# ------------------- INTERFAZ STREAMLIT -------------------

st.set_page_config(page_title="Predicción IMDb", page_icon="🎬", layout="wide")

st.markdown("<h1 style='text-align:center; color:#FF5733;'>🎬 Predicción de Calificación IMDb</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Datos de entrada")
    start_year = st.number_input("Año de estreno", min_value=1900, max_value=2100, value=2024)
    runtime = st.number_input("Duración (minutos)", min_value=1, value=120)
    presupuesto = st.number_input("Presupuesto (USD)", min_value=0, value=0)
    ingresos = st.number_input("Ingresos (USD)", min_value=0, value=0)
    popularidad = st.number_input("Popularidad TMDB", min_value=0, max_value=100, value=50)
    votos = st.number_input("Número de votos IMDb", min_value=0, value=1000)

    st.subheader("Categorías")
    generos = st.multiselect("Géneros", ["Action", "Adventure", "Drama", "Comedy", "Animation"])
    idioma = st.text_input("Idioma original", "English")
    pais = st.text_input("País", "USA")
    productora = st.text_input("Productora", "Warner Bros")

    st.subheader("Personas")
    actores = st.text_area("Actores (coma)", "Leonardo DiCaprio").split(",")
    actrices = st.text_area("Actrices (coma)", "Margot Robbie").split(",")
    directores = st.text_area("Directores (coma)", "Christopher Nolan").split(",")
    escritores = st.text_area("Escritores (coma)", "Quentin Tarantino").split(",")
    productores = st.text_area("Productores (coma)", "David Heyman").split(",")

if st.sidebar.button("🚀 Predecir"):
    input_usuario = {
        "start_Year": start_year,
        "runtime_Minutes": runtime,
        "Presupuesto": presupuesto,
        "Ingresos": ingresos,
        "popularity_tmdb": popularidad,
        "Número_de_votos_imdb": votos,
        "Other_Nominaciones": 0,
        "Other_Win": 0,
        "Score Ponderado ": 0,
        "Score log": 0,
        "generos": generos,
        "idioma": idioma,
        "pais": pais,
        "productora": productora,
        "actores": [a.strip() for a in actores if a.strip()],
        "actrices": [a.strip() for a in actrices if a.strip()],
        "directores": [d.strip() for d in directores if d.strip()],
        "escritores": [e.strip() for e in escritores if e.strip()],
        "productores": [p.strip() for p in productores if p.strip()]
    }

    calificacion_predicha = predecir_calificacion_pelicula(
        input_usuario,
        columnas_modelo=columnas_modelo,
        personas_dict=personas_dict,
        scaler=scaler,
        modelo=final_model,
        y_min=y_min,
        y_max=y_max
    )

    tab1, tab2 = st.tabs(["📊 Predicción", "ℹ️ Detalles"])

    with tab1:
        st.metric("⭐ Calificación IMDb esperada", f"{calificacion_predicha} / 10")
            
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calificacion_predicha,
            title={'text': "Predicción IMDb (0-10)"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "red"},
                    {'range': [5, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "green"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### 🎥 Datos ingresados")
        st.json(input_usuario)
