
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
model_filename = 'modelo_desercion.pkl'
loaded_model = joblib.load(model_filename)

st.set_page_config(page_title="Predictor de Deserción Estudiantil", page_icon="🎓")

st.title("🎓 Predictor de Deserción Estudiantil")
st.markdown("--- ")
st.write("### Ingresa los datos del estudiante para predecir si desertará o no.")

# Crear los widgets de entrada para cada variable
# Usaremos los rangos observados en los datos sintéticos como referencia.

# Columna 1: Datos demográficos y académicos
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Información Personal")
    edad = st.slider('Edad', min_value=18, max_value=30, value=20, step=1)

    st.subheader("Rendimiento Académico")
    promedio = st.slider('Promedio (0-5)', min_value=0.0, max_value=5.0, value=3.2, step=0.1)
    materias_perdidas = st.slider('Materias Perdidas', min_value=0, max_value=5, value=0, step=1)

with col2:
    st.subheader("Compromiso Académico")
    asistencia = st.slider('Asistencia (%)', min_value=0, max_value=100, value=85, step=1)
    horas_estudio = st.slider('Horas de Estudio/Semana', min_value=0, max_value=40, value=10, step=1)
    uso_plataforma = st.slider('Uso de Plataforma (%)', min_value=0, max_value=100, value=70, step=1)

with col3:
    st.subheader("Factores Externos")
    nivel_socioeconomico = st.selectbox('Nivel Socioeconómico (1=Bajo, 5=Alto)', options=[1, 2, 3, 4, 5], index=2)
    trabaja_map = {'No': 0, 'Sí': 1}
    trabaja_input = st.selectbox('¿El estudiante trabaja?', options=list(trabaja_map.keys()), index=0)
    trabaja = trabaja_map[trabaja_input]

    acceso_internet_map = {'Sí': 1, 'No': 0}
    acceso_internet_input = st.selectbox('¿Tiene acceso a Internet?', options=list(acceso_internet_map.keys()), index=0)
    acceso_internet = acceso_internet_map[acceso_internet_input]


st.markdown("--- ")

# Botón para hacer la predicción
if st.button('Predecir Deserción'):
    # Crear un DataFrame con los inputs del usuario
    input_data = pd.DataFrame([[edad, promedio, asistencia, horas_estudio, uso_plataforma,
                                materias_perdidas, nivel_socioeconomico, trabaja, acceso_internet]],
                              columns=['edad', 'promedio', 'asistencia', 'horas_estudio', 'uso_plataforma',
                                       'materias_perdidas', 'nivel_socioeconomico', 'trabaja', 'acceso_internet'])

    # Realizar la predicción
    prediction = loaded_model.predict(input_data)
    prediction_proba = loaded_model.predict_proba(input_data)

    st.write("### Resultado de la Predicción:")
    if prediction[0] == 1:
        st.error('¡ALERTA! El modelo predice que el estudiante tiene una ALTA probabilidad de desertar. ⚠️')
        st.write(f"Probabilidad de Deserción: **{prediction_proba[0][1]*100:.2f}%**")
        st.warning("Se recomienda una intervención o apoyo temprano para este estudiante.")
    else:
        st.success('El modelo predice que el estudiante NO desertará. 👍')
        st.write(f"Probabilidad de NO Deserción: **{prediction_proba[0][0]*100:.2f}%**")
        st.info("El estudiante parece estar en buen camino, pero siempre es bueno mantener el seguimiento.")

# Añadir un pie de página
st.markdown('''
---
Creado para el proyecto de predicción de deserción estudiantil.
''')

